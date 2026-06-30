import os
from pathlib import Path
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import argparse

from torch.utils.data import DataLoader, Dataset, Subset
import torch.utils.data.distributed
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast

from config.config import config, update_config

from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
from utils import exp_utils, train_utils, dist_utils
from dataset import dataset_utils, pulse_dataset, calopus_dataset
from func.train_anchor import train_epoch, validate
from model.vit import TimeSformer

import pandas as pd
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train hand reconstruction network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        "--eval", dest="eval", action="store_true",help="evaluate model")
    parser.add_argument(
        '--local_rank', default=-1, type=int, help="node rank for distributed training"
                                                   ", if localrank=-1 then don't use distributed training")
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='test')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # set device
    gpus = range(torch.cuda.device_count())
    distributed = torch.cuda.device_count() > 1
    device = torch.device('cuda') if len(gpus) > 0 else torch.device('cpu')

    # get model
    if (config.model.backbone_name == "swint") or \
    (config.model.backbone_name == "swinfpn") or (config.model.backbone_name == "dinov2"):
        model = ClipMatcher(config).to(device)
    else:
        model = TimeSformer(img_size=224, num_classes=3, num_frames=config.dataset.clip_num_frames, attention_type='divided_space_time', in_chans=1, pretrained=False).to(device)
    checkpoint = torch.load(config.model.spt_cpt_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print("_________________ LOADED FULL MODEL STATE DICT SUCCESSFULLY ____________________")
    for param in list(model.parameters()):
        param.requires_grad = False
    model.to(device)
    model.eval()

    df = pd.read_csv("/data/hert4942/pulsenet/calopus/gt_AS3_test.csv")
    testpids = df.pid.unique().tolist()
    testpids = [str(int(p)) for p in testpids]
    df = pd.read_csv("/data/hert4942/pulsenet/calopus/gt_AS3_val.csv")
    valpids = df.pid.unique().tolist()
    valpids = [str(int(p)) for p in valpids]
    df = pd.read_csv("/data/hert4942/pulsenet/calopus/gt_AS3_train.csv")
    trainpids = df.pid.unique().tolist()
    trainpids = [str(int(p)) for p in trainpids]

    pidlist = testpids + valpids + trainpids[:10]
    full_test_data = calopus_dataset.CalopusFullDst("/data/hert4942/pulsenet/calopus",
                                        img_size=(config.dataset.clip_size_fine,config.dataset.clip_size_fine),
                                        clip_num_frames=config.dataset.clip_num_frames_val,
                                        pidlist=pidlist)

    full_loader = DataLoader(full_test_data,
                            batch_size=config.test.batch_size, 
                            shuffle=False,
                            num_workers=int(config.workers),
                            pin_memory=True,
                            drop_last=False)

    class_mapping = {"AC":0, "FL": 1, "HC":2}
    time_meters = exp_utils.AverageMeters()
    for anat in ["AC", "HC", "FL"]:
        if config.query_used:
            query = calopus_dataset.get_query(anat, multiquery=config.multiquery)
        frame_paths = []; logits = []; clip_nums = []
        batch_idx = 0
        since=time.time()
        for batch_idx, sample in enumerate(tqdm(full_loader)):
            batch_start = time.time()
            with torch.no_grad():  # torch.no_grad makes no gradient calculations (same as
                sample = exp_utils.dict_to_cuda(sample)
                clips, img_paths, clip_num = sample['clip'],sample['img_path'], sample['clip_num']

                bs = clips.shape[0]

                # Save images from im loader to view it's what we expect
                # from PIL import Image
                # ts = clips.cpu().detach().shape[1]
                # testim = clips.cpu().detach().numpy()
                # testim = testim[0, 5,0,...]
                # testim = ((testim * 0.182) + 0.152) * 255
                # testim = np.asarray(testim).astype(np.uint8)
                # im = Image.fromarray(testim)
                # im.save(f"im_batch_{batch_idx}_{anat}.jpg")
                # qq = query.cpu().detach().numpy()
                # qq = qq[0,...]
                # qq = ((qq * 0.182) + 0.152) * 255
                # qq = np.asarray(qq).astype(np.uint8)
                # im = Image.fromarray(qq)
                # im.save(f"qq_batch_{batch_idx}_{anat}.jpg")
                # if batch_idx > 40:
                #     break

                after_data = time.time()
                time_meters.add_loss_value('Data time', after_data - batch_start)
                if config.query_used:
                    queries = torch.stack([query]*bs)
                    queries = queries.to(device)
                    preds = model(clips, queries, training=False, fix_backbone=True)
                else:
                    preds = model(clips)#, training=False, fix_backbone=True)
                preds = preds.detach().cpu().numpy()
                preds = preds.reshape(-1, preds.shape[-1])  # b,30,3,b -> bx30,3 or b,30 -> b,30 np.array
                # ^ (works for multi-class and single-class output?? maybe)

                time_meters.add_loss_value('Prediction time', time.time() - after_data)

                img_paths = zip(*img_paths)  # transpose this so shape = (batch,cliplen)
                frame_paths += img_paths  # list of tuples shape: (batches)
                # each element is a tuple of len=cliplen
                # need to flatten later to make shape (batches*cliplen)
                clip_num = clip_num.detach().cpu().numpy()
                clip_num = np.tile(clip_num, (config.dataset.clip_num_frames,1)).transpose().tolist()
                clip_nums += clip_num

                logits += list(preds)
                # type(logits)=list, len(logits)=(batches*cliplen), each element: np.array.shape:(3,) (float)

            time_meters.add_loss_value('Batch time', time.time() - batch_start)
            if batch_idx % config.print_freq == 0:  
                print('Time: data {data_time:.3f}s, pred {recon_time:.3f}s, all {batch_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                    data_time=time_meters.average_meters['Data time'].val,
                    recon_time=time_meters.average_meters['Prediction time'].val,
                    batch_time=time_meters.average_meters['Batch time'].val,
                    batch_time_avg=time_meters.average_meters['Batch time'].avg
                ), flush=True)
        torch.cuda.empty_cache()


        time_elapsed = time.time() - since
        print('Inference complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


        # save results as pandas csv
        frame_paths = [x for xs in frame_paths for x in xs]  # flatten the list of tuples
        clip_nums = [x for xs in clip_nums for x in xs]  # flatten the list of tuples
        pdf = pd.DataFrame({'frame': frame_paths, 'logits1': logits, 'clip_nums': clip_nums})
        if len(logits[0]) == 3:# logits[0] is either a np.array[3] or an np.float?
            pdf[list(class_mapping.keys())] = pd.DataFrame(pdf['logits1'].tolist(), index= pdf.index)
        elif len(logits[0]) == 2:
            pdf[["start_logits", "end_logits"]] = pd.DataFrame(pdf['logits1'].tolist(), index= pdf.index)
        else:
            pdf["logits"] = pd.DataFrame(pdf['logits1'].tolist(), index= pdf.index)
        pdf.drop('logits1', axis=1, inplace=True)
        pdf.sort_values(by=['frame'], inplace=True)
        save_dir = Path(config.model.spt_cpt_path).expanduser().resolve().parent / "calopus"
        save_dir.mkdir(exist_ok=True,parents=True)
        pdf.to_csv(save_dir / f"prediction_logits_calopus_all_{Path(config.model.spt_cpt_path).stem}_{anat}.csv", index=False)

    print(Path(save_dir / f"prediction_logits_calopus_all_{Path(config.model.spt_cpt_path).stem}_{anat}.csv"))

if __name__ == '__main__':
    main()
