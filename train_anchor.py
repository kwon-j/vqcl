import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import argparse
import time

from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset,WeightedRandomSampler
import torch.utils.data.distributed
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from utils.balanced_sampler import get_sample_weights_balanced_dloader

from config.config import config, update_config

from pathlib import Path
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
from model.vit import TimeSformer
from utils import exp_utils, train_utils
from dataset import pulse_dataset, calopus_dataset
from func.train_anchor import train_epoch, validate

import transformers
import wandb

this_script_dir = Path(__file__).parent.resolve()

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
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
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

    wandb_groupname = config.exp_name
    wandb_proj_name = config.exp_group
    wandb_runname =  'lr{}_bs{}_multq{}_frames{}'.\
        format(config.train.lr, config.train.batch_size, config.multiquery, config.dataset.clip_num_frames)
    wandb_runname += "_" + time.strftime("%Y_%m_%d-%H_%M_%S")

    wandb_run = wandb.init(project=wandb_proj_name, group=wandb_groupname, name=wandb_runname)#, name='smooth-puddle-94', resume=True)

    wandb.config.update({
        "exp_name": config.exp_name,
        "batch_size": config.train.batch_size,
        "total_epochs": config.train.total_epochs,
        "lr": config.train.lr,
        "weight_decay": config.train.weight_decay,
        "loss_weight_bbox_giou": config.loss.weight_bbox_giou,
        "loss_prob_bce_weight": config.loss.prob_bce_weight,
        "model_num_transformer": config.model.num_transformer,
        "model_resolution_transformer": config.model.resolution_transformer,
        "model_window_transformer": config.model.window_transformer,
    })

    # _______ GET MODEL ________
    if (config.model.backbone_name == "swint") or \
    (config.model.backbone_name == "swinfpn") or (config.model.backbone_name == "dinov2"):
        model = ClipMatcher(config).to(device)
    elif config.model.backbone_name == "timesformer":
        # model_file = this_script_dir/'model/TimeSformer_divST_64_224_SSv2.pyth'
        model = TimeSformer(img_size=224, num_classes=3, num_frames=config.dataset.clip_num_frames, attention_type='divided_space_time',  pretrained_model=str(config.model.spt_cpt_path), in_chans=1).to(device)
        print("_________ LOADED PRETRAINED WEIGHTS SUCCESSFULLY ________")
    else:
        print("______ INVALID MODEL NAME _____ IDK WHAT TO DO")
        raise(TypeError)

    if config.model.fix_backbone:
        assert ~(config.query_sim_loss), "if fix backbone, query_sim_loss won't do anything"
    # get optimizer
    optimizer = train_utils.get_optimizer(config, model)
    # schedular = train_utils.get_schedular(config, optimizer)
    # schedular = transformers.get_linear_schedule_with_warmup(optimizer,
    #                                                          num_warmup_steps=config.train.schedular_warmup_iter,
    #                                                          num_training_steps=config.train.total_iteration)
    schedular = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.train.schedular_warmup_iter,
                                                             num_training_steps=config.train.total_iteration)

    scaler = torch.cuda.amp.GradScaler()

    best_iou, best_prob = 0.0, 0.0
    ep_resume = None
    if config.train.resume:
        try:
            model, optimizer_, schedular_, scaler_, ep_resume_, best_iou, best_prob_ = train_utils.resume_training(
                                                                                model, optimizer, schedular, scaler, 
                                                                                output_dir,
                                                                                cpt_name=config.model.spt_cpt_path)
            print('LR after resume {}'.format(optimizer.param_groups[0]['lr']))
        except:
            print('Resume failed')

    # distributed training
    ddp = False
    local_rank = 0
    # model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        device_num = len(device_ids)
        # ddp = True

    # get dataset and dataloader    
    # train_data = dataset_utils.get_dataset(config, split='train')

    # If using gt_train.csv/_val.csv then before_query mask wrong so use full csv and split via torch subset
    # dst = pulse_dataset.PulseClipsDst(labels_csv="jong_gt2_timethresh.csv", transform=True,
    #                                    multiquery=config.multiquery, rm_pidlist=)
    # N = len(dst)
    # indices = np.random.permutation(np.arange(N))
    # tr_idx = indices[:int(0.7*N)]
    # val_idx = indices[int(0.7 * N):int(0.85 * N)]
    # # ts_idx = indices[int(0.85 * N):]
    # train_data = Subset(dst, tr_idx)
    # dst = pulse_dataset.PulseClipsDst(transform=False,multiquery=config.multiquery)
    # val_data = Subset(dst, val_idx)
    imsz = config.dataset.clip_size_fine

    cal_train_dsts = []
    cal_val_dsts = []
    pul_train_dsts = []
    pul_val_dsts = []
    curric_dict={"same_query":{"samequery":True},
                 "same_query_augment":{"samequery":True,"query_augment":True},
                 "std_query":{},
                 "std_query_augment":{"query_augment":True},
                 }
    for curric in ["same_query", "same_query_augment", "std_query", "std_query_augment"]:
        #,"varied_query_augment"]:
        cal_train_dsts.append(calopus_dataset.CalopusClipsDst(img_shape=(imsz,imsz),
                                                        clip_num_frames=config.dataset.clip_num_frames,
                                                        multi_class_out=config.multi_class_out,
                                                        interval=(1,4),  # used to be 1
                                                        **curric_dict[curric]))

        cal_val_dsts.append(calopus_dataset.CalopusClipsDst(labels_csv="gt_AS3_clips_val.csv",
                                                            img_shape=(imsz,imsz),
                                                        clip_num_frames=config.dataset.clip_num_frames,
                                                        multi_class_out=config.multi_class_out,
                                                        **curric_dict[curric]))

        pul_train_dsts.append(pulse_dataset.PulseClipsDst(labels_csv="pulse_manual_anno_train.csv",
                                                          img_shape=(imsz,imsz),
                                                          clip_num_frames=config.dataset.clip_num_frames,
                                                          multi_class_out=config.multi_class_out,
                                                          interval=(2,9),  # was 5 temporal augment
                                                          **curric_dict[curric]))
        pul_val_dsts.append(pulse_dataset.PulseClipsDst(labels_csv="pulse_manual_anno_val.csv",
                                                          img_shape=(imsz,imsz),
                                                          clip_num_frames=config.dataset.clip_num_frames,
                                                          multi_class_out=config.multi_class_out,
                                                          **curric_dict[curric]))

    dfct = cal_train_dsts[0].get_df()
    dfpt = pul_train_dsts[0].get_df()
    sample_weights = get_sample_weights_balanced_dloader(dfpt,dfct,0.7)  # oversample calopus 0.3,0.7

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    start_ep = ep_resume if ep_resume is not None else 0
    end_ep = config.train.total_epochs+1 #int(config.train.total_iteration / len(train_loader)) + 1
    # end_ep not used
    config.train.total_iteration = config.train.total_epochs * len(cal_train_dsts[0])
    
    schedular = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.train.schedular_warmup_iter,
                                                             num_training_steps=config.train.total_iteration)

    for i_c, curric in enumerate(["same_query", "same_query_augment", "std_query", "std_query_augment"]):#,"varied_query_augment"]):
        pulse_cal_dst = ConcatDataset([pul_train_dsts[i_c], cal_train_dsts[i_c]])
        pulse_cal_val_dst = ConcatDataset([pul_val_dsts[0],cal_val_dsts[0]])
        pulse_cal_std_val_dst = ConcatDataset([pul_val_dsts[2],cal_val_dsts[2]])
        train_loader = DataLoader(pulse_cal_dst,
                                batch_size=config.train.batch_size, 
                                # shuffle=True,
                                num_workers=int(config.workers), 
                                pin_memory=True, 
                                drop_last=True,
                                sampler=sampler)
        val_loader = DataLoader(pulse_cal_std_val_dst,
                            batch_size=config.test.batch_size, 
                            shuffle=False,
                            num_workers=int(config.workers), 
                            pin_memory=True, 
                            drop_last=False)
        val_sq_loader = DataLoader(pulse_cal_val_dst,
                            batch_size=config.test.batch_size, 
                            shuffle=False,
                            num_workers=int(config.workers), 
                            pin_memory=True, 
                            drop_last=False)

        # train
        for epoch in range(start_ep + i_c*config.train.epochs_curric,
                           start_ep+((i_c+1)*config.train.epochs_curric)):
            print(f"___________ epoch: {epoch} ____________")
            # train_sampler.set_epoch(epoch)
            train_epoch(config,
                        loader=train_loader,
                        model=model,
                        optimizer=optimizer,
                        schedular=schedular,
                        scaler=scaler,
                        epoch=epoch,
                        output_dir=output_dir,
                        device=device,
                        rank=local_rank,
                        ddp=ddp,
                        wandb_run=wandb_run
                        )
            torch.cuda.empty_cache()

            if local_rank == 0:
                train_utils.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'schedular': schedular.state_dict(),
                            'scaler': scaler.state_dict(),
                        }, 
                        checkpoint=output_dir, filename=f"cpt_epoch{epoch}.pth.tar")

            if epoch % 1 == 0:
                print('Doing validation...')
                prob_ = validate(config,
                                    loader=val_sq_loader,
                                    model=model,
                                    epoch=epoch,
                                    output_dir=output_dir,
                                    device=device,
                                    rank=local_rank,
                                    ddp=ddp,
                                    wandb_run=wandb_run,
                                    query_type="same_query",
                                    )
                torch.cuda.empty_cache()
                print('Doing validation on non-same query...')
                prob = validate(config,
                                    loader=val_loader,
                                    model=model,
                                    epoch=epoch,
                                    output_dir=output_dir,
                                    device=device,
                                    rank=local_rank,
                                    ddp=ddp,
                                    wandb_run=wandb_run,
                                    query_type="diff_query",
                                    )
                torch.cuda.empty_cache()
                # if iou > best_iou:
                #     best_iou = iou
                #     if local_rank == 0:
                #         train_utils.save_checkpoint(
                #         {
                #             'epoch': epoch + 1,
                #             'state_dict': model.state_dict(),
                #             'optimizer': optimizer.state_dict(),
                #             'schedular': schedular.state_dict(),
                #             'scaler': scaler.state_dict(),
                #             'best_iou': best_iou,
                #         }, 
                #         checkpoint=output_dir, filename="cpt_best_iou.pth.tar")

                if prob > best_prob:
                    best_prob = prob
                    if local_rank == 0:
                        train_utils.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'schedular': schedular.state_dict(),
                            'scaler': scaler.state_dict(),
                            'best_prob': best_prob,
                        }, 
                        checkpoint=output_dir, filename="cpt_best_prob.pth.tar")

                logger.info('Rank {}, best probability accuracy: {} (current {})'.format(local_rank, best_prob, prob))
            # dist.barrier()
            torch.cuda.empty_cache()

        train_utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedular': schedular.state_dict(),
                'scaler': scaler.state_dict(),
            }, 
            checkpoint=output_dir, filename=f"cpt_epoch{epoch}.pth.tar")
        # ______ LOAD BEST VAL WEIGHTS FOR SAME_QUERY? OKAY _______
        if config.train.load_best_after_curric & config.train.load_schedular_after_curric:
            model, optimizer, schedular, scaler, ep_resume, _, best_prob = train_utils.resume_training(
                                            model, optimizer, schedular, scaler, 
                                            output_dir,
                                            cpt_name="cpt_best_prob.pth.tar")
            print('LR after resume {}'.format(optimizer.param_groups[0]['lr']))
            print(f"_________ FINISHED {i_c}th CURRICULUM - LOADED PRETRAINED WEIGHTS SUCCESSFULLY ________")
        elif config.train.load_best_after_curric & ~(config.train.load_schedular_after_curric):
            model, optimizer_, schedular_, scaler_, ep_resume_, _, best_prob_ = train_utils.resume_training(
                                model, optimizer, schedular, scaler, 
                                output_dir,
                                cpt_name="cpt_best_prob.pth.tar")
            print('LR after resume {}'.format(optimizer.param_groups[0]['lr']))
            print(f"_________ FINISHED {i_c}th CURRICULUM - LOADED PRETRAINED WEIGHTS SUCCESSFULLY ________")


if __name__ == '__main__':
    main()
