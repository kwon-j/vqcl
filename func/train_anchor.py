import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
from torch.cuda.amp import autocast as autocast
import itertools
from utils import exp_utils, train_utils, loss_utils
import wandb
from einops import rearrange
from utils.loss_utils import GiouLoss, fast_auroc
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def train_epoch(config, loader, model, optimizer, schedular, scaler, epoch, output_dir, device, rank, wandb_run=None, ddp=True):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()
    prob_accuracy = torch.zeros(1)
    true_neg = torch.zeros(1)
    true_pos = torch.zeros(1)
    class_positives = torch.zeros(1)  # initialise: sometimes not calculated before wandb line runs
    train_utils.set_model_train(config, model, ddp)

    batch_end = time.time()
    for batch_idx, sample in enumerate(loader):
        iter_num = batch_idx + len(loader) * epoch

        # sample = {
        #     'clip': clip.float(),                           # [T,3,H,W]
        #     'clip_with_bbox': clip_with_bbox.float(),       # [T]
        #     'before_query': before_query.bool(),            # [T]
        #     'clip_bbox': clip_bbox.float().clamp(min=0.0, max=1.0),                 # [T,4]
        #     'query': query.float(),                         # [3,H2,W2] cropped query
        #     'clip_h': torch.tensor(clip_h),
        #     'clip_w': torch.tensor(clip_w),
        #     'query_frame': query_frame.float(),             # [3,H,W] entire query frame
        #     'query_frame_bbox': query_frame_bbox.float()    # [4]   (y_top, x_left, y_bot, x_ri)

        sample = exp_utils.dict_to_cuda(sample)
        # sample = dataset_utils.process_data(config, sample, iter=iter_num, split='train', device=device)     # normalize and data augmentations on GPU
        # # after a certain number of iterations augment the normalize clips, query, query frame
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        # reconstruction loss
        clips, queries = sample['clip'], sample['query']

        if config.query_used:
            preds = model(clips, queries, training=True, fix_backbone=config.model.fix_backbone, feat_out=config.query_sim_loss)
            # if config.query_sim_loss:
            #     preds, q_feat, clip_feat = preds
        else:
            preds = model(clips)

        time_meters.add_loss_value('Prediction time', time.time() - end)

        # preds = {
        #     'center': center,           # [b,t,N,2]
        #     'hw': hw,                   # [b,t,N,2]
        #     'bbox': bbox,               # [b,t,N,4]
        #     'prob': prob.squeeze(-1),   # [b,t,N]
        #     'anchor': anchors_xyxy      # [1,1,N,4]
        # }

        loss = loss_utils.get_pulse_losses(config, preds, sample)

        # losses, preds_top, sample = loss_utils.get_losses_with_anchor(config, preds, sample)
        total_loss = 0.0
        total_loss += loss.mean()
        loss_meters.add_loss_value("prob loss", loss.mean().detach().item())
        # for k, v in losses.items():
        #     if 'loss' in k:
        #         total_loss += losses[k.replace('loss_', 'weight_')] * v
        #         loss_meters.add_loss_value(k, v.detach().item())
        total_loss = total_loss / config.train.accumulation_step
        
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        total_loss.backward()
        if (batch_idx+1) % config.train.accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()
            schedular.step()

        if batch_idx % config.print_freq == 0:
            acc_calc_start_time = time.time()
            # msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
            #     'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, all {batch_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
            #     epoch, iter_num, rank,
            #     data_time=time_meters.average_meters['Data time'].val,
            #     recon_time=time_meters.average_meters['Prediction time'].val,
            #     batch_time=time_meters.average_meters['Batch time'].val,
            #     batch_time_avg=time_meters.average_meters['Batch time'].avg
            # )
            msg='______ Epoch {0}, Iter {1}: ____'.format(epoch, iter_num)

            gt_prob = sample['clip_with_bbox'].detach().cpu()      # [b,t]
            gt_bq = sample['before_query'].detach().cpu().bool()   # [b,t]
            gt_idx = gt_prob.bool()                                # [b,t]
            # gt_labels = torch.max(gt_prob.detach().cpu(),dim=-1)[1]
            # pp_labels = torch.max(preds.detach().cpu(),dim=-1)[1]
            if config.query_sim_loss:
                preds,_ = preds
            pp = preds.detach().cpu().squeeze(2)
            # # class_positives: same as true pos but taking max logits not on threshold
            # class_positives = (pp_labels[gt_idx] == gt_labels[gt_idx]).float().mean()  # accuracy if we mask only where gt is a positive
            # # num frames where max logits class is same class as the gt class/num frames that the gt for any anat is positive
            prob_accuracy = ((torch.sigmoid(pp[gt_bq]))>0.5) == gt_prob[gt_bq].bool()
            prob_accuracy = prob_accuracy.float().mean()
            true_neg = ((torch.sigmoid(pp[(gt_bq & ~gt_idx)]))>0.5) == gt_prob[(gt_bq & ~gt_idx)].bool()  # this looks wrong but is defo corrrect. can be <0.5 == ~gt_prob(gt_bq & ~gt_idx).bool()
            true_neg = true_neg.float().mean()
            true_pos = ((torch.sigmoid(pp[gt_idx]))>0.5) == gt_prob[gt_idx].bool()
            true_pos = true_pos.float().mean()  # of all actual yes, how many classfied yes?

            # My revised version:
            preds = ((torch.sigmoid(pp))>0.5) & gt_bq  # B,T bool
            tr_n = (~preds[gt_bq]) & (~gt_idx[gt_bq])  # this flattens to 1-D
            actual_negs = ~gt_idx[gt_bq]
            tnr = ((tr_n.sum().float()) / (actual_negs.sum().float() + 1e-6)).float().mean()
            all_prev = gt_prob.float().mean()
            prevalence=gt_idx.float().mean() # should be smaller becacuse 0.5->0
            # Ensure input tensors are boolean
            intersection = (preds & gt_idx).sum(dim=1)  # Sum over temporal dimension
            union = (preds | gt_idx).sum(dim=1)  # Sum over temporal dimension
            iou = (intersection.float() / (union.float() + 1e-6)).float().mean()  # IOU per batch element

            valid_indices = gt_bq.view(-1)
            try:
                sklearn_auroc = roc_auc_score(gt_idx.view(-1)[valid_indices].cpu().numpy(),
                                  pp.view(-1)[valid_indices].cpu().numpy())
            except:
                sklearn_auroc = 0.5

            # B, T = 2, 30
            # pred_logits = torch.randn(B, T)  # Random logits
            # gt_prob = torch.randint(0, 2, (B, T))  # Binary ground truth
            # gt_mask = torch.randint(0, 2, (B, T))  # Binary mask
            # auroc = fast_auroc(pred_logits, gt_prob, gt_mask, num_thresholds=10))

            loss_meters.add_loss_value("prob acc", prob_accuracy.item())
            loss_meters.add_loss_value("___old true neg", true_neg.item())
            loss_meters.add_loss_value("___ true pos", true_pos.item())
            loss_meters.add_loss_value("___ prevalence", prevalence.item())

            loss_meters.add_loss_value(" prevlnc_all", all_prev.item())
            loss_meters.add_loss_value("___JRev_ TNR", tnr.item())
            loss_meters.add_loss_value("___ iou", iou.item())
            loss_meters.add_loss_value("___tpr*Jtnr", (true_pos*tnr).item())
            # loss_meters.add_loss_value("___auroc", (auroc).item())
            loss_meters.add_loss_value("___skl auroc", sklearn_auroc)

            time_meters.add_loss_value('Accuracy time', time.time() - acc_calc_start_time)
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, all {batch_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].val,
                recon_time=time_meters.average_meters['Prediction time'].val,
                batch_time=time_meters.average_meters['Batch time'].val,
                batch_time_avg=time_meters.average_meters['Batch time'].avg
            )
            msg += f"acctime: {(time.time() - acc_calc_start_time):.4f}"
                
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)
        
        # if iter_num % config.vis_freq == 0 and rank == 0:
        #     vis_utils.vis_pred_clip(sample=sample,
        #                             iter_num=iter_num,
        #                             output_dir=output_dir,
        #                             subfolder='train')
        #     vis_utils.vis_pred_scores(sample=sample,
        #                             iter_num=iter_num,
        #                             output_dir=output_dir,
        #                             subfolder='train')

        batch_end = time.time()

        if rank == 0:
            wandb_log = {'Train/loss': total_loss.item(),
                        'Train/lr': optimizer.param_groups[0]['lr'],
                        "Train/accuracy": prob_accuracy.item(),
                        "Train/true neg": true_neg.item(),
                        "Train/true pos": true_pos.item(),
                        "Train/pos class acc": class_positives.item(),
                        "Train/prevalence": prevalence.item(),
                        "Train/true neg (rev)": tnr.item(),
                        "Train/tpr*tnr": (true_pos*tnr).item(),
                        "Train/iou": (iou).item(),
                        "Train/AUROC": sklearn_auroc,
                        "Epoch": epoch + (batch_idx / len(loader)),
                        }
            wandb_run.log(wandb_log)
        
        # dist.barrier()
        if batch_idx < 3:
            torch.cuda.empty_cache()



def validate(config, loader, model, epoch, output_dir, device, rank, wandb_run=None, ddp=True, query_type=None,):
    model.eval()
    metrics = {}

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            # if batch_idx % config.eval_vis_freq != 0:
            #     continue
            sample = exp_utils.dict_to_cuda(sample)
            # sample = dataset_utils.process_data(config, sample, split='val', device=device)     # normalize and data augmentations on GPU

            clips, queries = sample['clip'], sample['query']
            if config.query_used:
                preds = model(clips, queries, training=False, fix_backbone=config.model.fix_backbone)
            else:
                preds = model(clips)

            results = val_pulse_performance(config, preds, sample)
            try:
                for k, v in results.items():
                    if k in metrics.keys():
                        try:
                            metrics[k].append(v)
                        except:
                            print('1', k, v, metrics[k], batch_idx)
                    else:
                        metrics[k] = [v]
            except:
                print(metrics, batch_idx, len(loader), len(loader))

            # if rank == 0: #batch_idx % config.eval_vis_freq == 0 and rank == 0:
            #     vis_utils.vis_pred_clip(sample=sample,
            #                             pred=preds_top,
            #                             iter_num=batch_idx,
            #                             output_dir=output_dir,
            #                             subfolder='val')
            #     vis_utils.vis_pred_scores(sample=sample,
            #                             pred=preds_top,
            #                             iter_num=batch_idx,
            #                             output_dir=output_dir,
            #                             subfolder='val')
            # dist.barrier()
    if not(query_type is None):
        wandb_val_prefix = f"Valid_{query_type}"
    else:
        wandb_val_prefix = "Valid"

    if rank == 0:
        wandb_log = {}
        for k in metrics.keys():
            wandb_log[f'{wandb_val_prefix}/{k}'] = torch.tensor(metrics[k]).mean().item()
            wandb_log["Epoch"] = (epoch + (batch_idx / len(loader)))
        wandb_run.log(wandb_log)
    
    # return torch.tensor(metrics['prob_accuracy']).mean().item()
    mean_acc_tpr = 0.5*(torch.tensor(metrics['prob_accuracy']).mean().item() + torch.tensor(metrics['true_pos_rate']).mean().item())
    mean_auroc = torch.tensor(metrics['skl_auroc']).mean().item()
    return mean_auroc #mean_acc_tpr

def val_pulse_performance(config, pred_prob, gts, prob_theta=0.5):
        # gts = sample = {
        #   'clip': clip.float(),                           # [B,T,3,H,W]
        #   'clip_with_bbox': clip_with_bbox.float(),       # [B,T,3]
        #   'before_query': before_query.bool(),            # [B,T]
        #   'query': query.float()}                         # [B,3,H,W]
    device = pred_prob.device

    gt_prob = gts['clip_with_bbox']         # [b,t,3]
    gt_before_query = gts['before_query']   # [b,t]

    # if config.train.use_hnm:
    #     loss_prob = BCELogitsLoss_with_HNM(pred_prob, gt_prob, positive, gt_before_query, config.loss.prob_bce_weight)
    #     pred_prob = rearrange(pred_prob, 'b t N -> (b t N)')
    # else:
    pred_prob = pred_prob.squeeze(2)         # if [2,30,1] -> [2,30] otherwise [2,30,3] -> [2,30,3]
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    loss_prob = criterion(pred_prob[gt_before_query.bool()].float(),    # if gt_before_query is all ones then: [2,30,3]->[60,3]
                            gt_prob[gt_before_query.bool()].float())    # otherwise [x,3] (gets rid of batch dim)

    gt_idx = gt_prob.bool()        # temporal mask for when there is std planes [b,t]
    gt_bq = gt_before_query.bool() # [2,30,3] -> [2,30]
    # # class_positives: same as true pos but taking max logits not on threshold
    # class_positives = (pp_labels[gt_idx] == gt_labels[gt_idx]).float().mean()  # accuracy if we mask only where gt is a positive
    # # num frames where max logits class is same class as the gt class/num frames that the gt for any anat is positive

    # # prob_accuracy = ((torch.sigmoid(pred_prob[gt_before_query.bool()]) > prob_theta) == gt_prob[gt_before_query.bool()].bool()).float().mean()
    # # ^ p_acc wrong: increases w increasing num_classes because if for one frame the gt:[1,0,0] 1 hot encoded class vector 
    # #    if algo was [0,1,0] it would count as 1/3rd correct because the 3rd anat class (0s) matches
    # #    even though algo predicted as wrong anat class entirely for this frame...
    prob_accuracy = ((torch.sigmoid(pred_prob[gt_bq]))>prob_theta) == gt_prob[gt_bq].bool()
    prob_accuracy = prob_accuracy.float().mean()

    prob_accuracy_2 = ((torch.sigmoid(pred_prob[gt_bq]))> 0.6) == gt_prob[gt_bq].bool()
    prob_accuracy_2 = prob_accuracy_2.float().mean()

    true_neg = ((torch.sigmoid(pred_prob[(gt_bq & ~gt_idx)]))>prob_theta) == gt_prob[(gt_bq & ~gt_idx)].bool()
    true_neg = true_neg.float().mean()

    true_pos = ((torch.sigmoid(pred_prob[gt_idx]))>prob_theta) == gt_prob[gt_idx].bool()
    true_pos = true_pos.float().mean()

    # My revised version:
    preds = ((torch.sigmoid(pred_prob))>0.5) & gt_bq  # B,T bool
    tr_n = (~preds[gt_bq]) & (~gt_idx[gt_bq])
    actual_negs = ~gt_idx[gt_bq]
    tnr = ((tr_n.sum().float()) / (actual_negs.sum().float() + 1e-6)).float().mean()
    prevalence=gt_idx.float().mean() # should be smaller becacuse 0.5->0
    # Ensure input tensors are boolean
    intersection = (preds & gt_idx).sum(dim=1)  # Sum over temporal dimension
    union = (preds | gt_idx).sum(dim=1)  # Sum over temporal dimension
    iou = (intersection.float() / (union.float() + 1e-6)).float().mean()  # IOU per batch element

    valid_indices = gt_bq.view(-1)
    try:
        sklearn_auroc = roc_auc_score(gt_idx.view(-1)[valid_indices].cpu().numpy(),
                            pred_prob.view(-1)[valid_indices].cpu().numpy())
    except:
        sklearn_auroc = 0.5
    loss = {
        'BCE_loss_prob': loss_prob.mean(),
        'prob_accuracy': prob_accuracy.item(),      # temporal and anat class acc (only count as correct
        'prob_accuracy_0.6': prob_accuracy_2.item(),  # if logits for only correct anat>threshold)
        'true_pos_rate': true_pos.item(),  # same as above but only for +ve frames (where atleast 1 anat appears)
        'rev_TNR': tnr.item(),
        'prevalence': prevalence.item(),
        'iou': iou.item(),
        'tpr*tnr': (true_pos*tnr).item(),
        'skl_auroc': sklearn_auroc,
        }
    return loss


def val_performance(config, preds, gts, prob_theta=0.5):
    pred_center = preds['center']   # [b,t,N,2]
    pred_hw = preds['hw']           # [b,t,N,2], actually half of hw
    pred_bbox = preds['bbox']       # [b,t,N,4]
    pred_prob = preds['prob']       # [b,t,N]
    if 'prob_refine' in preds.keys():
        pred_prob_refine = preds['prob_refine']     # [b,t]
    b,t,N = pred_prob.shape

    pred_prob = rearrange(pred_prob, 'b t N -> (b t) N')
    pred_hw = rearrange(pred_hw, 'b t N c -> (b t) N c')
    pred_center = rearrange(pred_center, 'b t N c -> (b t) N c')
    pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t) N c')

    pred_prob, top_idx = torch.max(pred_prob, dim=-1)  # [b*t], [b*t]
    pred_bbox = torch.gather(pred_bbox, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,4)).squeeze()       # [b*t,4]
    pred_hw = torch.gather(pred_hw, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)).squeeze()           # [b*t,2]
    pred_center = torch.gather(pred_center, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)).squeeze()   # [b*t,2]

    if 'center' not in gts.keys():
        gts['center'] = (gts['clip_bbox'][...,:2] + gts['clip_bbox'][...,2:]) / 2.0
    if 'hw' not in gts.keys():
        gts['hw'] = gts['center'] - gts['clip_bbox'][...,:2]
    gt_center = rearrange(gts['center'], 'b t c -> (b t) c')
    gt_hw = rearrange(gts['hw'], 'b t c -> (b t) c')
    gt_bbox = rearrange(gts['clip_bbox'], 'b t c -> (b t) c')
    gt_prob = gts['clip_with_bbox'].reshape(-1)
    gt_before_query = gts['before_query'].reshape(-1)

    # bbox loss
    loss_center = F.l1_loss(pred_center[gt_prob.bool()], gt_center[gt_prob.bool()])
    loss_hw = F.l1_loss(pred_hw[gt_prob.bool()], gt_hw[gt_prob.bool()])
    iou_all, giou, loss_giou = GiouLoss(pred_bbox, gt_bbox, mask=gt_prob.bool())
    if torch.sum(gt_prob).item() > 0:
        iou = torch.mean(iou_all[gt_prob.bool()])
        iou_25 = torch.mean((iou_all[gt_prob.bool()] > 0.25).float())
        giou = torch.mean(giou[gt_prob.bool()])
    else:
        iou, iou_25, giou = 0.0, 0.0, 0.0
    
    # occurance loss
    weight = torch.tensor(config.loss.prob_bce_weight).to(gt_prob.device)
    weight_ = weight[gt_prob[gt_before_query.bool()].long()].reshape(-1)
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    loss_prob = (criterion(pred_prob[gt_before_query.bool()].float(), 
                           gt_prob[gt_before_query.bool()].float()) * weight_).mean()
    
    if 'prob_refine' in preds.keys():
        pred_prob = pred_prob_refine.reshape(-1)
    
    prob_accuracy = ((torch.sigmoid(pred_prob) > prob_theta) == gt_prob.bool()).float().mean()
    prob_accuracy_2 = ((torch.sigmoid(pred_prob) > 0.6) == gt_prob.bool()).float().mean()
    prob_accuracy_3 = ((torch.sigmoid(pred_prob) > 0.7) == gt_prob.bool()).float().mean()
    prob_accuracy_4 = ((torch.sigmoid(pred_prob) > 0.65) == gt_prob.bool()).float().mean()
    
    loss = {
        # losses
        'loss_bbox_center': loss_center.item(),
        'loss_bbox_hw': loss_hw.item(),
        'loss_bbox_giou': loss_giou.item(),
        'loss_prob': loss_prob.item(),
        # information
        'iou': iou.item(),
        'iou_25': iou_25.item(),
        'giou': giou.item(),
        'prob_accuracy': prob_accuracy.item(),
        'prob_accuracy_0.6': prob_accuracy_2.item(),
        'prob_accuracy_0.7': prob_accuracy_3.item(),
        'prob_accuracy_0.65': prob_accuracy_4.item(),
    }

    # get top prediction
    pred_prob = rearrange(pred_prob, '(b t) -> b t', b=b, t=t)           # [b,t]
    pred_bbox = rearrange(pred_bbox, '(b t) c -> b t c', b=b, t=t)       # [b,t,4]
    pred_top ={
        'bbox': pred_bbox,
        'prob': pred_prob
    }

    return loss, pred_top