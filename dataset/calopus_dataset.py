import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
# import cv2
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

NORMALIZE_MEAN = [int(it*255) for it in [0.485, 0.456, 0.406]]
NORMALIZE_STD = [int(it*255) for it in [0.229, 0.224, 0.225]]

class CalopusFrameDst(Dataset):
    """
    Args:
        root (string): Root directory where epic data is
        transform (bool, optional): Do transforms or not?
    Returns:
        Image as torch tensor dim C,H,W
        Label as an int
    """
    def __init__(self, root="/data/hert4942/pulsenet/calopus", labels_csv="gt_AS3_train.csv",
                 transform: bool = True):
        self.root = root
        self._transform = transform
        self._originalsize = (448,448) # H,W
        self.img_shape = (448,448)
        df = pd.read_csv(Path(root).resolve().expanduser() / labels_csv)
        self.df = df
        df["gt"].unique().tolist()
        # self.lab_vec_dict = {gt: num for (num, gt) in enumerate(df["gt"].unique().tolist())}
        self.lab_vec_dict = {"AC":0, "FL": 1, "HC":2}
        print(self.lab_vec_dict)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.loc[idx, "vid"], self.df.loc[idx, "frame_dir"],
                                self.df.loc[idx,"frame"] + '.png')
        im = get_img(img_path, self._transform, self.img_shape)
        label = self.df.loc[idx, "gt"]
        label = self.lab_vec_dict[label]
        return im, label

    def get_map(self):
        return self.lab_vec_dict

    def get_class_counts(self):
        return self.df["gt"].value_counts()

    def get_df(self):
        return self.df


class CalopusClipsDst(Dataset):
    """
    Args:
        root (string): Root directory where epic data is
        transform (bool, optional): Do transforms or not?
    Returns:
        Dict with clip, clip w bounding box, 
            query
    """
    def __init__(self, root="/data/hert4942/pulsenet/calopus", labels_csv="gt_AS3_clips_train.csv",
                 rm_poor: bool= False, transform: bool = True, multiquery: bool = False,
                 img_shape=(448,448), clip_num_frames=30, interval=1, multi_class_out:bool=False, samequery:bool=False,query_augment:bool=False):
        self.root = Path(root).resolve().expanduser()
        self._transform = transform
        self._multiquery = multiquery
        self._originalsize = (864,1136) # H,W
        self.img_shape = img_shape
        self.cnf = clip_num_frames
        self.interval = interval
        self.multi_class_out = multi_class_out
        self.samequery = samequery
        self.query_augment=query_augment

        df = pd.read_csv(Path(root).resolve().expanduser() / labels_csv)
        if rm_poor:
            df = df.loc[df.qual_subj != "Poor"]
        df.reset_index(inplace=True, drop=True)
        # reset clip_num if we drop stuff
        df['clip_num2']=pd.factorize(df['clip_num'].tolist())[0]
        df.drop(columns=["clip_num"] ,inplace=True)
        df.rename(columns={"clip_num2": "clip_num"}, inplace=True)
        self.df = df
        # df["gt"].unique().tolist()
        # self.lab_vec_dict = {gt: num for (num, gt) in enumerate(df["gt"].unique().tolist())}
        self.lab_vec_dict = {"AC":0, "FL": 1, "HC":2}
        print(self.lab_vec_dict)

    def __len__(self):
        return self.df.clip_num.max()+1

    def __getitem__(self, idx):
        clip_num_frames = self.cnf  # 30 frames in one clip
        # if interval: tuple then random sample for between those 2 values- temporal augment (speedup/down)
        if isinstance(self.interval, int):
            interval = self.interval
        else:
            interval = random.randint(self.interval[0], self.interval[1])
        sampling = "rand" if self._transform else "uniform"

        subdf = self.df.loc[self.df["clip_num"] == idx]
        sample = [subdf.frame.min(), subdf.frame.max()]
        clip_frame_nums = sample_frames_balance_calopus(clip_num_frames, interval,
                                                        subdf.vid_len.iloc[1], sample,
                                                        sampling)

        all_img_paths = sorted(list(Path(f"{self.root}/{subdf.vid.iloc[1]}")
                                    .resolve().iterdir()))
        img_paths = [str(all_img_paths[cfn]) for cfn in clip_frame_nums]
        clip = get_clip(img_paths, self._transform, self.img_shape)

        gt = subdf["gt"].iloc[1]
        clip_with_bbox = (clip_frame_nums >= subdf.frame.min()) & (clip_frame_nums <= subdf.frame.max())
        clip_with_bbox = torch.tensor(clip_with_bbox).float()  # whether std plane or not GT
        clip_with_bbox2= torch.tensor((clip_frame_nums == subdf.frame.min()) | (clip_frame_nums == subdf.frame.max())).float()*0.5
        clip_with_bbox3 = clip_with_bbox - clip_with_bbox2

        # tried to get a quick way for the presence of 2 classes at the same time, but couldn't figure it out
        # vdf = self.df.loc[("vid" == subdf.vid)] # need df of same size so pad w 0s?
        # vdf.loc[vdf.frame.isin(cfn)]

        # If using _train.csv then pre and post df may be missing so use full csv and split via torch subset
        before_query1 = np.array([True] * len(clip_with_bbox))
        after_query = np.array([True] * len(clip_with_bbox))
        if idx > 0:
            predf = self.df.loc[self.df["clip_num"] == idx -1]
            if (predf.vid.iloc[1] == subdf.vid.iloc[1]):
                before_query1 = clip_frame_nums > predf.frame.max()
        if idx < self.df["clip_num"].max(): #int(idx + 1) in self.df["clip_num"].values:
            postdf = self.df.loc[self.df["clip_num"] == idx +1]
            if (postdf.vid.iloc[1] == subdf.vid.iloc[1]):
                after_query = clip_frame_nums < postdf.frame.min()
        before_query = torch.tensor(before_query1 & after_query).bool()

        if self.multi_class_out:
            gt_planes = torch.zeros(clip_num_frames, len(self.lab_vec_dict))
            gt_planes[:,self.lab_vec_dict[gt]] = clip_with_bbox3
            clip_with_bbox3 = gt_planes

            gt_mask = torch.zeros(clip_num_frames, len(self.lab_vec_dict)).bool()
            gt_mask[:,self.lab_vec_dict[gt]] = before_query.bool()
            before_query = gt_mask

        # before_query is simply a temporal mask - of when the query will next appear
        # for our dataloader, we treat each clip separately, so if overlaps with other clips
        # we need to mask out the other clips when calculating loss and stuff

        # if (idx == 201):
        #     gt_class[:,self.lab_vec_dict["HC"]] = clip_with_bbox
        #     gt_mask = torch.ones(clip_num_frames, len(self.lab_vec_dict)).bool()
        # if (idx == 202):
        #     gt_class[:,self.lab_vec_dict["AC"]] = clip_with_bbox
        #     gt_mask = torch.ones(clip_num_frames, len(self.lab_vec_dict)).bool()
        # sameqpath = str(  (np.array(img_paths)[clip_with_bbox.bool()])[2] )  # 2 out of index for large frameintervals
        try:
            qpaths = (np.array(img_paths)[clip_with_bbox.bool()])
            sameqpath = str(qpaths[len(qpaths)//2])
            query = self.get_query(gt, multiquery=self._multiquery, img_shape=self.img_shape,samequery=self.samequery, sameq_path=sameqpath,query_augment=self.query_augment)  # cropped query but resized to full frame
        except:  # if you can't find self query
            print(f"_______ NO QUERY IN THIS ONE AT ALL ??? ________", flush=True)
            print(f"clip index (clip_num in gt_csv): {idx}", flush=True)
            print(f"_______ clip_frame_nums: {clip_frame_nums}", flush=True)
            print(f"__________ gt_frame_num: {sample}", flush=True)
            print(f"____ gt_tensor temporal: {clip_with_bbox}", flush=True)
            query = self.get_query(gt, multiquery=self._multiquery, img_shape=self.img_shape,samequery=False,query_augment=self.query_augment)  # cropped query but resized to full frame

        # clip ___ torch.Size([30, 3, 448, 448]) ___ torch.float32 ___ tensor(1.) ___ tensor(0.)
        # clip_with_bbox ___ torch.Size([30]) ___ torch.float32 ___ tensor(1.) ___ tensor(0.)
        # before_query ___ torch.Size([30]) ___ torch.bool ___ tensor(True) ___ tensor(True)
        # clip_bbox ___ torch.Size([30, 4]) ___ torch.float32 ___ tensor(0.8767) ___ tensor(0.)
        # query ___ torch.Size([3, 448, 448]) ___ torch.float32 ___ tensor(0.6667) ___ tensor(0.)
        # clip_h ___ torch.Size([]) ___ torch.int64 ___ tensor(224) ___ tensor(224)
        # clip_w ___ torch.Size([]) ___ torch.int64 ___ tensor(298) ___ tensor(298)
        # query_frame ___ torch.Size([3, 448, 448]) ___ torch.float32 ___ tensor(1.) ___ tensor(0.)
        # query_frame_bbox ___ torch.Size([4]) ___ torch.float32 ___ tensor(0.9637) ___ tensor(0.5025)

        results = {
            'clip': clip.float(),                           # [T,3,H,W]
            'clip_with_bbox': clip_with_bbox3, #gt_class.float(),       # [T]
            'before_query': before_query, #gt_mask,            # [T]
            #'clip_bbox': clip_bbox.float().clamp(min=0.0, max=1.0),                 # [T,4]
            'query': query.float(),                         # [3,H2,W2]
            # 'clip_h': torch.tensor(clip_h),
            # 'clip_w': torch.tensor(clip_w),
            #'query_frame': query_frame.float(),             # [3,H,W]
            #'query_frame_bbox': query_frame_bbox.float()    # [4]
            'img_path': img_paths,
            'clip_num': idx,
            'anat': self.lab_vec_dict[gt], # for 1 class output - we need anat information

        }
        return results

    def get_query(self, anat, multiquery=False, img_shape=(448,448), samequery=False, sameq_path=None,query_augment:bool=False):
        if query_augment:
            # ShiftScaleRotate-scale_limit: Let (W, H) be the original image dimensions and (W', H') be the output dimensions. The scale factor s is sampled from the range [1 + scale_limit[0], 1 + scale_limit[1]]. Then, W' = W * s and H' = H * s.
            # RandomResizedCrop-scale: A target area A is sampled from the range [scale[0] * input_area, scale[1] * input_area]
            transform = A.Compose(
                    [   A.Rotate(30,p=1),
                        A.ShiftScaleRotate(scale_limit=(-0.5,0.2),rotate_limit=(0),p=0.9),#,border_mode=cv2.BORDER_CONSTANT, value=0.3),
                        A.RandomResizedCrop(height=img_shape[0],width=img_shape[1],scale=(0.4, 1.0), ratio=(0.75, 1.333)),
                        A.HorizontalFlip(p=0.5),
                        A.ColorJitter (brightness=0.5, contrast=0.5, p=1),
                        A.Normalize(mean=(0.152), std=(0.182), max_pixel_value=1.),
                        ToTensorV2()])
        else:
            transform = A.Compose([A.Resize(img_shape[0],img_shape[1]), A.Normalize(mean=(0.152), std=(0.182),max_pixel_value=1.), ToTensorV2()])
        if multiquery:

            # PULSE QUERY OR CALOPUS QUERY? RN THIS IS AVE PULSE QUERY

            qpath = Path(f"/data/biomedia1/pulse/ave_query_feat_{anat}.pth").expanduser().resolve()
            query = torch.load(qpath)
            return query
        elif samequery:
            # ____ for query from same video as the one being trained on ___
            query = Image.open(str(sameq_path)).convert('L')
            query = np.asarray(query).astype(np.float32)
            query = query/255.
            imdict = transform(image=query)  # C,H,W
            return imdict["image"]
        else:
            ACquery_path = (self.root / "305_step1/305_step1_f00111.png").resolve()
                    # AC/2018-10-03T095126
            HCquery_path = (self.root / "325_step1/325_step1_f00153.png").resolve()
            FLquery_path = (self.root / "373_step3.2/373_step3.2_f00295.png").resolve()
                    # FL/2019-02-25T110550 frames1
            anat2query = {"AC":ACquery_path, "HC":HCquery_path, "FL":FLquery_path}
            query_path = anat2query[anat]
            query = Image.open(str(query_path)).convert('L')
            query = np.asarray(query).astype(np.float32)
            query = query/255.
            imdict = transform(image=query)  # C,H,W
            return imdict["image"]

    def get_map(self):
        return self.lab_vec_dict

    def get_class_counts(self):
        return self.df["gt"].value_counts()

    def get_df(self):
        return self.df


def get_clip(img_paths, transform=True, img_shape=(448,448), mean=0.152,std=0.182,):
    imlist = []
    for img_path in img_paths:
        im = Image.open(img_path).convert('L')
        im = np.asarray(im).astype(np.float32)
        im = im/255.
        imlist.append(im)

    n_imgs = len(imlist)
    add_targs = {(f"image{n}" if n>0 else "image"):"image" for n in range(n_imgs)}
    imkwargs = {(f"image{n}" if n>0 else "image"):im for (n, im) in enumerate(imlist)}
    add_targs["image"] = "image"

    if transform:
        transform = A.Compose(
            [   A.Rotate(30,p=1),
                A.ShiftScaleRotate(scale_limit=(-0.5,0.2),rotate_limit=(0),p=0.9),#,border_mode=cv2.BORDER_CONSTANT, value=0.3),
                A.RandomResizedCrop(height=img_shape[0],width=img_shape[1],scale=(0.4, 1.0), ratio=(0.75, 1.333)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter (brightness=0.5, contrast=0.5, p=1),
                A.Normalize(mean=(0.152), std=(0.182), max_pixel_value=1.),
                ToTensorV2()],
            additional_targets=add_targs)
        imdict = transform(**imkwargs)  # C,H,W
        imlist2 = [imdict[f"image{n}"] if n>0 else imdict["image"] for n in range(n_imgs)]
    else:
        transform = A.Compose([
            A.Resize(img_shape[0],img_shape[1]),
            A.Normalize(mean=(0.152), std=(0.182), max_pixel_value=1.), ToTensorV2()],
            additional_targets=add_targs)
        imdict = transform(**imkwargs)  # C,H,W
        imlist2 = [imdict[f"image{n}"] if n>0 else imdict["image"] for n in range(n_imgs)]
    
    clip = torch.stack(imlist2)
    return clip


def get_img(img_path, transform=True, img_shape=(448,448)):
    im = Image.open(img_path)
    if transform:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_shape, scale=(0.7, 1.0), ratio=(0.75,1.25), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.Normalize(mean=0.152, std=0.182),
        ])
        im = torch.from_numpy(np.asarray(im).astype(np.float32))
        im = im.unsqueeze(0)  # add fake dimensions (for channels)
        # im = im.permute(2, 0, 1)
        im = im/255.
        im = transform(im)
    else:
        im = im.resize(reversed(img_shape))  # PIL resize is W,H not H,W
        im = ImageOps.grayscale(im)
        im = torch.from_numpy(np.asarray(im).astype(np.float32))
        im = im.unsqueeze(0)  # add fake dimensions (for channels)
        im = im/255.
        im = F.normalize(im, mean=0.152, std=0.182)
    return im


def sample_frames_balance_calopus(num_frames, frame_interval, end_frame, anno_valid_idx_range, sampling='rand'):
    '''
    sample clips with balanced negative and postive samples
    params:
        num_frames: total number of frames to sample
        frame_interval: frame interval, where value 1 is for no interval (consecutive frames)
        sample: data annotations
        sampling: only effective for frame_interval larger than 1
    return: 
        frame_idxs: length [num_frames]
    '''
    required_len = (num_frames - 1) * frame_interval + 1
    anno_len = anno_valid_idx_range[1] - anno_valid_idx_range[0] + 1
    if anno_len <= required_len:
        if anno_len < required_len:
            num_valid = anno_len // frame_interval  # num_valid is the number of valid frames we can sample
        else:
            num_valid = num_frames
        num_invalid = num_frames - num_valid
        if anno_valid_idx_range[1] < required_len:  # if we get a start frame num < 0 then do:
            # idx_start is any point before the first frame of bbox anno, idx_end is this added on
            idx_start = random.choice(range(anno_valid_idx_range[0])) if anno_valid_idx_range[0] > 0 else 0
            idx_end = idx_start + required_len
        elif anno_valid_idx_range[0] + required_len > end_frame:
            idx_end = random.choice(range(anno_valid_idx_range[1],end_frame+1)) if anno_valid_idx_range[1] < end_frame else end_frame
            idx_start = idx_end - required_len
        else:
            num_prior = random.choice(range(num_invalid)) if num_invalid != 0 else 0
            num_post = num_invalid - num_prior
            idx_start = anno_valid_idx_range[0] - frame_interval * num_prior
            idx_end = anno_valid_idx_range[1] + frame_interval * num_post + 1
        intervals = np.linspace(start=idx_start, stop=idx_end, num=num_frames+1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1]))
        if sampling == 'rand':  # random num of frames between consecutive frames
            # this is done taking set frame_interval and randomly sampling between each of those frames
            # so average frame interval between consecutive frames is frame_interval
            # (min and max frame_interval between 2 consecutive frames as 1, 2*frame_interval -1 respectively)
            # (min and max frame_interval between frame 1 and frame 3 is 1*frame_interval +1, 3*frame_interval-1 respectively)
            frame_idxs_pos = [random.choice(range(x[0], x[1])) for x in ranges]
        elif sampling == 'uniform':
            frame_idxs_pos = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        # if anno_valid_idx_range[1] < 470:        # else: # was gna elegant soln but whatever
        num_addition = anno_len - required_len
        start = random.choice(range(num_addition))
        intervals = range(0,frame_interval * (num_frames+1), frame_interval)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1]))
        if sampling == 'rand':
            frame_idxs_pos = [anno_valid_idx_range[0] + start + random.choice(range(x[0], x[1])) for x in ranges]
        elif sampling == 'uniform':
            frame_idxs_pos = [(x[0] + x[1]) // 2 for x in ranges]
            frame_idxs_pos = [anno_valid_idx_range[0] + start + ((x[0] + x[1]) // 2) for x in ranges]
    return np.clip(np.array(frame_idxs_pos),0,end_frame)

class CalopusFullDst(Dataset):
    """
    Args:
        root (string): Root directory where epic data is
        transform (bool, optional): Do transforms or not?
    """
    def __init__(self, root: str, pidlist=None, img_size=(448,448), clip_num_frames=30):
        self.root = root
        self._originalsize = (864,1136) # H,W
        self._img_size = img_size # H,W
        interval = 2
        cl = interval * clip_num_frames
        self.cl = cl
        self.interval = interval
        self.cnf = clip_num_frames
        fulldirlist = sorted(list(Path(root).glob('[0-9]*/')))
        if pidlist is None:
            dirlist = fulldirlist
        else:
            dirlist = [dir for dir in fulldirlist if dir.name[:3] in pidlist]
        # pidlist = list(set([x[:3] for x in Path(root).glob('[0-9]*/')]))
        vid2framlistdict = {dir.name: sorted(dir.glob("*.png")) for dir in dirlist}
        vid2vidlendict = {v:len(flist) for v,flist in vid2framlistdict.items()}
        vid2numclipsdict = {v:-(vl//-cl) for v,vl in vid2vidlendict.items()}
        df = pd.DataFrame(vid2numclipsdict.items(), columns=['vid', 'num_clips'])
        df["clip_num"] = df.num_clips.cumsum()
        df["vid_len"] = list(vid2vidlendict.values())

        self.df = df

    def __len__(self):
        return self.df.clip_num.max()

    def __getitem__(self, idx):
        df = self.df.iloc[self.df['clip_num'].gt(idx).idxmax()]
        clip_spacing = df.vid_len / df.num_clips
        framelist = sorted((Path(self.root)/df.vid).glob('*.png'))
        if df.num_clips == 1:
            overlap=0
        else:
            overlap = ((self.cl * df.num_clips) % df.vid_len) / (df.num_clips-1) # needs to be an int...
        nth_clip = idx - (df.clip_num - df.num_clips)
        start_num = (nth_clip * self.cl) - int(overlap*nth_clip)
        end_num = start_num + self.cl
        clip_frame_nums = start_num + np.arange(0, self.cnf) * self.interval
        clip_frame_nums = np.clip(clip_frame_nums,0,df.vid_len-1)
        img_paths = [str(framelist[cfn]) for cfn in clip_frame_nums]
        clip = get_clip(img_paths, transform=False, img_shape=self._img_size)
        results = {
            'clip': clip,                       # [(B,) T,1,H,W]  B, T-30, 1 channel, height, width
            # 'query': query,                             # [3,H2,W2]
            'img_path': img_paths,
            'clip_num': idx,
        }
        return results

def get_query(anat, root="/data/hert4942/pulsenet/calopus", multiquery=False, img_shape=(448,448)):
    transform = A.Compose([A.Resize(img_shape[0],img_shape[1]), A.Normalize(mean=(0.152), std=(0.182),max_pixel_value=1.), ToTensorV2()])
    root = Path(root).expanduser().resolve()
    if multiquery:
        qpath = Path(f"/data/biomedia1/pulse/ave_query_feat_{anat}.pth").expanduser().resolve()
        query = torch.load(qpath)
        return query
    else:
        ACquery_path = (root / "305_step1/305_step1_f00111.png").resolve()
                # AC/2018-10-03T095126
        HCquery_path = (root / "325_step1/325_step1_f00153.png").resolve()
        FLquery_path = (root / "373_step3.2/373_step3.2_f00295.png").resolve()
                # FL/2019-02-25T110550 frames1
        anat2query = {"AC":ACquery_path, "HC":HCquery_path, "FL":FLquery_path}
        query_path = anat2query[anat]
        query = Image.open(str(query_path)).convert('L')
        query = np.asarray(query).astype(np.float32)
        query = query/255.
        imdict = transform(image=query)  # C,H,W
    return imdict["image"]

