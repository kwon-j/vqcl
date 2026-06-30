import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d, positionalencoding2d
from utils.model_utils import BasicBlock_Conv2D, BasicBlock_MLP
# from utils.anchor_utils import generate_anchor_boxes_on_regions
# from dataset.dataset_utils import bbox_xyhwToxyxy
from einops import rearrange
import math
import torchvision
# from dataset import dataset_utils
from model.mae import vit_base_patch16
from model.swin import SwinTransformer
from model.jongnet import SwinJongNetClassifier

default_aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32)                                # 3 types of aspect ratio
base_sizes=torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32)    # 4 types of size
aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32)                                # 3 types of aspect ratio
n_base_sizes = base_sizes.shape[0]
n_aspect_ratios = aspect_ratios.shape[0]


def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dino':
        assert type in ['vitb8', 'vitb16', 'vits8', 'vits16']
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_{}'.format(type))
        down_rate = int(type.replace('vitb', '').replace('vits', ''))
        backbone_dim = 768
        if type == 'vitb16' and config.model.bakcbone_use_mae_weight:
            mae_weight = torch.load('/vision/hwjiang/episodic-memory/VQ2D/checkpoint/mae_pretrain_vit_base.pth')['model']
            backbone.load_state_dict(mae_weight)
    elif name == 'dinov2':
        assert type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(type))  # add kwargs like img_size and in_channels? 
        down_rate = 14
        if type == 'vitb14':
            backbone_dim = 768
        elif type == 'vits14':
            backbone_dim = 384

        if (config.model.cpt_path != "") and (config.model.cpt_path is not None):
            checkpoint = torch.load(config.model.cpt_path)
            remove_prefix = 'backbone.'
            state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) \
                          else k: v for k, v in checkpoint["state_dict"].items()}
            state_dict.pop("prediction_head.2.weight", None)
            state_dict.pop("prediction_head.2.bias", None)
            backbone.load_state_dict(state_dict, strict=True)
            print("_________________ LOADED STATE DICT SUCCESSFULLY ____________________")

    elif name == 'mae':
        backbone = vit_base_patch16()
        cpt = torch.load('/vision/hwjiang/download/model_weight/mae_pretrain_vit_base.pth')['model']
        backbone.load_state_dict(cpt, strict=False)
        down_rate = 16
        backbone_dim = 768

    elif name == 'swint':
        backbone = SwinTransformer(img_size=448,in_chans=1, num_classes=0,
                                #   depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                                )
        if (config.model.cpt_path == "") or (config.model.cpt_path is None):
            cpt = torch.load("./model/swin_tiny_patch4_window7_224_22k.pth")
            state_dict = cpt["model"]
            # print(state_dict.keys())
            pe_weight = state_dict["patch_embed.proj.weight"][:,0,:,:]
            state_dict["patch_embed.proj.weight"] = torch.unsqueeze(pe_weight,1)
            # remove_prefix = 'head.'
            # state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) \
            #             else k: v for k, v in checkpoint["model"].items()}
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            state_dict.pop("layers.0.blocks.1.attn_mask", None)
            state_dict.pop("layers.1.blocks.1.attn_mask", None)
            state_dict.pop("layers.2.blocks.1.attn_mask", None)
            state_dict.pop("layers.2.blocks.3.attn_mask", None)
            state_dict.pop("layers.2.blocks.5.attn_mask", None)
            # state_dict.pop("layers.3.blocks.1.attn_mask", None)
            backbone.load_state_dict(state_dict, strict=False)
            print("_________ LOADED SWIN T PRETRAINED IMAGENET WEIGHTS SUCCESSFULLY ________")
        else:
            cpt = torch.load(config.model.cpt_path) #, map_location=dev)
            backbone.load_state_dict(cpt['state_dict'], strict=True)
            print("_________ LOADED SWIN T WEIGHTS SUCCESSFULLY ________")
        backbone_dim = 768; down_rate=64  # TODO: is downrate correct??
        # ViTB14  - 768*32*32 - bbdim=768, down_rate=14 -> 448/32 = 14 so downrate=14
        # swint   - 196, 768 -> (14*14), 768 ?? downrate = 5 448/14=32 so downrate=32

    elif name == "swinfpn":
        if (config.model.cpt_path == "") or (config.model.cpt_path is None):
            backbone = SwinJongNetClassifier(img_size=448,in_chans=1, imgnet_pretrained=True)
        else:
            backbone = SwinJongNetClassifier(img_size=448,in_chans=1, imgnet_pretrained=False)
            cpt = torch.load(config.model.cpt_path) #, map_location=dev)
            backbone.load_state_dict(cpt['state_dict'], strict=True)
            print("_________ LOADED SWIN-FPN(jongnet) WEIGHTS SUCCESSFULLY ________")
            backbone_dim = 768; down_rate=64  # TODO: is downrate correct??
            # ViTB14  - 768*32*32 - bbdim=768, down_rate=14 -> 448/32 = 14 so downrate=14
            # swint   - 196, 768 -> (14*14), 768 ?? downrate = 5 448/14=32 so downrate=32
    return backbone, down_rate, backbone_dim


class ClipMatcher(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)

        self.multiquery = config.multiquery

        self.backbone_name = config.model.backbone_name

        self.query_size = config.dataset.query_size
        self.clip_size_fine = config.dataset.clip_size_fine
        self.clip_size_coarse = config.dataset.clip_size_coarse

        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        self.type_transformer = config.model.type_transformer
        assert self.type_transformer in ['local', 'global']
        self.window_transformer = config.model.window_transformer
        self.resolution_transformer = config.model.resolution_transformer
        self.resolution_anchor_feat = config.model.resolution_anchor_feat

        # self.anchors_xyhw = generate_anchor_boxes_on_regions(image_size=[self.clip_size_coarse, self.clip_size_coarse],
        #                                                 num_regions=[self.resolution_anchor_feat, self.resolution_anchor_feat])
        # self.anchors_xyhw = self.anchors_xyhw / self.clip_size_coarse   # [R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis
        # self.anchors_xyxy = bbox_xyhwToxyxy(self.anchors_xyhw)

        # query down heads -- this is completely unused??!
        self.query_down_heads = []  # this is completely unused??!
        for _ in range(int(math.log2(self.query_feat_size))):  # qfs=14, then int(log2(14)) = 3 so 3 downheads
            self.query_down_heads.append(
                nn.Sequential(
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(self.backbone_dim),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.query_down_heads = nn.ModuleList(self.query_down_heads)

        # feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        
        # clip-query correspondence
        self.CQ_corr_transformer = []
        for _ in range(1):
            self.CQ_corr_transformer.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=256,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
                )
            )
        self.CQ_corr_transformer = nn.ModuleList(self.CQ_corr_transformer)

        # feature downsample layers -- this is only used until shape=8,8 so for ViTb14: 32->16->8 so only 3 layers even though there will be 5 down layers made log2(32)=5
        self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        if (self.backbone_name == "swint") or (self.backbone_name == "swinfpn"):
            padd = 2  # if padd=1, then 14->7->4 which misses 8,8, so need padd=2 for swint
        else:
            padd = 1
        for i in range(self.num_head_layers-1):
            self.in_channel = 256 if i != 0 else self.backbone_dim
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=padd),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        self.down_heads = nn.ModuleList(self.down_heads)

        # spatial-temporal PE
        self.pe_3d = positionalencoding3d(d_model=256, 
                                          height=self.resolution_transformer, 
                                          width=self.resolution_transformer, 
                                          depth=config.dataset.clip_num_frames,
                                          type=config.model.pe_transformer).unsqueeze(0)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        # spatial-temporal transformer layer
        self.feat_corr_transformer = []
        self.num_transformer = config.model.num_transformer
        for _ in range(self.num_transformer):
            self.feat_corr_transformer.append(
                    torch.nn.TransformerEncoderLayer(
                        d_model=256, 
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                ))
        self.feat_corr_transformer = nn.ModuleList(self.feat_corr_transformer)
        self.temporal_mask = None

        # output head
        # self.head = Head(in_dim=256, in_res=self.resolution_transformer, out_res=self.resolution_anchor_feat)
        self.head = ProbHead(in_dim=256, in_res=self.resolution_transformer, out_res=self.resolution_anchor_feat)

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'mae':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.forward_features(x) # [b,1+h*w,c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out[:,1:].reshape(b, h, w, dim).permute(0,3,1,2)  # [b,c,h,w]
            out = F.interpolate(out, size=(16,16), mode='bilinear')
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == "swint":
            out = self.backbone.forward_features(x) # [b,1+h*w,c]
            if return_h_w:
                print("_____ NO H,W OPTION FOR THIS")
                raise KeyError
            return out
        elif self.backbone_name == "swinfpn":
            out = self.backbone.forward(x) # [b,1+h*w,c]
            if return_h_w:
                print("_____ NO H,W OPTION FOR THIS")
                raise KeyError
            return out
        
        
    def replicate_for_hnm(self, query_feat, clip_feat):
        '''
        query_feat in shape [b,c,h,w]
        clip_feat in shape [b*t,c,h,w]
        '''
        b = query_feat.shape[0]
        bt = clip_feat.shape[0]
        t = bt // b
        
        clip_feat = rearrange(clip_feat, '(b t) c h w -> b t c h w', b=b, t=t)

        new_clip_feat, new_query_feat = [], []
        for i in range(b):
            for j in range(b):
                new_clip_feat.append(clip_feat[i])
                new_query_feat.append(query_feat[j])

        new_clip_feat = torch.stack(new_clip_feat)      # [b^2,t,c,h,w]
        new_query_feat = torch.stack(new_query_feat)    # [b^2,c,h,w]

        new_clip_feat = rearrange(new_clip_feat, 'b t c h w -> (b t) c h w')
        return new_clip_feat, new_query_feat


    def forward(self, clip, query, query_frame_bbox=None, training=False, fix_backbone=True, feat_out=False):
        '''
        clip: in shape [b,t,c,h,w]   =  [2, 30, 1, 448, 448]
        query: in shape [b,c,h2,w2]  =  [2, 1, 448, 448]
        '''
        b, t = clip.shape[:2]
        clip = rearrange(clip, 'b t c h w -> (b t) c h w')  # (b=2, 30, 1, 448, 448) -> (2*30, 1, 448, 448)

        # get backbone features
        if fix_backbone:
            with torch.no_grad():
                if not self.multiquery:
                    query_feat = self.extract_feature(query)
                else:
                    query_feat = query
                clip_feat = self.extract_feature(clip)
        else:
            if not self.multiquery:
                query_feat = self.extract_feature(query)
            else:
                query_feat = query
            clip_feat = self.extract_feature(clip)          # (b t) c h w
        h, w = clip_feat.shape[-2:]
        # clip_feat.shape: torch.Size([60, 768, 32, 32])
        # query_feat.shape: torch.Size([2, 768, 32, 32])
        # print(f"query_feat: {query_feat.shape},\t query: {query.shape},\t clip_feat: {clip_feat.shape}")
        # reduce channel size
        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)  # q:[2, 256, 32, 32], c:[60, 256, 32, 32]

        if self.config.train.use_hnm and training:
            clip_feat, query_feat = self.replicate_for_hnm(query_feat, clip_feat)   # b -> b^2
            b = b**2
        
        # find spatial correspondence between query-frame
        query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')  # [b*t,n,c]=[60, 1024, 256]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')                                         # [b*t,n,c]=[60, 1024, 256]

        for layer in self.CQ_corr_transformer:
            clip_feat = layer(clip_feat, query_feat)                                                     # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)                               # [b*t,c,h,w]=[60, 256, 32, 32]
        # clip_feat.shape: torch.Size([60, 256, 32, 32])
        # query_feat.shape: torch.Size([60, 1024, 256])
        # down-size features and find spatial-temporal correspondence
        for head in self.down_heads:
            clip_feat = head(clip_feat)
            # clip_feat_0.shape: torch.Size([b*t=60, 256, 16, 16])
            # clip_feat_1.shape: torch.Size([60, 256, 8, 8])
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w -> b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(clip_feat, 'b (t h w) c -> (b t) c h w', b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break
        # clip_feat.shape: torch.Size([60, 256, 8, 8])

        # refine anchors
        # anchors_xyhw = self.anchors_xyhw.to(clip_feat.device)                   # [3072,4]
        # anchors_xyxy = self.anchors_xyxy.to(clip_feat.device)                   # [3072, 4]
        # anchors_xyhw = anchors_xyhw.reshape(1,1,-1,4)                           # [1,1,N,4]
        # anchors_xyxy = anchors_xyxy.reshape(1,1,-1,4)                           # [1,1,N,4]

        # bbox_refine, prob = self.head(clip_feat)                                # [b*t,N=h*w*n*m,c] [60, 3072, 4],[60, 3072, 1]
        # bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  # [b,t,N,4], in xyhw frormulation
        # prob = rearrange(prob, '(b t) N c -> b t N c', b=b, t=t)                # [b,t,N,1]
        # bbox_refine += anchors_xyhw                                             # [b,t,N,4]

        # center, hw = bbox_refine.split([2,2], dim=-1)                           # represented by [c_x, c_y, h, w]
        # hw = 0.5 * hw                                                           # anchor's hw is defined as real hw
        # bbox = torch.cat([center - hw, center + hw], dim=-1)                    # [b,t,N,4]

        # result = {
        #     'center': center,           # [b,t,N,2]
        #     'hw': hw,                   # [b,t,N,2]
        #     'bbox': bbox,               # [b,t,N,4]
        #     'prob': prob.squeeze(-1),   # [b,t,N]
        #     'anchor': anchors_xyxy      # [1,1,N,4]
        # }

        prob = self.head(clip_feat)                                                 # bt,w,h,c = [60,1,1,3]
        prob = rearrange(prob, '(b t) w h c -> b t (w h c)', b=b, t=t)              # [b=2,t=30,3]
        if feat_out and training:
            return prob, all_feat
        else:
            return prob

    def get_mask(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            hw = src.shape[1] // t
            thw = src.shape[1]
            mask = torch.ones(thw, thw).float() * float('-inf')

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*hw)
                max_idx = min(thw, (i+window_size+1)*hw)
                mask[i*hw: (i+1)*hw, min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask
    


class Head(nn.Module):
    def __init__(self, in_dim=256, in_res=8, out_res=16, n=n_base_sizes, m=n_aspect_ratios):
        super(Head, self).__init__()

        self.in_dim = in_dim
        self.n = n
        self.m = m
        self.num_up_layers = int(math.log2(out_res // in_res))
        self.num_layers = 3
        
        if self.num_up_layers > 0:
            self.up_convs = []
            for _ in range(self.num_up_layers):
                self.up_convs.append(torch.nn.ConvTranspose2d(in_dim, in_dim, kernel_size=4, stride=2, padding=1))
            self.up_convs = nn.Sequential(*self.up_convs)

        self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head = nn.Conv2d(in_dim, n * m * 4, kernel_size=3, padding=1)
        self.classification_head = nn.Conv2d(in_dim, n * m * 1, kernel_size=3, padding=1)

        self.regression_head.apply(self.init_weights_conv)
        self.classification_head.apply(self.init_weights_conv)

    def init_weights_conv(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        '''
        x in shape [B,c,h=8,w=8]
        '''
        # print(f"original input x.shape: {x.shape}")               # [60,256,8,8]
        if self.num_up_layers > 0:
            x = self.up_convs(x)     # [B,c,h=16,w=16]

        B, c, h, w = x.shape          # [60,256,16,16]
        # print(f" after upconv  x.shape: {x.shape}")               # [60,256,16,16]
        feat_reg, feat_cls = self.in_conv(x).split([c, c], dim=1)   # both [B,c,h,w] = [60,256,16,16]
        # dpout pos 1, seems better
        feat_reg = self.droupout_feat(feat_reg)                     # [60,256,16,16]
        feat_cls = self.droupout_cls(feat_cls)                      # [60,256,16,16]

        feat_reg = self.regression_conv(feat_reg)        # [B,n*m*4,h,w] = [60,256,16,16]
        feat_cls = self.classification_conv(feat_cls)    # [B,n*m*1,h,w] = [60,256,16,16]
        # dpout pos 2
        out_reg = self.regression_head(feat_reg)        # out_reg [60,48,16,16] (48=n*m*4) (4 bbox coords)
        out_cls = self.classification_head(feat_cls)    # out_cls [60,12,16,16] (12=n*m*1) (1 prob scalar)
        out_reg = rearrange(out_reg, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=4)
        out_cls = rearrange(out_cls, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=1)

        return out_reg, out_cls

class ProbHead(nn.Module):
    def __init__(self, in_dim=256, in_res=8, out_res=16, n=n_base_sizes, m=n_aspect_ratios):
        super().__init__()

        self.in_dim = in_dim
        self.n = n
        self.m = m
        self.num_up_layers = int(math.log2(out_res // in_res))
        self.num_layers = 3
        num_classes = 1

        if self.num_up_layers > 0:
            self.up_convs = []
            for _ in range(self.num_up_layers):
                self.up_convs.append(torch.nn.ConvTranspose2d(in_dim, in_dim, kernel_size=4, stride=2, padding=1))
            self.up_convs = nn.Sequential(*self.up_convs)

        self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=in_dim)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.classification_head = nn.Conv2d(in_dim, 3, kernel_size=5) # [B*30,256,16,16]->[B*30,3,12,12]
        # self.classification_head = nn.Sequential(nn.Conv2d(in_dim, 3, kernel_size=5),
        #                                         nn.BatchNorm2d(3),
        #                                         nn.LeakyReLU(inplace=True)
        #                                         )
        self.final_layer = nn.Sequential(#nn.AdaptiveAvgPool2d(6),  # [B*30,3,12,12]->[B*30,3,6,6]
                                        #  nn.MaxPool2d(4),           # B,3,12,12 -> B,3,9,9
                                         nn.MaxPool2d(4,stride=2, padding=1),  # B,3,12,12 -> B,3,6,6
                                         nn.Flatten(),
                                         nn.Linear(3*6*6, num_classes))

        self.classification_head.apply(self.init_weights_conv)

    def init_weights_conv(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        '''
        x in shape [B,c,h=8,w=8]
        '''
        # print(f"original input to prob head x.shape: {x.shape}", flush=True)       # [60,256,8,8] (60=b*t)
        if self.num_up_layers > 0:
            x = self.up_convs(x)     # [B,c,h=16,w=16]

        B, c, h, w = x.shape                                # [60,256,16,16]

        feat_cls = self.in_conv(x)                          # [60,256,16,16]

        # dpout pos 1, seems better
        feat_cls = self.droupout_cls(feat_cls)              # [60,256,16,16]

        feat_cls = self.classification_conv(feat_cls)       # [60,256,16,16]

        # dpout pos 2
        # print(f"feature shape before class head shape: {feat_cls.shape}", flush=True)
        out_cls = self.classification_head(feat_cls)        # [60,1,1,1]

        out_cls = self.final_layer(out_cls)

        out_cls.unsqueeze_(-1).unsqueeze_(-1)

        return out_cls
