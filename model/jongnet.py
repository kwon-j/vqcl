import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from model.swin import SwinTransformer


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Feature Pyramid Network (FPN) with lateral connections.
        Args:
            in_channels: List of input channels from different Swin Transformer stages.
            out_channels: Number of output channels in each FPN level.
        """
        super(FPN, self).__init__()

        # 1x1 Convolutions to unify channels
        # self.lateral_convs = nn.ModuleList(
        #     [nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels]) # to common 

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_c, out_c, kernel_size=1) for in_c, out_c in zip(in_channels, in_channels[1:])]
            + [nn.Identity()])
        # increase channels of largest scale feat map to 2nd largest, 2nd to 3rd etc. so they can be added

        # 3x3 Convolutions to smooth final feature maps
        self.smooth_convs = nn.ModuleList(
            [nn.Conv2d(in_c, out_channels, kernel_size=3, padding=1) for in_c in in_channels[1:]]
            + [nn.Identity()])
        # smoothing convolution not applied to smallest scale feat map because it doesn't get added by anything
        
    def forward(self, features):
        """
        Forward pass of FPN.
        Args:
            features: List of feature maps from different Swin Transformer stages.
        Returns:
            List of multi-scale feature maps.
        """
        # Apply lateral 1x1 convolutions
        fpn_features = [lateral_conv(f) for f, lateral_conv in zip(features, self.lateral_convs)]
        
        # Top-down feature fusion - upsample smaller non-conv feature vec to the large size and add
        for i in range(len(fpn_features) - 2, -1, -1):  
            fpn_features[i] += F.interpolate(features[i + 1], size=features[i].shape[2:], mode='nearest')

        # # Top-down feature fusion - upsample smaller feature vec to the large size
        # for i in range(len(fpn_features) - 2, -1, -1):  
        #     fpn_features[i] += F.interpolate(fpn_features[i + 1], size=fpn_features[i].shape[2:], mode='nearest')

        # Apply smoothing convolutions
        fpn_features = [smooth_conv(f) for f, smooth_conv in zip(fpn_features, self.smooth_convs)]
        
        return fpn_features



# ______________________ below is for swin-t

    # def forward_features(self, x):
    #     x = self.patch_embed(x)
    #     if self.ape:
    #         x = x + self.absolute_pos_embed
    #     x = self.pos_drop(x)
    #     for iii, layer in enumerate(self.layers):
    #         x = layer(x)
    #         # print(f"x after layer_{iii} shape: {x.shape}", flush=True)
    #         # for swin-T:
    #         # x after layer_0 shape: torch.Size([16, 3136, 192])
    #         # x after layer_1 shape: torch.Size([16, 784, 384])
    #         # x after layer_2 shape: torch.Size([16, 196, 768])
    #         # x after layer_3 shape: torch.Size([16, 196, 768])

    #     x = self.norm(x)  # B L C

    #     if self.feat_only:
    #         x = torch.flatten(x, 1)
    #         x = F.normalize(x, dim=1)
    #     else:
    #         x = self.avgpool(x.transpose(1, 2))  # BLC -> B C 1  for swin-T: [B, 196, 768]) -> [B, 768, 1]
    #         x = torch.flatten(x, 1)
# ___________________________________


class SwinFPNClassifier(nn.Module):
    def __init__(self, num_classes=0):
        super(SwinFPNClassifier, self).__init__()

        # Load a pre-trained Swin Transformer model
        # self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)
        self.swin = SwinTransformer(img_size=448,in_chans=1, num_classes=0,
                            #   depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                                )

        checkpoint = torch.load("./model/swin_tiny_patch4_window7_224_22k.pth")
        state_dict = checkpoint["model"]
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
        # # state_dict.pop("layers.3.blocks.1.attn_mask", None)
        self.swin.load_state_dict(state_dict, strict=False)
        print("_____________ LOADED SWIN-TRANSFORMER STATE DICT SUCCESSFULLY _______________")

        # print(f"________ How many layers: {len(self.swin.layers)} ____")
        # print(f"________ Patch embedding size: {self.swin.embed_dim} ____")

        # Feature dimensions from Swin Transformer
        self.feature_dims = [192, 384, 768, 768]  # Channels at each stage

        # self.out_channels = self.feature_dims[1:]  # feature_dims except 1st one.
        self.out_channels = self.feature_dims[-1]

        # Feature Pyramid Network
        self.fpn = FPN(self.feature_dims, out_channels=self.out_channels)

        # Global Pooling + Classification
        # self.classifier = nn.Linear(256, num_classes)
        # self.classifier = nn.Linear(out_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Separate classification heads for each scale
        # self.classifiers = nn.ModuleList([
        #     nn.Linear(256, num_classes) for _ in self.feature_dims
        # ])

    def forward(self, x):
        # features = self.swin(x)  # this only outputs final feat vec

        # Manually extract features at every scale (after patch-merging)

        #input shape: B,C,H,W = 2*32,1,448,448
        x = self.swin.patch_embed(x)
        #patch embed: B,(H/4)*(W/4),embed_dim = 2*32,12544,96
        x = self.swin.pos_drop(x)  # doesn't change shape

        features = []
        for iii, layer in enumerate(self.swin.layers):
            x = layer(x)
            # for swin-T with 448 input after patch merging:
            # patch merging at end of each layer not at beginning of next layer as in diagram in paper. (no patchmerge in final layer)
            # afer layer_0 and patchmerge: torch.Size([B, 3136, 192]) - B,(H/8)*(W/8),2C
            # afer layer_1 and patchmerge: torch.Size([B, 784, 384]) - B,(H/16)*(W/16),4C
            # afer layer_2 and patchmerge: torch.Size([B, 196, 768]) - B,(H/32)*(W/32),8C
            # afer layer_3 (no patchmerge): torch.Size([B, 196, 768])


            # Reshape to [B, C, H, W]
            h_w = int(x.shape[1] ** 0.5)
            features.append(x.permute(0, 2, 1).reshape(x.shape[0], self.feature_dims[iii], h_w, h_w))

        # Pass features through FPN
        fpn_features = self.fpn(features)

        # _________ Downsample to smallest scale and average feature vector for contrastive learning __________
        # Downsample all features to smallest scale (last feature map)
        target_size = fpn_features[-1].shape[2:]  
        aligned_features = [F.interpolate(f, size=target_size, mode='nearest') for f in fpn_features]
        
        # # Average across scales
        # combined_features = torch.stack(aligned_features, dim=0).mean(dim=0)

        # Weighted average where first layer is weighted less (has less expressiveness less semantivally meaningful usually, so penalise this less in loss)
        # could do torch.tensor([0.0, 0.1, 0.3, 0.5]) which may work.
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=aligned_features[0].device).view(-1, 1, 1, 1, 1)
        combined_features = torch.stack(aligned_features, dim=0) * weights
        combined_features = combined_features.sum(dim=0)  # Weighted sum across scales
        
        # # Flatten & pass through contrastive projection head
        # feature_vector = F.adaptive_avg_pool2d(combined_features, 1).view(combined_features.shape[0], -1)

        combined_features = torch.flatten(combined_features, 1)  # B,NxC
        feature_vector = F.normalize(combined_features, dim=1)

        return feature_vector  # Feature vector for contrastive loss

        # # Global Average Pooling on highest-level FPN feature
        # x = F.adaptive_avg_pool2d(fpn_features[-1], 1).view(x.shape[0], -1)

        # # Classification head
        # x = self.classifier(x)

        # return x
# # Example usage:
# model = SwinFPNClassifier(num_classes=1000)
# inp = torch.randn(1, 3, 224, 224)
# out = model(inp)
# print(out.shape)  # Should be [1, 1000]


class SwinJongNetClassifier(nn.Module):
    def __init__(self, img_size=448,in_chans=1, imgnet_pretrained:bool=True):
        super(SwinJongNetClassifier, self).__init__()
        
        self.n_chan = 96  # intermediate number of channels during downsample

        # Load a pre-trained Swin Transformer model
        # self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)
        self.swin = SwinTransformer(img_size=img_size,in_chans=in_chans,num_classes=0,
                            #   depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                                )
        if imgnet_pretrained:
            checkpoint = torch.load("./model/swin_tiny_patch4_window7_224_22k.pth")
            state_dict = checkpoint["model"]
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
            # # state_dict.pop("layers.3.blocks.1.attn_mask", None)
            self.swin.load_state_dict(state_dict, strict=False)
            print("_____________ LOADED SWIN-TRANSFORMER STATE DICT SUCCESSFULLY _______________")

        # print(f"________ How many layers: {len(self.swin.layers)} ____")
        # print(f"________ Patch embedding size: {self.swin.embed_dim} ____")

        # Feature dimensions from Swin Transformer
        self.feature_dims = [192, 384, 768, 768]  # Channels at each stage

        # self.out_channels = self.feature_dims[1:]  # feature_dims except 1st one.
        self.out_channels = self.feature_dims[-1]

        # 192,56,56 --> n_chan,14,14
        # self.layer0tofinal = nn.Sequential(
        #   nn.Conv2d(192,self.n_chan,kernel_size=4,stride=4),
        #   nn.LeakyReLU(),
        #   )
        # Or could make 10% more params, but less flops and less aggressive downsample / more expressiveness?
        self.layer0tofinal = nn.Sequential(
            nn.Conv2d(192, int((192+self.n_chan)/2), kernel_size=3, stride=2, padding=1),  # 56 → 28
            nn.LeakyReLU(),
            nn.Conv2d(int((192+self.n_chan)/2), self.n_chan, kernel_size=3, stride=2, padding=1),  # 28 → 14
        )

        # 384,28,28 --> n_chan,14,14
        self.layer1tofinal = nn.Sequential(
          nn.Conv2d(384,self.n_chan,kernel_size=2,stride=2),
          nn.LeakyReLU(),
          )
        # Or could make 10% more params, but less flops and less aggressive downsample / more expressiveness?
        # self.layer1tofinal = nn.Sequential(
        #     nn.Conv2d(384, self.n_chan, kernel_size=3, stride=2, padding=1),  # 28 → 14
        #     nn.LeakyReLU(),
        #     nn.Conv2d((384+self.n_chan)/2, self.n_chan, kernel_size=3, stride=2, padding=1),  # 14 (keeps it)
        # )        # 768+n_chan+n_chan=864,14,14--> 768,14,14
        self.finalfinal = nn.Sequential(
          nn.Conv2d(768+(self.n_chan*2),768,kernel_size=1),
          )

    def forward(self, x):
        # features = self.swin(x)  # this only outputs final feat vec

        # Manually extract features at every scale (after patch-merging)

        #input shape: B,C,H,W = 2*32,1,448,448
        x = self.swin.patch_embed(x)
        #patch embed: B,(H/4)*(W/4),embed_dim = 2*32,12544,96
        x = self.swin.pos_drop(x)  # doesn't change shape

        features = []
        for iii, layer in enumerate(self.swin.layers):
            x = layer(x)
            # for swin-T with 448 input after patch merging:
            # patch merging at end of each layer not at beginning of next layer as in diagram in paper. (no patchmerge in final layer)
            # afer layer_0 and patchmerge: torch.Size([B, 3136, 192]) - B,(H/8)*(W/8),2C
            # afer layer_1 and patchmerge: torch.Size([B, 784, 384]) - B,(H/16)*(W/16),4C
            # afer layer_2 and patchmerge: torch.Size([B, 196, 768]) - B,(H/32)*(W/32),8C
            # afer layer_3 (no patchmerge): torch.Size([B, 196, 768])

            # Reshape to [B, C, H, W]
            h_w = int(x.shape[1] ** 0.5)
            features.append(x.permute(0, 2, 1).reshape(x.shape[0], self.feature_dims[iii], h_w, h_w))

        # _________ Downsample to smallest scale and average feature vector for contrastive learning __________
        # Downsample all features to smallest scale (last feature map)
        # Skip connections - connect each scale directly to final feat map by downsample then conv1x1 to get features
        jong_feats0 = self.layer0tofinal(features[0])
        jong_feats1 = self.layer1tofinal(features[1])
        jong_feats = self.finalfinal(torch.cat([jong_feats0,jong_feats1,features[-1]],dim=1))
        # jong_feats = torch.flatten(jong_feats, 1)  # B,NxC
        # jong_feats = F.normalize(jong_feats, dim=1)
        return jong_feats  # Feature vector for contrastive loss
