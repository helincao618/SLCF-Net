# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.init as init
from models.modules import SegmentationHead
from models.CRP3D import CPMegaVoxels
from models.modules import Process, Upsample, Downsample
from models.modules import extract_features


class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        batch_size_per_gpu,
        feature, # num of features/channels feed in unet3d
        context_prior=False,
        bn_momentum=0.1,
        reg_value = 0.9, # regulerization value
    ):
        super(UNet3D, self).__init__()
        self.business_layer = []
        self.full_scene_size = full_scene_size
        self.feature = feature
        self.context_prior = context_prior
        self.reg_value = reg_value
        self.batch_size_per_gpu = batch_size_per_gpu
        self.initial_hidden_l1 = nn.Parameter(torch.randn([1, 64, 128, 128, 16]))
        self.initial_hidden_l2 = nn.Parameter(torch.randn([1, 128, 64, 64, 8]))
        # init learnable init hidden layer with kaiming init
        init.kaiming_normal_(self.initial_hidden_l1, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.initial_hidden_l2, mode='fan_in', nonlinearity='relu')
        self.hidden_l1 = None
        self.hidden_l2 = None
        size_l1 = (self.full_scene_size[0] // 2, self.full_scene_size[1] // 2, self.full_scene_size[2] // 2)
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        dilations = [1, 2, 3]
        self.process_l1 = nn.Sequential(
            Process(self.feature, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 2, norm_layer, bn_momentum),
        )
        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.up_l1_lfull = Upsample(
            self.feature, self.feature // 2, norm_layer, bn_momentum
        )

        self.ssc_head = SegmentationHead(
            self.feature // 2, self.feature // 2, class_num, dilations
        )

        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature * 4, size_l3, bn_momentum=bn_momentum
            )

    def forward(self, input_dict):
        res = {}
        x3d_l1 = input_dict['x3d'] #torch.Size([batch_size, 64, 128, 128, 16])
        x3d_l2 = self.process_l1(x3d_l1) #torch.Size([batch_size, 128, 64, 64, 8])
        x3d_l3 = self.process_l2(x3d_l2) #torch.Size([batch_size, 256, 32, 32, 4])

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_l3)
            x3d_l3 = ret['x']
            for k in ret.keys():
                res[k] = ret[k]
        if input_dict['if_first_frame']:
            self.hidden_l1 = self.initial_hidden_l1.repeat(self.batch_size_per_gpu, 1, 1, 1, 1)
            self.hidden_l2 = self.initial_hidden_l2.repeat(self.batch_size_per_gpu, 1, 1, 1, 1)
        aligned_hidden_l1 = extract_features(self.hidden_l1, input_dict["map_vox_2"], input_dict["overlap_mask_2"])
        aligned_hidden_l2 = extract_features(self.hidden_l2, input_dict["map_vox_4"], input_dict["overlap_mask_4"])
        
        x3d_up_l2 = self.reg_value *(self.up_13_l2(x3d_l3) + x3d_l2) + (1-self.reg_value)* aligned_hidden_l2
        x3d_up_l1 = self.reg_value * (self.up_12_l1(x3d_up_l2) + x3d_l1) + (1-self.reg_value) * aligned_hidden_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        ssc_logit_full = self.ssc_head(x3d_up_lfull)

        # save the hidden
        self.hidden_l1 = x3d_up_l1
        self.hidden_l2 = x3d_up_l2
        res['ssc_logit'] = ssc_logit_full
        return res
    
