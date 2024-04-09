import torch
import torch.nn as nn
from models.DDR import Bottleneck3D


class ASPP(nn.Module):
    '''
    ASPP 3D
    Adapt from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    '''
    def __init__(self, planes, dilations_conv_list):
        super().__init__()

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

    def forward(self, x_in):

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        return x_in


class SegmentationHead(nn.Module):
    '''
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Conv3D, ASPP block, Conv3D.
    Taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    '''

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

        self.conv_classes = nn.Conv3d(
            planes, nbr_classes, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x_in):

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in


class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[
                Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(
            feature,
            feature // 4,
            bn_momentum=bn_momentum,
            expansion=expansion,
            stride=2,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    feature,
                    int(feature * expansion / 4),
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                norm_layer(int(feature * expansion / 4), momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.main(x)
    
# class ImageSemanticHead(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(ImageSemanticHead, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=2, dilation=2)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.relu1 = nn.ReLU()

#         self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.relu2 = nn.ReLU()

#         self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)

#         x = self.classifier(x)
        
#         return x

def extract_features(hidden_feature, map_vox, overlap_mask):
    """
    Extracts features from a hidden tensor based on the provided 3D indices.
        
    Parameters:
    ----------
    hidden_feature : torch.Tensor of shape (N, C, H, W, D)
        The source tensor from which features need to be extracted.
        
    map_vox : torch.Tensor of shape (N, M, 3)
        The 3D indices for feature extraction. M is the number of voxels.
        
    overlap_mask : torch.Tensor of shape (N, M,)
        A boolean mask to ensure the indices in map_vox are valid and do not 
        exceed the boundary of the hidden_feature.     
    Returns:
    -------
    extracted_features : torch.Tensor of shape (N, C, H, W, D)
        The tensor containing the extracted features. Features are set to zero 
        wherever overlap_mask is False.
    """
        
    N, C, H, W, D = hidden_feature.shape
    
    # Convert 3D indices to 1D linear indices
    linear_index = map_vox[:, :, 0] * W * D + map_vox[:, :, 1] * D + map_vox[:, :, 2]
    
    # Reshape tensors for batched gather operation
    hidden_feature_flattened = hidden_feature.view(N, C, -1)
    
    # Append a column of zeros to the flattened tensor
    zeros_vec = torch.zeros(N, C, 1, device=hidden_feature.device, dtype=hidden_feature.dtype)
    hidden_feature_flattened = torch.cat([hidden_feature_flattened, zeros_vec], dim=2)
    
    # Set the out-of-bounds indices to point to the zeros column
    linear_index[~overlap_mask] = H * W * D
    
    # Use gather operation
    selected_features = hidden_feature_flattened.gather(2, linear_index.unsqueeze(1).expand(-1, C, -1))
    
    selected_features_reshaped = selected_features.view(N, C, H, W, D)

    return selected_features_reshaped