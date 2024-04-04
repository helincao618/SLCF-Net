import torch
import torch.nn as nn


class GDP(nn.Module):
    def __init__(self, scene_size, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, scale_2d, fov_mask, pix_z, depth_img):
        '''
        src is the feature from rgb_net, len(pix_x) == 256*256*32
        calculate the indecies form pix_x, pix_y, and gather the feather from src according to indecies,
        the the feature of image is projected to 3d volume
        '''
        c, h, w = x2d.shape
        src = x2d.view(c, -1)
        zeros_vec = torch.zeros(c, 1, device=src.device).type_as(src)
        src = torch.cat([src, zeros_vec], 1) 
        scaled_pix_x, scaled_pix_y = projected_pix[:, 0]//scale_2d, projected_pix[:, 1]//scale_2d
        scaled_img_indices = scaled_pix_y * w + scaled_pix_x
        # if one voxel not in fov (False in fov_mask), set it to last order, 
        # we add 0 as last one, so it should be the zero vector
        scaled_img_indices[~fov_mask] = h * w
        scaled_img_indices = scaled_img_indices.expand(c, -1).long()  # c, HWD
        src_feature = torch.gather(src, 1, scaled_img_indices)
        # Flatten the depth image and gather the depths according to indices.
        depth_flat = depth_img.contiguous().view(1, -1)
        zeros_item = torch.zeros(1, 1, device=src.device).type_as(depth_flat)
        depth_flat = torch.cat([depth_flat, zeros_item], 1)
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices[~fov_mask] = depth_img.shape[1] * depth_img.shape[2]
        img_indices = img_indices.expand(1, -1).long()
        depth_values = torch.gather(depth_flat, 1, img_indices)
        weights = self.gaussian_decay_weight(pix_z, depth_values, mu=0.0, sigma=1.0).float()  # Call the Gaussian decay function
        # Apply the Gaussian weights to the features.
        weighted_feature = src_feature * weights

        x3d = weighted_feature.reshape(
            c,
            self.scene_size[0] // self.project_scale,
            self.scene_size[1] // self.project_scale,
            self.scene_size[2] // self.project_scale,
        )
        return x3d


    def gaussian_decay_weight(self, pix_z, depth_value, mu=0.0, sigma=1.0):
        """
        This function calculates a Gaussian weight for a given distance.

        Parameters:
        pix_z: pixel depth values (torch.Tensor)
        depth_value: depth values (torch.Tensor)
        mu: mean of the Gaussian distribution (torch.Tensor or float)
        sigma: standard deviation of the Gaussian distribution (torch.Tensor or float)

        Returns:
        Gaussian weight for the given distance (torch.Tensor)
        """
        x = pix_z - depth_value
        gaussian_weight = torch.exp(-0.5 * ((x - mu) / (sigma / self.project_scale)) ** 2)
        # Replace the Gaussian weight with 1.0 where depth_value is 0
        mask = (depth_value == 0.0)
        one_item = torch.tensor(1.0, device=gaussian_weight.device, dtype=gaussian_weight.dtype)
        gaussian_weight_with_mask = torch.where(mask, one_item, gaussian_weight)

        return gaussian_weight_with_mask


