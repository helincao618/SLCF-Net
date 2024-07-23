"""
Preprocess the Semantic Kitti dataset to generate the following files:
    -sparsedepthmask: sparse depth mask for each frame
    -depthimage: depth image for each frame
    -label: label for each frame with scale 1/1, 1/8
    -invalid: invalid mask for each frame with scale 1/1, 1/8

This code is based on the original Semantic Kitti dataset preprocessing code
"""
import glob
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
import data.semantic_kitti.io_data as SemanticKittiIO

# Constants
VISUALIZE = False
CAM2TOCAM0 = np.array([[1,0,0,-0.06],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
R0_RECT = np.array([[9.999239000000e-01,9.837760000000e-03,-7.445048000000e-03,0],
                    [-9.869795000000e-03,9.999421000000e-01,-4.278459000000e-03,0],
                    [7.402527000000e-03,4.351614000000e-03,9.999631000000e-01,0],
                    [0,0,0,1]])
DOWNSCALE = [1, 8]
# Functions definition
def _downsample_label(label, invalid, voxel_size, downscale):
    if downscale == 1:
        return label, invalid
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    invalid = np.zeros_like(label)
    invalid[np.isclose(label, 255)] = 1
    return label_downscale, invalid

def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):
                sub_m = grid[
                    (xx * k_size) : (xx * k_size) + k_size,
                    (yy * k_size) : (yy * k_size) + k_size,
                    (zz * k_size) : (zz * k_size) + k_size,
                ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result

def dot(transform, pts):
    if pts.shape[1] == 3:
        pts = np.concatenate([pts,np.ones((len(pts),1))],1)
    return (transform @ pts.T).T

# def img2point(u, v, d, P):
#     # Create a homogeneous image coordinate
#     uv1 = np.array([u, v, 1])
#     # Compute the homogeneous 3D point in camera coordinates
#     X_c = np.linalg.pinv(P) @ (d * uv1)
#     # Normalize the homogeneous coordinate by dividing by the last element
#     X_c /= X_c[-1]
#     return X_c[:-1]

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_slcfnet_path = os.path.join(
        'SLCF-Net','slcfnet','config','slcfnet.yaml'
        )
    config_kitti_path = os.path.join(
        'SLCF-Net','slcfnet','config','semantic-kitti.yaml'
        )
    config_slcfnet = load_config(config_slcfnet_path)
    remap_lut = SemanticKittiIO.get_remap_lut(config_kitti_path)
    splits = {
        'train': ['00', '01', '02', '03', '04', '05', '06', '07'],
        'val': ['09','10'],
        'test': ['08'],
    }
    scene_size =[256, 256, 32]
    imgh, imgw = 376, 1241
    split = 'val'
    sequences = splits[split]

    for sequence in sequences:
        # Read paths and store in lists
        voxel_path = os.path.join(
            config_slcfnet['kitti_voxel_root'], 'dataset', 'sequences', sequence
        )
        velodyne_path = os.path.join(
            config_slcfnet['kitti_pointcloud_root'], 'dataset', 'sequences', sequence,
        )
        if split == 'train' or split == 'val':
            label_paths = sorted(
                glob.glob(os.path.join(voxel_path, 'voxels', '*.label'))
            )
            invalid_paths = sorted(
                glob.glob(os.path.join(voxel_path, 'voxels', '*.invalid'))
            )
            # Create output directories
            label_outdir = os.path.join(config_slcfnet['kitti_preprocess_root'], sequence, 'labels')
            invalid_outdir = os.path.join(config_slcfnet['kitti_preprocess_root'], sequence, 'invalid')
            os.makedirs(label_outdir, exist_ok=True)
            os.makedirs(invalid_outdir, exist_ok=True)
        pts_paths = sorted(
            glob.glob(os.path.join(velodyne_path, 'velodyne', '*.bin'))
        )
        poses = SemanticKittiIO.read_poses_SemKITTI(os.path.join(voxel_path, 'poses.txt'))
        calib = SemanticKittiIO.read_calib_SemKITTI(os.path.join(voxel_path, 'calib.txt'))
        p2 = calib['P2']
        T_velo_2_cam = calib['Tr']

        # Create output directories
        sparsedepthmask_outdir = os.path.join(config_slcfnet['kitti_preprocess_root'], sequence, 'sparsedepthmask')
        os.makedirs(sparsedepthmask_outdir, exist_ok=True)

        for i in tqdm(range((len(pts_paths)-1)//5+1)):
            frame_id, extension = os.path.splitext(os.path.basename(pts_paths[5*i]))
            ## Semantic and depth mask and image
            sparsedepthmask = np.ones([imgh,imgw], dtype = int)*200
            # point cloud from velodyne
            pose_velo = np.linalg.inv(T_velo_2_cam).dot(poses[5*i].dot(T_velo_2_cam))
            scan = SemanticKittiIO.read_pointcloud_SemKITTI(pts_paths[5*i])
            scan_global = dot(pose_velo, scan)
            pose_cam2 = np.linalg.inv(CAM2TOCAM0).dot(np.linalg.inv(poses[5*i]).dot(T_velo_2_cam))
            
            if split == 'train' or split == 'val':
                
                ## Reading and Downscaling labels and invalid masks
            
                LABEL = SemanticKittiIO.read_label_SemKITTI(label_paths[i])
                INVALID = SemanticKittiIO.read_invalid_SemKITTI(invalid_paths[i])
                LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
                LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
                LABEL = LABEL.reshape(scene_size)
                for scale in DOWNSCALE:
                    filename = frame_id + '_' + str(scale) + '.npy'
                    label_filename = os.path.join(label_outdir, filename)
                    invalid_filename = os.path.join(invalid_outdir, filename)
                    # If files have not been created...
                    if not os.path.exists(label_filename) & os.path.exists(invalid_filename):
                        LABEL_ds, INVALID_ds = _downsample_label(LABEL, INVALID, scene_size, scale)
                        np.save(label_filename, LABEL_ds)
                        np.save(invalid_filename, INVALID_ds)
            
            # generate sparse depth mask fuse 5 frame
            fused_scan_global = np.array([[0,0,0,0]])
            for j in range(1):
                if 5*i+j <len(pts_paths):
                    single_scan = SemanticKittiIO.read_pointcloud_SemKITTI(pts_paths[5*i+j])
                    pose_single_velo = np.linalg.inv(T_velo_2_cam).dot(poses[5*i+j].dot(T_velo_2_cam))
                    scan_global = dot(pose_single_velo, single_scan)
                    fused_scan_global = np.concatenate([fused_scan_global,scan_global],0)
            pos_scans = dot(pose_cam2, fused_scan_global[:,:3])
            pos_scans = pos_scans[pos_scans[:,2]>0]
            pos_img_scans = p2 @ R0_RECT @ pos_scans.T
            pos_img_scans[:2] /= pos_img_scans[2,:]
            pos_img_scans = pos_img_scans.T
            pos_img_scans = pos_img_scans[pos_img_scans[:, 0] > 0]
            pos_img_scans = pos_img_scans[pos_img_scans[:, 0] < imgw]
            pos_img_scans = pos_img_scans[pos_img_scans[:, 1] > 0]
            pos_img_scans = pos_img_scans[pos_img_scans[:, 1] < imgh]
            for idx, pos_img_scan in enumerate(pos_img_scans):
                sparsedepthmask[int(pos_img_scan[1]),int(pos_img_scan[0])] = pos_img_scan[2]
            np.save(sparsedepthmask_outdir +'/'+ frame_id +'.npy',sparsedepthmask.astype(np.float32))
            # draw on img
            # cmap = plt.cm.jet
            # norm_depth_map = (sparsedepthmask - 0) / (100 - 0)
            # norm_depth_map[sparsedepthmask == 200] = 0
            # color_image = cmap(norm_depth_map)
            # color_image[sparsedepthmask == 200, 0] = 1.0
            # color_image[sparsedepthmask == 200, 1] = 1.0
            # color_image[sparsedepthmask == 200, 2] = 1.0
            # plt.imsave(sparsedepthimg_outdir +'/'+ frame_id+'.png', color_image)

if __name__ == '__main__':
    main()