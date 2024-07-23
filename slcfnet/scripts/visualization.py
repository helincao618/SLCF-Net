## Visualize the fused and seprate sequence and save the entire sequence
from dvis import dvis
import numpy as np
import yaml
from tqdm import tqdm
import os
import argparse
import pickle
import sys
# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
from data.semantic_kitti.io_data import read_poses_SemKITTI

SEQ = ['08']  # set the seq you want to visualize

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--iffuse', type=bool, default=False, help='if visualize all the catagories')
    return parser.parse_args()

def dot(transform, pts):
    if pts.shape[1] == 3:
        pts = np.concatenate([pts,np.ones((len(pts),1))],1)
    return (transform @ pts.T).T

def process_scene_coords(target, pose_velo, color_map, learning_map_inv, fov_mask, unknown_mask):
    #invalid_mask labels the unknown
    valid_label_mask = (target > 0) & (target < 255)
    valid_labels = target[valid_label_mask]
    valid_fov_mask = fov_mask[valid_label_mask]
    valid_label_inds = np.stack(np.nonzero(valid_label_mask), 1)
    valid_unknown_mask = unknown_mask[valid_label_mask]

    vox2scene = np.eye(4)
    vox2scene[:3, :3] = np.diag([1/5, 1/5, 1/5])
    vox2scene[:3, 3] = np.array([-25.5, -25.5, -1.9])#np.array([0.1, -25.5, -1.9])
    valid_scene_coords = dot(vox2scene, valid_label_inds)
    valid_scene_coords_global = valid_scene_coords[:, :3]#.dot(pose_velo.T)[:, :3]

    valid_colors = np.zeros((len(valid_scene_coords), 3))
    for label in np.unique(valid_labels):
        label_mask = valid_labels == label
        valid_colors[label_mask] = color_map[learning_map_inv[label]][::-1]
    
    darker_value = 90
    darken_array = np.ones_like(valid_colors) * darker_value
    valid_colors[~valid_fov_mask] -= darken_array[~valid_fov_mask]
    valid_colors = np.clip(valid_colors, 0, 255)

    # convert to aligned voxel
    valid_scene_coords_global = valid_scene_coords_global * 5
    valid_scene_coords_global.astype(int)
    valid_scene_coords_global = valid_scene_coords_global / 5
    valid_scene_coords_col = np.concatenate([valid_scene_coords_global, valid_colors], 1)
    valid_scene_coords_col_known = valid_scene_coords_col[valid_unknown_mask == 0]
    valid_scene_coords_col_unknown = valid_scene_coords_col[valid_unknown_mask == 1]
    return valid_scene_coords_col_known, valid_scene_coords_col_unknown

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_slcfnet_path = os.path.join(
        'SLCF-Net','slcfnet','config','slcfnet.yaml'
        )
    config_slcfnet = load_config(config_slcfnet_path)
    preprocess_root = os.path.join(config_slcfnet['kitti_preprocess_root'])
    output_path = os.path.join(config_slcfnet['output_path'])
    config_kitti_path = os.path.join('SLCF-Net', 'slcfnet', 'config', 'semantic-kitti.yaml')
    config_kitti = yaml.safe_load(open(config_kitti_path))
    color_map = config_kitti['color_map']
    learning_map_inv = config_kitti['learning_map_inv']
    args = parse_args()
    for seq in SEQ[:1]:
        data_path = os.path.join(output_path, seq)
        path_poses = os.path.join(config_slcfnet['kitti_pointcloud_root'], 'dataset', 'sequences', seq, 'poses.txt')
        poses = read_poses_SemKITTI(path_poses)
        volume_colors = np.array([[0,0,0,0,0,0]])
        startframe = 0
        endframe = len(poses)
        for i in tqdm(range(startframe, endframe,5)):
            path_labels = os.path.join(data_path, "{:06}.pkl".format(i))
            unknown_path = os.path.join(preprocess_root, seq, 'invalid' , "{:06}".format(i) + '_1.npy')
            unknown_mask = np.load(unknown_path).reshape(256,256,32)
            if os.path.exists(path_labels):
                with open(path_labels, "rb") as handle:
                    b = pickle.load(handle)
                velo2cam = b['T_velo_2_cam']
                target = b["target_1_1"]
                pred = b['y_pred']
                fov_mask = b['fov_mask_1'].reshape(256,256,32)
                np.save('fov_mask.npy', fov_mask)
                pose_velo = np.linalg.inv(velo2cam).dot(poses[i].dot(velo2cam))
                processed_target, processed_target_unknown = process_scene_coords(target, pose_velo, color_map, learning_map_inv, fov_mask, unknown_mask)
                processed_pred, processed_pred_unknown  = process_scene_coords(pred, pose_velo, color_map, learning_map_inv, fov_mask, unknown_mask)
                if not args.iffuse:
                    dvis(processed_target, l=1, t=i, vs=1/5, name='target/semantic volume'+ str(i))
                    dvis(processed_target_unknown, l=1, t=i, vs=1/5, vis_conf={'transparent': True, 'opacity':0.2}, name='target/semantic volume'+ str(i))
                    dvis(processed_pred, l=2, t=i, vs=1/5, name='prediction/semantic volume'+ str(i))
                    dvis(processed_pred_unknown, l=2, t=i, vs=1/5, vis_conf={'transparent': True, 'opacity':0.2}, name='prediction/semantic volume'+ str(i))
                if args.iffuse:  
                    volume_colors = np.concatenate([volume_colors, processed_target],0)

        if args.iffuse:
            volume_colors = np.unique(volume_colors, axis=0)# delete the repeated voxel
            dvis(volume_colors, l=4, vs=1/5, ms=1000000, name='volume/fused volume')


if __name__ == "__main__":
    main()
