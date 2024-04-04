import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from data.utils.data_process import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
    inter_frame_mapping,
)
import data.semantic_kitti.io_data as SemanticKittiIO


class KittiDataset(Dataset):
    def __init__(
        self,
        split,
        voxel_root,
        preprocess_root,
        # project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
    ):
        super().__init__()

        self.voxel_root = voxel_root
        self.preprocess_root = preprocess_root
        self.img_W = 1220
        self.img_H = 370
        self.voxel_size = 0.2  # 0.2m
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0.1,-25.5,-1.9])
        self.n_classes = 20
        splits = {
            'train': ['00', '01', '02', '03', '04', '05', '06', '07'],
            'val': ['09','10'],
            'test': ['08'],
        }
        self.split = split
        self.grid_dimensions = [256, 32, 256]  # (W, H, D)
        self.sequences = splits[split]
        self.frustum_size = frustum_size
        
        self.fliplr = fliplr
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.scans = []
        for sequence in self.sequences:
            calib = SemanticKittiIO.read_calib_SemKITTI(
                os.path.join(self.voxel_root, 'dataset', 'sequences', sequence, 'calib.txt')
            )
            P = calib['P2']
            T_velo_2_cam = calib['Tr']
            poses = SemanticKittiIO.read_poses_SemKITTI(
                os.path.join(self.voxel_root, 'dataset', 'sequences', sequence, 'poses.txt')
            ) # pose for each frame
            glob_path = os.path.join(
                self.voxel_root, 'dataset', 'sequences', sequence, 'voxels', '*.bin'
            )
            voxel_paths = sorted(glob.glob(glob_path)) # voxel path for each 5 frames
            for i, voxel_path in enumerate(voxel_paths):
                is_first_frame_in_seq = (i == 0)
                self.scans.append(
                    {
                        'sequence': sequence,
                        'P': P,
                        'T_velo_2_cam': T_velo_2_cam,
                        'voxel_path': voxel_path,
                        'is_first_frame_in_seq': is_first_frame_in_seq,
                        'pose': poses[5*i],
                    }
                )

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]

        )

    def __getitem__(self, index):
        scan = self.scans[index]
        if self.split == 'train':
            if scan['is_first_frame_in_seq']:
                index += 1
                return self.load_single_frame(index-1, exsit_previous_pose = False), self.load_single_frame(index)
            else:
                return self.load_single_frame(index-1, exsit_previous_pose = False), self.load_single_frame(index)             
        else:
            if scan['is_first_frame_in_seq']:
                return self.load_single_frame(index, exsit_previous_pose = False)
            else:
                return self.load_single_frame(index)

    def load_single_frame(self, index, exsit_previous_pose = True):
        scan = self.scans[index]
        voxel_path = scan['voxel_path']
        P = scan['P']
        sequence = scan['sequence']
        pose_current = scan['pose']
        if exsit_previous_pose:
            pose_previous = self.scans[index-1]['pose']
        else:
            pose_previous = pose_current
        T_velo_2_cam = scan['T_velo_2_cam']

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = os.path.join(
            self.voxel_root, 'dataset', 'sequences', sequence, 'image_2', frame_id + '.png'
        )

        data = {
            'split':self.split,
            'frame_id': frame_id,
            'sequence': sequence,
            'P': P,
            'T_velo_2_cam': T_velo_2_cam,
        }
        scale_3ds = [1,2]
        data['scale_3ds'] = scale_3ds
        data['inter_frame_downscales'] = [1,2,4]
        cam_k = P[0:3, 0:3]
        data['cam_k'] = cam_k
        for scale_3d in scale_3ds:
            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                cam_E = T_velo_2_cam,
                P = P, 
                vox_origin = self.vox_origin,
                voxel_size = self.voxel_size * scale_3d,
                img_W = self.img_W,
                img_H = self.img_H, 
                scene_size = self.scene_size,
            )            

            data['projected_pix_{}'.format(scale_3d)] = projected_pix
            data['pix_z_{}'.format(scale_3d)] = pix_z
            data['fov_mask_{}'.format(scale_3d)] = fov_mask

        for downscale in data['inter_frame_downscales']:
            map_vox, overlap_mask = inter_frame_mapping(pose_previous, pose_current, T_velo_2_cam, downscale)
            data['map_vox_{}'.format(downscale)] = map_vox
            data['overlap_mask_{}'.format(downscale)] = overlap_mask

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != 'test':
            target_1_path = os.path.join(self.preprocess_root, sequence, 'labels', frame_id + '_1.npy')
            target_8_path = os.path.join(self.preprocess_root, sequence, 'labels', frame_id + '_8.npy')
            target_1_1 = np.load(target_1_path)
            target_1_8 = np.load(target_8_path)
            CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
            projected_pix_output = data['projected_pix_1']
            pix_z_output = data['pix_z_1']
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target_1_1,
                self.img_W,
                self.img_H,
                dataset='kitti',
                n_classes=20,
                size=self.frustum_size,
            )
            #semantic mask as gt
            semanticmask_path = os.path.join(self.preprocess_root, sequence, 'semanticmask', frame_id + '.npy')
            semanticmask = np.load(semanticmask_path)[:370, :1220]
            data['semanticmask'] = semanticmask
        else:
            frustums_masks = None
            frustums_class_dists = None
            target_1_1 = None
            CP_mega_matrix = None
        
        data['target_1_1'] = target_1_1
        data['CP_mega_matrix'] = CP_mega_matrix
        data['frustums_masks'] = frustums_masks
        data['frustums_class_dists'] = frustums_class_dists
        
        img = Image.open(rgb_path).convert('RGB')

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:370, :1220, :]  # crop image
        sparsedepthmask_path = os.path.join(self.preprocess_root, sequence, 'sparsedepthmask', frame_id + '.npy')
        sparsedepthmask = np.load(sparsedepthmask_path)
        sparsedepthmask = np.expand_dims(sparsedepthmask, axis=2)[:370, :1220, :]

        # Fliplr the image and volume
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            sparsedepthmask = np.ascontiguousarray(np.fliplr(sparsedepthmask))
            semanticmask = np.ascontiguousarray(np.fliplr(semanticmask))
            for scale in scale_3ds:
                key = 'projected_pix_' + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:, 0]

        data['img'] = self.normalize_rgb(img)
        data['sparsedepthmask'] = sparsedepthmask.reshape(1, 370, 1220)

        return data

    def __len__(self):
        if self.split == 'train':
            return len(self.scans)-1 # consider load 2 frames at same time
        else:
            return len(self.scans)