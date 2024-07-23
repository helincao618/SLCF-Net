import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
import yaml

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
from models.slcfnet import SLCFNet
from data.semantic_kitti.kitti_dm import KittiDataModule
import data.semantic_kitti.io_data as SemanticKittiIO

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    torch.set_grad_enabled(False)
    config_slcfnet_path = os.path.join(
        'SLCF-Net','slcfnet','config','slcfnet.yaml'
        )
    config_kitti_path = os.path.join(
        'SLCF-Net','slcfnet','config','semantic-kitti.yaml'
        )
    config_slcfnet = load_config(config_slcfnet_path)
    learning_map_inv = SemanticKittiIO.get_inv_map(config_kitti_path)
    # Setup dataloader
    feature = 64
    project_scale = 2
    full_scene_size = (256, 256, 32)
    data_module = KittiDataModule(
        voxel_root = config_slcfnet['kitti_voxel_root'],
        preprocess_root=config_slcfnet['kitti_preprocess_root'],
        frustum_size=config_slcfnet['frustum_size'],
        batch_size=config_slcfnet['inference_batch_size'],
        num_workers=int(config_slcfnet['num_workers_per_gpu']),
    )
    data_module.setup()
    data_loader = data_module.val_dataloader() # use this if you want to infer on val set
    # data_loader = data_module.test_dataloader() # use this if you want to infer on test set

    # Load pretrained models

    model_path = os.path.join(
        config_slcfnet['kitti_logdir'], 'trained_models', 'kitti.ckpt'
    )

    model = SLCFNet.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config_slcfnet['fp_loss'],
        full_scene_size=full_scene_size,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data 
    output_path = os.path.join(config_slcfnet['output_path'])
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch['img'] = batch['img'].cuda()
            batch['sparsedepthmask'] = batch['sparsedepthmask'].cuda()
            if batch['frame_id'][0] == '000000':
                batch['if_first_frame'] = True
            else:
                batch['if_first_frame'] = False
            pred = model(batch)
            y_pred = torch.softmax(pred['ssc_logit'], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            for i in range(config_slcfnet['inference_batch_size']):
                out_dict = {'y_pred': y_pred[i].astype(np.uint16)}
                if 'target_1_1' in batch:
                    out_dict['target_1_1'] = (
                        batch['target_1_1'][i].detach().cpu().numpy().astype(np.uint16)
                    )
                write_path = os.path.join(output_path, batch['sequence'][i])
                filepath = os.path.join(write_path, batch['frame_id'][i] + '.pkl')
                # submission_dir = os.path.join(output_path, 'submission', 'sequences', batch['sequence'][i], 'predictions')
                # submission_path = os.path.join(submission_dir, batch['frame_id'][i] + '.label')
                    
                out_dict['fov_mask_1'] = (
                    batch['fov_mask_1'][i].detach().cpu().numpy()
                )
                out_dict['cam_k'] = batch['cam_k'][i].detach().cpu().numpy()
                out_dict['T_velo_2_cam'] = (
                    batch['T_velo_2_cam'][i].detach().cpu().numpy()
                )
                
                os.makedirs(write_path, exist_ok=True)
                with open(filepath, 'wb') as handle:
                    pickle.dump(out_dict, handle)
                    # print('wrote to', filepath)
                # submission
                # os.makedirs(submission_dir, exist_ok=True)
                # y_transformed = learning_map_inv[out_dict['y_pred'].reshape(-1)].astype(np.uint16)
                # y_transformed.tofile(submission_path)


if __name__ == '__main__':
    main()
