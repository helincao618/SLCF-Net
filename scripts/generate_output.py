import sys
import os
# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
from models.slcfnet import SLCFNet
from data.semantic_kitti.kitti_dm import KittiDataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle
import data.semantic_kitti.io_data as SemanticKittiIO


@hydra.main(config_name='../config/sfc.yaml')
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    config_path = os.path.join(
            get_original_cwd(),
            'SLCF-Net',
            'SLCF-Net',
            'config',
            'semantic-kitti.yaml',
        )
    learning_map_inv = SemanticKittiIO.get_inv_map(config_path)
    # Setup dataloader
    feature = 64
    project_scale = 2
    full_scene_size = (256, 256, 32)
    data_module = KittiDataModule(
        voxel_root = config.kitti_voxel_root,
        preprocess_root=config.kitti_preprocess_root,
        frustum_size=config.frustum_size,
        batch_size=config.inference_batch_size,
        num_workers=int(config.num_workers_per_gpu),
    )
    data_module.setup()
    data_loader = data_module.val_dataloader()
    # data_loader = data_module.test_dataloader() # use this if you want to infer on test set

    # Load pretrained models

    model_path = os.path.join(
        get_original_cwd(), 'SLCF-Net', 'trained_models', 'mIoU=0.15071.ckpt'
    )

    model = SLCFNet.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data 
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.output_path, config.dataset)
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
            for i in range(config.inference_batch_size):
                out_dict = {'y_pred': y_pred[i].astype(np.uint16)}
                if 'target_1_1' in batch:
                    out_dict['target_1_1'] = (
                        batch['target_1_1'][i].detach().cpu().numpy().astype(np.uint16)
                    )

                # if config.dataset == 'NYU':
                #     write_path = output_path
                #     filepath = os.path.join(write_path, batch['name'][i] + '.pkl')
                #     out_dict['cam_pose'] = batch['cam_pose'][i].detach().cpu().numpy()
                #     out_dict['vox_origin'] = (
                #         batch['vox_origin'][i].detach().cpu().numpy()
                #     )
                # else:
                write_path = os.path.join(output_path, batch['sequence'][i])
                filepath = os.path.join(write_path, batch['frame_id'][i] + '.pkl')
                submission_dir = os.path.join(output_path, 'submission', 'sequences', batch['sequence'][i], 'predictions')
                submission_path = os.path.join(submission_dir, batch['frame_id'][i] + '.label')
                    
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
                    print('wrote to', filepath)
                # submission
                os.makedirs(submission_dir, exist_ok=True)
                y_transformed = learning_map_inv[out_dict['y_pred'].reshape(-1)].astype(np.uint16)
                y_transformed.tofile(submission_path)


if __name__ == '__main__':
    main()
