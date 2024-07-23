from pytorch_lightning import Trainer
import sys
import os
import yaml
# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
from models.slcfnet import SLCFNet
from data.semantic_kitti.kitti_dm import KittiDataModule
import torch


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_slcfnet_path = os.path.join(
        'SLCF-Net','slcfnet','config','slcfnet.yaml'
        )
    config_slcfnet = load_config(config_slcfnet_path)
    torch.set_grad_enabled(False)

    # config.batch_size = 1
    n_classes = 20
    feature = 64
    project_scale = 2
    full_scene_size = (256, 256, 32)
    data_module = KittiDataModule(
        voxel_root=config_slcfnet['kitti_voxel_root'],
        preprocess_root=config_slcfnet['kitti_preprocess_root'],
        frustum_size=config_slcfnet['frustum_size'],
        batch_size=int(config_slcfnet['batch_size'] / config_slcfnet['n_gpus']),
        num_workers=int(config_slcfnet['num_workers_per_gpu']),
    )
    trainer = Trainer(
        sync_batchnorm=True, deterministic=True, gpus=config_slcfnet['n_gpus'], accelerator='ddp'
    )
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
    model.eval()
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    trainer.test(model, test_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
