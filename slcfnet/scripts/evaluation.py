from pytorch_lightning import Trainer
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

from hydra.utils import get_original_cwd


@hydra.main(config_name='../config/slcfnet.yaml')
def main(config: DictConfig):
    torch.set_grad_enabled(False)

    # config.batch_size = 1
    n_classes = 20
    feature = 64
    project_scale = 2
    full_scene_size = (256, 256, 32)
    data_module = KittiDataModule(
        voxel_root=config.kitti_voxel_root,
        preprocess_root=config.kitti_preprocess_root,
        frustum_size=config.frustum_size,
        batch_size=int(config.batch_size / config.n_gpus),
        num_workers=int(config.num_workers_per_gpu),
    )

    trainer = Trainer(
        sync_batchnorm=True, deterministic=True, gpus=config.n_gpus, accelerator='ddp'
    )


    model_path = os.path.join(
        get_original_cwd(), 'SLCF-Net', 'trained_models', 'kitti.ckpt'
    )

    model = SLCFNet.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
    )
    model.eval()
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    trainer.test(model, test_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
