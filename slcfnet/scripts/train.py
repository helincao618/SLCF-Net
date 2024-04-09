from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import sys
from datetime import datetime

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
from data.semantic_kitti.kitti_dm import KittiDataModule
from data.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names,
)
from models.slcfnet import SLCFNet


hydra.output_subdir = None


@hydra.main(config_name='../config/slcfnet.yaml')
def main(config: DictConfig):
    now = datetime.now()
    exp_name = config.exp_prefix
    exp_name += now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name += '_{}_{}'.format(config.dataset, config.run)
    exp_name += '_FrusSize_{}'.format(config.frustum_size)
    exp_name += '_nRelations{}'.format(config.n_relations)
    exp_name += '_WD{}_lr{}'.format(config.weight_decay, config.lr)
    if config.CE_ssc_loss:
        exp_name += '_CEssc'
    if config.geo_scal_loss:
        exp_name += '_geoScalLoss'
    if config.sem_scal_loss:
        exp_name += '_semScalLoss'
    if config.fp_loss:
        exp_name += '_fpLoss'
    if config.relation_loss:
        exp_name += '_CERel'
    if config.context_prior:
        exp_name += '_3DCRP'
    if config.inter_frame_loss:
        exp_name += '_interframe'

    # Setup dataloaders
    class_names = kitti_class_names
    max_epochs = 10
    logdir = config.kitti_logdir
    full_scene_size = (256, 256, 32)
    project_scale = 2
    feature = 64
    num_classes = 20
    class_weights = torch.from_numpy(
        1 / np.log(semantic_kitti_class_frequencies + 0.001)
    )
    data_module = KittiDataModule(
        voxel_root = config.kitti_voxel_root,
        preprocess_root=config.kitti_preprocess_root,
        frustum_size=config.frustum_size,
        batch_size=int(config.batch_size / config.n_gpus),
        num_workers=int(config.num_workers_per_gpu),
    )

    project_res = ['1']
    if config.project_1_2:
        exp_name += '_2'
        project_res.append('2')
    if config.project_1_4:
        exp_name += '_4'
        project_res.append('4')
    if config.project_1_8:
        exp_name += '_8'
        project_res.append('8')

    print(exp_name)

    # Initialize SFC model
    model = SLCFNet(
        dataset=config.dataset,
        frustum_size=config.frustum_size,
        feature=feature,
        full_scene_size=full_scene_size,
        num_classes=num_classes,
        class_names=class_names,
        context_prior=config.context_prior,
        fp_loss=config.fp_loss,
        relation_loss=config.relation_loss,
        CE_ssc_loss=config.CE_ssc_loss,
        sem_scal_loss=config.sem_scal_loss,
        geo_scal_loss=config.geo_scal_loss,
        lr=config.lr,
        weight_decay=config.weight_decay,
        class_weights=class_weights,
        project_scale=project_scale,
        project_res=project_res,
    )

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version='')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor='val/mIoU',
                save_top_k=-1,
                mode='max',
                filename='{epoch:03d}-{val/mIoU:.5f}',
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False
    model_path = os.path.join(logdir, 'exp' + exp_name[22:], 'checkpoints/last.ckpt')
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        print('Loading checkpoint from ' + model_path)
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator='ddp',
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator='ddp',
        )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
