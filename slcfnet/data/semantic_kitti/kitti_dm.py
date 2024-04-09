from torch.utils.data.dataloader import DataLoader
from data.semantic_kitti.kitti_dataset import KittiDataset
import pytorch_lightning as pl
from data.semantic_kitti.collate import collate_fn
from data.utils.torch_util import worker_init_fn


class KittiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        voxel_root,
        preprocess_root,
        project_scale=4,
        frustum_size=4,
        batch_size=4,
        num_workers=6,
    ):
        super().__init__()
        self.voxel_root = voxel_root
        self.preprocess_root = preprocess_root
        self.project_scale = project_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size

    def setup(self, stage=None):
        self.train_ds = KittiDataset(
            split='train',
            voxel_root = self.voxel_root,
            preprocess_root=self.preprocess_root,
            # project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
        )

        self.val_ds = KittiDataset(
            split='val',
            voxel_root = self.voxel_root,
            preprocess_root=self.preprocess_root,
            # project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
        )

        self.test_ds = KittiDataset(
            split='test',
            voxel_root = self.voxel_root,
            preprocess_root=self.preprocess_root,
            # project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
