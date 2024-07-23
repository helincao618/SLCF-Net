import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from models.GDP import GDP
from models.unet2d import UNet2D
from models.unet3d import UNet3D
from loss.CRP_loss import compute_super_CP_multilabel_loss
from loss.sscMetrics import SSCMetrics
from loss.ssc_loss import CE_ssc_loss, KL_sep, geo_scal_loss, inter_frame_loss


class SLCFNet(pl.LightningModule):
    def __init__(
        self,
        num_classes, # 20 in semantickitti
        class_names, 
        feature, # num of features/channels feed in unet3d
        class_weights,
        project_scale,
        full_scene_size,
        context_prior=True,
        fp_loss=True,
        frustum_size=4,
        img_sem_loss = True,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        inter_frame_loss = True,
        lr=1e-4,
        weight_decay=1e-4,
        project_res=[],
    ):
        super().__init__()
        self.project_res = project_res
        self.project_scale = project_scale
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.fp_loss = fp_loss
        self.img_sem_loss = img_sem_loss
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.inter_frame_loss = inter_frame_loss
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = GDP(
                full_scene_size, self.project_scale
            )
        self.projects = nn.ModuleDict(self.projects)
        self.num_classes = num_classes
        self.net_3d_decoder = UNet3D(
            class_num = self.num_classes,
            norm_layer = nn.BatchNorm3d,
            full_scene_size=full_scene_size,
            batch_size_per_gpu = 1,
            feature=feature,
            context_prior=context_prior,
        )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)
        # log hyperparameters
        self.save_hyperparameters()
        self.train_metrics = SSCMetrics(self.num_classes)
        self.val_metrics = SSCMetrics(self.num_classes)
        self.test_metrics = SSCMetrics(self.num_classes)
    
    def forward(self, batch):
        img = batch["img"]
        depth_img = batch["sparsedepthmask"]
        bs = len(img)
        out = {}
        x_rgb = self.net_rgb(img)
        x3ds = []
        map_vox_2s = []
        overlap_mask_2s = [] 
        map_vox_4s = []
        overlap_mask_4s = []
        x_depth = depth_img
        for i in range(bs):
            x3d = None
            for scale_2d in self.project_res:
                # project features at each 2D scale to target 3D scale
                scale_2d = int(scale_2d)
                projected_pix = batch["projected_pix_{}".format(self.project_scale)][i].cuda()
                fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].cuda()
                pix_z = batch["pix_z_{}".format(self.project_scale)][i].cuda()
                # Sum all the 3D features
                if x3d is None:
                    x3d = self.projects[str(scale_2d)](
                        x2d = x_rgb["1_" + str(scale_2d)][i],
                        projected_pix = projected_pix,
                        scale_2d = scale_2d, # because the correspondence of voxel & pixel is changed by scale
                        fov_mask = fov_mask,
                        pix_z = pix_z, #Howerver the depth is not influenced by scale
                        depth_img = x_depth[i],
                    )
                else:
                    x3d += self.projects[str(scale_2d)](
                        x2d = x_rgb["1_" + str(scale_2d)][i],
                        projected_pix = projected_pix,
                        scale_2d = scale_2d,
                        fov_mask = fov_mask,
                        pix_z = pix_z,
                        depth_img = x_depth[i],
                    )
            map_vox_2s.append(batch["map_vox_2"][i].cuda())
            overlap_mask_2s.append(batch["overlap_mask_2"][i].cuda())
            map_vox_4s.append(batch["map_vox_4"][i].cuda())
            overlap_mask_4s.append(batch["overlap_mask_4"][i].cuda())
            x3ds.append(x3d)
        input_dict = {
            "x3d": torch.stack(x3ds),
            "if_first_frame": batch["if_first_frame"],
            "map_vox_2": torch.stack(map_vox_2s),
            "overlap_mask_2": torch.stack(overlap_mask_2s),
            "map_vox_4": torch.stack(map_vox_4s),
            "overlap_mask_4": torch.stack(overlap_mask_4s),
            }
        out = self.net_3d_decoder(input_dict)
        return out

    def step(self, batch, step_type, metric):
        bs = len(batch['img'])
        loss = 0.0
        out_dict = self(batch)
        ssc_pred = out_dict['ssc_logit']
        target = batch['target_1_1']
        if self.context_prior:
            P_logits = out_dict['P_logits']
            CP_mega_matrices = batch['CP_mega_matrices']
            if self.relation_loss:
                loss_rel_ce = compute_super_CP_multilabel_loss(
                    P_logits, CP_mega_matrices
                )
                loss += loss_rel_ce
                self.log(
                    step_type + '/loss_relation_ce_super',
                    loss_rel_ce.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )
        class_weight = self.class_weights.type_as(batch['img'])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + '/loss_ssc',
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )
        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + '/loss_geo_scal',
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )
        if self.fp_loss and step_type != 'test':
            frustums_masks = torch.stack(batch['frustums_masks'])
            frustums_class_dists = torch.stack(
                batch['frustums_class_dists']
            ).float()  # (bs, n_frustums, num_classes)
            n_frustums = frustums_class_dists.shape[1]
            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, num_classes)
            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, num_classes, H, W, D
                prob = prob.reshape(bs, self.num_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.num_classes, -1)
                cum_prob = prob.sum(dim=1)  # num_classes
                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # num_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + '/loss_frustums',
                frustum_loss.detach(),
                on_epoch=True,
                sync_dist=True,
            )
        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)
        self.log(step_type + '/loss', loss.detach(), on_epoch=True, sync_dist=True)
        return loss, ssc_pred

    def training_step(self, batch, batch_idx):
        #{} in test/val  [{},{}] in train
        first_frame, second_frame = batch
        first_frame['if_first_frame'] = True
        second_frame['if_first_frame'] = False
        loss1, pred1 = self.step(first_frame, 'train', self.train_metrics)
        loss2, pred2 = self.step(second_frame, 'train', self.train_metrics)
        lambda1, lambda2 = 1.0, 1.0
        combined_loss = lambda1 * loss1 + lambda2 * loss2
        if self.inter_frame_loss:
            class_weight = self.class_weights.type_as(first_frame['img'])
            inter_frame_loss_ssc = inter_frame_loss(pred1, pred2, torch.stack(second_frame["map_vox_1"]), torch.stack(second_frame["overlap_mask_1"]), class_weight)
            self.log(
                'train/inter_frame_loss_ssc',
                inter_frame_loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )
            lambda3 = 0.5
            combined_loss += lambda3 * inter_frame_loss_ssc
        self.log('train' + '/loss', combined_loss.detach(), on_epoch=True, sync_dist=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        if batch['frame_id'][0] == '000000' or batch['frame_id'][0] == '000005':
            batch['if_first_frame'] = True
        else:
            batch['if_first_frame'] = False
        self.step(batch, 'val', self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [('train', self.train_metrics), ('val', self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    '{}_SemIoU/{}'.format(prefix, class_name),
                    stats['iou_ssc'][i],
                    sync_dist=True,
                )
            self.log('{}/mIoU'.format(prefix), stats['iou_ssc_mean'], sync_dist=True)
            self.log('{}/IoU'.format(prefix), stats['iou'], sync_dist=True)
            self.log('{}/Precision'.format(prefix), stats['precision'], sync_dist=True)
            self.log('{}/Recall'.format(prefix), stats['recall'], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        if batch['frame_id'][0] == '000000':
            batch['if_first_frame'] = True
        else:
            batch['if_first_frame'] = False
        self.step(batch, 'test', self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [('test', self.test_metrics)]
        for prefix, metric in metric_list:
            print('{}======'.format(prefix))
            stats = metric.get_stats()
            print(
                'Precision={:.4f}, Recall={:.4f}, IoU={:.4f}'.format(
                    stats['precision'] * 100, stats['recall'] * 100, stats['iou'] * 100
                )
            )
            print('class IoU: {}, '.format(classes))
            print(
                ' '.join(['{:.4f}, '] * len(classes)).format(
                    *(stats['iou_ssc'] * 100).tolist()
                )
            )
            print('mIoU={:.4f}'.format(stats['iou_ssc_mean'] * 100))
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        return [optimizer], [scheduler]
