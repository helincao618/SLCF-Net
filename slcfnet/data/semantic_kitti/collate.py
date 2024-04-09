import torch

def collate_fn(batch):#batch[({},{})*batchsize] in train [{}*batchsize] in val and test
    len_seq = len(batch[0])
    if len_seq == 2: # train
        batch_now = [batch[0][0]] 
        batch_next = [batch[0][1]]
        ret_data_now = collate_fn_single(batch_now)
        ret_data_next = collate_fn_single(batch_next)
        return ret_data_now, ret_data_next
    else: # test/val
        ret_data = collate_fn_single(batch)
        return ret_data


def collate_fn_single(batch):#[{}]
    data = {}
    imgs = []
    sparsedepthmasks = []
    semanticmasks = []
    CP_mega_matrices = []
    targets_1_1 = []
    frame_ids = []
    sequences = []
    Ps = []
    cam_ks = []
    T_velo_2_cams = []
    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]['scale_3ds']
    for scale_3d in scale_3ds:
        data['projected_pix_{}'.format(scale_3d)] = []
        data['fov_mask_{}'.format(scale_3d)] = []
        data['pix_z_{}'.format(scale_3d)] = []
    
    inter_frame_downscales = batch[0]['inter_frame_downscales']
    for downscale in inter_frame_downscales:
        data['map_vox_{}'.format(downscale)] = []
        data['overlap_mask_{}'.format(downscale)] = []

    for input_dict in batch:
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        Ps.append(torch.from_numpy(input_dict['P']).double())
        T_velo_2_cams.append(torch.from_numpy(input_dict['T_velo_2_cam']).float())
        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))
        imgs.append(input_dict['img'])
        frame_ids.append(input_dict['frame_id'])
        sequences.append(input_dict['sequence'])
        sparsedepthmasks.append(torch.from_numpy(input_dict['sparsedepthmask']))
        if input_dict['split'] != 'test':
            frustums_masks.append(torch.from_numpy(input_dict['frustums_masks']))
            frustums_class_dists.append(torch.from_numpy(input_dict['frustums_class_dists']).float())
            semanticmasks.append(torch.from_numpy(input_dict['semanticmask']))
            targets_1_1.append(torch.from_numpy(input_dict['target_1_1']))
            CP_mega_matrices.append(torch.from_numpy(input_dict['CP_mega_matrix']))

            ret_data = {
                'frame_id': frame_ids,
                'sequence': sequences,
                'frustums_class_dists': frustums_class_dists,
                'frustums_masks': frustums_masks,
                "cam_k": cam_ks,
                'P': Ps,
                'T_velo_2_cam': T_velo_2_cams,
                'img': torch.stack(imgs),
                'sparsedepthmask': torch.stack(sparsedepthmasks),
                'semanticmask': torch.stack(semanticmasks),
                'CP_mega_matrices': CP_mega_matrices,
                'target_1_1': torch.stack(targets_1_1),
            }
        else:
            ret_data = {
                'frame_id': frame_ids,
                'sequence': sequences,
                "cam_k": cam_ks,
                'P': Ps,
                'T_velo_2_cam': T_velo_2_cams,
                'img': torch.stack(imgs),
                'sparsedepthmask': torch.stack(sparsedepthmasks),
                # 'target_1_1': torch.stack(targets_1_1),
            }
        
    for key in data:
        ret_data[key] = data[key]
    return ret_data
