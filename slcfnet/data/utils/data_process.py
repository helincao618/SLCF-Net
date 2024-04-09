import numpy as np

def compute_CP_mega_matrix(target, is_binary=False):
    '''
    Parameters
    ---------
    target: (H, W, D)
        contains voxels semantic labels

    is_binary: bool
        if True, return binary voxels relations else return 4-way relations
    '''
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i//2 for i in target.shape]
    if is_binary:
        matrix = np.zeros((2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)
    else:
        matrix = np.zeros((4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                label_col_megas = np.array([
                    target[xx * 2,     yy * 2,     zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2],
                    target[xx * 2,     yy * 2 + 1, zz * 2],
                    target[xx * 2,     yy * 2,     zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                    target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N)  * label_col_mega
                    if not is_binary:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0), col_idx] = 1.0 # non non same
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1.0 # non non diff
                        matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        matrix[3, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty
                    else:
                        matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0 # diff
                        matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0 # same
    return matrix

def dot(transform, pts):
    if pts.shape[1] == 3:
        pts = np.concatenate([pts,np.ones((len(pts),1))],1)
    return (transform @ pts.T).T

def cam2pix(cam_pts, intr):
    '''Convert camera coordinates to pixel coordinates.'''
    R0_RECT = np.array([[9.999239000000e-01,9.837760000000e-03,-7.445048000000e-03,0],
                    [-9.869795000000e-03,9.999421000000e-01,-4.278459000000e-03,0],
                    [7.402527000000e-03,4.351614000000e-03,9.999631000000e-01,0],
                    [0,0,0,1]])
    pos_img = intr @ R0_RECT @ cam_pts.T
    pos_img[:2] /= pos_img[2,:]
    pos_img = pos_img.T
    return pos_img


def vox2pix(cam_E, P, 
            vox_origin, voxel_size, 
            img_W, img_H, 
            scene_size):
    '''
    compute the 2D projection of voxels centroids
    
    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    P: 3x4
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2
    
    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels, N is the number of voxels (256*256*32)
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    '''
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vox_coords = np.stack(np.nonzero(np.ones([int(scene_size[0]/voxel_size),int(scene_size[1]/voxel_size),int(scene_size[2]/voxel_size)])),1)
    vox2pts = np.eye(4)
    vox2pts[:3,:3] = np.diag([voxel_size,voxel_size,voxel_size])
    vox2pts[:3,3] = vox_origin  
    velo_pts = dot(vox2pts, vox_coords)

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = dot(cam_E, velo_pts)

    # Project camera coordinates to pixel positions
    projected_pix = cam2pix(cam_pts, P).astype(int)
    pix_x, pix_y, pix_z = projected_pix[:, 0], projected_pix[:, 1], projected_pix[:, 2]
    projected_pix_xy = projected_pix[:, :2]
    # Eliminate pixels outside view frustum
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))

    return projected_pix_xy, fov_mask, pix_z


def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    valid_pix = np.logical_and(pix_x >= min_x,
                np.logical_and(pix_x < max_x,
                np.logical_and(pix_y >= min_y,
                np.logical_and(pix_y < max_y,
                pix_z > 0))))
    return valid_pix

def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    '''
    Compute the local frustums mask and their class frequencies
    
    Parameters:
    ----------
    projected_pix: (N, 2)
        2D projected pix of all voxels
    pix_z: (N,)
        Distance of the camera sensor to voxels
    target: (H, W, D)
        Voxelized sematic labels
    img_W: int
        Image width
    img_H: int
        Image height
    dataset: str
        ='NYU' or 'kitti' (for both SemKITTI and KITTI-360)
    n_classes: int
        Number of classes (12 for NYU and 20 for SemKITTI)
    size: int
        determine the number of local frustums i.e. size * size
    
    Returns
    -------
    frustums_masks: (n_frustums, N)
        List of frustums_masks, each indicates the belonging voxels  
    frustums_class_dists: (n_frustums, n_classes)
        Contains the class frequencies in each frustum
    '''
    H, W, D = target.shape
    ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == 'kitti':
                mask = (target != 255) & local_frustum.reshape(H, W, D)
            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
    return frustums_masks, frustums_class_dists


def inter_frame_mapping(pose_previous, pose_current, T_velo_2_cam, downscale = 1):
    '''
    compute the mapping of voxels centroids in different scale
    
    Parameters:
    ----------
    pose_previous: 4x4
        camera pose of previous frame
    pose_current: 4x4
        camera pose of current frame
    Returns
    -------
    map_vox: (N, 3)
        Projected 3D positions of voxels, N is the number of voxels (256*256*32)
    overlap_mask: (N,)
        Voxels mask indice voxels inside previous volume defination
    '''
    # generate indicies of volume
    shape = (256//downscale, 256//downscale, 32//downscale)
    indices = np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')).reshape(3, -1).T
    
    # define tranform matrix
    pose_current_velo = np.linalg.inv(T_velo_2_cam).dot(pose_current.dot(T_velo_2_cam))
    pose_previous_velo = np.linalg.inv(T_velo_2_cam).dot(pose_previous.dot(T_velo_2_cam))
   
    vox2pts = np.eye(4)
    vox2pts[:3,:3] = np.diag([0.2 * downscale, 0.2 * downscale, 0.2 * downscale])
    vox2pts[:3,3] = np.array([0.1, -25.5, -1.9])
    
    # transform current volume to pts in global frame
    pts_current = dot(vox2pts, indices)
    pts_current_global = dot(pose_current_velo, pts_current)
    pts_current_in_pose_previous = dot(np.linalg.inv(pose_previous_velo), pts_current_global)
    vox_current_in_pose_previous = dot(np.linalg.inv(vox2pts), pts_current_in_pose_previous)

    map_vox = np.round(vox_current_in_pose_previous[:,0:3]).astype(int)

    # overlap mask with updated shape values
    overlap_mask = (map_vox[:, 0] >= 0) & (map_vox[:, 0] < shape[0]) & \
                   (map_vox[:, 1] >= 0) & (map_vox[:, 1] < shape[1]) & \
                   (map_vox[:, 2] >= 0) & (map_vox[:, 2] < shape[2])

    return map_vox, overlap_mask