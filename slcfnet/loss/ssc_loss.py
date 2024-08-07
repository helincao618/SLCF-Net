import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import extract_features


def KL_sep(p, target):
    '''
    KL divergence on nonzeros classes
    '''
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction='sum')
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def CE_ssc_loss(pred, target, class_weights):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction='mean'
    )
    loss = criterion(pred, target.long())

    return loss

def inter_frame_loss(pred_1, pred_2, map_vox, overlap_mask, class_weights):
    '''
    Computes the inter-frame loss between predictions of two consecutive frames.
    
    :param pred_1: The output for the first frame, shape: [N, C, H, W, D]
    :param pred_2: The output for the second frame, shape: [N, C, H, W, D]
    :param map_vox: Tensor containing 3D indices for mapping pred_1 to the space of pred_2.
    :param overlap_mask: Boolean tensor to ensure valid mapping between frames.
    :param class_weights: Tensor of class weights.
    
    :return: Loss value computed based on the pseudo-labels from pred_1 and the predictions in pred_2.
    '''
    N, C, H, W, D = pred_1.shape
    # Map pred_1 to the new volume using extract_features function
    aligned_pred_1 = extract_features(pred_1, map_vox, overlap_mask)
    
    # Use the argmax of aligned_pred_1 as the pseudo-label for comparison
    pseudo_labels = torch.argmax(aligned_pred_1, dim=1).long()
    
    # Modify pseudo_labels outside the overlapping regions to be ignored
    pseudo_labels_flat = pseudo_labels.view(N, -1)
    pseudo_labels_flat[~overlap_mask] = 255  # Set regions outside overlap to 255
    pseudo_labels_modified = pseudo_labels_flat.view(N, H, W, D)
    
    # Define the criterion with ignore_index for areas outside overlap region
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean')
    
    # Compute the loss only on overlapping regions
    loss = criterion(pred_2, pseudo_labels_modified)

    return loss

