import numpy as np
import torch
import math

from AB3DMOT.AB3DMOT_libs.box import Box3D


def get_2d_center_distance_matrix(gt_boxes, track_boxes):
  '''
  Calculate the distance matrix using 2d ground plane center distance
  Input:
    In ab3dmot kitti coordinate system
    xz is the ground plane, y is height
    gt_boxes: [M, 7=self.dim_z]
    track_boxes: [N, 7=self.dim_z]
    7:  [x, y, z, theta, l, w, h]
  Output:
    distance_matrix: [M, N]
  '''
  gt_boxes_extended = torch.unsqueeze(gt_boxes, 1)
  gt_boxes_extended = torch.cat([gt_boxes_extended[:,:, 0:1], gt_boxes_extended[:,:,2:3]], dim=2)
  #print('gt_boxes_extended.shape: ', gt_boxes_extended.shape)
  # [M, 1, 2]

  track_boxes_extended = torch.unsqueeze(track_boxes, 0)
  track_boxes_extended = torch.cat([track_boxes_extended[:,:, 0:1], track_boxes_extended[:,:,2:3]], dim=2)
  #print('track_boxes_extended.shape: ', track_boxes_extended.shape)
  # [1, N, 2]

  distance_matrix = torch.norm(gt_boxes_extended - track_boxes_extended, dim=2)
  #print('distance_matrix: ', distance_matrix)
  #print('distance_matrix.shape: ', distance_matrix.shape)
  # [M, N]

  return distance_matrix


def get_neg_log_likelihood_loss(mean_boxes, covariance, gt_boxes):
  '''
  Input:
    mean_boxes: torch [N, 7], 7:  [x, y, z, theta, l, w, h]
    covariance: torch [N, 7, 7], same order
    gt_boxes: torch [G, 7], same order
  Output:
    neg_log_likelihood: torch [N]
  '''
  N, V = mean_boxes.shape
  #print('mean_boxes: ', mean_boxes) # (14, 7)
  #print('covariance.shape: ', covariance.shape) # (14, 7, 7)
  #print('gt_boxes: ', gt_boxes) # (5, 7)

  covariance_diag = torch.diagonal(covariance, dim1 = -2, dim2 = -1)
  #print('covariance_diag: ', covariance_diag) # (14, 7)

  distance_matrix = get_2d_center_distance_matrix(gt_boxes, mean_boxes)
  #print('distance_matrix.shape: ', distance_matrix.shape)
  # (G, N) = (5, 14)
  min_per_det = torch.min(distance_matrix, dim=0)
  min_distance_per_det = min_per_det.values
  matched_gt_idx_per_det = min_per_det.indices

  #print('min_distance_per_det: ', min_distance_per_det)
  #print('matched_gt_idx_per_det: ', matched_gt_idx_per_det)
  # (N) = (14)

  matched_gt_per_det = gt_boxes[matched_gt_idx_per_det] 
  #print('matched_gt_per_det: ', matched_gt_per_det)
  # (N, 7) = (14, 7)

  # Do we need distance threshold here? 
  # ignore FP detection or make them to learn large R?
  # better to have distance threshold
  # other wise model will need to learn large variance
  # for FP, while its input feature may be similar to TP
  # and the loss will be dominant by FP, instead of let
  # model distinguish very good and good detection
  # only consider det whose distance to closet gt < 2 meters
  matched_mask_per_det = min_distance_per_det < 2
  #print('matched_mask_per_det: ', matched_mask_per_det)
  # (N)
  matched_det_count = torch.sum(matched_mask_per_det)

  # original 
  neg_log_likelihood_separate = (
    torch.log(covariance_diag * math.sqrt(2 * math.pi)) + 
    0.5 * ((mean_boxes - matched_gt_per_det) / 2) ** 2
  )
  # (N, V)
  neg_log_likelihood_separate = neg_log_likelihood_separate * matched_mask_per_det.unsqueeze(dim=1)
  #print('neg_log_likelihood_separate: ', neg_log_likelihood_separate)

  
  diff = mean_boxes - matched_gt_per_det
  #print('diff.shape: ', diff.shape)
  # (N, V)

  covariance_diag_norm = torch.norm(covariance_diag, dim=1)
  #print('covariance_diag_norm.shape: ', covariance_diag_norm.shape)
  # (N)

  # covariance is diagonal matrix
  neg_log_likelihood = (
    0.5 * V * math.log(2 * math.pi) +
    0.5 * torch.log(covariance_diag_norm) +
    0.5 * torch.sum(diff ** 2 / covariance_diag, dim=1)
  )
  (N)
  #print('neg_log_likelihood: ', neg_log_likelihood)
  #print('neg_log_likelihood.shape: ', neg_log_likelihood.shape)

  # element wise 
  neg_log_likelihood = neg_log_likelihood * matched_mask_per_det
  #print('neg_log_likelihood: ', neg_log_likelihood)

  return neg_log_likelihood, matched_det_count


def transform_dets_dict_to_kf_format(dets_dict, dtype, device):
  '''
  Input:
    dets_dict: {cav_id: [Box3D]]}
  Output:
    dets: [num_total_dets, 7], [x, y, z, theta, l, w, h]
  '''
  dets = []
  for cav_id in dets_dict.keys():
    for box3d in dets_dict[cav_id]:
      dets.append(Box3D.bbox2array(box3d))

  dets = np.stack(dets, axis=0)
  dets = torch.tensor(dets, dtype=dtype, device=device)
  return dets


def get_matched_gt_id_for_each_det(dets, gt_boxes, gt_ids, dataset, match_gt_threshold=1):
  '''
  For each det, get the matched gt ids.
  A gt is considered to be a match for a det if their 
  center 2D distance is less than (2 * tolerance_factor) meters.
  If more than 1 gt match, choose the closest 

  For nuscenes, we check xy plane
  For kitti, we check xz plane
  For v2v4real, we have transform data into ab3dmot kitti format, use xz
  
  dets and gt_boxes: [x, y, z, theta, l, w, h]
  '''
  #print("get_matched_gt_ids_for_each_det")
  #print("dets: ", dets)
  #print("gt_boxes: ", gt_boxes)
  
  #print("dets.shape: ", dets.shape)
  if dataset ==  'nuscenes':
    dets_xy = dets[:,:2].unsqueeze(1) # [M, 1, 2]
    gts_xy = gt_boxes[:, :2].unsqueeze(0) # [1, N, 2]
  else: # ab3dmot kitti, my v2v4real
    dets_xy = torch.cat([dets[:,0:1], dets[:,2:3]], dim=1).unsqueeze(1) # [M, 1, 2]
    gts_xy = torch.cat([gt_boxes[:, 0:1], gt_boxes[:, 2:3]], dim=1).unsqueeze(0) # [1, N, 2]
  #print('dets_xy.shape: ', dets_xy.shape)
  #print('gts_xy.shape: ', gts_xy.shape)

  #assert False

  diff = dets_xy - gts_xy # [M, N, 2]
  distance_matrix = torch.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2)
  #print("distance_matrix: ", distance_matrix)

  closest_info = torch.min(distance_matrix, dim=1)
  closest_gts = closest_info.indices
  closest_gt_ids = [gt_ids[closest_gts[d]] for d in range(dets.shape[0])]
  closest_dists = closest_info.values

  matched_mask = closest_dists <= match_gt_threshold 
  
  #print("closest_gts: ", closest_gts)
  #print("closest_gt_ids: ", closest_gt_ids)
  #print("closest_dists: ", closest_dists)
  #print("matched_mask: ", matched_mask)

  matched_dist_sum = torch.sum(closest_dists[matched_mask])

  return closest_gt_ids, matched_mask


def get_samples_masks(negative_sample_mode, closest_gts_per_det, matched_mask_per_det, closest_gts_per_trk, matched_mask_per_trk):
  '''
  negative_sample_mode == 0:
  First implement the one from uber paper PnPNet (https://arxiv.org/pdf/2005.14711.pdf) 
    Valid pairs: both det and trk match gt
    Positive samples: det and trk match to the same gt
    Negative samples: det and trk match to different gts
    Ignore calculating loss for invalid pairs

  TODO:
  negative_sample_mode == 1
  Another options is that including invalid pairs in the negative samples
  '''
  dtype = matched_mask_per_det.dtype
  device = matched_mask_per_det.device
  #positive_samples_mask = torch.zeros((len(closest_gts_per_det), len(closest_gts_per_trk)), dtype=dtype, device=device)

  #print("closest_gts_per_det, matched_mask_per_det, closest_gts_per_trk, matched_mask_per_trk: ", closest_gts_per_det, matched_mask_per_det, closest_gts_per_trk, matched_mask_per_trk)
  valid_mask = matched_mask_per_det.unsqueeze(1) & matched_mask_per_trk.unsqueeze(0)
  #print("valid_mask.shape: ", valid_mask.shape)

  # gt tracking_ids are strings
  have_same_closest_gt_id_mask = torch.zeros_like(valid_mask)
  #print("len(closest_gts_per_det): ", len(closest_gts_per_det))
  #print("len(closest_gts_per_trk): ", len(closest_gts_per_trk))
  
  for d in range(len(closest_gts_per_det)):
    for t in range(len(closest_gts_per_trk)):
      have_same_closest_gt_id_mask[d][t] = True if closest_gts_per_det[d] == closest_gts_per_trk[t] else False

  #positive_samples_mask = valid_mask & (closest_gts_per_det.unsqueeze(1) == closest_gts_per_trk.unsqueeze(0))
  positive_samples_mask = valid_mask & have_same_closest_gt_id_mask

  if negative_sample_mode == 0:
    negative_samples_mask = valid_mask & ~have_same_closest_gt_id_mask
  elif negative_sample_mode == 1:
    negative_samples_mask = ~positive_samples_mask
  else:
    assert(False)

  return positive_samples_mask, negative_samples_mask


def get_loss_from_samples(dtype, device, distance_matrix, positive_samples_mask, negative_samples_mask, contrastive_margin, match_dist_threshold, positive_sample_loss_weight):
  #print('distance_matrix: ', distance_matrix)
  #print('contrastive_margin: ', contrastive_margin)
  #print('match_dist_threshold: ', match_dist_threshold)

  #print("positive_samples_mask: ", positive_samples_mask)
  #print("negative_samples_mask: ", negative_samples_mask)
  num_positive_samples = torch.sum(positive_samples_mask)
  num_negative_samples = torch.sum(negative_samples_mask)
  #print("num_positive_samples: ", num_positive_samples)
  #print("num_negative_samples: ", num_negative_samples)

  if num_positive_samples == 0 or num_negative_samples == 0:
    #print('num samples 0')
    return torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(0.0, dtype=dtype, device=device), 0

  assert(not torch.any(torch.isnan(distance_matrix)))

  positive_distances = distance_matrix[positive_samples_mask]
  negative_distances = distance_matrix[negative_samples_mask]
  #print("positive_distances: ", positive_distances)
  #print("negative_distances: ", negative_distances)

  # note we use distance measurement, we want positive_distances smaller
  diff = negative_distances.unsqueeze(0) - positive_distances.unsqueeze(1) # [P, N]
  contrastive_max_margin_diff = torch.max(torch.zeros_like(diff), contrastive_margin - diff)
  contrastive_loss = torch.mean(contrastive_max_margin_diff)
  #print('contrastive_loss: ', contrastive_loss)

  # each positive or nagetive sample should be larger or smaller than the threshold by a margin
  if True:
    positive_max_margin_diff = torch.max(torch.zeros_like(positive_distances), contrastive_margin/2 - (match_dist_threshold - positive_distances))
    positive_max_margin_loss = torch.mean(positive_max_margin_diff)
    #print('positive_max_margin_loss: ', positive_max_margin_loss)
    negative_max_margin_diff = torch.max(torch.zeros_like(negative_distances), contrastive_margin/2 - (negative_distances - match_dist_threshold))
    negative_max_margin_loss = torch.mean(negative_max_margin_diff)
    #print('negative_max_margin_loss: ', negative_max_margin_loss)
    single_margin_loss = (positive_max_margin_loss * num_positive_samples * num_negative_samples * positive_sample_loss_weight + negative_max_margin_loss * num_negative_samples * num_positive_samples * (1 - positive_sample_loss_weight)) / (num_positive_samples * num_negative_samples * 1.0)
    #print('single_margin_loss: ', single_margin_loss)

  return single_margin_loss, contrastive_loss, num_positive_samples * num_negative_samples


def get_association_loss(dets_dict, trackers, gt_boxes, gt_ids, prev_gt_boxes, prev_gt_ids):
  #mahalanobis_threshold, filter_threshold, negative_sample_mode, contrastive_margin, match_distance, binary_distance_matrix, sort_distance_matrix, merged_distance_matrix, positive_sample_loss_weight, dataset):
  '''
  Input:
    dets_dict: {cav_id: [Box3D]]}
      conncted autonomous vehicle id to its detection boxes
    trackers: a list of class DKF, each contains one track's previous state before predict step
      dkf.x: [x, y, z, theta, l, w, h]
    gt_boxes:  ground truth box [num_gts, 7], [x, y, z, theta, l, w, h]
    gt_ids: ground truth object id: [num_gts],  number of objects in this frame
    prev_gt_boxes: previous frame's ground truth box [prev_num_gts, 7], [x, y, z, theta, l, w, h]
    prev_gt_ids: previous frame's ground truth object id: [prev_num_gts],  number of objects in previous frame
  '''
  dtype = gt_boxes.dtype
  device = gt_boxes.device

  # no loss for first frame in a sequence
  if prev_gt_boxes is None:
    return torch.tensor(0.0, dtype=dtype, device=device), 0

  # no gt box in this frame
  if len(gt_boxes) == 0 or len(prev_gt_boxes) == 0:
    # no positive sample => loss == 0
    # TODO: no loss or set all pair of det and trk as negative samples?
    #assert(False)
    #print('gt boxes 0')
    return torch.tensor(0.0, dtype=dtype, device=device), 0


  # get each track's [x,y] in the previous frame (before predict)
  # 0: dkf's shape [7, 1]
  prev_trks = torch.stack([trk.dkf.prev_x[:7, 0] for trk in trackers])
  #print("prev_trks.shape: ", prev_trks.shape) # [14, 7]

  # transform det into the same format of trk and gt: [num_objects, [x, y, z, theta, l, w, h]]
  dets = transform_dets_dict_to_kf_format(dets_dict, gt_boxes.dtype, gt_boxes.device)
  #print('dets.shape: ', dets.shape) # [12, 7]


  # no detection or track
  if dets.shape[0] == 0 or prev_trks.shape[0] == 0:
    #print('det or trk 0')
    return torch.tensor(0.0, dtype=dtype, device=device), 0
 
  #print("type(gt_boxes[0]): ", type(gt_boxes[0])) #nuscenes.eval.tracking.data_classes.TrackingBox
  dataset = 'v2v4real'
  #print('gt_boxes: ', gt_boxes)
  #print('prev_gt_boxes: ', prev_gt_boxes)
  closest_gts_per_det, matched_mask_per_det = get_matched_gt_id_for_each_det(dets, gt_boxes, gt_ids, dataset)
  closest_gts_per_trk, matched_mask_per_trk = get_matched_gt_id_for_each_det(prev_trks, prev_gt_boxes, prev_gt_ids, dataset)

  negative_sample_mode = 1
  positive_samples_mask, negative_samples_mask = get_samples_masks(
    negative_sample_mode, 
    closest_gts_per_det, matched_mask_per_det, 
    closest_gts_per_trk, matched_mask_per_trk)
  
  # get each track's [x,y] in the current frame (after predict)
  # 0: dkf's shape [7, 1]
  # without clone():
  # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
  trks = torch.stack([trk.dkf.x[:7, 0].clone() for trk in trackers])
  #print('trks: ', trks)
  assert(prev_trks.shape == trks.shape)

  # get loss from positive and negative samples
  # when using 2d center distance
  match_dist_threshold = 2 # we hope positive pair has dist < 2, negative pair has dist > 2
  # for contrastive loss
  contrastive_margin = 2 # we hope (dist of negative pair) - (dist of positive pair) > 2
  # for single loss
  # we hope positive pair has dist < 2 - 1, negative pair has dist > 2 + 1 
  # num_positive_samples and num_negative_samples can differ a lot
  positive_sample_loss_weight = 0.5

  distance_matrix = get_2d_center_distance_matrix(dets, trks)
  #print('distance_matrix: ', distance_matrix)

  single_margin_loss, contrastive_loss, loss_count = get_loss_from_samples(dtype, device, distance_matrix, positive_samples_mask, negative_samples_mask, contrastive_margin, match_dist_threshold, positive_sample_loss_weight)
  #print("single_margin_loss, contrastive_loss, loss_count: ", single_margin_loss, contrastive_loss, loss_count)

  association_loss_sum = (single_margin_loss + contrastive_loss) * loss_count
  association_loss_count = loss_count
  #print('association_loss_sum: ', association_loss_sum)
  #print('association_loss_count: ', association_loss_count)
  return association_loss_sum, association_loss_count
