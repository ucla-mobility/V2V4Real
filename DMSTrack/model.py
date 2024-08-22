from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
from AB3DMOT.AB3DMOT_libs.box import Box3D
from AB3DMOT.AB3DMOT_libs.matching import data_association
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_miscellaneous import print_log
import numpy as np, os, copy, math

from differentiable_kalman_filter import DKF
import torch
import torch.nn as nn
from loss import get_2d_center_distance_matrix, get_association_loss, get_neg_log_likelihood_loss
import time


class ObservationCovarianceNet(torch.nn.Module):
  def __init__(self, differentiable_kalman_filter_config, feature):
    '''
    Input
      differentiable_kalman_filter_config
      feature: one of ['fusion', 'bev', 'pos']
    '''
    super(ObservationCovarianceNet, self).__init__()
    self.dim_x = differentiable_kalman_filter_config['dim_x'] # num_state 10
    self.dim_z = differentiable_kalman_filter_config['dim_z'] # num_observation 7
    self.dkf_type = differentiable_kalman_filter_config['dkf_type'] # single_sensor or multi_sensor
    observation_covariance_setting = differentiable_kalman_filter_config['observation_covariance_setting']
    self.feature = feature
    self.bev_feature_channel_size = observation_covariance_setting['feature_channel_size'] # 512 # 320
    self.bev_feature_region_size = observation_covariance_setting['feature_region_size'] # 5 # 20
    self.positional_embedding_size = 18 * 256 # positional feature size 18 * dim 256 = 4608

    if self.dkf_type == 'multi_sensor':
      # The following 4 * 4 is the output spatial size of self.bev_conv_and_max_pool()
      self.fusion_channel_size = self.bev_feature_channel_size * 4 * 4 + self.positional_embedding_size # 9728
    else: # single_sensor, such as cobevt + dkf
      self.fusion_channel_size = self.bev_feature_channel_size * 3 * 3 + self.positional_embedding_size # 9216
  

    # for multi_sensor
    # input channel size 320, spatial dim [20, 20]
    self.bev_conv_and_max_pool = nn.Sequential(
      nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
        kernel_size=3, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
        kernel_size=3, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
        kernel_size=3, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
        kernel_size=3, padding=0, bias=True),
      nn.ReLU(inplace=True),
    )
    # output channel size 320, spatial dim [4, 4]


    


    # input channel size 320 * 4 * 4 = 5120
    self.bev_linear = nn.Sequential(
      nn.Linear(self.bev_feature_channel_size * 4 * 4, self.bev_feature_channel_size),
      nn.ReLU(inplace=True),
      nn.Linear(self.bev_feature_channel_size, self.dim_x)
    )
    # output size 10


    # input size 18 * 256 = 4608
    self.positional_encoding_linear = nn.Sequential(
      nn.Linear(self.positional_embedding_size, self.positional_embedding_size  // 2),
      nn.ReLU(inplace=True),
      nn.Linear(self.positional_embedding_size // 2, self.positional_embedding_size  // 4),
      nn.ReLU(inplace=True),
      nn.Linear(self.positional_embedding_size // 4, self.positional_embedding_size  // 8),
      nn.ReLU(inplace=True),
      nn.Linear(self.positional_embedding_size // 8, self.positional_embedding_size  // 16),
      nn.ReLU(inplace=True),
      nn.Linear(self.positional_embedding_size // 16, self.dim_x)
    )
    # output size 10


    # input size 5120 + 4608 = 9728
    self.fusion_linear = nn.Sequential(
      nn.Linear(self.fusion_channel_size, self.fusion_channel_size // 2),
      nn.ReLU(inplace=True),
      nn.Linear(self.fusion_channel_size // 2, self.dim_x),
    )
    # output size 10

    # smaller fusion net
    if self.dkf_type == 'multi_sensor':
      # input channel size 320, spatial dim [20, 20]
      self.fusion_bev_conv_and_max_pool = nn.Sequential(
        nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
          kernel_size=3, padding=0, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
          kernel_size=3, padding=0, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        # [8, 8]
        nn.AvgPool2d(kernel_size=2),
      )
      # output channel size 320, spatial dim [4, 4]
    else:
      # for single_sensor, such as cobevt + dkf
      # input channel size 512, spatial dim [5, 5]
      self.fusion_bev_conv_and_max_pool = nn.Sequential(
        nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
          kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size,
          kernel_size=3, padding=0, bias=True),
        nn.ReLU(inplace=True),
      )
      # output channel size 512, spatial dim [3, 3]


    # input size 18 * 256 = 4608
    self.fusion_positional_encoding_linear = nn.Sequential(
      nn.Linear(self.positional_embedding_size, self.positional_embedding_size),
      nn.ReLU(inplace=True),
      nn.Linear(self.positional_embedding_size, self.positional_embedding_size),
      nn.ReLU(inplace=True),
    )
    # output size 4608


    # Not used
    # embedding experiment
    #self.max_num_detection_per_frame = 500
    #self.max_num_frames = 400
    #self.detection_embedding = nn.Embedding(self.max_num_detection_per_frame * self.max_num_frames, 10)


  # choose different feature and model given self.feature
  def forward(self, x, frame, transformation_matrix, dets_in_gt_order):
    '''
    Input
      x: det feature: [N, 320, 20, 20] 
      transformation_matrix: [4, 4] cav sensor to ego coordinate
      dets_in_gt_order: [N, 7], 
        7: [x, y, z, theta, l, w, h] in ab3dmot kitti, xz is the ground plane
    Output: covariance diagonal residual : [N, self.dim_x=10]
    '''
    N, C, W, H = x.shape
    if N == 0:
      return torch.zeros([0, self.dim_x], dtype=x.dtype, device=x.device)

    if self.feature == 'bev':
      return self.forward_bev(x, frame, transformation_matrix, dets_in_gt_order)
    elif self.feature == 'pos':
      return self.forward_pos(x, frame, transformation_matrix, dets_in_gt_order)
    else: # fusion
      return self.forward_fusion(x, frame, transformation_matrix, dets_in_gt_order)


  def forward_fusion(self, x, frame, transformation_matrix, dets_in_gt_order):

    # positional encoding
    positional_embedding, distance_2d_det_to_ego, distance_2d_det_to_sensor, distance_2d_sensor_to_ego= self.get_positional_embedding(transformation_matrix, dets_in_gt_order)
    #print('positional_embedding.shape: ', positional_embedding.shape)
    # [14, 18, 256]
    #print('positional_embedding[:, -1, :5]: ', positional_embedding[:, -1, :5])
    positional_embedding = torch.flatten(positional_embedding, start_dim=1)
    # [14, 4608]
    positional_embedding = self.fusion_positional_encoding_linear(positional_embedding)
    # [14, 4608]

    # bev feature
    N, C, H, W = x.shape
    bev_feature = self.fusion_bev_conv_and_max_pool(x)
    #print('bev_feature.shape: ', bev_feature.shape) #                 # [5, 320, 4, 4]
    bev_feature = bev_feature.reshape([N, -1])
    # [14, 320 * 4 * 4 = 5120]

    fusion_feature = torch.cat([positional_embedding, bev_feature], dim=1)
    #print('fusion_feature.shape: ', fusion_feature.shape)
    # [14, 4608 + 5120 = 9728]
    
    output = self.fusion_linear(fusion_feature)
    #print('output.shape: ', output.shape)
    # [14, 10]

    return output


  # MY_EXPERIMENT
  def forward_embedding(self, x, frame):
    '''
    embedding experiment
    '''
    start = self.max_num_detection_per_frame * frame
    end = self.max_num_detection_per_frame * frame + x.shape[0]
    indices = torch.LongTensor(range(start, end)).to(x.device)
    return self.detection_embedding(indices)

  
  def get_positional_embedding(self, transformation_matrix, dets_in_gt_order, hidden_dim=256):
    '''
    Input
      transformation_matrix: [4, 4] cav sensor to ego coordinate
      dets_in_gt_order: [N, 7] in ego coordinate
        7 : [x, y, z, theta, l, w, h] in ab3dmot kitti, xz is the ground plane
    Output:
      positional_embedding: [N, (7 + 1 + 5 + 5),  K=positional_dim=256]
        7 : [x, y, z, theta, l, w, h]
        1 : [distance_2d_det_to_ego]
        5 : [sensor_to_ego_x, sensor_to_ego_y, sensor_to_ego_z, sensor_to_ego_theta, distance_2d_sensor_to_ego]
        5 : [det_to_sensor_x, det_to_sensor_y, det_to_sensor_z, det_to_sensor_theta, distance_2d_det_to_sensor]
    '''
    N, dim_z = dets_in_gt_order.shape

    # Do not normalize here so that we have raw feature to print for debug
    # normalize all distance by dividing by max distance 200 meters
    #max_distance = 200
    #normalize_distance_factor = torch.ones(7, dtype=dets_in_gt_order.dtype, device=dets_in_gt_order.device) * max_distance
    #normalize_distance_factor[3] = 1
    #print('normalize_distance_factor: ', normalize_distance_factor)
    #dets_in_gt_order = dets_in_gt_order / normalize_distance_factor
    #normalize_distance_factor = torch.ones([4,4], dtype=transformation_matrix.dtype, device=transformation_matrix.device)
    #normalize_distance_factor[0:3, 3] = max_distance
    #print('normalize_distance_factor: ', normalize_distance_factor)
    #transformation_matrix = transformation_matrix / normalize_distance_factor


    det_to_ego_x = dets_in_gt_order[:, 0]
    det_to_ego_y = dets_in_gt_order[:, 1]
    det_to_ego_z = dets_in_gt_order[:, 2]
    distance_2d_det_to_ego = torch.sqrt(det_to_ego_x ** 2 + det_to_ego_z ** 2)
    #print('distance_2d_det_to_ego: ', distance_2d_det_to_ego)

    # sensor to ego, which can be infer the position of sensor
    #print('transformation_matrix: ', transformation_matrix)
    # transformation_matrix is in v2v4real coordinate system
    # need to swap y and z 
    # TODO: move this coordinate transformation in v2v4real data extraction
    sensor_to_ego_x = transformation_matrix[0, 3]
    sensor_to_ego_y = transformation_matrix[2, 3]
    sensor_to_ego_z = transformation_matrix[1, 3]
    distance_2d_sensor_to_ego = torch.sqrt(sensor_to_ego_x ** 2 + sensor_to_ego_z ** 2)
    #print('distance_2d_sensor_to_ego: ', distance_2d_sensor_to_ego)
    sensor_to_ego_theta = torch.arctan2(transformation_matrix[1,0], transformation_matrix[0,0])
    #print('sensor_to_ego_theta: ', sensor_to_ego_theta)

    
    det_to_sensor_x = det_to_ego_x - sensor_to_ego_x
    det_to_sensor_y = det_to_ego_y - sensor_to_ego_y
    det_to_sensor_z = det_to_ego_z - sensor_to_ego_z
    distance_2d_det_to_sensor = torch.sqrt((det_to_sensor_x) ** 2 + (det_to_sensor_z) ** 2)
    #print('distance_2d_det_to_sensor: ', distance_2d_det_to_sensor)
    det_to_sensor_theta = dets_in_gt_order[:, 3] - sensor_to_ego_theta
    #print('det_to_sensor_theta: ', det_to_sensor_theta)

    
    positional_feature = torch.cat(
      [
        dets_in_gt_order,
        distance_2d_det_to_ego.unsqueeze(1),
        sensor_to_ego_x.unsqueeze(0).unsqueeze(0).expand(N, 1),
        sensor_to_ego_y.unsqueeze(0).unsqueeze(0).expand(N, 1),
        sensor_to_ego_z.unsqueeze(0).unsqueeze(0).expand(N, 1),
        sensor_to_ego_theta.unsqueeze(0).unsqueeze(0).expand(N, 1),
        distance_2d_sensor_to_ego.unsqueeze(0).unsqueeze(0).expand(N, 1),
        det_to_sensor_x.unsqueeze(1),
        det_to_sensor_y.unsqueeze(1),
        det_to_sensor_z.unsqueeze(1),
        det_to_sensor_theta.unsqueeze(1),
        distance_2d_det_to_sensor.unsqueeze(1)
      ],
      dim=1
    )
    #print('positional_feature.shape: ', positional_feature.shape)
    # (N, 18 = (7 + 1 + 5 + 5))
    #print('positional_feature: ', positional_feature)

    # normalize all distance by dividing by max distance 200 meters,
    # before applying sin cos positional embedding
    # TODO: move this constant to dataset dependent config
    max_distance = 200
    positional_feature /= max_distance
    # revert the above change on all theta
    positional_feature[:, 3] *= max_distance
    positional_feature[:, 11] *= max_distance
    positional_feature[:, 16] *= max_distance
    #print('positional_feature: ', positional_feature)
    # after this normalization, the range of distances is [-1, 1]


    # 
    half_hidden_dim = hidden_dim // 2
    # because the range of distance is [-1, 1]
    # we want scale = math.pi
    # the original code scale = 2 * math.pi, is for range [0, 1]
    scale = math.pi

    dim_t = torch.arange(half_hidden_dim, dtype=positional_feature.dtype, device=positional_feature.device)
    dim_t = 2 ** (2 * dim_t / hidden_dim)
    #print('dim_t[:5]: ', dim_t[:5])
    # (128)
    
    positional_embedding = positional_feature.unsqueeze(dim=2)
    #print('positional_embedding.shape: ', positional_embedding.shape)
    # (N, 18, 1)

    positional_embedding = positional_embedding * scale / dim_t
    #print('positional_embedding.shape: ', positional_embedding.shape)
    # (N, 18, 128)
    #print('positional_embedding[0, -1, :5]: ', positional_embedding[0, -1, :5])

    positional_embedding = torch.cat([positional_embedding.sin(), positional_embedding.cos()], dim=2)
    #print('positional_embedding.shape: ', positional_embedding.shape)
    # (N, 18, 256)
    # print encoded distance_2d_det_to_sensor
    #print('positional_embedding[:, -1, :]: ', positional_embedding[:, -1, :])


    # also return other distances for visualization and debug
    return positional_embedding, distance_2d_det_to_ego, distance_2d_det_to_sensor, distance_2d_sensor_to_ego



  # MY_DEBUG
  # new approach using positional encoding
  def forward_pos(self, x, frame, transformation_matrix, dets_in_gt_order): 
    '''
    Input
      x: det feature: [N, 320, 20, 20] 
      transformation_matrix: [4, 4] cav sensor to ego coordinate
      dets_in_gt_order: [N, 7], 
        7: [x, y, z, theta, l, w, h] in ab3dmot kitti, xz is the ground plane
    '''
    #print('x.shape: ', x.shape) # [5, 256, 5, 5] # [5, 320, 20, 20]
    N, C, W, H = x.shape
    

    # learnable approach
    # positional encoding
    positional_embedding, distance_2d_det_to_ego, distance_2d_det_to_sensor, distance_2d_sensor_to_ego= self.get_positional_embedding(transformation_matrix, dets_in_gt_order)
    #print('positional_embedding.shape: ', positional_embedding.shape)
    # [14, 18, 256]
    #print('positional_embedding[:, -1, :5]: ', positional_embedding[:, -1, :5])

    positional_embedding = torch.flatten(positional_embedding, start_dim=1)
    # [14, 4608]
    #print('positional_embedding.shape: ', positional_embedding.shape)
    #assert False
    diagonal_residual = self.positional_encoding_linear(positional_embedding)

    #print('diagonal_residual: ', diagonal_residual)
    #assert False


    # heuristic adjust observation covariance residual by distance to sensor
    # broadcast elementwise multiplication
    #diagonal_residual = torch.ones([N, 10], dtype=x.dtype, device=x.device) * distance_2d_det_to_sensor.unsqueeze(dim=1)  
    #diagonal_residual = -torch.ones([N, 10], dtype=x.dtype, device=x.device) / distance_2d_det_to_sensor.unsqueeze(dim=1) 
    #print('diagonal_residual: ', diagonal_residual)



    #assert False
    #return torch.zeros([N, 10], dtype=x.dtype, device=x.device)
    return diagonal_residual


  # MY_DEBUG
  # original way using local BEV feature
  def forward_bev(self, x, frame, transformation_matrix, dets_in_gt_order): 
    '''
    Using local feature map per detection to generate the 
    residual of diagonal of init covariance P and observation covariance R
    assume P and R share the same numbers.   
     
    P's dimension is dim_x
    R's dimension is dim_z

    Here we generate output vector of size dim_x directly
    R will only use the first dim_z values

    None-diagonal values in P and R will be set to 0

    Input:
      x: local feature map per detection, shape: (N, 256, 20, 20)
    Output:
      diagonal covariance residual, shape: (N, dim_x)
    '''
    #print('x.shape: ', x.shape) # [5, 320, 20, 20]
    N, C, W, H = x.shape

    x = self.bev_conv_and_max_pool(x)
    #print('x.shape: ', x.shape)  # [5, 320, 4, 4]
    x = x.reshape([N, -1])

    x = self.bev_linear(x) 
    #print('x.shape: ', x.shape) # [5, 10]

    # covar values are at scale of 0.01
    # neural network init values of network output has scale -0.09 ~ 0.09
    # make it smaller to the scale of -0.009 ~ 0.009
    #x *= 0.1
    #x *= 10
    #print("x: ", x)

    return x


class DMSTrack(AB3DMOT):
  '''
  Differentiable Probabilistic 3D Multi-Object Collaberative Tracker
  '''
  def __init__(self, cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=None, ID_init=0, 
      dtype=None, device=None, differentiable_kalman_filter_config=None, 
      observation_covariance_net_dict=None, force_gt_as_predicted_track=False, use_static_default_R=False,
      use_multiple_nets=False):
    super().__init__(cfg, cat, calib, oxts, img_dir, vis_dir, hw, log)
    # self.trackers is a list of class DKF, one DKF per track
    # self.trackers = [] is initialized in above AB3DMOT constructor

    self.dtype = dtype
    self.device = device
    self.dim_x = differentiable_kalman_filter_config['dim_x']
    self.dim_z = differentiable_kalman_filter_config['dim_z']
    observation_covariance_setting = differentiable_kalman_filter_config['observation_covariance_setting']
    self.feature_channel_size = observation_covariance_setting['feature_channel_size']
    self.feature_region_size = observation_covariance_setting['feature_region_size']
    self.gt_data_association_threshold = differentiable_kalman_filter_config['gt_data_association_threshold']

    #self.observation_covariance_net = ObservationCovarianceNet(
    #  self.dim_x, self.dim_z, self.feature_channel_size, self.feature_region_size).to(device)
    self.observation_covariance_net_dict = observation_covariance_net_dict

    # Previous frame's ground-truth boxs and ids
    # for calculating association loss
    self.prev_gt_boxes = None
    self.prev_gt_ids = None
    # Two frames earlier
    self.prev_prev_gt_boxes = None
    self.prev_prev_gt_ids = None

    # during training, force gt as predicted track before data association and update
    self.force_gt_as_predicted_track = force_gt_as_predicted_track

    # not using the learnable R
    self.use_static_default_R = use_static_default_R

    # use multiple observation covariance nets for multiple self-driving cars
    self.use_multiple_nets = use_multiple_nets


  def get_param(self, cfg, cat):
    if cfg.dataset == 'v2v4real':
      # use default ab3dmot kitti car setting
      algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
      # MY_DEUG
      # iou_2d
      #algm, metric, thres, min_hits, max_age = 'hungar', 'iou_2d', 0.25, 3, 2
      # dist_2d
      #algm, metric, thres, min_hits, max_age = 'greedy', 'dist_2d', 4, 3, 2
      # mahalanobis
      #algm, metric, thres, min_hits, max_age = 'greedy', 'm_dis', 1, 3, 2

    else:
      assert False

    # add negative due to it is the cost
    if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1
    self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
      algm, metric, thres, max_age, min_hits

    # define max/min values for the output affinity matrix
    if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
    elif self.metric in ['iou_2d', 'iou_3d']:          self.max_sim, self.min_sim = 1.0, 0.0
    elif self.metric in ['giou_2d', 'giou_3d']:        self.max_sim, self.min_sim = 1.0, -1.0


  def within_range_torch(self, angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    if angle > math.pi:
      angle -= 2 * math.pi
    if angle < -math.pi:
      angle += 2 * math.pi
    return angle


  def transform_gt_as_predicted_track(self, frame): 
    '''
    Similar to prediction(), but use previous two frames' gt to predict 
    as the track at the current frame
    '''
    #print('self.prev_gt_boxes: ', self.prev_gt_boxes)
    #print('self.prev_gt_ids: ', self.prev_gt_ids)
    #print('self.prev_prev_gt_boxes: ', self.prev_prev_gt_boxes)
    #print('self.prev_prev_gt_ids: ', self.prev_prev_gt_ids)


    # drop all the tracks
    self.trackers = []
    trks = []

    for i in range(self.prev_gt_boxes.shape[0]):
      prev_box = self.prev_gt_boxes[i]
      gt_id = self.prev_gt_ids[i]
      #print('gt_id: ', gt_id)
      i_in_prev_prev_frame = (self.prev_prev_gt_ids == gt_id).nonzero(as_tuple=True)[0]
      #print('i_in_prev_prev_frame: ', i_in_prev_prev_frame)

      bbox3D = prev_box.clone()
      #print('bbox3D: ', bbox3D)

      if len(i_in_prev_prev_frame) != 0:
        prev_prev_box = self.prev_prev_gt_boxes[i_in_prev_prev_frame][0]
        bbox3D[:2] += prev_box[:2] - prev_prev_box[:2]
        #print('bbox3D: ', bbox3D)
    
      info = np.array([0,    2,    0,    0,    0,    0,    1.0])
      # instead of using 'gt', just use 'ego' to make the code run
      trk = DKF(bbox3D.detach().cpu().numpy(), info, gt_id.item(), self.dtype, self.device, 
        self.use_static_default_R, frame, 'ego', gt_id.item())

      # force it to show up in tracking results
      trk.hits = 3
      # for calculating association loss if needed
      trk.dkf.prev_x = torch.cat([prev_box.reshape([7, 1]), torch.zeros([3, 1], dtype=self.dtype, device=self.device)], dim=0)
      #print('trk.dkf.prev_x: ', trk.dkf.prev_x)

      self.trackers.append(trk)

      trk_tmp = trk.dkf.x.reshape((-1))[:7]
      trk_tmp = trk_tmp.detach().cpu().numpy()
      trks.append(Box3D.array2bbox(trk_tmp))

    return trks


  def prediction(self):
    '''
    self.trackers perform Kalman filter prediction step
    this function will return the track states after prediciton
    in numpy for future data association
    '''

    # get predicted locations from existing tracks

    trks = []
    for t in range(len(self.trackers)):

      # propagate locations
      kf_tmp = self.trackers[t]
      if kf_tmp.id == self.debug_id:
        print('\n before prediction')
        print(kf_tmp.dkf.x.reshape((-1)))
        print('\n current velocity')
        print(kf_tmp.get_velocity())
      
      #print('kf_tmp.dkf.x: ', kf_tmp.dkf.x)
      kf_tmp.dkf.predict(None)
      #print('kf_tmp.dkf.x: ', kf_tmp.dkf.x)
      if kf_tmp.id == self.debug_id:
        print('After prediction')
        print(kf_tmp.dkf.x.reshape((-1)))
      kf_tmp.dkf.x[3] = self.within_range_torch(kf_tmp.dkf.x[3])

      # for visualization
      kf_tmp.reset_matched_detection_id_dict()

      # update statistics
      kf_tmp.time_since_update += 1
      trk_tmp = kf_tmp.dkf.x.reshape((-1))[:7]
      trk_tmp = trk_tmp.detach().cpu().numpy()
      trks.append(Box3D.array2bbox(trk_tmp))

    return trks


  def get_trks_for_match(self):
    # generate tracks' Box3D format for match
    # similar to prediction() but does not perform kalman filter prediction

    trks = []
    for t in range(len(self.trackers)):

      # propagate locations
      kf_tmp = self.trackers[t]
      if kf_tmp.id == self.debug_id:
        print('\n before prediction')
        print(kf_tmp.dkf.x.reshape((-1)))
        print('\n current velocity')
        print(kf_tmp.get_velocity())
      # do not perform kalman filter prediction
      #kf_tmp.dkf.predict(None)
      if kf_tmp.id == self.debug_id:
        print('After prediction')
        print(kf_tmp.dkf.x.reshape((-1)))
      #print('kf_tmp.dkf.x[3]: ', kf_tmp.dkf.x[3])
      kf_tmp.dkf.x[3] = self.within_range_torch(kf_tmp.dkf.x[3])
      #print('kf_tmp.dkf.x[3]: ', kf_tmp.dkf.x[3])

      # do not update statistics
      #kf_tmp.time_since_update += 1
      trk_tmp = kf_tmp.dkf.x.reshape((-1))[:7]
      trk_tmp = trk_tmp.detach().cpu().numpy()
      #print('trk_tmp: ', trk_tmp)
      trks.append(Box3D.array2bbox(trk_tmp))

    return trks


  def orientation_correction_torch(self, theta_pre, theta_obs):
    # update orientation in propagated tracks and detected boxes so that they are within 90 degree

    # make the theta still in the range
    theta_pre = self.within_range_torch(theta_pre)
    theta_obs = self.within_range_torch(theta_obs)

    # if the angle of two theta is not acute angle, then make it acute
    if abs(theta_obs - theta_pre) > math.pi / 2.0 and abs(theta_obs - theta_pre) < math.pi * 3 / 2.0:
      theta_pre += math.pi
      theta_pre = self.within_range_torch(theta_pre)

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(theta_obs - theta_pre) >= math.pi * 3 / 2.0:
      if theta_obs > 0: theta_pre += math.pi * 2
      else: theta_pre -= math.pi * 2

    return theta_pre, theta_obs

 
  def get_learnable_observation_covariance(self, default_init_P , diagonal_residual):
    '''
    Use diagonal_residual as the additive residual
    to the KF's default observation covariance matrix,
    for constructing the learnable observation matrix

    Note that we use dim_x = 10
    because we will also use it as track's initial covariance
    if the detection is used to initialize a new track

    Input:
      default_init_P: (10, 10)
      diagonal_residual: (N, 10)
    Output:
      learnable_init_P: (N, 10, 10)
      learnable_R = learnable_init_P[:, :7, :7] (N, 7, 7)

    '''
    #print('default_init_P.shape: ', default_init_P.shape)
    #print('diagonal_residual.shape: ', diagonal_residual.shape)

    diagonal_P = (torch.sqrt(torch.diag(default_init_P)) + diagonal_residual) ** 2
    #print('diagonal_P.shape: ', diagonal_P.shape) # (N, 10)

    # prevent negative or 0 or too small covariance value
    # which will introduce invalid values or super large loss
    diagonal_P = torch.clamp(diagonal_P, min=1e-2)
    #print('diagonal_P: ', diagonal_P)

    learnable_init_P = []
    for i in range(diagonal_P.shape[0]):
      single_learnable_init_P = torch.diag(diagonal_P[i])
      #print('single_learnable_init_P.shape: ', single_learnable_init_P.shape)
      learnable_init_P.append(single_learnable_init_P)

    if len(learnable_init_P) == 0:
      # no detection
      return torch.zeros([0, 10], dtype=self.dtype, device=self.device), torch.zeros([0, 7], dtype=self.dtype, device=self.device)

    learnable_init_P = torch.stack(learnable_init_P, dim=0)
    #print('learnable_init_P.shape: ', learnable_init_P.shape)

    learnable_R = learnable_init_P[:, :7, :7]
    #print('learnable_R.shape: ', learnable_R.shape)

    return learnable_init_P, learnable_R


  def update(self, matched, unmatched_trks, dets, info, learnable_R_dict, frame, cav_id):
    # update matched trackers with assigned detections

    #print('len(dets): ', len(dets))
    #print('observation_covariance.shape: ', observation_covariance.shape)
    assert(len(dets) == learnable_R_dict[cav_id].shape[0])

    dets = copy.copy(dets)
    for t, trk in enumerate(self.trackers):

      if t not in unmatched_trks:

        d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
        assert len(d) == 1, 'error'
        # d[0] is the matched detection id in dets

        # update statistics
        trk.time_since_update = 0               # reset because just updated
        trk.hits += 1

        # update orientation in propagated tracks and detected boxes so that they are within 90 degree
        bbox3d = Box3D.bbox2array(dets[d[0]])
        trk.dkf.x[3], bbox3d[3] = self.orientation_correction_torch(trk.dkf.x[3], bbox3d[3])

        if trk.id == self.debug_id:
          print('After ego-compoensation')
          print(trk.dkf.x.reshape((-1)))
          print('matched measurement')
          print(bbox3d.reshape((-1)))
          # print('uncertainty')
          # print(trk.dkf.P)
          # print('measurement noise')
          # print(trk.dkf.R)

        # kalman filter update with observation
        # implement regular kf without learnable R first
        # extend it with learnable R later
        #default_R = trk.default_R
        #diagonal_residual_R = observation_covariance[d[0]][:self.dim_z]
        #learnable_R = self.get_learnable_observation_covariance(default_R, diagonal_residual_R)
        learnable_R = learnable_R_dict[cav_id][d[0]]

        skip_update = False
        if trk.last_updated_frame == frame:
          # more than one cav detect this object at the same frame
          skip_update = True
        skip_update = False

        if not skip_update:
          #print('before update')
          #print('trk.last_updated_frame: ', trk.last_updated_frame)
          #print('trk.last_updated_cav_id: ', trk.last_updated_cav_id)
          #print('trk.dkf.x: ', trk.dkf.x)
          trk.dkf.update(bbox3d, learnable_R, None)
          trk.last_updated_frame = frame
          trk.last_updated_cav_id = cav_id
          # matched_detection_id_dict reset in prediction step
          trk.matched_detection_id_dict[cav_id] = d[0]
          #print('after update')
          #print('trk.last_updated_frame: ', trk.last_updated_frame)
          #print('trk.last_updated_cav_id: ', trk.last_updated_cav_id)
          #print('trk.dkf.x: ', trk.dkf.x)
        

        if trk.id == self.debug_id:
          print('after matching')
          print(trk.dkf.x.reshape((-1)))
          print('\n current velocity')
          print(trk.get_velocity())

        trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])
        trk.info = info[d, :][0]

      # debug use only
      # else:
        # print('track ID %d is not matched' % trk.id)


  def birth(self, dets, info, unmatched_dets, frame, cav_id, learnable_init_P_dict):
    # create and initialise new trackers for unmatched detections

    # dets = copy.copy(dets)
    new_id_list = list()                                    # new ID generated for unmatched detections
    for i in unmatched_dets:                                # a scalar of index
      trk = DKF(
        Box3D.bbox2array(dets[i]), 
        info[i, :], self.ID_count[0], 
        self.dtype, self.device, self.use_static_default_R,
        frame, cav_id, i, 
        learnable_init_P_dict[cav_id][i])
      self.trackers.append(trk)
      new_id_list.append(trk.id)
      # print('track ID %s has been initialized due to new detection' % trk.id)

      self.ID_count[0] += 1

    return new_id_list


  def output(self):
    # output exiting tracks that have been stably associated, i.e., >= min_hits
    # and also delete tracks that have appeared for a long time, i.e., >= max_age

    num_trks = len(self.trackers)
    results = []
    matched_detection_id_dict = []
    track_P = []

    for trk in reversed(self.trackers):
      # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
      trk_tmp = trk.dkf.x.reshape((-1))[:7]
      trk_tmp = trk_tmp.detach().cpu().numpy()
      d = Box3D.array2bbox(trk_tmp)     # bbox location self
      d = Box3D.bbox2array_raw(d)

      if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
        results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))

        # for visuailization and debug
        matched_detection_id_dict.append(trk.matched_detection_id_dict)
        track_P.append(trk.dkf.P.detach().cpu().numpy())


      num_trks -= 1

      # deadth, remove dead tracklet
      if (trk.time_since_update >= self.max_age):
        self.trackers.pop(num_trks)

    return results, matched_detection_id_dict, track_P



  def greedy_match(self, distance_matrix):
    '''
    Find the one-to-one matching using greedy allgorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    '''
    matched_indices = []
    matched_mask = np.zeros_like(distance_matrix)

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks
    for sort_i in range(index_2d.shape[0]):
      detection_id = int(index_2d[sort_i][0])
      tracking_id = int(index_2d[sort_i][1])
      if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
        tracking_id_matches_to_detection_id[tracking_id] = detection_id
        detection_id_matches_to_tracking_id[detection_id] = tracking_id
        matched_indices.append([detection_id, tracking_id])
        matched_mask[detection_id][tracking_id] = 1

    matched_indices = np.array(matched_indices)
    return matched_indices, matched_mask


  def get_regression_loss_per_pair(self, gt_box, tracking_box):
    '''
    Calculate the regression loss by
    between a ground truth box and a tracking result box
    Input:
      gt_box: ground truth box: [7=self.dim_z]
      tracking_box: tracking result box: [7=self.dim_z]
        7: [x, y, z, theta, l, w, h]
    Output:
      regression_loss_per_pair: a scalar tensor
    '''
    #print('gt_box: ', gt_box)
    #print('tracking_box: ', tracking_box)
    gt_box[3], tracking_box[3] = self.orientation_correction_torch(gt_box[3], tracking_box[3])
    regression_loss_per_pair = torch.norm(gt_box - tracking_box)
    #print('gt_box: ', gt_box)
    #print('tracking_box: ', tracking_box)
    #print('regression_loss_per_pair: ', regression_loss_per_pair)
    #print('regression_loss_per_pair.shape: ', regression_loss_per_pair.shape)
    # torch.Size([])

    return regression_loss_per_pair


  def get_regression_loss(self, gt_boxes):
    '''
    Calculate the regression loss by
    data association using center point distance 
    between tracking result boxes and ground truth boxes
    Input:
      gt_boxes: ground truth boxes: [N, 7=self.dim_z]
        7: [x, y, z, theta, l, w, h]
      self.trackers: list of class DKF
        DFK.dkf: class DifferentiableKalmanFilter
          DFK.dfk.x: tracking object state: [10]
            10: [x, y, z, theta, l, w, h, dx, dy, dz]
    Output:
      regression_loss_sum: sum of l2 loss of matched track gt pairs
      regression_loss_count: number of matched track gt pairs
    '''
    tracking_boxes = []
    for m in range(len(self.trackers)):
      tracking_boxes.append(self.trackers[m].dkf.x[:self.dim_z, 0])
    tracking_boxes = torch.stack(tracking_boxes, dim=0)

    #print('tracking_boxes: ', tracking_boxes)
    #print('gt_boxes: ', gt_boxes)

    #print('tracking_boxes.shape: ', tracking_boxes.shape)
    #print('gt_boxes.shape: ', gt_boxes.shape)

    # this data association does not need grad
    # we only use the matching indices
    with torch.no_grad():
      distance_matrix = get_2d_center_distance_matrix(gt_boxes, tracking_boxes)
      #print('distance_matrix: ', distance_matrix)
    distance_matrix = distance_matrix.detach().cpu().numpy()

    matched_indices, matched_mask = self.greedy_match(distance_matrix)
    #print('matched_indices: ', matched_indices)

    regression_loss = []
    for i in range(matched_indices.shape[0]):
      matched_index = matched_indices[i]
      #print('matched_index: ', matched_index)
      distance = distance_matrix[matched_index[0]][matched_index[1]]
      #print('distance: ', distance)
      if distance < self.gt_data_association_threshold:
        regression_loss_per_pair = self.get_regression_loss_per_pair(
          gt_boxes[matched_index[0]],
          tracking_boxes[matched_index[1]])
        regression_loss.append(regression_loss_per_pair)

    regression_loss_count = len(regression_loss)

    regression_loss = torch.stack(regression_loss, dim=0)
    #print('regression_loss: ', regression_loss)
    regression_loss_sum = torch.sum(regression_loss)
    #print('regression_loss: ', regression_loss)

    return regression_loss_sum, regression_loss_count


  def reset_dkf_gradients(self):
    '''
    Reset each class DKF's 
     class DifferentiableKalmanFilter
       in self.trackers,
    in order to run loss.backward() and optimizer.step 
    multiple times during tracking a sequence,
    by using detach and clone tensor values of DifferentiableKalmanFilter
    '''
    for tracker in self.trackers:
      tracker.reset_gradients()

  def process_dets_to_gt_order(self, dets):
    '''
    Input
      dets - list of  numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
    Output
      dets_in_gt_order: numpy [N, 7], [x, y, z, theta, l, w, h]
    '''
    #print('dets: ', dets)

    if len(dets) == 0:
      return np.zeros([0, 7])

    dets = np.stack(dets, axis=0)
    #print('dets.shape: ', dets.shape)

    dets_in_gt_order = np.concatenate(
      [
        dets[:, 3:7],
        dets[:, 2:3],
        dets[:, 1:2],
        dets[:, 0:1]
      ],
      axis=1
    )
    
    #print('dets_in_gt_order: ', dets_in_gt_order)
    #print('dets_in_gt_order.shape: ', dets_in_gt_order.shape)

    return dets_in_gt_order



  def track_multi_sensor_differentiable_kalman_filter(self, dets_all_dict, frame, seq_name, cav_id_list, dets_feature_dict, gt_boxes, gt_ids, transformation_matrix_dict):
    """
    Params:
      dets_all_dict: dict
        dets_all_dict[cav_id] is a dict: cav_id is in cav_id_list
          dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
          info: a array of other info for each det
        frame:    str, frame number, used to query ego pose
        cav_id_list: match tracks with detection from vehicles in the order of cav_id_list
        dets_feature_dict: object detection feature per object
        gt_boxes: ground truth box [N, 7], [x, y, z, theta, l, w, h] 
        gt_ids: ground truth object id: [N], N: number of objects in this frame
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    Output:
      results: tracking result boxes
      affi: TODO
      loss_dict: loss dictionary {
        'regression' : {
          'sum' :
          'count' : 
        },
        'association' : {
          'sum' :
          'count' : 
        }
      }
      for each type of loss, we store both sum and count,
      in order to calculate average loss of all tracking boxes
      after finish tracking a sequence

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    # MY_DEBUG
    # when measuring run time during inference, ignore loss calculation
    measure_run_time = False

    # experiment that change the order
    #cav_id_list = ['1', 'ego']
    loss_dict = {}


    dets_dict = {}
    info_dict = {}
    for cav_id in dets_all_dict.keys():
      dets_dict[cav_id] = dets_all_dict[cav_id]['dets'] # dets: N x 7, float numpy array
      info_dict[cav_id] = dets_all_dict[cav_id]['info']

    if self.debug_id: print('\nframe is %s' % frame)

    # logging
    print_str = '\n\n*****************************************\n\nprocessing seq_name/frame %s/%d' % (seq_name, frame)
    print_log(print_str, log=self.log, display=False)
    self.frame_count += 1

    # recall the last frames of outputs for computing ID correspondences during affinity processing
    self.id_past_output = copy.copy(self.id_now_output)
    self.id_past = [trk.id for trk in self.trackers]

    # my process detection to gt order
    # from  [h,w,l,x,y,z,theta] to [x, y, z, theta, l, w, h]
    dets_in_gt_order_dict = {}
    for cav_id in dets_all_dict.keys():
      dets_in_gt_order_dict[cav_id] = self.process_dets_to_gt_order(dets_dict[cav_id])

    # process detection format
    # from [h,w,l,x,y,z,theta] to Box3D()
    for cav_id in dets_all_dict.keys():
      dets_dict[cav_id] = self.process_dets(dets_dict[cav_id])

      

    default_init_P, _, _ = DKF.get_ab3dmot_default_covariance_matrices(self.dtype, self.device, dim_x=10, dim_z=7)

    # get observation covariance matrix from ObservationCovarianceNet
    observation_covariance_dict = {}
    learnable_init_P_dict = {}
    learnable_R_dict = {}
    det_neg_log_likelihood_loss_dict = {}
    det_neg_log_likelihood_loss_sum = []
    det_neg_log_likelihood_loss_count = 0
    for cav_id in dets_all_dict.keys():
      dets_feature = dets_feature_dict[cav_id]
      dets_feature = torch.tensor(dets_feature, dtype=self.dtype, device=self.device)
      transformation_matrix = transformation_matrix_dict[cav_id]
      transformation_matrix = torch.tensor(transformation_matrix, dtype=self.dtype, device=self.device)
      dets_in_gt_order = dets_in_gt_order_dict[cav_id]
      dets_in_gt_order = torch.tensor(dets_in_gt_order, dtype=self.dtype, device=self.device)
      #print('cav_id: ', cav_id)
      #print('transformation_matrix: ', transformation_matrix)

      #start_time = time.time()
      # if 0 detection (val seq 0007 frame 32), return torch.zeros([0, 10])
      if self.use_multiple_nets:
        observation_covariance_dict[cav_id] = self.observation_covariance_net_dict[cav_id](
          dets_feature, frame, transformation_matrix, dets_in_gt_order)
      else:
        # if use only one net, use the ego's model which is set in optimizer
        observation_covariance_dict[cav_id] = self.observation_covariance_net_dict['ego'](
          dets_feature, frame,  transformation_matrix, dets_in_gt_order)

      #print('cav_id: ', cav_id)
      #print('observation_covariance_dict[cav_id]: ', observation_covariance_dict[cav_id])
      #print('observation_covariance_dict[cav_id].shape: ', observation_covariance_dict[cav_id].shape)
      #print('torch.mean(observation_covariance_dict[cav_id]): ', torch.mean(observation_covariance_dict[cav_id]))

      # get learnable_R_dict
      learnable_init_P_dict[cav_id], learnable_R_dict[cav_id] = self.get_learnable_observation_covariance(default_init_P, observation_covariance_dict[cav_id])
      #end_time = time.time()
      #print('Covariance Net runtime: %f' % (end_time - start_time))

      if not measure_run_time:
        # calculate negative loglikelihood loss
        # between pair of det and gt
        if dets_in_gt_order.shape[0] == 0:
          # no detection # val seq 0007 frame 32
          continue
        det_neg_log_likelihood_loss_dict[cav_id], matched_det_count = get_neg_log_likelihood_loss(
          dets_in_gt_order, learnable_R_dict[cav_id], gt_boxes)
        det_neg_log_likelihood_loss_sum.append(det_neg_log_likelihood_loss_dict[cav_id])
        det_neg_log_likelihood_loss_count += matched_det_count

    # total det_neg_log_likelihood_loss for all cav
    #print('det_neg_log_likelihood_loss_sum: ', det_neg_log_likelihood_loss_sum)
    #print('det_neg_log_likelihood_loss_count: ', det_neg_log_likelihood_loss_count)
    if measure_run_time:
      loss_dict['det_neg_log_likelihood'] = {
        'sum' : torch.zeros(1, dtype=self.dtype, device=self.device),
        'count' : 1 
      }
    else:
      det_neg_log_likelihood_loss_sum = torch.cat(det_neg_log_likelihood_loss_sum, dim=0)
      det_neg_log_likelihood_loss_sum = torch.sum(det_neg_log_likelihood_loss_sum)
      loss_dict['det_neg_log_likelihood'] = {
        'sum' : det_neg_log_likelihood_loss_sum,
        'count' :det_neg_log_likelihood_loss_count.detach().cpu().numpy() 
      }



    # KF prediction step
    # tracks propagation based on velocity
    #start_time = time.time()
    if self.force_gt_as_predicted_track and self.prev_prev_gt_boxes is not None:
      trks = self.transform_gt_as_predicted_track(frame)
    else:
      trks = self.prediction()
    #end_time = time.time()
    #print('Kalman Filter Prediction Step runtime: %f', (end_time - start_time))

    # ego motion compensation, adapt to the current frame of camera coordinate
    if (frame > 0) and (self.ego_com) and (self.oxts is not None):
      # we do not have self.oxts for v2v4real
      assert False
      trks = self.ego_motion_compensation(frame, trks)

    # visualization
    if self.vis and (self.vis_dir is not None):
      assert False
      img = os.path.join(self.img_dir, f'{frame:06d}.png')
      save_path = os.path.join(self.vis_dir, f'{frame:06d}.jpg'); mkdir_if_missing(save_path)
      self.visualization(img, dets, trks, self.calib, self.hw, save_path)

    # get data association loss
    # before the actual mathcing and update step, after the predict step
    # so the track contains the state right before and after the predict step
    # prev state is used to get the matched gt id in the previous frame
    # current state is used to get the distance to detection box to get loss
    if measure_run_time:
      loss_dict['association'] = {
        'sum' : torch.zeros(1, dtype=self.dtype, device=self.device),
        'count' : 1
      }
    else:
      association_loss_sum, association_loss_count = get_association_loss(dets_dict, self.trackers, gt_boxes, gt_ids, self.prev_gt_boxes, self.prev_gt_ids)
      loss_dict['association'] = {
        'sum' : association_loss_sum,
        'count' : association_loss_count
      }

    # For multi_sensor_kalman_filter,
    # perform match and update for each detection set from each vehicle in the order of cav_id_list

    #print('2 cav_id_list: ', cav_id_list)
    for cav_id in cav_id_list:
      #print('cav_id: ', cav_id)
      if cav_id not in dets_dict.keys():
        # this cav does not detect any object in this frame
        continue

      # matching
      trk_innovation_matrix = None
      if self.metric == 'm_dis':
        trk_innovation_matrix = [trk.compute_innovation_matrix().detach().cpu().numpy() for trk in self.trackers]

      
      #start_time = time.time()
      # this data association is calculated in numpy
      matched, unmatched_dets, unmatched_trks, cost, affi = \
        data_association(dets_dict[cav_id], trks, self.metric, self.thres, self.algm, trk_innovation_matrix)
      # print_log('detections are', log=self.log, display=False)
      # print_log(dets, log=self.log, display=False)
      # print_log('tracklets are', log=self.log, display=False)
      # print_log(trks, log=self.log, display=False)
      # print_log('matched indexes are', log=self.log, display=False)
      # print_log(matched, log=self.log, display=False)
      # print_log('raw affinity matrix is', log=self.log, display=False)
      # print_log(affi, log=self.log, display=False)
      #end_time = time.time()
      #print('Data Association runtime: %f', (end_time - start_time))


      #start_time = time.time()
      # update trks with matched detection measurement
      #print('learnable_R_dict[cav_id].shape: ', learnable_R_dict[cav_id].shape)
      self.update(matched, unmatched_trks, dets_dict[cav_id], info_dict[cav_id], 
        learnable_R_dict, frame, cav_id)
      #end_time = time.time()
      #print('Kalman Filter Update Step runtime: %f', (end_time - start_time))

      # create and initialise new trackers for unmatched detections
      new_id_list = self.birth(dets_dict[cav_id], info_dict[cav_id], unmatched_dets, frame, cav_id, learnable_init_P_dict)

      #start_time = time.time()
      # generate the trk from updated tracks,
      # for next match with the next vehicle's detection set
      trks = self.get_trks_for_match()
      #end_time = time.time()
      #print('Generate Track for next update step runtime: %f', (end_time - start_time))

    # End of match and update tracks with all detection sets


    # calculate loss during training
    # regression loss: measureing the difference between tracking results and gt boxes
    if measure_run_time:
      loss_dict['regression'] = {
        'sum' : torch.zeros(1, dtype=self.dtype, device=self.device),
        'count' : 1
      }
    else:
      regression_loss_sum, regression_loss_count = self.get_regression_loss(gt_boxes)
      loss_dict['regression'] = {
        'sum' : regression_loss_sum,
        'count' : regression_loss_count
      }


    # save gt info for next frame's loss calculation
    self.prev_prev_gt_boxes = self.prev_gt_boxes
    self.prev_prev_gt_ids = self.prev_gt_ids
    self.prev_gt_boxes = gt_boxes
    self.prev_gt_ids = gt_ids

    # output existing valid tracks
    results, matched_detection_id_dict, track_P = self.output()

    if len(results) > 0: results = [np.concatenate(results)]                # h,w,l,x,y,z,theta, ID, other info, confidence
    else:                    results = [np.empty((0, 15))]
    self.id_now_output = results[0][:, 7].tolist()                                  # only the active tracks that are outputed

    # post-processing affinity to convert to the affinity between resulting tracklets
    if self.affi_process:
      # implement this if we want more debug and visualization
      assert False
      affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)
      # print_log('processed affinity matrix is', log=self.log, display=False)
      # print_log(affi, log=self.log, display=False)

    # logging
    #print_log('\ntop-1 cost selected', log=self.log, display=False)
    #print_log(cost, log=self.log, display=False)
    #for result_index in range(len(results)):
    #  print_log(results[result_index][:, :8], log=self.log, display=False)
    #  print_log('', log=self.log, display=False)


      

    return results, affi, loss_dict, matched_detection_id_dict, learnable_R_dict, track_P, det_neg_log_likelihood_loss_dict
