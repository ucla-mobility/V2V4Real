import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

import torch
from AB3DMOT.AB3DMOT_libs.box import Box3D


class Filter(object):
  def __init__(self, bbox3D, info, ID):

    self.initial_pos = bbox3D
    self.time_since_update = 0
    self.id = ID
    self.hits = 1               # number of total hits including the first detection
    self.info = info            # other information associated  

class DKF(Filter):
  def __init__(self, bbox3D, info, ID, dtype, device, use_static_default_R, frame, 
      cav_id, det_id_in_cav, learnable_init_P):
    '''
    Init a new track with the detection information
    '''

    super().__init__(bbox3D, info, ID)

    self.dtype = dtype
    self.device = device
    self.use_static_default_R = use_static_default_R

    # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
    # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz 
    # while all others (theta, l, w, h, dx, dy, dz) remain the same
    F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
                  [0,1,0,0,0,0,0,0,1,0],
                  [0,0,1,0,0,0,0,0,0,1],
                  [0,0,0,1,0,0,0,0,0,0],  
                  [0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,0,1]])     
    F = torch.tensor(F, dtype=self.dtype, device=self.device)

    # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
    H = np.array([[1,0,0,0,0,0,0,0,0,0],      
                  [0,1,0,0,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,1,0,0,0]])
    H = torch.tensor(H, dtype=self.dtype, device=self.device)

    # Uses the ab3dmot parameters as the baseline
    P, Q, R = self.get_ab3dmot_default_covariance_matrices(self.dtype, self.device, dim_x=10, dim_z=7)
    self.default_P = P.detach().clone()
    self.default_Q = Q.detach().clone()
    self.default_R = R.detach().clone()

    # MY_DEBUG
    # set init P to constant value
    # see whether we can prevent learnable R grow unbounded
    if not self.use_static_default_R:
      # use the detection box's observation covairance as
      # track's initial covairance
      P = learnable_init_P

    # Get trainable covariance matrices
    #P, Q, R = self.get_covariance_matrices(observation_covar_kf_residual)

    # set initial velocity to 0
    initial_state_mean = np.concatenate([self.initial_pos.reshape((7, 1)), np.array([[0], [0], [0]])], axis=0)
    initial_state_mean = torch.tensor(initial_state_mean, dtype=self.dtype, device=self.device)

    # Implement the default one to reproduce ab3dmot result for now
    # Extend it to be a trainable version later
    # original variable name
    
    self.dkf = DifferentiableKalmanFilter(F, H, initial_state_mean, P, Q, R,
      False, None, False, None, None, use_static_default_R)       
    #print('self.dkf.x: ', self.dkf.x)
    #assert False

    self.last_updated_frame = frame
    self.last_updated_cav_id = cav_id

    self.matched_detection_id_dict = {'ego': -1, '1': -1}
    self.matched_detection_id_dict[cav_id] = det_id_in_cav
    

  def reset_matched_detection_id_dict(self):
    self.matched_detection_id_dict = {'ego': -1, '1': -1}


  def reset_gradients(self):
    self.dkf.x = self.dkf.x.detach().clone()
    self.dkf.P = self.dkf.P.detach().clone()
    self.dkf.Q = self.dkf.Q.detach().clone()
    self.dkf.R = self.dkf.R.detach().clone()
    

  @staticmethod
  def get_ab3dmot_default_covariance_matrices(dtype, device, dim_x, dim_z):
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
    # kf.R[0:,0:] *= 10.   
    # default identity matrix
    R = torch.tensor(kf.R, dtype=dtype, device=device)

    # initial state uncertainty at time 0
    # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
    #kf.P[7:, 7:] *= 1000.   
    #kf.P *= 10.

    # MY_DEBUG
    # for my learnable covariance, starting from identity matrix P and R
    # comment out above two lines let P = 1

    P = torch.tensor(kf.P, dtype=dtype, device=device) * 1

    # Use the same Q as the original ab3dmot kitti
    # process uncertainty, make the constant velocity part more certain
    kf.Q[7:, 7:] *= 0.01
    Q = kf.Q
    Q = torch.tensor(kf.Q, dtype=dtype, device=device)

    return P, Q, R


  def compute_innovation_matrix(self):
    """ compute the innovation matrix for association with mahalanobis distance
    """
    # should be torch tensor version
    return torch.matmul(torch.matmul(self.dkf.H, self.dkf.P), self.dkf.H.t()) + self.dkf.R


  def get_velocity(self):
    # return the object velocity in the state
    return self.kf.x[7:]


class DifferentiableKalmanFilter(object):
  '''
  PyTorch simplified version of filterpy.kalman
  Mainly follow the naming convention of 
  "https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html"
  
  "https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python"
  '''
  def __init__(self, F, H, x, P, Q, R, use_mlp_process_model, process_model, use_sensor_distance, ego_position, tracking_name, use_static_default_R):
    '''
    torch:
      F: process model matrix
      H: obervation model matrix
      x: init state mean
      P: init state covariance matrix
      Q: process model noise covariance
      R: observation model noise covariance
    '''
    self.dtype = P.dtype
    self.device = P.device
    self.dim_x = F.shape[0]
    self.dim_z = H.shape[0]
    self.dim_h = process_model.hidden_size if process_model is not None else 0

    #print('self.dtype, self.device, self.dim_x, self.dim_z: ', self.dtype, self.device, self.dim_x, self.dim_z)
    #print('F.shape, H.shape, x.shape, P.shape, Q.shape, R.shape: ', F.shape, H.shape, x.shape, P.shape, Q.shape, R.shape)

    self.F = F
    self.H = H
    self.x = x
    self.P = P
    self.Q = Q
    #if torch.any(self.Q < 0):
    #  print('self.Q: ', self.Q)
    #  assert(False)

    self.R = R
    self.init_P = P.clone()

    self.use_mlp_process_model = use_mlp_process_model
    self.process_model = process_model
    self.use_sensor_distance = use_sensor_distance
    self.tracking_name = tracking_name
    self.use_static_default_R = use_static_default_R

    # for lstm process model
    if use_mlp_process_model:
      self.lstm_hidden = (torch.randn(1, 1, self.dim_h).to(dtype = self.dtype, device=self.device), torch.randn(1, 1, self.dim_h).to(dtype=self.dtype, device=self.device))
    else:
      self.lstm_hidden = None

  def predict(self, ego_position):
    #if torch.any(self.Q < 0):
    #  print('self.Q: ', self.Q)
    #  assert(False)
    self.prev_x = self.x.clone()
    self.prev_P = self.P.clone()

    if not self.use_mlp_process_model:
      self.x = torch.matmul(self.F, self.x)
      self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.t()) + self.Q
      #if torch.any(self.P < 0):
      #  print('self.Q: ', self.Q)
      #  print('self.prev_P: ', self.prev_P)
      #  print('self.P: ', self.P)
      #  assert(False)
    else:
      self.x, self.P, self.lstm_hidden = self.process_model(self.x, self.P, self.F, self.Q, self.use_mlp_process_model, self.lstm_hidden)
      #print('self.prev_x: ', self.prev_x)
      #print('self.prev_P: ', self.prev_P)
      #print('self.x: ', self.x)
      #print('self.P: ', self.P)

    # TODO: S and K depends on R 
    # for observation covar per detecton, need to move to another function
    # also need to change  associate funcrion which calculate m distance
    # sanity check after this code change
    #if True and self.use_sensor_distance:
    #  sensor_distance = torch.sqrt((self.prev_x[0] - ego_position[0])**2 + ((self.prev_x[1] - ego_position[1])**2))
    #  ratio = 1 + (sensor_distance / NUSCENES_CLASS_RANGE[self.tracking_name]) * self.use_sensor_distance
    #  self.S = torch.matmul(torch.matmul(self.H, self.P), self.H.t()) + self.R * ratio
    #else:
    #  self.S = torch.matmul(torch.matmul(self.H, self.P), self.H.t()) + self.R

    #only for sanity check during develop
    #self.S = torch.matmul(torch.matmul(self.H, self.P), self.H.t()) + self.R

    #self.K = torch.matmul(torch.matmul(self.P, self.H.t()), torch.inverse(self.S))

  def get_S_for_each_detections_without_updating(self, R):
    '''
    Single case equation:
      self.S = torch.matmul(torch.matmul(self.H, self.P), self.H.t()) + self.R

    Input:
      R : observation noise covariance matrix for each detection: [N, dim_z, dim_z]
    Output:
      S : [N, dim_z, dim_z]
    '''
    #if torch.any(self.Q < 0):
    #  print('self.Q: ', self.Q)
    #  assert(False)
    #print("R.shape: ", R.shape)
    S = torch.matmul(torch.matmul(self.H, self.P), self.H.t())
    #print('kalman filter S 0: ', S)
    #print("S.shape: ", S.shape)
    # broadcast
    S = S + R
    #print('kalman filter R [2]:', R[2])
    #print("S.shape: ", S.shape)
    #print('kalman filter S + R [2]: ', S[2])
    return S 


  def update(self, z, learnable_R, ego_position, debug=False):
    #if torch.any(self.Q < 0):
    #  print('self.Q: ', self.Q)
    #  assert(False)
    #print('type(z): ', type(z))
    #print('z.shape: ', z.shape)
    #print('z: ', z)
    z = z.reshape([self.dim_z, 1])
    z = torch.tensor(z, dtype=self.dtype, device=self.device)
    #print('z.shape: ', z.shape)
    #print('z: ', z)

    #print('ego_position: ', ego_position)
    identity = torch.eye(self.dim_x, dtype=self.dtype, device=self.device)

    #print("R: ", R)

    if not self.use_static_default_R:
      self.R = learnable_R

    self.S = torch.matmul(torch.matmul(self.H, self.P), self.H.t()) + self.R
    self.K = torch.matmul(torch.matmul(self.P, self.H.t()), torch.inverse(self.S))

    if False and self.use_sensor_distance:
      # recalculate the S and K, adjust R based on the distance between sensor and detection
      sensor_distance = torch.sqrt((z[0] - ego_position[0])**2 + ((z[1] - ego_position[1])**2))
      #print('sensor_distance: ', sensor_distance)

      ratio = 1 + (sensor_distance / NUSCENES_CLASS_RANGE[self.tracking_name]) * self.use_sensor_distance
      #print('ratio: ', ratio)
      self.S = self.S - self.R + self.R * ratio
      self.K = torch.matmul(torch.matmul(self.P, self.H.t()), torch.inverse(self.S))


    # kalman filter update on mean
    #print('dkf before update x: ', self.x, self.K, z)
    diff  = z - torch.matmul(self.H, self.x)
    #print('diff: ', diff)
    #additional = torch.matmul(self.K, z - torch.matmul(self.H, self.x))
    additional = torch.matmul(self.K, diff)
    #if additional[9] > 0.5:
    #  print('additional: ', additional)
    #self.x = self.x + torch.matmul(self.K, z - torch.matmul(self.H, self.x))
    self.x = self.x + additional
    #print('dkf after update x: ', self.x)
    if False and (self.x[9] > 0.5 or debug):
      print('Positive z velocity: ', self.x[9] > 0.5)
      print('deubg: ', debug)
      print('self.prev_x: ', self.prev_x)
      print('self.x: ', self.x)
      print('z: ', z)
      print('self.K.shape: ', self.K.shape)
      print('self.K: ', self.K)
      print('self.prev_P: ', self.prev_P)
      print('self.init_P: ', self.init_P)
      print('self.P: ', self.P)
      print('self.Q: ', self.Q)
      print('self.R: ', self.R)
      exit(-1)

    # force accepting the full detection
    # self.x[:self.dim_z] = z
    #self.x = torch.cat((z,  (self.x + torch.matmul(self.K, z - torch.matmul(self.H, self.x)))[self.dim_z:]))

    #print('identity.shape: ', identity.shape)
    #print('self.K.shape: ', self.K.shape)
    #print('self.H.shape: ', self.H.shape)
    #print('self.P.shape: ', self.P.shape)
    temp = self.P.clone()
    self.P = torch.matmul(identity - torch.matmul(self.K, self.H), self.P)
    #if torch.any(self.P < 0):
    #  print('temp self.P: ', temp)
    #  print('self.S: ', self.S)
    #  print('self.K: ', self.K)
    #  print('self.P: ', self.P)
    #  assert(False)


  def get_updated_state_without_updating(self, x, z, R):
    '''
    Use x as current state, z as observation, with self.P,Q,R to 
    get updated state but without really updating self dkf
    '''
    #if torch.any(self.Q < 0):
    #  print('self.Q: ', self.Q)
    #  assert(False)
    #print('type(z): ', type(z))
    #print('z.shape: ', z.shape)
    #print('z: ', z)
    #print('ego_position: ', ego_position)
    identity = torch.eye(self.dim_x, dtype=self.dtype, device=self.device)

   
    S = torch.matmul(torch.matmul(self.H, self.P), self.H.t()) + (self.R if self.use_static_kf_update_R else R)
    K = torch.matmul(torch.matmul(self.P, self.H.t()), torch.inverse(S))


    # kalman filter update on mean
    #print('dkf before update x: ', self.x, self.K, z)
    diff  = z - torch.matmul(self.H, x)
    #print('diff: ', diff)
    #additional = torch.matmul(self.K, z - torch.matmul(self.H, self.x))
    additional = torch.matmul(K, diff)
    #if additional[9] > 0.5:
    #  print('additional: ', additional)
    #self.x = self.x + torch.matmul(self.K, z - torch.matmul(self.H, self.x))
    x = x + additional
    #print('dkf after update x: ', self.x)
    # force accepting the full detection
    # self.x[:self.dim_z] = z
    #self.x = torch.cat((z,  (self.x + torch.matmul(self.K, z - torch.matmul(self.H, self.x)))[self.dim_z:]))

    #print('identity.shape: ', identity.shape)
    #print('self.K.shape: ', self.K.shape)
    #print('self.H.shape: ', self.H.shape)
    #print('self.P.shape: ', self.P.shape)
    #self.P = torch.matmul(identity - torch.matmul(self.K, self.H), self.P)
    return x
