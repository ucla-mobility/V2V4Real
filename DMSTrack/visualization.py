import os
import numpy as np
import open3d as o3d
import time

from V2V4Real.opencood.utils import box_utils, common_utils
from V2V4Real.opencood.visualization.vis_utils import color_encoding, bbx2linset, LineMesh

import random
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_visualization import random_colors


def transform_tracking_boxes(ab3dmot_tracking_results):
  tracks = ab3dmot_tracking_results[:, :7]
  #print('tracks: ', tracks)

  # ab3dmot kitti to v2v4real: swapping yz
  tracks = np.concatenate([tracks[:, 0:4], tracks[:, 5:6], tracks[:, 4:5], tracks[:, 6:7]], axis=1)
  #print('tracks: ', tracks)

  # [h,w,l,x,y,z,theta] to [x, y, z, h, w, l, theta]
  tracks = np.concatenate([tracks[:, 3:6], tracks[:, 0:3], tracks[:, 6:7]], axis=1)
  #print('tracks: ', tracks)

  # to corners
  tracks = box_utils.boxes_to_corners_3d(tracks, order='hwl')
  #print('tracks: ', tracks)
  
  return tracks


def transform_gt_boxes(gt_boxes):
  # ab3dmot kitti to v2v4real: swapping yz
  #print('gt_boxes: ', gt_boxes)
  gt_boxes = np.concatenate([gt_boxes[:, 0:1], gt_boxes[:, 2:3], gt_boxes[:, 1:2], gt_boxes[:, 3:7]], axis=1)
  #print('gt_boxes: ', gt_boxes)

  #  [x, y, z, theta, l, w, h] to [x, y, z, h, w, l, theta]
  gt_boxes = np.concatenate([gt_boxes[:, 0:3], gt_boxes[:, 6:7], gt_boxes[:, 5:6], gt_boxes[:, 4:5], gt_boxes[:, 3:4]], axis=1)
  #print('gt_boxes: ', gt_boxes)

  # to corners
  gt_boxes = box_utils.boxes_to_corners_3d(gt_boxes, order='hwl')
  #print('gt_boxes: ', gt_boxes)

  return gt_boxes


def load_pcd(detection_feature_path, global_timestamp_index):
  # point cloud
  pcd_file = os.path.join(detection_feature_path, '%04d_pcd.npy' % global_timestamp_index)
  pcd = np.load(pcd_file)
  #print('pcd.shape: ', pcd.shape) # (49274, 4)
  return pcd


def transform_detection_boxes(dets_frame_dict):
  '''
  Input:
    dets_frame_dict: {cav_id: {'dets': dets, 'info': info}}
      dets: [h,w,l,x,y,z,theta] in ab3dmot coordinate
      info: [0, 2, 0, 0, 0, 0, score]
  Output:
    det_boxes_dict: {cav_id: (N, 8, 3)} in v2v4real coordinate
  '''
  #print('dets_frame_dict: ', dets_frame_dict)
  det_boxes_dict = {}
  for cav_id in dets_frame_dict.keys():
    det_boxes = dets_frame_dict[cav_id]['dets']
    #print('det_boxes: ', det_boxes)
    
    # ab3dmot kitti to v2v4real: swapping yz
    det_boxes = np.concatenate([det_boxes[:, 0:4], det_boxes[:, 5:6], det_boxes[:, 4:5], det_boxes[:, 6:7]], axis=1)
    #print('det_boxes: ', det_boxes)

    # [h,w,l,x,y,z,theta] to [x, y, z, h, w, l, theta]
    det_boxes = np.concatenate([det_boxes[:, 3:6], det_boxes[:, 0:3], det_boxes[:, 6:7]], axis=1)
    #print('det_boxes: ', det_boxes)

    # to corners
    det_boxes = box_utils.boxes_to_corners_3d(det_boxes, order='hwl')
    #print('det_boxes: ', det_boxes)

    det_boxes_dict[cav_id] = det_boxes

  return det_boxes_dict
  

def text_3d(text, pos, direction=None, degree=-45.0, density=1,
  font='usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', font_size=200, text_color=(255, 255, 255)):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    #font_obj = ImageFont.load_default()
    
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=text_color)
    img = np.asarray(img)

    # keep non-background pixels
    #img_mask = img[:, :, 0] < 128
    img_mask = np.sum(img, axis=2) < 255 * 3

    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def custom_draw_geometry(pcd_list, boxes_list, seq_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #vis.create_window(width=640, height=480)
    #vis.create_window(width=1792, height=665)


    opt = vis.get_render_option()
    opt.line_width = 20
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    #print('original camera_params.extrinsic: ', camera_params.extrinsic)

    for pcd in pcd_list:
        vis.add_geometry(pcd)

    for boxes in boxes_list:
        for ele in boxes:
            vis.add_geometry(ele)

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    #print('after add_geometry() camera_params.extrinsic: ', camera_params.extrinsic)
    new_camera_params_extrinsic = np.copy(camera_params.extrinsic)

    # camera parameters to cover all point cloud
    #new_camera_params_extrinsic[0, 3] = -40
    #new_camera_params_extrinsic[1, 3] = 0
    #new_camera_params_extrinsic[2, 3] = 100
    # camera parameters to zoom-in val 0003 for paper figure
    #new_camera_params_extrinsic[0, 3] = -40
    #new_camera_params_extrinsic[1, 3] = 0
    #new_camera_params_extrinsic[2, 3] = 40

    #camera_params.extrinsic = new_camera_params_extrinsic
    #print('new camera_params.extrinsic: ', camera_params.extrinsic)
    #ctr.convert_from_pinhole_camera_parameters(camera_params)
    #camera_params = ctr.convert_to_pinhole_camera_parameters()
    #print('after convert camera_params.extrinsic: ', camera_params.extrinsic)


    # camera parameters for video: val 0003
    camera_file = 'ScreenCamera_for_video_val_%s.json' % seq_name
    new_camera_params_file = os.path.join('./DMSTrack/visualization_parameters', camera_file)

    new_camera_params = o3d.io.read_pinhole_camera_parameters(new_camera_params_file)
    ctr.convert_from_pinhole_camera_parameters(new_camera_params)



    return vis


def offscreen_render(pcd_list, boxes_list):
  render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
  render.scene.set_background([0, 0, 0, 0])
  
  for pcd in pcd_list:
    render.scene.add_geometry(pcd)

  for boxes in boxes_list:
    for ele in boxes:
      render.scene.add_geometry(ele)

  img_o3d = render.render_to_image()
  img = np.array(img_o3d)
  print('img.shape: ', img.shape) # (681, 1792, 3)
  print('np.mean(img): ', np.mean(img))
  print('np.min(img): ', np.min(img))
  print('np.max(img): ', np.max(img))

  assert False


def bbx2linset_color_coded_id(bbx_corner, color_map, ids, order='hwl'):
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color_map : 
        
    ids : (n)

    Returns
    -------
    line_set : list
        The list containing linsets.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    ids = ids.astype(int)
    color_per_object = [color_map[id*5566%len(color_map)] for id in ids]
    #print('color_per_object: ', color_per_object)

    # Use the same color for all lines
    #colors = [list(color) for _ in range(len(lines))]
    color_per_object = [ [list(color) for _ in range(len(lines))] for color in color_per_object]



    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        #line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.colors = o3d.utility.Vector3dVector(color_per_object[i])

        #line_mesh1 = LineMesh(np.array(bbx), lines, colors, radius=0.1)
        line_mesh1 = LineMesh(np.array(bbx), lines, color_per_object[i], radius=0.1)
        line_mesh1_geoms = line_mesh1.cylinder_segments

        bbx_linset += [*line_mesh1_geoms]

    return bbx_linset


def visualize_single_sample_output_gt(
  seq_name,
  pred_tensor,
  gt_tensor,
  pcd,
  show_vis=True,
  save_vis=False,
  save_path='',
  mode='constant',
  det_boxes_dict=None,
  gt_ids=None,
  trk_ids=None,
  matched_detection_id_dict=None,
  learnable_R_dict=None,
  track_P=None):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.

    # for my DMSTrack
    det_boxes_dict: {cav_id: (N, 8, 3)}
    """
    # MY_DEBUG
    visualization_for_video = True
    include_text_for_debug = False

    # for paper figure
    color_map_1 = {
      'gt': (0, 1, 0), # green
      'trk': (1, 0, 0), # red
      'ego': (1, 1, 0), # yellow
      '1': (0, 1, 1) # cyan
    }
    color_map_255 = {
      'gt': (0, 255, 0), # green
      'trk': (255, 0, 0), # red
      'ego': (255, 255, 0), # yellow
      '1': (0, 255, 255) # cyan
    }
    # for text go up
    text_coordinate_map = {
      'gt': 1,
      'trk': 3,
      'ego': 5,
      '1': 6
    }
    # for accompanying video, color coded tracking id
    random.seed(0)
    max_color = 30
    random_color_map = random_colors(max_color)


    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, -1], mode=mode)
    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)
    pcd_list = [o3d_pcd]


    if visualization_for_video:
      oabbs_pred = bbx2linset_color_coded_id(pred_tensor, random_color_map, trk_ids)
      oabbs_gt = []
    else: # for paper figure or debug
      oabbs_pred = bbx2linset(pred_tensor, color=color_map_1['trk'])
      oabbs_gt = bbx2linset(gt_tensor, color=color_map_1['gt'])


    boxes_list = [oabbs_pred,  oabbs_gt]


    if not visualization_for_video:
      # for paper figure or debug
      # include det box and id in each cav
      if det_boxes_dict is not None:
        for cav_id in det_boxes_dict.keys():
          oabbs_det = bbx2linset(det_boxes_dict[cav_id], color=color_map_1[cav_id])
          boxes_list.append(oabbs_det)

          if include_text_for_debug:
            det_boxes = det_boxes_dict[cav_id]
            for i in range(det_boxes.shape[0]):
              R_diagonal = np.diag((learnable_R_dict[cav_id][i]).detach().cpu().numpy())
              R_mean = np.mean(R_diagonal)
              R_str = '/'.join(['%.2f' % r for r in R_diagonal])

              #text = str(i) + ' R: %.2f' % R_mean
              #text = '   ' + 'avg diag(R): %.2f' % R_mean
              #text += ' / ' + R_str
              text = '   ' + '%.2f' % R_mean

              pos = det_boxes[i, text_coordinate_map[cav_id], :3]
              det_id_text_3d = text_3d(text, pos, text_color=color_map_255[cav_id])
              pcd_list.append(det_id_text_3d)


      # include gt id and track id
      if include_text_for_debug:
        if gt_ids is not None:
          for i in range(gt_tensor.shape[0]):
            text = '   ' + 'ground-truth ID: ' + str(int(gt_ids[i]))
            pos = gt_tensor[i, text_coordinate_map['gt'], :3]
            gt_id_text_3d = text_3d(text, pos, text_color=color_map_255['gt'])
            #pcd_list.append(gt_id_text_3d)
        if trk_ids is not None:
          for i in range(pred_tensor.shape[0]):
            # include track's matched detection info
            ego_det_id = matched_detection_id_dict[i]['ego']
            cav_1_det_id = matched_detection_id_dict[i]['1']
            # include track's state covariance P
            P_diagonal = np.diag(track_P[i])
            P_mean = np.mean(P_diagonal)
            P_str = '/'.join(['%.2f' % p for p in P_diagonal])

            #text = str(int(trk_ids[i])) + ' ego: %d, cav_1: %d' % (ego_det_id, cav_1_det_id) + ' P: %.2f' % P_mean
            #text = '   ' + 'tracking ID: ' + str(int(trk_ids[i])) + '  , avg diag(Sigma): %.2f' % P_mean
            #text += ' / ' + P_str
            text = '   ' + '%.2f' % P_mean

            pos = pred_tensor[i, text_coordinate_map['trk'], :3]
            trk_id_text_3d = text_3d(text, pos, text_color=color_map_255['trk'])
            #pcd_list.append(trk_id_text_3d)



    if show_vis:
        #custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
        vis = custom_draw_geometry(pcd_list, boxes_list, seq_name)
        vis.run()
        vis.destroy_window()
    if False and save_vis:
        # does not work
        assert False
        offscreen_render(pcd_list, boxes_list)
    if save_vis:
        vis = custom_draw_geometry(pcd_list, boxes_list, seq_name)
        vis.poll_events()
        vis.update_renderer()
        #time.sleep(10)
        o3d_screenshot_mat = vis.capture_screen_float_buffer(True)
        # scale and convert to uint8 type
        o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
        #print('o3d_screenshot_mat.shape: ', o3d_screenshot_mat.shape) # (681, 1792, 3)
        #print('np.mean(o3d_screenshot_mat): ', np.mean(o3d_screenshot_mat))
        #print('np.min(o3d_screenshot_mat): ', np.min(o3d_screenshot_mat))
        #print('np.max(o3d_screenshot_mat): ', np.max(o3d_screenshot_mat))
        # originally above all 0, set capture_screen_float_buffer(True) to see real values

        # save to image file
        import PIL
        image = PIL.Image.fromarray(o3d_screenshot_mat , "RGB")
        image.save(save_path)
        vis.destroy_window()


def visualize(show_vis, save_vis, save_vis_dir, ab3dmot_tracking_results, gt_boxes, gt_ids, 
    dets_frame_dict,
    detection_feature_path, seq_name, frame, global_timestamp_index, 
    matched_detection_id_dict, learnable_R_dict, track_P):
  '''
  Transform the input to v2v4real coordinate, 
  visulize tracking results and ground-truth boxes

  Input:
    show_vis: show open3d interactive visualization
    save_vis: save image and video visualization
    save_vis_dir: directory to save image and video visualization 
    ab3dmot_tracking_results: numpy [N, 15] in ab3dmot coordinate
      [h,w,l,x,y,z,theta] + [tracking_id] + [0, 2, 0, 0, 0, 0] + [score]
    gt_boxes: numpy (N, 7):
      same order of kalman filter state in ab3dmot coordinate:
      [x, y, z, theta, l, w, h]
    gt_ids: numpy (N)
    dets_frame_dict: {cav_id: {'dets': dets, 'info': info}}
      dets: [h,w,l,x,y,z,theta] in ab3dmot coordinate
      info: [0, 2, 0, 0, 0, 0, score]

  Target:
    pred_tensor : torch.Tensor or numpy
        (N, 8, 3) prediction.
    gt_tensor : torch.Tensor or numpy
        (N, 8, 3) groundtruth bbx
    pcd : torch.Tensor or numpy
        PointCloud, (N, 4)
    det_boxes_dict: {cav_id: (N, 8, 3)} in v2v4real coordinate
  '''
  trk_boxes = transform_tracking_boxes(ab3dmot_tracking_results)
  trk_ids = ab3dmot_tracking_results[:, 7]
  gt_boxes = transform_gt_boxes(gt_boxes)
  pcd = load_pcd(detection_feature_path, global_timestamp_index)

  det_boxes_dict = transform_detection_boxes(dets_frame_dict)

  save_file_name = 'seq_%s_frame_%04d.png' % (seq_name, frame)
  save_path = os.path.join(save_vis_dir, save_file_name)
  print(save_file_name)

  visualize_single_sample_output_gt(seq_name, trk_boxes, gt_boxes, pcd, show_vis=show_vis, save_vis=save_vis, save_path=save_path, det_boxes_dict=det_boxes_dict, gt_ids=gt_ids, trk_ids=trk_ids, matched_detection_id_dict=matched_detection_id_dict, learnable_R_dict=learnable_R_dict, track_P=track_P)
  


