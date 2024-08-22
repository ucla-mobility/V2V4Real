import argparse
import os
import time
import numpy as np

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from opencood.utils import box_utils


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion', 'no_fusion_keep_all']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)
    print(hypes['fusion']['core_method'])
    # IntermediateFusionDataset

    print("opencood_dataset.len_record: ", opencood_dataset.len_record)
    # train set has 32 sequences
    # [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648, 3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]
    # test set has 9 sequences
    # [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
    print("opencood_dataset.scenario_database.keys(): ", opencood_dataset.scenario_database.keys())
    # odict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8])
    npy_save_path = os.path.join(opt.model_dir, 'npy')


    # for debug
    #transform_and_save_detection_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path, {'ego', '1'})
    #assert False

    # only need to generate gt label once
    #transform_and_save_tracking_label_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path)
    #assert False

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())

    total_time = 0.0

    cav_id_set = set()

    for i, batch_data in enumerate(data_loader):
        print(i)
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            elif opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif opt.fusion_method == 'no_fusion_keep_all':
                start_time = time.time()
                pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict = \
                    infrence_utils.inference_no_fusion_keep_all(batch_data,
                                                                model,
                                                                opencood_dataset)
                end_time = time.time()
                total_time += (end_time - start_time)

                transformation_matrix_dict = {}
                for cav_id in batch_data.keys():
                  transformation_matrix_dict[cav_id] = batch_data[cav_id]['transformation_matrix']
                #print('transformation_matrix_dict: ', transformation_matrix_dict)

                #print('pred_box_dict: ', pred_box_dict)
                #print('pred_score_dict: ', pred_score_dict)
                #print('pred_feature_dict: ', pred_feature_dict)

                # For no_fusion_keep_all,
                # the detection evaluation still use no_fusion approach
                # but we will save all cav's detection results for tracking
                pred_box_tensor = pred_box_dict['ego']
                pred_score = pred_score_dict['ego']

                for cav_id in pred_box_dict.keys():
                  cav_id_set.add(cav_id)

            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                start_time = time.time()
                # MY_DEBUG: new return format in dict
                #pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset)
                end_time = time.time()
                total_time += (end_time - start_time)
                
                transformation_matrix_dict = {}
                for cav_id in batch_data.keys():
                  transformation_matrix_dict[cav_id] = batch_data[cav_id]['transformation_matrix']

                # This cobevt detection have the same mAP as the v2v4real paper: 0.665
                pred_box_tensor = pred_box_dict['ego']
                pred_score = pred_score_dict['ego']

                for cav_id in pred_box_dict.keys():
                  cav_id_set.add(cav_id)

                #print("batch_data['ego']: ", batch_data['ego'])
                #print("batch_data['ego'].keys(): ", batch_data['ego'].keys())

                #print("batch_data['ego']['record_len']: ", batch_data['ego']['record_len']) # 1
                #print("batch_data['ego']['object_ids']: ", batch_data['ego']['object_ids']) # 1

                if gt_box_tensor.shape[0] != gt_object_id_tensor.shape[0]:
                  # i == 98
                  #print("batch_data['ego']['object_ids']: ", batch_data['ego']['object_ids']) # [1, 2, 11]
                  print('gt_box_tensor.shape[0]: ', gt_box_tensor.shape[0]) # 2
                  print('gt_object_id_tensor: ', gt_object_id_tensor)
                  assert False
                
                #print("batch_data['ego']['scenario_index']: ", batch_data['ego']['scenario_index'])
                #print("batch_data['ego']['timestamp_index']: ", batch_data['ego']['timestamp_index'])

                # num_objects, num_corners_per_box, 3 dim coordinates
                #print('pred_box_tensor.shape: ', pred_box_tensor.shape) # [3, 8, 3]
                #print('pred_box_tensor: ', pred_box_tensor)

                #print('pred_score.shape: ', pred_score.shape) # [3]
                #print('pred_score: ', pred_score)

                #print('gt_box_tensor.shape: ', gt_box_tensor.shape) # [1, 8, 3]
                #print('gt_box_tensor: ', gt_box_tensor)
                #if i == 144:
                #  assert False

            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            # overall calculating
            #print('gt_box_tensor: ', gt_box_tensor)
            #print('len(gt_box_tensor): ', len(gt_box_tensor))
            #print('gt_object_id_tensor: ', gt_object_id_tensor)
            #assert False
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            # short range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,
                                       left_range=0,
                                       right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.7,
                                       left_range=0,
                                       right_range=30)

            # middle range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,
                                       left_range=30,
                                       right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.7,
                                       left_range=30,
                                       right_range=50)

            # right range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,
                                       left_range=50,
                                       right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.7,
                                       left_range=50,
                                       right_range=100)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path,
                                                  pred_score,
                                                  gt_object_id_tensor)

                if opt.fusion_method == 'no_fusion_keep_all' or 'intermediate':
                  for cav_id in pred_box_dict.keys():
                    npy_cav_id_save_path = os.path.join(opt.model_dir, 'npy', cav_id)
                    if not os.path.exists(npy_cav_id_save_path):
                        os.makedirs(npy_cav_id_save_path)
                    pred_box_tensor = pred_box_dict[cav_id]
                    pred_score = pred_score_dict[cav_id]
                    pred_feature = pred_feature_dict[cav_id]
                    pred_early_feature = pred_early_feature_dict[cav_id]
                    transformation_matrix = transformation_matrix_dict[cav_id]
                    #print('pred_feature.shape: ', pred_feature.shape)
                    infrence_utils.save_prediction_gt(pred_box_tensor,
                                                      gt_box_tensor,
                                                      batch_data['ego'][
                                                          'origin_lidar'][0],
                                                      i,
                                                      npy_cav_id_save_path,
                                                      pred_score,
                                                      gt_object_id_tensor,
                                                      pred_feature,
                                                      pred_early_feature,
                                                      transformation_matrix)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    total_frames = i + 1
    speed = total_frames / total_time
    print('Detection speed: %f frames per second, total frames: %d, total time: %f ' % 
          (speed, total_frames, total_time))

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)
    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir,
                                  "short")
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir,
                                  "middle")
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir,
                                  "long")


    
    # MY_CODE
    if opt.save_npy:
      transform_and_save_detection_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path, cav_id_set)
      transform_and_save_tracking_label_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path)

    if opt.show_sequence:
        vis.destroy_window()

def transform_and_save_detection_to_ab3dmot_format(len_record, npy_save_path, cav_id_set):
  # opencood_dataset.len_record
  print('len_record: ', len_record)
  # [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]

  # add '' to also tranform data in npy_save_path
  cav_id_set.add('')

  for cav_id in cav_id_set:
    ab3dmot_detection_save_path = os.path.join(npy_save_path, 'ab3dmot_detection', cav_id)
    print('ab3dmot_detection_save_path: ', ab3dmot_detection_save_path)
    if not os.path.exists(ab3dmot_detection_save_path):
      os.makedirs(ab3dmot_detection_save_path)

    for scenario_index in range(len(len_record)):
      if scenario_index == 0:
        start_global_timestamp_index = 0
      else:
        start_global_timestamp_index = len_record[scenario_index - 1]
      end_global_timestamp_index = len_record[scenario_index] - 1
      #print('start_global_timestamp_index: ', start_global_timestamp_index)
      #print('end_global_timestamp_index: ', end_global_timestamp_index)
      
      ab3dmot_detection_save_file = os.path.join(ab3dmot_detection_save_path, '%04d.txt' % scenario_index)
      #print('ab3dmot_detection_save_file: ', ab3dmot_detection_save_file)
    
      with open(ab3dmot_detection_save_file, 'w') as f:
        for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
          local_timestamp_index = global_timestamp_index - start_global_timestamp_index
          # load v2v4real detection output file
          # reversed operation of 
          # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/tools/infrence_utils.py#L116
          #print('global_timestamp_index: ', global_timestamp_index)

          v2v4real_detection_file = os.path.join(npy_save_path, cav_id, '%04d_pred.npy' % global_timestamp_index)
          v2v4real_detection = np.load(v2v4real_detection_file)
          #print('v2v4real_detection.shape: ', v2v4real_detection.shape)
          # (9, 8, 3)
        
          v2v4real_detection_score_file = os.path.join(npy_save_path, cav_id, '%04d_pred_score.npy' % global_timestamp_index)
          v2v4real_detection_score = np.load(v2v4real_detection_score_file)
          #print('v2v4real_detection_score.shape: ', v2v4real_detection_score.shape)
          # (9) 

        
          # https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md
          # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/utils/box_utils.py#L14
          # (N, 8, 3) to (N , [xyz, h, w, l, theta])
          #print('v2v4real_detection: ', v2v4real_detection)
          boxes_3d = box_utils.corner_to_center(v2v4real_detection, order='hwl')
          #print('boxes_3d: ', boxes_3d)
          # (N, [xyz, h, w, l, theta]) to (N, (h, w, l, x, y, z, rot_y))
          boxes_3d = np.concatenate([boxes_3d[:, 3:6], boxes_3d[:, 0:3], boxes_3d[:, 6:7]], axis=1)
          #print('boxes_3d: ', boxes_3d)

          # transform to ab3dmot kitti coordinate system
          # swap y, z
          boxes_3d = np.concatenate([boxes_3d[:, 0:4], boxes_3d[:, 5:6], boxes_3d[:, 4:5], boxes_3d[:, 6:7]], axis=1)
          #print('boxes_3d: ', boxes_3d)

          for detection_id in range(v2v4real_detection.shape[0]):
            frame = local_timestamp_index
            type = 2 # type index of Car in ab3dmot KITTI
            box_2d = '0,0,0,0' # ignore 2d box
            score = v2v4real_detection_score[detection_id]
            box_3d = ','.join([str(value) for value in boxes_3d[detection_id]])
            #print('box_3d: ', box_3d)
            alpha = 0 # ignore observation angle
            ab3dmot_detection_string = '%d,%d,%s,%f,%s,%f\n' % (
              local_timestamp_index, type, box_2d, score, box_3d, alpha)
            #print('ab3dmot_detection_string: ', ab3dmot_detection_string)
            f.write(ab3dmot_detection_string)


def transform_and_save_tracking_label_to_ab3dmot_format(len_record, npy_save_path):
  # opencood_dataset.len_record
  print('len_record: ', len_record)
  # [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]


  ab3dmot_tracking_label_save_path = os.path.join(npy_save_path, 'ab3dmot_tracking_label')
  if not os.path.exists(ab3dmot_tracking_label_save_path):
    os.makedirs(ab3dmot_tracking_label_save_path)

  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1
    #print('start_global_timestamp_index: ', start_global_timestamp_index)
    #print('end_global_timestamp_index: ', end_global_timestamp_index)
      
    ab3dmot_tracking_label_save_file = os.path.join(ab3dmot_tracking_label_save_path, '%04d.txt' % scenario_index)
    
    with open(ab3dmot_tracking_label_save_file, 'w') as f:
      for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
        local_timestamp_index = global_timestamp_index - start_global_timestamp_index
        # load v2v4real gt files
        # reversed operation of 
        # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/tools/infrence_utils.py#L116
        #print('global_timestamp_index: ', global_timestamp_index)

        v2v4real_gt_file = os.path.join(npy_save_path, '%04d_gt.npy' % global_timestamp_index)
        v2v4real_gt = np.load(v2v4real_gt_file)
        #print('v2v4real_gt.shape: ', v2v4real_gt.shape)
        # (1, 8, 3)
        
        v2v4real_gt_object_id_file = os.path.join(npy_save_path, '%04d_gt_object_id.npy' % global_timestamp_index)
        v2v4real_gt_object_id = np.load(v2v4real_gt_object_id_file)
        #print('v2v4real_gt_object_id.shape: ', v2v4real_gt_object_id.shape)
        # (1,)
        
        # https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md
        # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/utils/box_utils.py#L14
        # (N, 8, 3) to (N , [xyz, h, w, l, theta])
        #print('v2v4real_gt: ', v2v4real_gt)
        boxes_3d = box_utils.corner_to_center(v2v4real_gt, order='hwl')
        #print('boxes_3d: ', boxes_3d)
        # (N, [xyz, h, w, l, theta]) to (N, (h, w, l, x, y, z, rot_y))
        boxes_3d = np.concatenate([boxes_3d[:, 3:6], boxes_3d[:, 0:3], boxes_3d[:, 6:7]], axis=1)
        #print('boxes_3d: ', boxes_3d)

        # transform to ab3dmot kitti coordinate system
        # swap y, z
        boxes_3d = np.concatenate([boxes_3d[:, 0:4], boxes_3d[:, 5:6], boxes_3d[:, 4:5], boxes_3d[:, 6:7]], axis=1)
        #print('boxes_3d: ', boxes_3d)
        #assert False

        # https://github.com/xinshuoweng/AB3DMOT/blob/master/scripts/KITTI/label/0000.txt
        # https://github.com/xinshuoweng/AB3DMOT/blob/master/scripts/KITTI/evaluate.py#L268
        for gt_id in range(v2v4real_gt.shape[0]):
          frame = local_timestamp_index
          track_id = v2v4real_gt_object_id[gt_id]
          object_type = 'Car' # type string of Car in ab3dmot label KITTI
          truncation = 0
          occlusion = 0
          obs_angle = 0
          box_2d = '0 0 0 0' # ignore 2d box
          box_3d = ' '.join([str(value) for value in boxes_3d[gt_id]])
          #print('box_3d: ', box_3d)
          ab3dmot_tracking_label_string = '%d %d %s %d %d %d %s %s\n' % (
            local_timestamp_index, track_id, object_type, truncation, occlusion, obs_angle,
            box_2d, box_3d)
          #print('ab3dmot_tracking_label_string: ', ab3dmot_tracking_label_string)
          f.write(ab3dmot_tracking_label_string)


if __name__ == '__main__':
    main()
