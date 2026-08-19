"""
PATCH -- add this function to opencood/tools/infrence_utils.py, right next
to the existing save_prediction_gt(). It's a superset of it (same files,
same names, same format) plus one new optional file, so nothing that
already reads *_pcd.npy / *_gt.npy / *_pred.npy breaks.

Call it exactly where save_prediction_gt(...) is currently called in
inference_vn.py's --save_npy branch, just pass pred_var_tensor through
(it's already sitting right there in that loop, from
infrence_utils.inference_intermediate_fusion_with_variance(...)).
"""

def save_prediction_gt_with_variance(pred_tensor, gt_tensor, pcd,
                                     timestamp, save_path,
                                     pred_var_tensor=None,
                                     pred_score=None):
    """
    Same as save_prediction_gt(), plus optionally saves the per-box
    variance and per-box confidence score, so a real frame's uncertainty
    can be plotted later without re-running the model.

    Files written (N = number of surviving predicted boxes):
        %04d_pcd.npy    (P, 4)   lidar points, xyzi           -- unchanged
        %04d_gt.npy     (G, 8, 3) GT box corners               -- unchanged
        %04d_pred.npy   (N, 8, 3) predicted box corners        -- unchanged
        %04d_var.npy    (N, 7)   predicted variance per box    -- NEW, only
                                   if pred_var_tensor is not None
        %04d_score.npy  (N,)     predicted confidence per box  -- NEW, only
                                   if pred_score is not None
    """
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)

    if pred_tensor is not None:
        pred_np = torch_tensor_to_numpy(pred_tensor)
        np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    else:
        print(f"[Warning] No prediction at frame {timestamp}")
        return

    if pred_var_tensor is not None:
        var_np = torch_tensor_to_numpy(pred_var_tensor)
        np.save(os.path.join(save_path, '%04d_var.npy' % timestamp), var_np)

    if pred_score is not None:
        score_np = torch_tensor_to_numpy(pred_score)
        np.save(os.path.join(save_path, '%04d_score.npy' % timestamp), score_np)


# In inference_vn.py, the --save_npy block currently calls:
#
#   infrence_utils.save_prediction_gt(pred_box_tensor, gt_box_tensor,
#                                      batch_data['ego']['origin_lidar'][0],
#                                      i, npy_save_path)
#
# change it to:
#
#   infrence_utils.save_prediction_gt_with_variance(
#       pred_box_tensor, gt_box_tensor,
#       batch_data['ego']['origin_lidar'][0],
#       i, npy_save_path,
#       pred_var_tensor=pred_var_tensor,
#       pred_score=pred_score)
#
# Do the same in inference_mc.py once its post_process_with_variance
# wiring is added (see the earlier audit PDF, Section 7).
