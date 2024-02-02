
# video 0003, current best model checkpoint
for SEQ_EVAL_MODE in 0007 0006 0005 0004 0003 0002 0001 0000 
do
  python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backprop 10 --num_frames_per_sub_seq -1 --num_epochs 0  --use_multiple_nets --seq_eval_mode $SEQ_EVAL_MODE --run_evaluation_every_epoch  --training_split train --evaluation_split val --regression_loss_weight 1 --association_loss_weight 0 --det_neg_log_likelihood_loss_weight 0 --feature fusion  --clip_grad_norm 1 --load_model_path ~/my_cooperative_tracking/AB3DMOT/all_results/fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1/evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_3/model_ego_epoch_3.pth --save_dir_prefix video2_fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1  --save_vis
done























# paper figure 0003, current best model checkpoint
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backprop 10 --num_frames_per_sub_seq -1 --num_epochs 0  --use_multiple_nets --seq_eval_mode 0003 --run_evaluation_every_epoch  --training_split train --evaluation_split val --regression_loss_weight 1 --association_loss_weight 0 --det_neg_log_likelihood_loss_weight 0 --feature fusion  --clip_grad_norm 1 --load_model_path ~/my_cooperative_tracking/AB3DMOT/all_results/fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1/evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_3/model_ego_epoch_3.pth --save_dir_prefix vis_fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1  --save_vis

# all, current best model checkpoint
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backprop 10 --num_frames_per_sub_seq -1 --num_epochs 0  --use_multiple_nets --seq_eval_mode all --run_evaluation_every_epoch  --training_split train --evaluation_split val --regression_loss_weight 1 --association_loss_weight 0 --det_neg_log_likelihood_loss_weight 0 --feature fusion  --clip_grad_norm 1 --load_model_path ~/my_cooperative_tracking/AB3DMOT/all_results/fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1/evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_3/model_ego_epoch_3.pth --save_dir_prefix vis_fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1  --show_vis








# debug
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --seq_eval_mode 0003   --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/debug_evaluation_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_1/model_ego_epoch_1.pth --training_split val --evaluation_split val --feature fusion --save_dir_prefix eval_debug --show_vis



# pos_frame
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --seq_eval_mode 0003   --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/pos_frame_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_9/model_ego_epoch_9.pth --save_dir_prefix eval_pos_frame --show_vis


# bev_frame
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --seq_eval_mode 0003   --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/bev_frame_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_9/model_ego_epoch_9.pth --save_dir_prefix eval_bev_frame --show_vis




# positional_embedding, P_learnable_1
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --seq_eval_mode 0003   --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/positional_embedding_P_learnable_1_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_99/model_ego_epoch_99.pth --save_dir_prefix eval_positional_embedding_P_learnable_1 --show_vis


# positional_embedding, P_static_1
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --seq_eval_mode 0003   --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/positional_embedding_P_static_1_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_99/model_ego_epoch_99.pth --save_dir_prefix eval_positional_embedding_P_static_1 --show_vis





# debug no load_model_path, no use_static_default_R
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --seq_eval_mode 0003   --save_dir_prefix vis_debug --show_vis



# distance_residual
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --seq_eval_mode 0003   --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/distance_residual_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_0/model_ego_epoch_0.pth --save_dir_prefix eval_distance_residual --show_vis


# default R
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --seq_eval_mode 0003 --num_frames_backpro 1 --num_epochs 1 --use_static_default_R --run_evaluation_every_epoch --save_dir_prefix eval_static_default_R --show_vis










# overfit single frame
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --seq_eval_mode 0003 --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/debug_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_0/model_ego_epoch_0.pth --save_dir_prefix eval_debug --show_vis


# learnable_P_R
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --seq_eval_mode 0003 --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/learnable_P_R_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_9/model_ego_epoch_9.pth --save_dir_prefix eval_learnable_P_R --show_vis



# regular 
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --seq_eval_mode 0003 --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/regular_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_50/model_ego_epoch_50.pth --save_dir_prefix eval_regular --show_vis


# embedding experiment
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --seq_eval_mode 0003 --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/embedding_lr_0p1_multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_9/model_ego_epoch_9.pth --save_dir_prefix eval_embedding_lr_0p1 --show_vis

# force gt predict next track
#python3 main_dkf.py --dataset v2v4real --det_name multi_sensor_differentiable_kalman_filter --seq_eval_mode 0003 --num_frames_backpro -1 --num_epochs 1 --run_evaluation_every_epoch --use_multiple_nets --load_model_path ~/my_cooperative_tracking/AB3DMOT/results/v2v4real/multi_sensor_differentiable_kalman_filter_Car_val_0003_H1_epoch_49/model_ego_epoch_49.pth --save_dir_prefix eval_gt_force --show_vis

