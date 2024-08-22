# Train

You can use the following command to train a DMSTrack model.

## **1. Training:**
```shell
cd DMSTrack
python3 main_dkf.py \
--dataset v2v4real \
--det_name multi_sensor_differentiable_kalman_filter \
--num_frames_backprop 10 \
--num_frames_per_sub_seq -1 \
--num_epochs 20 \
--use_multiple_nets \
--seq_eval_mode all \
--run_evaluation_every_epoch \
--training_split train \
--evaluation_split val \
--regression_loss_weight 1 \
--association_loss_weight 0 \
--det_neg_log_likelihood_loss_weight 0 \
--feature fusion \
--clip_grad_norm 1 \
--save_dir_prefix training_result
```

This command will also run tracking inference and evaluation on the val set after training every epoch on the training set.

The training model checkpoints will be save in the folders:
```
DMSTrack
├── DMSTrack/
│   ├── results/
│   │   ├── v2v4real/
│   │   │   ├── training_result/
│   │   │   │   ├── training_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_1/
│   │   │   │   │   ├── model_ego_epoch_1.pth
│   │   │   │   │   ├── model_1_epoch_1.pth
│   │   │   │   ├── training_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_2/
│   │   │   │   │   ├── model_ego_epoch_2.pth
│   │   │   │   │   ├── model_1_epoch_2.pth
...
```


The val set tracking and evaluation result will be saved in the folders:
```
DMSTrack
├── DMSTrack/
│   ├── results/
│   │   ├── v2v4real/
│   │   │   ├── reproducing_official_result/
│   │   │   │   ├── evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_1/
│   │   │   │   │   ├── summary_car_average_eval3D.txt
│   │   │   │   │   ├── data_0/
│   │   │   │   ├── evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_2/
│   │   │   │   │   ├── summary_car_average_eval3D.txt
│   │   │   │   │   ├── data_0/
...
```
