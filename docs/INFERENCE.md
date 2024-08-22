# Inference

You can use my DMSTrack model checkpoint to run cooperative tracking inference and evaluation for reproducing the tracking accuracy in the DMSTrack paper.

## **1. Inference and Evaluation:**
```shell
cd DMSTrack
python3 main_dkf.py \
--dataset v2v4real \
--det_name multi_sensor_differentiable_kalman_filter \
--num_frames_backprop 10 \
--num_frames_per_sub_seq -1 \
--num_epochs 0 \
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
--load_model_path ./official_models/model_ego_epoch_3.pth \
--save_dir_prefix reproducing_official_result
```

The tracking and evaluation result will be saved in the folder:
```
DMSTrack
├── DMSTrack/
│   ├── results/
│   │   ├── v2v4real/
│   │   │   ├── reproducing_official_result/
│   │   │   │   ├── evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_0/
│   │   │   │   │   ├── summary_car_average_eval3D.txt
│   │   │   │   │   ├── data_0/
```

We will see the tracking accuracy AMOTA 43.52.


## **2. Visualization:**

You can append --show_vis to the above command to use Open3D to show the visualization of the tracking result frame by frame interactively.  
Or you can append --save_vis to the above command to use Open3D to show and save the visualization of the tracking result frame by frame automatically.
The tracking result images will be saved in the visualization folder in the above path.


## **3. (Optional) V2V4Real Baseline Inference and Evaluation:**
If you want to run the tracking inference and evaluation using my implementation of the V2V4Real's cooperative tracking baseline method CoBEVT + AB3DMOT, you can follow the commands: 

Tracking inference command:
```shell
cd ../AB3DMOT
python3 main.py --dataset v2v4real --det_name cobevt 
```

Evaluation command:
```shell
python3 scripts/KITTI/evaluate.py cobevt_Car_val_H1 1 3D 0.25 
```

We will see the tracking accuracy AMOTA 37.16.



