import os
import argparse
import cv2

from AB3DMOT.Xinshuo_PyToolbox.xinshuo_video import generate_video_from_list
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import load_list_from_folder


def parse_args():
  parser = argparse.ArgumentParser(description='generate_video')
  parser.add_argument('--save_dir_prefix', type=str, default='video2_fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1', help='folder of tracking save_dir_prefix')

  args = parser.parse_args()
  return args


def generate_single_video(image_list, video_file):
  #print('image_list: ', image_list)
  #generate_video_from_list(image_list[:100], video_file, framerate=10)

  frame = cv2.imread(image_list[0])
  height, width, layers = frame.shape
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  video = cv2.VideoWriter(video_file, fourcc, 10, (width, height))
  for image in image_list:
    video.write(cv2.imread(image))

  cv2.destroyAllWindows()
  video.release()



def generate_video(save_dir_prefix):
  # sample image file
  # /home/eddy/my_cooperative_tracking/AB3DMOT/results/v2v4real/video_fusion_loss_r1a0d0_subseq_n1_backprop_10_cgn_1/evaluation_multi_sensor_differentiable_kalman_filter_Car_val_0008_H1_epoch_0/visualization/seq_0008_frame_0000.png

  save_folder = os.path.join('../results/v2v4real/', save_dir_prefix)

  image_list = []
  num_images = 0

  seq_name_list = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008']
  seq_name_list = ['0005', '0003', '0006', '0007', '0000']

  for seq_name in seq_name_list:
    save_seq_folder = os.path.join(
      save_folder, 
      'evaluation_multi_sensor_differentiable_kalman_filter_Car_val_%s_H1_epoch_0' % seq_name,
      'visualization')
    seq_image_list, seq_num_images = load_list_from_folder(save_seq_folder)
    #seq_image_list = [os.path.join(save_seq_folder, f) for f in os.listdir(save_seq_folder) if os.path.isfile(join(save_seq_folder, f))].sort()

    #seq_video_file = os.path.join(save_folder, 'tracking_result_video_%s.mp4' % seq_name)
    #generate_single_video(seq_image_list, seq_video_file)

    image_list.extend(seq_image_list)
    num_images += seq_num_images

  seq_video_file = os.path.join(save_folder, 'tracking_result_video_53670.mp4')
  generate_single_video(image_list, seq_video_file)



def main(args):
  generate_video(args.save_dir_prefix)


if __name__ == '__main__':
  args = parse_args()
  main(args)
