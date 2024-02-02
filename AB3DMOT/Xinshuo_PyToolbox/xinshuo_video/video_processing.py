# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import cv2, os, sys
from skvideo.io import FFmpegWriter

from xinshuo_miscellaneous import is_path_exists, islistofstring, ispositiveinteger, reverse_list
from xinshuo_visualization import visualize_image
from xinshuo_io import mkdir_if_missing, load_image, load_list_from_folder
from xinshuo_images import image_resize

def extract_images_from_video_opencv(video_file, save_dir, debug=True):
	'''
	extract a list of images from a video file using opencv package
	Note that if the VideoCapture does not work, uninstall python-opencv and reinstall the newest version

	Parameters:
		video_file:		a file path to a video file
		save_dir:		a folder to save the images extracted from the video
		debug:			boolean, debug mode to check format

	Returns:
	'''
	if debug: assert is_path_exists(video_file), 'the input video file does not exist'
	mkdir_if_missing(save_dir)
	cap = cv2.VideoCapture(video_file)
	frame_id = 0

	while(True):
		ret, frame = cap.read()
		if not ret: break
		save_path = os.path.join(save_dir, 'image%05d.png' % frame_id)
		visualize_image(frame, bgr2rgb=True, save_path=save_path)
		frame_id += 1
		print('processing frame %d' % frame_id)

	cap.release()

	return

def extract_images_from_video_ffmpeg(video_file, save_dir, format='frame%06d.png', startnum=0, verbose=True, debug=True):
	'''
	extract a list of images from a video file using system ffmpeg library

	Parameters:
		video_file:		a file path to a video file
		save_dir:		a folder to save the images extracted from the video
		format:			string format for the extracted output images
		verbose:		boolean, display logging information
		debug:			boolean, debug mode to check format

	Returns:
	'''
	if debug: assert is_path_exists(video_file), 'the input video file does not exist'
	mkdir_if_missing(save_dir)
	if verbose:
		command = 'ffmpeg -i %s -start_number %d %s/%s' % (video_file, startnum, save_dir, format)
	else:
		command = 'ffmpeg -loglevel panic -i %s -start_number %d %s/%s' % (video_file, startnum, save_dir, format)
	os.system(command)

	return

def extract_images_from_video_ffmpeg_python(video_file, save_dir, debug=True):
	'''
	extract a list of images from a video file using python ffmpeg library

	Returns:
		frames: the frames of the video as a list of 3D tensors
	        (channels, width, height)"""
	'''

	vid = imageio.get_reader(filename, 'ffmpeg')
	frames = []
	for i in range(0, num_frames):
		image = vid.get_data(i)
		frames.append(image)

	return frames

def generate_video_from_list(image_list, save_path, framerate=30, downsample=1, display=True, warning=True, debug=True):
	'''
	create video from a list of images with a framerate
	note that: the height and widht of the images should be a multiple of 2

	parameters:
		image_list:			a list of image path
		save_path:			the path to save the video file
		framerate:			fps 
	'''
	if debug: 
		assert islistofstring(image_list), 'the input is not correct'
		assert ispositiveinteger(framerate), 'the framerate is a positive integer'
	mkdir_if_missing(save_path)
	inputdict = {'-r': str(framerate)}
	outputdict = {'-r': str(framerate), '-crf': '18', '-vcodec': 'libx264', '-profile:V': 'high', '-pix_fmt': 'yuv420p'}
	video_writer = FFmpegWriter(save_path, inputdict=inputdict, outputdict=outputdict)
	count = 1
	num_images = len(image_list)
	for image_path in image_list:
		if display:
			sys.stdout.write('processing frame %d/%d\r' % (count, num_images))
			sys.stdout.flush()
		image = load_image(image_path, resize_factor=downsample, warning=warning, debug=debug)

		# make sure the height and width are multiple of 2
		height, width = image.shape[0], image.shape[1]
		if not (height % 2 == 0 and width % 2 == 0):
			height += height % 2
			width += width % 2
			image = image_resize(image, target_size=[height, width], warning=warning, debug=debug)

		video_writer.writeFrame(image)
		count += 1

	video_writer.close()

def generate_video_from_folder(images_dir, save_path, framerate=30, downsample=1, depth=1, reverse=False, display=True, warning=True, debug=True):
	image_list, num_images = load_list_from_folder(images_dir, ext_filter=['.jpg', '.png', '.jpeg'], depth=depth, debug=debug)
	if reverse: image_list = reverse_list(image_list, warning=warning, debug=debug)
	if display:
		print('%d images loaded' % num_images)
	generate_video_from_list(image_list, save_path, framerate=framerate, downsample=downsample, display=display, warning=warning, debug=debug)
