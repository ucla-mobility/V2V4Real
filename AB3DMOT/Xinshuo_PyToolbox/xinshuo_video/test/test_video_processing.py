# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from xinshuo_video.video_processing import generate_video_from_list

def test_generate_video_from_list():
	print('test basic')
	image_list = ['../image0001.jpg', '../image0002.jpg', '../image0003.jpg', '../image0004.jpg', '../image0005.jpg']
	generate_video_from_list(image_list, '../test.mp4')

	print('test slow framerate')
	image_list = ['../image0001.jpg', '../image0002.jpg', '../image0003.jpg', '../image0004.jpg', '../image0005.jpg']
	generate_video_from_list(image_list, '../test_fr5.mp4', framerate=5)

	print('test skipped frames')
	image_list = ['../image0001.jpg', '../image0003.jpg', '../image0005.jpg']
	generate_video_from_list(image_list, '../test_skipped.mp4', framerate=3)

	print('test large frames')
	image_list = ['../image35997.jpg', '../image35998.jpg', '../image35999.jpg', '../image36000.jpg', '../image36001.jpg']
	generate_video_from_list(image_list, '../test_large.mp4', framerate=5)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_generate_video_from_list()