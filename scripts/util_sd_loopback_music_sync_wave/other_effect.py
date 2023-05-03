from PIL import Image
import cv2
import numpy as np

debug_c = 0
def debug_save_img(img,comment):
	global debug_c
	im = Image.fromarray(img)
	im.save( f"scripts/testpngs/{debug_c}_{comment}.png")

	debug_c += 1

# https://note.nkmk.me/python-numpy-generate-gradation-image/
def get_gradient_2d(start, stop, width, height, is_horizontal):
	if is_horizontal:
		return np.tile(np.linspace(start, stop, width), (height, 1))
	else:
		return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
	result = np.zeros((height, width, len(start_list)), dtype=np.float)

	for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
		result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

	return result

def get_gradient_circular_2d(start, stop, width, height):
	x_axis = np.linspace(-1, 1, width)[None,:]
	y_axis = np.linspace(-1, 1, height)[:,None]
	arr = np.sqrt(x_axis ** 2 + y_axis ** 2)

	inner = np.array([start])[None, None, :]
	outer = np.array([stop])[None, None, :]
	arr /= arr.max()
	arr = arr[:, :, None]
	arr = arr * outer + (1 - arr) * inner

	return arr

_gradiention_mask = None
_gradiention_mask_stat = [-1,0,0]

def initialize_cache():
	global _gradiention_mask
	global _gradiention_mask_stat
	_gradiention_mask = None
	_gradiention_mask_stat = [-1,0,0]


def get_gradient_mask(mask_type, w, h):
	global _gradiention_mask
	global _gradiention_mask_stat

	if mask_type != _gradiention_mask_stat[0] or \
		w != _gradiention_mask_stat[1] or \
		h != _gradiention_mask_stat[2] or \
		_gradiention_mask is None:

		_gradiention_mask_stat = [mask_type, w, h]

		# "None","R","L","D","U","RD","LD","RU","LU","C Out","C In"
		if mask_type == 1:		# R
			mask = get_gradient_2d(0.0,1.0,w,h,True)
			mask = mask.reshape(*mask.shape, 1)
		elif mask_type == 2:	# L
			mask = get_gradient_2d(1.0,0.0,w,h,True)
			mask = mask.reshape(*mask.shape, 1)
		elif mask_type == 3:	# D
			mask = get_gradient_2d(0.0,1.0,w,h,False)
			mask = mask.reshape(*mask.shape, 1)
		elif mask_type == 4:	# U
			mask = get_gradient_2d(1.0,0.0,w,h,False)
			mask = mask.reshape(*mask.shape, 1)
		elif mask_type == 5:	# RD
			mask1 = get_gradient_2d(0.0,1.0,w,h,True)
			mask2 = get_gradient_2d(0.0,1.0,w,h,False)
			mask1 = mask1 * mask2
			mask1 /= mask1.max()
			mask = mask1.reshape(*mask1.shape, 1)
		elif mask_type == 6:	# LD
			mask1 = get_gradient_2d(1.0,0.0,w,h,True)
			mask2 = get_gradient_2d(0.0,1.0,w,h,False)
			mask1 = mask1 * mask2
			mask1 /= mask1.max()
			mask = mask1.reshape(*mask1.shape, 1)
		elif mask_type == 7:	# RU
			mask1 = get_gradient_2d(0.0,1.0,w,h,True)
			mask2 = get_gradient_2d(1.0,0.0,w,h,False)
			mask1 = mask1 * mask2
			mask1 /= mask1.max()
			mask = mask1.reshape(*mask1.shape, 1)
		elif mask_type == 8:	# LU
			mask1 = get_gradient_2d(1.0,0.0,w,h,True)
			mask2 = get_gradient_2d(1.0,0.0,w,h,False)
			mask1 = mask1 * mask2
			mask1 /= mask1.max()
			mask = mask1.reshape(*mask1.shape, 1)
		elif mask_type == 9:	# C Out
			mask = get_gradient_circular_2d(0.0,1.0,w,h)
			mask = mask * mask
		else:					# C In
			mask = get_gradient_circular_2d(1.0,0.0,w,h)
			mask = mask * mask
		
		_gradiention_mask = mask

	return _gradiention_mask




def apply_blur(img_array, blur_str):
	blur_str = max( int(blur_str), 0)
	if blur_str == 0:
		return img_array
	blur_str = (blur_str//2)*2 + 1

	img_array = cv2.GaussianBlur(img_array,(blur_str,blur_str),0,cv2.BORDER_DEFAULT)

	return img_array


def apply_hue_gradiation(img_array, gradiation_type, hue):
	gradiation_type = max(int(gradiation_type), 0)
	hue = int(hue)
	# 0 <= hue < 180
	hue %= 180

	hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
	hsv = hsv.astype(np.uint16)
	hsv[:,:,0] = (hsv[:,:,0]+hue)%180
	hsv = hsv.astype(np.uint8)
	colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	if gradiation_type == 0:
		return colored.astype(np.uint8)
	
	h,w,_ = img_array.shape

	mask = get_gradient_mask(gradiation_type, w, h)

	#output = np.zeros((h,w,3), np.uint16)
	output = img_array * (1 - mask) + colored * mask
	return output.astype(np.uint8)


def apply_other_effect(img:Image, blur_str, hue_type, hue ):
	
	img_array = np.array(img)
	
	if blur_str != 0:
		img_array = apply_blur(img_array, blur_str)
	
	if hue_type != -1:
		img_array = apply_hue_gradiation(img_array, hue_type, hue)
		#debug_save_img(img_array,"hue")
	
	return Image.fromarray(img_array)



