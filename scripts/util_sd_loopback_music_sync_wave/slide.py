import cv2
import numpy as np
from PIL import Image

debug_c = 0

def debug_save_img_array(img_array, comment):
	debug_save_img( Image.fromarray(img_array), comment)

def debug_save_img(img:Image,comment):
	global debug_c
	img.save( f"scripts/testpngs/{debug_c}_{comment}.png")

	debug_c += 1

def slide_image(img_array, slide_type, slide_val, slide_direction, border, is_y ):
	h, w, c = img_array.shape
	output = output = np.zeros((h, w, 4), np.uint8)

	if slide_type == 0:
		# open / close
		if slide_direction > 0:
			dst_start = 0
			dst_end = border - slide_val
			src_start = slide_val
			src_end = border
			if src_start < src_end:
				if is_y:
					output[:,dst_start:dst_end] = img_array[:,src_start:src_end]
				else:
					output[dst_start:dst_end,:] = img_array[src_start:src_end,:]

			dst_start = border + slide_val
			dst_end = w if is_y else h
			src_start = border
			src_end = (w if is_y else h) - slide_val
			if src_start < src_end:
				if is_y:
					output[:,dst_start:dst_end] = img_array[:,src_start:src_end]
				else:
					output[dst_start:dst_end,:] = img_array[src_start:src_end,:]

		else:
			dst_start = slide_val
			dst_end = border
			src_start = 0
			src_end = border - slide_val
			if src_start < src_end:
				if is_y:
					output[:,dst_start:dst_end] = img_array[:,src_start:src_end]
				else:
					output[dst_start:dst_end,:] = img_array[src_start:src_end,:]

			dst_start = border
			dst_end = (w if is_y else h) - slide_val
			src_start = border + slide_val
			src_end = w if is_y else h
			if src_start < src_end:
				if is_y:
					output[:,dst_start:dst_end] = img_array[:,src_start:src_end]
				else:
					output[dst_start:dst_end,:] = img_array[src_start:src_end,:]
	else:
		# cross
		if slide_direction > 0:
			dst_start = 0
			dst_end = (h if is_y else w) - slide_val
			src_start = slide_val
			src_end = (h if is_y else w)
			if src_start < src_end:
				if is_y:
					output[dst_start:dst_end,:border] = img_array[src_start:src_end,:border]
				else:
					output[:border,dst_start:dst_end] = img_array[:border,src_start:src_end]

			dst_start = slide_val
			dst_end = (h if is_y else w)
			src_start = 0
			src_end = (h if is_y else w) - slide_val
			if src_start < src_end:
				if is_y:
					output[dst_start:dst_end,border:] = img_array[src_start:src_end,border:]
				else:
					output[border:,dst_start:dst_end] = img_array[border:,src_start:src_end]

		else:
			dst_start = slide_val
			dst_end = (h if is_y else w)
			src_start = 0
			src_end = (h if is_y else w) - slide_val
			if src_start < src_end:
				if is_y:
					output[dst_start:dst_end,:border] = img_array[src_start:src_end,:border]
				else:
					output[:border,dst_start:dst_end] = img_array[:border,src_start:src_end]
			
			dst_start = 0
			dst_end = (h if is_y else w) - slide_val
			src_start = slide_val
			src_end = (h if is_y else w)
			if src_start < src_end:
				if is_y:
					output[dst_start:dst_end,border:] = img_array[src_start:src_end,border:]
				else:
					output[border:,dst_start:dst_end] = img_array[border:,src_start:src_end]
	
	return output


def slide_y_image(img_array, slide_type, slide_val, slide_border_pos):
	h, w, c = img_array.shape

	if slide_type == -1:
		return img_array
	
	slide_border_pos = min(max(0, slide_border_pos), 1)
	
	border = int(slide_border_pos * w)
	slide_direction = 1 if slide_val > 0 else -1

	if slide_type == 0:
		# open / close
		if slide_val > 0:
			slide_val = int(w * slide_val)

		else:
			slide_val = -1 * int(w * slide_val)

	else:
		# up /down
		if slide_val > 0:
			slide_val = int(h * slide_val)

		else:
			slide_val = -1 * int(h * slide_val)
	
	output = slide_image(img_array, slide_type, slide_val, slide_direction, border, True)
	
	#debug_save_img_array(img_array, "y_input")
	#debug_save_img_array(output, "y_slide")
	return output

def slide_x_image(img_array, slide_type, slide_val, slide_border_pos):
	h, w, c = img_array.shape

	if slide_type == -1:
		return img_array
	
	slide_border_pos = min(max(0, slide_border_pos), 1)
	
	border = int(slide_border_pos * h)
	slide_direction = 1 if slide_val > 0 else -1

	if slide_type == 0:
		# open / close
		if slide_val > 0:
			slide_val = int(h * slide_val)

		else:
			slide_val = -1 * int(h * slide_val)

	else:
		# cross
		if slide_val > 0:
			slide_val = int(w * slide_val)

		else:
			slide_val = -1 * int(w * slide_val)
	
	output = slide_image(img_array, slide_type, slide_val, slide_direction, border, False)
	
	#debug_save_img_array(img_array, "x_input")
	#debug_save_img_array(output, "x_slide")
	return output

def SlideImage(img:Image, slide_x_inputs, slide_y_inputs):

	img_array = np.asarray(img)
	h, w, c = img_array.shape

	img_array = slide_x_image(img_array, *slide_x_inputs)

	img_array = slide_y_image(img_array, *slide_y_inputs)

	return Image.fromarray(img_array)
