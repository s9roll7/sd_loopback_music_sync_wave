import os
import time
import PIL.Image
from modules import processing,images
from modules.processing import Processed

def upscale(p, us_map, output_path, us_width, us_height, us_method, us_denoising_strength):

	if us_width == -1 and us_height == -1:
		return
	if us_method == 'None':
		return
	
	print("upscale start")

	calc_time_start = time.perf_counter()

	os.makedirs(output_path, exist_ok=True)

	org_w = p.width
	org_h = p.height

	if us_width != -1:
		us_width = us_width // 8 * 8
	if us_height != -1:
		us_height = us_height // 8 * 8

	if us_width == -1:
		us_width = int((us_height * org_w / org_h) // 8 * 8)
	elif us_height == -1:
		us_height = int((us_width * org_h / org_w) // 8 * 8)

	print("({0},{1}) upscale to ({2},{3})".format(org_w, org_h, us_width, us_height))

	total = len(us_map)

	for i,img_no in enumerate(us_map):
		img_path = os.path.join(p.outpath_samples, f"{img_no}.png")
		if not os.path.isfile(img_path):
			print("warning file not found : ",img_path)
			continue

		im = PIL.Image.open(img_path)
		_seed = us_map[img_no]["seed"]
		_prompt = us_map[img_no]["prompt"]
		_info = us_map[img_no]["info"]

		if us_method != 'latent':
			resized_img = images.resize_image(0, im, us_width, us_height, us_method )
		else:
			p.resize_mode = 3
			p.width = us_width
			p.height = us_height
			p.init_images = [im]
			p.seed = _seed
			p.prompt = _prompt
			p.denoising_strength = us_denoising_strength

			processed = processing.process_images(p)

			resized_img = processed.images[0]

		images.save_image(resized_img, output_path, "", _seed, _prompt, info=_info, forced_filename=str(img_no), p=p)

		print(f"{i}/{total}")
	

	calc_time_end = time.perf_counter()
	print("upscale elapsed_time (sec) : ", calc_time_end - calc_time_start)	





