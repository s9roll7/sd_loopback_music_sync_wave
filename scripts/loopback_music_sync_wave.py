import os
import platform
import math
import subprocess as sp
import random
import re
import glob
import time
from PIL import Image
import json
import numpy as np
import cv2
import copy

import modules.scripts
import gradio as gr

from modules import processing,images
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state, sd_upscalers

import scripts.util_sd_loopback_music_sync_wave.affine
import scripts.util_sd_loopback_music_sync_wave.slide
import scripts.util_sd_loopback_music_sync_wave.sync_effect
import scripts.util_sd_loopback_music_sync_wave.bpm
import scripts.util_sd_loopback_music_sync_wave.other_effect
import scripts.util_sd_loopback_music_sync_wave.sam
import scripts.util_sd_loopback_music_sync_wave.controlnet
import scripts.util_sd_loopback_music_sync_wave.upscale
from scripts.util_sd_loopback_music_sync_wave.regex import create_regex, create_regex_text

skip_process_for_debug = False

debug_c = 0

def debug_save_img_array(img_array, comment):
	debug_save_img( Image.fromarray(img_array), comment)

def debug_save_img(img:Image,comment):
	global debug_c
	img.save( f"scripts/testpngs/{debug_c}_{comment}.png")

	debug_c += 1




# @func
wave_completed_regex = create_regex(r'@','wave_completed',2)
wave_remaining_regex = create_regex(r'@','wave_remaining',2)
wave_amplitude_regex = create_regex(r'@','wave_amplitude',2)
wave_shape_regex = create_regex(r'@','wave_shape',2)
wave_progress_regex = create_regex(r'@','wave_progress',2)
total_progress_regex = create_regex(r'@','total_progress',2)
random_regex = create_regex(r'@','random',1,1)

# #func
vel_x_regex = create_regex(r'#','vel_x',1)
vel_y_regex = create_regex(r'#','vel_y',1)
rot_regex = create_regex(r'#','rot',1)
zoom_regex = create_regex(r'#','zoom',1)
center_regex = create_regex(r'#','center',2)
rot_x_regex = create_regex(r'#','rot_x',1)
rot_y_regex = create_regex(r'#','rot_y',1)

blur_regex = create_regex(r'#','blur',1)
hue_regex = create_regex(r'#','hue',2)

inpaint_regex = create_regex_text(r'#','__inpaint',1,1)

slide_x_regex = create_regex(r'#','slide_x',2,1)
slide_y_regex = create_regex(r'#','slide_y',2,1)

postprocess_regex = create_regex(r'#','post_process',1)


# @@bpm[]
#	-> bpm.py

# $func
#	-> sync_effect.py

extend_prompt_range_regex = r'([0-9]+)\-([0-9]+)'
#wild_card_regex = r'(?:\A|\W)__(\w+)__(?:\W|\Z)'
wild_card_regex = r'(\A|\W)__(\w+)__(\W|\Z)'


wave_func_map = {
	"zero": lambda p : 0,
	"one": lambda p : 1,
	"wave": lambda p : 1 - abs(math.cos(math.radians( (p + 0.5) * 180 ))),
	"wave2": lambda p : 1 - abs(math.cos(math.radians( p * 180 ))),
	"wave3": lambda p : p*p,
	"wave4": lambda p : (1-p)*(1-p),
}

wave_prompt_change_timing_map = {
	"zero": 0,
	"one": 0,
	"wave": 0,
	"wave2": 0.5,
	"wave3": 0,
	"wave4": 0,
}

def get_wave_type_list():
	return list(wave_func_map.keys())


def resize_img_array(img_array, w, h):
	if img_array.shape[0] + img_array.shape[1] < h + w:
		interpolation = interpolation=cv2.INTER_CUBIC
	else:
		interpolation = interpolation=cv2.INTER_AREA
	return cv2.resize(img_array, (w, h), interpolation=interpolation)

def resize_img(image:Image, w, h):
	_w,_h = image.size
	if _w == w and _h == h:
		return image
	
	im = resize_img_array(np.array(image), w, h)
	return Image.fromarray(im)

def image_open_and_resize(path, w, h):
	image = Image.open(path)
	return resize_img(image, w, h)


def get_wild_card_dir():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    path = os.path.join(path, "wildcards")
    return os.path.normpath(path)

def get_video_frame_path(project_dir, i):
	path = os.path.join(project_dir, "video_frame")
	path = os.path.join(path, f"{str(i).zfill(5)}.png")
	return path

def run_cmd(cmd, silent = False):
	cmd = list(map(lambda arg: str(arg), cmd))
	if not silent:
		print("Executing %s" % " ".join(cmd))
	popen_params = {"stdout": sp.DEVNULL, "stderr": sp.PIPE, "stdin": sp.DEVNULL}

	if os.name == "nt":
		popen_params["creationflags"] = 0x08000000

	proc = sp.Popen(cmd, **popen_params)
	out, err = proc.communicate()  # proc.wait()
	proc.stderr.close()

	if proc.returncode:
		raise IOError(err.decode("utf8"))

	del proc

def encode_video(input_pattern, starting_number, output_dir, fps, quality, encoding, create_segments, segment_duration, ffmpeg_path, sound_file_path):
	two_pass = (encoding == "VP9 (webm)")
	alpha_channel = ("webm" in encoding)
	suffix = "webm" if "webm" in encoding else "mp4"
	output_location = output_dir + f".{suffix}"

	encoding_lib = {
	  "VP9 (webm)": "libvpx-vp9",
	  "VP8 (webm)": "libvpx",
	  "H.264 (mp4)": "libx264",
	  "H.265 (mp4)": "libx265",
	}[encoding]

	args = [
		"-framerate", fps,
		"-start_number", int(starting_number),
		"-i", input_pattern
		]
	
	if sound_file_path:
		args += ["-i", sound_file_path]
	
	args+=[
		"-c:v", encoding_lib, 
		"-b:v","0", 
		"-crf", quality,
	]

	if encoding_lib == "libvpx-vp9":
		args += ["-pix_fmt", "yuva420p"]
		
	if(ffmpeg_path == ""):
		ffmpeg_path = "ffmpeg"
		if(platform.system == "Windows"):
			ffmpeg_path += ".exe"

	print("\n\n")
	if two_pass:
		first_pass_args = args + [
			"-pass", "1",
			"-an", 
			"-f", "null",
			os.devnull
		]

		second_pass_args = args + [
			"-pass", "2",
			output_location
		]

		print("Running first pass ffmpeg encoding")		  

		run_cmd([ffmpeg_path] + first_pass_args)
		print("Running second pass ffmpeg encoding.	 This could take awhile...")
		run_cmd([ffmpeg_path] + second_pass_args)
	else:
		print("Running ffmpeg encoding.	 This could take awhile...")
		run_cmd([ffmpeg_path] + args + [output_location])

	if(create_segments):
		print("Segmenting video")
		run_cmd([ffmpeg_path] + [
		  "-i", output_location,
		  "-f", "segment",
		  "-segment_time", segment_duration,
		  "-vcodec", "copy",
		  "-acodec", "copy",
		  f"{output_dir}.%d.{suffix}"
		])

def extract_sound(sound_file_path, output_dir, ffmpeg_path):
	ext = os.path.splitext(os.path.basename(sound_file_path))[1]

	if ext in (".mp4",".MP4"):

		if(ffmpeg_path == ""):
			ffmpeg_path = "ffmpeg"
			if(platform.system == "Windows"):
				ffmpeg_path += ".exe"

		tmp_path = os.path.join( output_dir, "sound.mp4" )
		run_cmd([ffmpeg_path] + [
		  "-i", sound_file_path,
		  "-vn",
		  "-acodec", "copy",
		  tmp_path
		])
		print("tmp_path : ",tmp_path)
		if os.path.isfile(tmp_path):
			sound_file_path = tmp_path

	return sound_file_path




def set_weights(match_obj, wave_progress):
	weight_0 = 0
	weight_1 = 0
	if match_obj.group(1) is not None:
		weight_0 = float(match_obj.group(1))
	if match_obj.group(2) is not None:
		weight_1 = float(match_obj.group(2))

	max_weight = max(weight_0, weight_1)
	min_weight = min(weight_0, weight_1)

	weight_range = max_weight - min_weight
	weight = min_weight + weight_range * wave_progress
	return f"{weight:.3f}"

def set_weights2(match_obj, wave_progress):
	weight_0 = 0
	weight_1 = 0
	if match_obj.group(1) is not None:
		weight_0 = float(match_obj.group(1))
	if match_obj.group(2) is not None:
		weight_1 = float(match_obj.group(2))

	min_weight = weight_0
	max_weight = weight_1

	weight_range = max_weight - min_weight
	weight = min_weight + weight_range * wave_progress
	return f"{weight:.3f}"

#def get_weights(match_obj, out_list):
#	out_list.append( float(match_obj.group(1)) )
#	return ""

#def get_weights2(match_obj, out_list1, out_list2):
#	out_list1.append( float(match_obj.group(1)) )
#	out_list2.append( float(match_obj.group(2)) )
#	return ""

def get_weights(match_obj, *list_of_out_list):
	vals = [ float(x) for x in match_obj.groups() if x is not None ]
	for i, v in enumerate( vals ):
		list_of_out_list[i].append( v )
	return ""

def get_weights_text(match_obj, out_list1, out_list2):
	out_list1.append( str(match_obj.group(1)) )
	if match_obj.group(2) is not None:
		out_list2.append( str(match_obj.group(2)) )
	return ""

def get_random_value(match_obj):
	m1 = float(match_obj.group(1))
	m2 = 0
	if match_obj.group(2) is not None:
		m2 = float(match_obj.group(2))
	
	v = random.uniform(m1,m2)
	return "{:.3f}".format(v)

def replace_wild_card_token(match_obj, wild_card_map):
	m1 = match_obj.group(1)
	m3 = match_obj.group(3)

	dict_name = match_obj.group(2)

	if dict_name in wild_card_map:
		token_list = wild_card_map[dict_name]
		token = token_list[random.randint(0,len(token_list)-1)]
		return m1+token+m3
	else:
		return match_obj.group(0)


def get_positive_prompt_from_image(img):
	from modules.generation_parameters_copypaste import parse_generation_parameters
	geninfo, _ = images.read_info_from_image( img )
	res = parse_generation_parameters(geninfo)
	return res["Prompt"]


def str_to_wave_list(raw_wave_list):
	if not raw_wave_list:
		raise IOError(f"Invalid input in wave list: {raw_wave_list}")
	wave_list=[]
	lines = raw_wave_list.split("\n")

	#start_msec,type,(strength)

	for wave_line in lines:
		params = wave_line.split(",")
		params = [x.strip() for x in params]
		if len(params) == 2:
			wave_list.append( {"start_msec": int(params[0]), "type": params[1], "strength": 1.0 })
		elif len(params) == 3:
			wave_list.append( {"start_msec": int(params[0]), "type": params[1], "strength": float(params[2]) })
		else:
			raise IOError(f"Invalid input in wave list line: {wave_line}")
	wave_list = sorted(wave_list, key=lambda x: x['start_msec'])

	start_times = [x["start_msec"] for x in wave_list]
	start_times.pop(0)
	start_times.append(start_times[-1] + 1)

	for i in range( len(wave_list) ):
		wave_list[i]["end_msec"] = start_times[i]-1
	
	print(wave_list)

	if wave_list[-1]["type"] != "end":
		print("!!!!!!!!!!! Warning. last element in wave list is not [end]")
		wave_list[-1]["type"] = "end"

	return wave_list

def wave_list_to_str(wave_list):
	wave_str_list = []
	for w in wave_list:
		if w["type"] in ("zero", "end") or w["strength"] == 1.0:
			wave_str_list.append( f'{w["start_msec"]},{w["type"]}' )
		else:
			wave_str_list.append( f'{w["start_msec"] },{w["type"]},{w["strength"]}' )	
	
	return "\n".join(wave_str_list)

def merge_wave_list(org_list,add_list,start_msec,end_msec):
	length = org_list[-1]["start_msec"]
	end_msec = min(length-1, end_msec)

	org_list = [x for x in org_list if not (start_msec <= x["start_msec"] <= end_msec)]
	add_list = [x for x in add_list if start_msec <= x["start_msec"] <= end_msec]
	
	wave_list = org_list + add_list

	wave_list = sorted(wave_list, key=lambda x: x['start_msec'])

	start_times = [x["start_msec"] for x in wave_list]
	start_times.pop(0)
	start_times.append(start_times[-1] + 1)

	for i in range( len(wave_list) ):
		wave_list[i]["end_msec"] = start_times[i]-1

	return wave_list

def process_image(p, loopback_count, str_for_loopback, is_controlnet, img_for_controlnet):
	if skip_process_for_debug:
		print("skip process for debug")
		return processing.Processed(p,[p.init_images[0]],p.seed)
	
	if is_controlnet:
		scripts.util_sd_loopback_music_sync_wave.controlnet.enable_controlnet(p, img_for_controlnet)
	else:
		scripts.util_sd_loopback_music_sync_wave.controlnet.disable_controlnet(p)

	while True:
		processed = processing.process_images(p)
		loopback_count -= 1

		if loopback_count <= 0:
			break

		p.init_images = [processed.images[0]]
		p.seed = processed.seed + 1
		p.denoising_strength = str_for_loopback
	return processed

def outpainting(p, img, org_mask_array, op_mask_blur, op_inpainting_fill, op_str, op_seed, is_controlnet, img_for_controlnet):
	p.init_images = [ img ]
	p.mask_blur = op_mask_blur * 2
	p.inpainting_fill = op_inpainting_fill
	p.inpaint_full_res = False
	p.denoising_strength = op_str
	p.seed = op_seed

	#image_mask
	k_size = int(op_mask_blur*2) // 2 * 2 + 1
	if k_size > 2:
		kernel = np.ones((k_size,k_size),np.uint8)
		mask_array = cv2.dilate(org_mask_array, kernel, iterations=1 )
	else:
		mask_array = org_mask_array
#	debug_save_img_array(mask_array,"out_image_mask")
	p.image_mask = Image.fromarray(mask_array, mode="L")

	#latent_mask
	k_size = int(op_mask_blur / 2) // 2 * 2 + 1
	if k_size > 2:
		kernel = np.ones((k_size,k_size),np.uint8)
		mask_array = cv2.dilate(org_mask_array, kernel, iterations=1 )
	else:
		mask_array = org_mask_array
#	debug_save_img_array(mask_array,"out_latent_mask")
	p.latent_mask = Image.fromarray(mask_array, mode="L")

	if is_controlnet:
		scripts.util_sd_loopback_music_sync_wave.controlnet.enable_controlnet(p, img_for_controlnet)
	else:
		scripts.util_sd_loopback_music_sync_wave.controlnet.disable_controlnet(p)

	processed = processing.process_images(p)

	return processed.images[0]


def affine_image(_p, op_mask_blur, op_inpainting_fill, op_str, op_seed, affine_input, is_controlnet, img_for_controlnet):
	print("affine_image")
	p = copy.copy(_p)

	img = p.init_images[0].convert('RGBA')
	img.putalpha(255)
	img = scripts.util_sd_loopback_music_sync_wave.affine.AffineImage(img, *affine_input)
	org_mask_array = np.array(img)[:, :, 3]

	if org_mask_array.min() == 255:
		print("skip outpainting")
		return img

	org_mask_array = 255 - org_mask_array
	img = img.convert("RGB")

	return outpainting(p, img, org_mask_array, op_mask_blur, op_inpainting_fill, op_str, op_seed, is_controlnet, img_for_controlnet)

def apply_slide(_p, op_mask_blur, op_inpainting_fill, op_str, op_seed, slide_inputs, is_controlnet, img_for_controlnet):
	print("apply_slide")
	p = copy.copy(_p)

	img = p.init_images[0].convert('RGBA')
	img.putalpha(255)

	img = scripts.util_sd_loopback_music_sync_wave.slide.SlideImage(img, *slide_inputs)

	org_mask_array = np.array(img)[:, :, 3]

	if org_mask_array.min() == 255:
		print("skip outpainting")
		return img

	org_mask_array = 255 - org_mask_array
	img = img.convert("RGB")

	return outpainting(p, img, org_mask_array, op_mask_blur, op_inpainting_fill, op_str, op_seed, is_controlnet, img_for_controlnet)


def apply_inpaint(_p, mask_prompt, inpaint_prompt, op_mask_blur, op_inpainting_fill, op_str, op_seed, is_controlnet, img_for_controlnet):
	print("apply_inpaint")
	p = copy.copy(_p)

	img = p.init_images[0]

	masks = scripts.util_sd_loopback_music_sync_wave.sam.get_mask_from_sam( img, mask_prompt, 0.3, 0 )
	if not masks:
		print("get_mask_from_sam failed.")
		return img

	p.prompt = inpaint_prompt

	org_mask_array = np.asarray( masks[0] )

	return outpainting(p, img, org_mask_array, op_mask_blur, op_inpainting_fill, op_str, op_seed, is_controlnet, img_for_controlnet)


def debug_info_suffix(mode_setting, base_denoising_strength, additional_denoising_strength, inner_lb_count, inner_lb_str):
	s=[
		"lb" if mode_setting == "loopback" else "i2i",
		f"bstr{ str(int(base_denoising_strength*100)).zfill(3) }",
		f"astr{ str(int(additional_denoising_strength*100)).zfill(3) }",
	]

	if inner_lb_count > 1:
		s += [
			f"i{ inner_lb_count }",
			f"istr{ str(int(inner_lb_str*100)).zfill(3) }",
		]
	return "_".join(s)

def create_output_dir(base_name, suffix, sample_path, project_dir):
	if(base_name==""):
		base_name = time.strftime("%Y%m%d-%H%M%S")
	else:
		base_name = base_name + "-" + time.strftime("%Y%m%d-%H%M%S")

	base_name += "_" + suffix

	loopback_wave_path = os.path.join(sample_path, "loopback-music-sync-wave")
	if os.path.isdir( project_dir ):
		loopback_wave_path = project_dir
	loopback_wave_images_path = os.path.join(loopback_wave_path, base_name)

	os.makedirs(loopback_wave_images_path, exist_ok=True)

	return loopback_wave_path, loopback_wave_images_path

def main_wave_loop(p, wave_list, current_time, total_progress, mode_setting, initial_denoising_strength, denoising_strength_change_amplitude, fps, wave_status, common_prompt_map, extend_prompt_map, init_image_per_wave_map, wild_card_map, effects, bpm_event):

	wave_index = wave_status["wave_index"]
	prompt_changed = wave_status["prompt_changed"]
	current_common_prompt = wave_status["current_common_prompt"]
	current_extend_prompt = wave_status["current_extend_prompt"]

	init_image = None


	while True:
		wave = wave_list[wave_index]
		wave_start_time = wave["start_msec"]
		wave_end_time = wave["end_msec"]
		wave_strength = wave["strength"]
		wave_type = wave["type"]

		if wave_type == "end":
			return False

		wave_prompt_change_timing = wave_prompt_change_timing_map[ wave_type ]

		while True:

			wave_progress = (current_time - wave_start_time)/(wave_end_time - wave_start_time)
			print("wave_progress = ", wave_progress)

			if prompt_changed == False:
				if wave_progress >= wave_prompt_change_timing:
					print("prompt_change_timing")
					prompt_changed = True

					# prompt change
					if common_prompt_map:
						if wave_index in common_prompt_map:
							current_common_prompt = common_prompt_map[ wave_index ]["prompt"]
					
					if extend_prompt_map:
						if wave_index in extend_prompt_map:
							current_extend_prompt = extend_prompt_map[ wave_index ]
						else:
							current_extend_prompt = ""
					
					if mode_setting == "loopback":
						# force init_image change
						if init_image_per_wave_map:
							if wave_index in init_image_per_wave_map:
								new_init_image_path = init_image_per_wave_map[ wave_index ]
								init_image = image_open_and_resize(new_init_image_path, p.width, p.height)

								try:
									prompt = get_positive_prompt_from_image(init_image)
									if prompt:
										current_common_prompt = prompt
								except Exception as e:
									print("get_positive_prompt_from_image failed. ",new_init_image_path)
					
					# register bpm event
					current_common_prompt = bpm_event.parse_prompt(current_common_prompt, current_time)
					current_extend_prompt = bpm_event.parse_prompt(current_extend_prompt, current_time)

					# wild card
					if wild_card_map:
						current_common_prompt = re.sub(wild_card_regex, lambda x: replace_wild_card_token(x, wild_card_map ), current_common_prompt)
						current_extend_prompt = re.sub(wild_card_regex, lambda x: replace_wild_card_token(x, wild_card_map ), current_extend_prompt)

					# random
					current_common_prompt = re.sub(random_regex, lambda x: get_random_value(x ), current_common_prompt)
					current_extend_prompt = re.sub(random_regex, lambda x: get_random_value(x ), current_extend_prompt)

					# effect
					current_common_prompt = effects.parse_prompt(current_common_prompt)
					current_extend_prompt = effects.parse_prompt(current_extend_prompt)

					print("current_common_prompt: ", current_common_prompt)
					print("current_extend_prompt: ", current_extend_prompt)
			
			if wave_end_time < current_time:
				break
			
			wave_amp = wave_func_map[ wave_type ](wave_progress)
			
			wave_amp_str = wave_amp * wave_strength
			
			denoising_strength = initial_denoising_strength + denoising_strength_change_amplitude * wave_amp_str

			print("wave_amp = ", wave_amp)

			raw_prompt = current_common_prompt + "," + current_extend_prompt

			# @func
			new_prompt = re.sub(wave_completed_regex, lambda x: set_weights(x, wave_progress), raw_prompt)
			new_prompt = re.sub(wave_remaining_regex, lambda x: set_weights(x, 1 - wave_progress), new_prompt)
			new_prompt = re.sub(wave_amplitude_regex, lambda x: set_weights2(x, wave_amp_str), new_prompt)
			new_prompt = re.sub(wave_shape_regex, lambda x: set_weights2(x, wave_amp), new_prompt)
			new_prompt = re.sub(wave_progress_regex, lambda x: set_weights2(x, wave_progress), new_prompt)
			new_prompt = re.sub(total_progress_regex, lambda x: set_weights2(x, total_progress), new_prompt)

			# exit
			wave_status["wave_index"] = wave_index
			wave_status["prompt_changed"] = prompt_changed
			wave_status["current_common_prompt"] = current_common_prompt
			wave_status["current_extend_prompt"] = current_extend_prompt

			wave_status["init_image"] = init_image
			wave_status["denoising_strength"] = denoising_strength

			wave_status["new_prompt"] = new_prompt

			return True

		print("main end wave ", wave_index)
		wave_index += 1
		prompt_changed = False
		print("main start wave ", wave_index)

def sub_wave_loop(p, wave_list, current_time, total_progress, wave_status, extend_prompt_map, wild_card_map, effects, bpm_event):

	wave_index = wave_status["wave_index"]
	prompt_changed = wave_status["prompt_changed"]
	current_extend_prompt = wave_status["current_extend_prompt"]

	while True:
		wave = wave_list[wave_index]
		wave_start_time = wave["start_msec"]
		wave_end_time = wave["end_msec"]
		wave_strength = wave["strength"]
		wave_type = wave["type"]

		if wave_type == "end":
			return False

		wave_prompt_change_timing = wave_prompt_change_timing_map[ wave_type ]

		while True:

			wave_progress = (current_time - wave_start_time)/(wave_end_time - wave_start_time)
			print("sub wave_progress = ", wave_progress)

			if prompt_changed == False:
				if wave_progress >= wave_prompt_change_timing:
					print("sub prompt_change_timing")
					prompt_changed = True

					# prompt change
					if wave_index in extend_prompt_map:
						current_extend_prompt = extend_prompt_map[ wave_index ]
					else:
						current_extend_prompt = ""
					
					# register bpm event
					current_extend_prompt = bpm_event.parse_prompt(current_extend_prompt, current_time)
					
					# wild card
					current_extend_prompt = re.sub(wild_card_regex, lambda x: replace_wild_card_token(x, wild_card_map ), current_extend_prompt)

					# random
					current_extend_prompt = re.sub(random_regex, lambda x: get_random_value(x ), current_extend_prompt)

					# effect
					current_extend_prompt = effects.parse_prompt(current_extend_prompt)

					print("sub current_extend_prompt: ", current_extend_prompt)
			
			if wave_end_time < current_time:
				break
			
			wave_amp = wave_func_map[ wave_type ](wave_progress)
			
			wave_amp_str = wave_amp * wave_strength
			
			print("sub wave_amp = ", wave_amp)

			raw_prompt = current_extend_prompt

			# @func
			new_prompt = re.sub(wave_completed_regex, lambda x: set_weights(x, wave_progress), raw_prompt)
			new_prompt = re.sub(wave_remaining_regex, lambda x: set_weights(x, 1 - wave_progress), new_prompt)
			new_prompt = re.sub(wave_amplitude_regex, lambda x: set_weights2(x, wave_amp_str), new_prompt)
			new_prompt = re.sub(wave_shape_regex, lambda x: set_weights2(x, wave_amp), new_prompt)
			new_prompt = re.sub(wave_progress_regex, lambda x: set_weights2(x, wave_progress), new_prompt)
			new_prompt = re.sub(total_progress_regex, lambda x: set_weights2(x, total_progress), new_prompt)

			# exit
			wave_status["wave_index"] = wave_index
			wave_status["prompt_changed"] = prompt_changed
			wave_status["current_extend_prompt"] = current_extend_prompt

			wave_status["new_prompt"] = new_prompt

			return True

		print("sub end wave ", wave_index)
		wave_index += 1
		prompt_changed = False
		print("sub start wave ", wave_index)

def create_extend_prompt_map(prompts, w_list):
	result_map = {}

	if not prompts or not w_list:
		return result_map

	lines = prompts.split("\n")
	for prompt_line in lines:
		# wave_range::ex_prompt
		# -1
		# 5
		# 1,3,8
		# 5-8

		params = prompt_line.split("::")
		if len(params) == 2:
			raw_range = params[0].replace(" ", "")
			m = re.match( extend_prompt_range_regex, raw_range )
			range_list = []
			if m:
				range_list = list(range(int(m.group(1)) , int(m.group(2)) + 1))
			elif raw_range == "-1":
				range_list = list(range( len(w_list) ))
			else:
				range_list = [ int(x) for x in raw_range.split(",")]
			
			print(raw_range, range_list)

			for n in range_list:
				ext_prompt = ""
				if n in result_map:
					ext_prompt = result_map[n] + ","
				
				ext_prompt += params[1]
				result_map[n] = ext_prompt
			
		else:
			raise IOError(f"Invalid input in extend prompt line: {prompt_line}")
	
	for n in result_map:
		result_map[n] = result_map[n].replace(',', ' , ')
	
	return result_map

def create_wild_card_map(wild_card_dir):
	result = {}
	if os.path.isdir(wild_card_dir):
		txt_list = glob.glob( os.path.join(wild_card_dir ,"**/*.txt"), recursive=True)
		print("wild card txt_list : ", txt_list)
		for txt in txt_list:
			basename_without_ext = os.path.splitext(os.path.basename(txt))[0]
			with open(txt, encoding='utf-8') as f:
				try:
					result[basename_without_ext] = [s.rstrip() for s in f.readlines()]
				except Exception as e:
					print(e)
					print("can not read ", txt)
	return result

def parse_sharp_func(prompt, fps):
	velx=[]
	vely=[]
	rot=[]
	zoom=[]
	cx=[]
	cy=[]
	rot_x=[]
	rot_y=[]

	prompt = re.sub(vel_x_regex, lambda x: get_weights(x, velx), prompt)
	prompt = re.sub(vel_y_regex, lambda x: get_weights(x, vely), prompt)
	prompt = re.sub(rot_regex, lambda x: get_weights(x, rot), prompt)
	prompt = re.sub(zoom_regex, lambda x: get_weights(x, zoom), prompt)
	prompt = re.sub(center_regex, lambda x: get_weights(x, cx, cy), prompt)
	prompt = re.sub(rot_x_regex, lambda x: get_weights(x, rot_x), prompt)
	prompt = re.sub(rot_y_regex, lambda x: get_weights(x, rot_y), prompt)

	_velx = 0 if not velx else (sum(velx) / fps)
	_vely = 0 if not vely else (sum(vely) / fps)
	_rot = 0 if not rot else (sum(rot) / fps)
	_zoom = 1 if not zoom else (1 + ((sum(zoom) - len(zoom))/ fps))
	_cx = 0.5 if not cx else (sum(cx)/len(cx))
	_cy = 0.5 if not cy else (sum(cy)/len(cy))
	_rot_x = 0 if not rot_x else (sum(rot_x) / fps)
	_rot_y = 0 if not rot_y else (sum(rot_y) / fps)

	affine_input = [_velx,_vely,_rot,_zoom,_cx,_cy, _rot_x, _rot_y]

	is_affine_need = (_velx != 0) or (_vely != 0) or (_rot != 0) or (_zoom != 1) or (_rot_x != 0) or (_rot_y != 0)

	return is_affine_need, affine_input, prompt

def parse_slide_func(prompt, fps):
	slide_x_type = [-1]
	slide_x_speed = [-1]
	slide_x_border = [0.5]
	prompt = re.sub(slide_x_regex, lambda x: get_weights(x, slide_x_type, slide_x_speed, slide_x_border), prompt)

	slide_y_type = [-1]
	slide_y_speed = [-1]
	slide_y_border = [0.5]
	prompt = re.sub(slide_y_regex, lambda x: get_weights(x, slide_y_type, slide_y_speed, slide_y_border), prompt)
	slide_inputs = [(int(slide_x_type[-1]), slide_x_speed[-1]/ fps, slide_x_border[-1]), (int(slide_y_type[-1]), slide_y_speed[-1]/ fps, slide_y_border[-1]) ]
	is_slide_need = (slide_x_type[-1] != -1) or (slide_y_type[-1] != -1)

	return is_slide_need, slide_inputs, prompt

def save_param_file(file_path, params):
	print("save param : ", params)
	with open(file_path, 'w') as f:
		json.dump(params, f, indent=4)

def load_param_file(file_path):
	params = {}
	with open(file_path, "r") as f:
		params = json.load(f)
	print("load param : ", params)
	return params

class Script(modules.scripts.Script):
	def title(self):
		return "Loopback Music Sync Wave"

	def show(self, is_img2img):
		return is_img2img

	def ui(self, is_img2img):

		param_file_path = gr.Textbox(label="Load inputs txt Path( Use parameters stored in *-inputs.txt )", lines=1, value="")


		fps = gr.Slider(minimum=1, maximum=120, step=1, label='Frames per second', value=8)

		project_dir = gr.Textbox(label="Project Directory(optional)", lines=1, value="")
		sound_file_path = gr.Textbox(label="Sound File Path(optional)", lines=1, value="")

		denoising_strength_change_amplitude = gr.Slider(minimum=0, maximum=1, step=0.01, label='Max additional denoise', value=0.6)

		with gr.Accordion(label="Cheat Sheet", open=False):
			gr.Textbox(label="ver 0.012", lines=5, interactive=False,
	      		value=
			    "-------------------------------\n"
			    "------ Wave List format -------\n"
			    "-------------------------------\n"
			    "time,wave type,wave strength\n"
				"\n"
			    "time ... Milliseconds from start\n"
				"wave type ... Select from zero, one, wave, wave2, wave3, wave4, end\n"
				"wave strength ... Default is 1.0 and is optional\n"
				"\n"
				"Can be generated automatically in [Loopback Music Sync Wave] Tab\n"
				"\n"
			    "-------------------------------\n"
				"----- Prompt Changes format ---\n"
			    "-------------------------------\n"
				"index of wave list::prompt\n"
				"\n"
				"index of wave list ... Index of wave list starting from 0\n"
				"                       0:: refers to the first wave \n"
				"prompt ... In this script, prompts are managed separately for common and additional parts,\n"
				"              where the common part is overwritten\n"
				"\n"
			    "-------------------------------\n"
				"----- Extend Prompt format ----\n"
			    "-------------------------------\n"
				"index of wave list::prompt\n"
				"\n"
				"index of wave list ... Index of wave list starting from 0\n"
				"                       0:: refers to the first wave \n"
				"                       1,3,6:: refers to waves of indexes 1, 3, and 6 \n"
				"                       5-8:: refers to waves of indexes 5, 6, 7, and 8 \n"
				"                       -1:: refers to the every wave \n"
				"prompt ... This will be added to the additional part of the prompt.\n"
				"\n"
				"As an example, if you do not want to change the basic picture, \n"
				"but want to change the background for each wave, \n"
				"you can simply write a line like this\n"
				"-1::__background__\n"
				"(However, this requires a background.txt file)\n"
				"\n"
			    "-------------------------------\n"
				"---------- Wild Card ----------\n"
			    "-------------------------------\n"
				"In this script, wildcards can be used with the following statement\n"
				"(No other extensions need to be installed as they are implemented within this script)\n"
				"__test__ ... which will be replaced by a random line in test.txt\n"
				"\n"
				"Wildcard files are searched from the following locations\n"
				"extensions\sd_loopback_music_sync_wave\wildcards\n"
				"\n"
			    "-------------------------------\n"
			    "---------- @function ----------\n"
			    "-------------------------------\n"
				"@wave_completed(min,max)\n"
				"@wave_remaining(min,max)\n"
				"@wave_amplitude(start_val,end_val)\n"
				"@wave_shape(start_val,end_val)\n"
				"@wave_progress(start_val,end_val)\n"
				"@total_progress(start_val,end_val)\n"
				"@random(min,max)\n"
				"\n"
			    "-------------------------------\n"
				"---------- #function ----------\n"
			    "-------------------------------\n"
				"#vel_x(x)\n"
				"#vel_y(y)\n"
				"#rot(deg)\n"
				"#zoom(z)\n"
				"#center(cx,cy)\n"
				"#rot_x(deg)\n"
				"#rot_y(deg)\n"
				"\n"
				"#slide_x(type,slide_val,border_pos)\n"
				"#slide_y(type,slide_val,border_pos)\n"
				"\n"
				"#blur(blur_str)\n"
				"#hue(type, hue)\n"
				"#post_process(flag)\n"
				"\n"
			    "-------------------------------\n"
				"---------- $function ----------\n"
			    "-------------------------------\n"
				"$shake_x(duration, amp)\n"
				"$shake_y(duration, amp)\n"
				"$shake_rot(duration, amp)\n"
				"$shake_rot_x(duration, amp)\n"
				"$shake_rot_y(duration, amp)\n"
				"$shake_zoom(duration, amp)\n"
				"$vibration(duration, amp)\n"
				"\n"
				"$random_xy(duration, x_amp, y_amp, resolution_msec=1000)\n"
				"$random_zoom(duration, z_amp, resolution_msec=1000)\n"
				"$random_rot(duration, r_amp, resolution_msec=1000)\n"
				"$random_rot_x(duration, r_amp, resolution_msec=1000)\n"
				"$random_rot_y(duration, r_amp, resolution_msec=1000)\n"
				"$random_center(duration, amp_x, amp_y, cx=0.5, cy=0.5, resolution_msec=1000 )\n"
				"\n"
				"$pendulum_xy(duration, x1, x2, y1, y2 )\n"
				"$pendulum_rot(duration, angle1, angle2 )\n"
				"$pendulum_rot_x(duration, angle1, angle2 )\n"
				"$pendulum_rot_y(duration, angle1, angle2 )\n"
				"$pendulum_zoom(duration, z1, z2 )\n"
				"$pendulum_center(duration, cx1, cx2, cy1, cy2 )\n"
				"\n"
				"$beat_blur(duration, amp)\n"
				"$random_blur(duration, amp, resolution_msec=1000)\n"
				"$pendulum_hue(duration, type, angle1, angle2)\n"
				"$random_hue(duration, type, start_angle, amp_angle, resolution_msec=1000)\n"
				"\n"
				"$beat_slide_x(duration, type, amp_slide_val, border_pos=0.5, amp_border=0)\n"
				"$beat_slide_y(duration, type, amp_slide_val, border_pos=0.5, amp_border=0)\n"
				"$random_slide_x(duration, type, amp_slide_val, border_pos=0.5, amp_border=0, resolution_msec=1000)\n"
				"$random_slide_y(duration, type, amp_slide_val, border_pos=0.5, amp_border=0, resolution_msec=1000)\n"
				"\n"
				"$inpaint(mask_prompt, inpaint_prompt)\n"
				"\n"
			    "-------------------------------\n"
				"-------- function usage -------\n"
			    "-------------------------------\n"
				"I'm providing an example of its use as a wildcard, so take a look there.\n"
				"\n"
			    "------------------------------------------------------------\n"
				"---------- How to write prompts that ignore waves ----------\n"
			    "------------------------------------------------------------\n"
				"You can specify a prompt that ignores the wave list with the following statement\n"
				"@@bpmBPM@DURATION[prompt]\n"
				"\n"
				"For example, the following in Extend Prompt would mean to add a prompt at 55.3 bpm for 8 seconds after the first wave\n"
				"0::@@bpm55.3@8000[$beat_slide_x(500, 1, 0.05,0.65)]\n"
				"\n"
	      		)

		with gr.Accordion(label="Main Wave", open=True):
			wave_list = gr.Textbox(label="Wave List (Main)", lines=5, value="")
			common_prompts = gr.Textbox(label="Prompt Changes (Main)", lines=5, value="")
			extend_prompts = gr.Textbox(label="Extend Prompt (Main)", lines=5, value="")
		
		with gr.Accordion(label="Sub Wave", open=True):
			sub_wave_list = gr.Textbox(label="Wave List (Sub)", lines=5, value="")
			sub_extend_prompts = gr.Textbox(label="Extend Prompt (Sub)", lines=5, value="")

		save_video = gr.Checkbox(label='Save results as video', value=True)
		output_name = gr.Textbox(label="Video Name", lines=1, value="")
		video_quality = gr.Slider(minimum=0, maximum=60, step=1, label='Video Quality (crf) ', value=22)
		video_encoding = gr.Dropdown(label='Video encoding ', value="H.264 (mp4)", choices=["VP9 (webm)", "VP8 (webm)", "H.265 (mp4)", "H.264 (mp4)"])
		
		with gr.Accordion(label="Mode Settings", open=True):
			use_video_frame_for_controlnet_in_loopback_mode = gr.Checkbox(label='Use video_frame for controlnet in loopback mode', value=False)
			mode_setting = gr.Radio(label='Mode', choices=["loopback","img2img"], value="loopback", type="value")
			use_controlnet_for_lb = gr.Checkbox(label='Use Controlnet for loopback', value=False)
			use_controlnet_for_img2img = gr.Checkbox(label='Use Controlnet for img2img', value=True)
			use_controlnet_for_inpaint = gr.Checkbox(label='Use Controlnet for inpaint', value=True)
			use_controlnet_for_outpaint = gr.Checkbox(label='Use Controlnet for outpaint', value=False)

		with gr.Accordion(label="OutPainting Setting", open=True):
			op_mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
			op_inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", elem_id=self.elem_id("inpainting_fill"))
			op_str = gr.Slider(minimum=0, maximum=1, step=0.01, label='Denoising Strength for OutPainting', value=0.8)
		
		with gr.Accordion(label="Upscale Setting", open=False):
			us_width = gr.Number(value=-1, label="Width", precision=0, interactive=True)
			us_height = gr.Number(value=-1, label="Height", precision=0, interactive=True)
			us_method = gr.Radio(label='Method', choices=['latent',*[x.name for x in sd_upscalers]], value=sd_upscalers[0].name, type="value")
			us_denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='Denoising Strength for latent method', value=0.35)

		with gr.Accordion(label="Inner Loopback", open=False):
			inner_lb_count = gr.Slider(minimum=1, maximum=10, step=1, label='Inner Loopback Count', value=1)
			inner_lb_str = gr.Slider(minimum=0, maximum=1, step=0.01, label='Denoising Strength for Inner Loopback', value=0.25)

		with gr.Accordion(label="Advanced Settings", open=False):
			save_prompts = gr.Checkbox(label='Save prompts as text file', value=True)
			initial_image_number = gr.Number(minimum=0, label='Initial generated image number', value=0)
			ffmpeg_path = gr.Textbox(label="ffmpeg binary.	Only set this if it fails otherwise.", lines=1, value="")
			segment_video = gr.Checkbox(label='Cut video in to segments ', value=False)
			video_segment_duration = gr.Slider(minimum=10, maximum=60, step=1, label='Video Segment Duration (seconds)', value=20)

		return [param_file_path, wave_list, sub_wave_list, project_dir, sound_file_path, mode_setting, use_video_frame_for_controlnet_in_loopback_mode, op_mask_blur, op_inpainting_fill, op_str, inner_lb_count, inner_lb_str, denoising_strength_change_amplitude, initial_image_number, common_prompts,extend_prompts, sub_extend_prompts, save_prompts, save_video, output_name, fps, video_quality, video_encoding, ffmpeg_path, segment_video, video_segment_duration, use_controlnet_for_lb,use_controlnet_for_img2img,use_controlnet_for_inpaint,use_controlnet_for_outpaint,us_width,us_height,us_method,us_denoising_strength]

		
	def run(self, p, param_file_path, raw_wave_list, raw_sub_wave_list, project_dir, sound_file_path, mode_setting, use_video_frame_for_controlnet_in_loopback_mode, op_mask_blur, op_inpainting_fill, op_str, inner_lb_count, inner_lb_str, denoising_strength_change_amplitude, initial_image_number, common_prompts, extend_prompts, sub_extend_prompts, save_prompts, save_video, output_name, fps, video_quality, video_encoding, ffmpeg_path, segment_video, video_segment_duration,use_controlnet_for_lb,use_controlnet_for_img2img,use_controlnet_for_inpaint,use_controlnet_for_outpaint,us_width,us_height,us_method,us_denoising_strength):
		calc_time_start = time.perf_counter()

		processing.fix_seed(p)

		scripts.util_sd_loopback_music_sync_wave.other_effect.initialize_cache()
		scripts.util_sd_loopback_music_sync_wave.controlnet.initialize(p)

		if param_file_path:
			if not os.path.isfile(param_file_path):
				raise IOError(f"Invalid input in param_file_path: {param_file_path}")
			else:
				params = load_param_file(param_file_path)
				raw_wave_list = params["raw_wave_list"]
				raw_sub_wave_list = params["raw_sub_wave_list"]
				project_dir = params["project_dir"]
				sound_file_path = params["sound_file_path"]
				mode_setting = params["mode_setting"]
				use_video_frame_for_controlnet_in_loopback_mode = params["use_video_frame_for_controlnet_in_loopback_mode"]
				op_mask_blur = params["op_mask_blur"]
				op_inpainting_fill = params["op_inpainting_fill"]
				op_str = params["op_str"]
				inner_lb_count = params["inner_lb_count"]
				inner_lb_str = params["inner_lb_str"]
				denoising_strength_change_amplitude = params["denoising_strength_change_amplitude"]
				initial_image_number = params["initial_image_number"]
				common_prompts = params["common_prompts"]
				extend_prompts = params["extend_prompts"]
				sub_extend_prompts = params["sub_extend_prompts"]
				save_prompts = params["save_prompts"]
				save_video = params["save_video"]
				output_name = params["output_name"]
				fps = params["fps"]
				video_quality = params["video_quality"]
				video_encoding = params["video_encoding"]
				ffmpeg_path = params["ffmpeg_path"]
				segment_video = params["segment_video"]
				video_segment_duration = params["video_segment_duration"]
				use_controlnet_for_lb = params["use_controlnet_for_lb"]
				use_controlnet_for_img2img = params["use_controlnet_for_img2img"]
				use_controlnet_for_inpaint = params["use_controlnet_for_inpaint"]
				use_controlnet_for_outpaint = params["use_controlnet_for_outpaint"]

				us_width = params["us_width"]
				us_height = params["us_height"]
				us_method = params["us_method"]
				us_denoising_strength = params["us_denoising_strength"]

				p.denoising_strength = params["p_denoising_strength"]
				p.prompt = params["p_prompt"]
				p.negative_prompt = params["p_negative_prompt"]
				p.seed = params["p_seed"]
				p.sampler_name = params["p_sampler_name"]
				p.cfg_scale = params["p_cfg_scale"]
				p.width = params["p_width"]
				p.height = params["p_height"]


		p.extra_generation_params = {
			"Max Additional Denoise": denoising_strength_change_amplitude,
		}

		#input validation
		raw_wave_list = raw_wave_list.strip()
		wave_list = str_to_wave_list(raw_wave_list)

		sub_wave_list = []
		raw_sub_wave_list = raw_sub_wave_list.strip()
		if raw_sub_wave_list:
			sub_wave_list = str_to_wave_list(raw_sub_wave_list)

		if sound_file_path:
			if not os.path.isfile(sound_file_path):
				raise IOError(f"Invalid input in sound_file_path: {sound_file_path}")


		#calc frames
		total_length = wave_list[-1]["end_msec"]
		frames = total_length * fps / 1000
		print( "end_time = ",total_length )
		print( "fps = ",fps )
		print( "frames = ",frames )
		frames = int(frames)


		# We save them ourselves for the sake of ffmpeg
		p.do_not_save_samples = True

		p.batch_size = 1
		p.n_iter = 1

		initial_seed = None
		initial_info = None

		grids = []
		all_images = []
		original_init_image = p.init_images
		state.job_count = frames * inner_lb_count

		initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
		initial_denoising_strength = p.denoising_strength

		suffix = debug_info_suffix(mode_setting, initial_denoising_strength, denoising_strength_change_amplitude, inner_lb_count, inner_lb_str)
		loopback_wave_path,loopback_wave_images_path = create_output_dir(output_name, suffix, p.outpath_samples, project_dir)
		
		p.outpath_samples = loopback_wave_images_path

		org_sound_file_path = sound_file_path
		sound_file_path = extract_sound(sound_file_path, loopback_wave_images_path, ffmpeg_path)

		common_prompts = common_prompts.strip()
		extend_prompts = extend_prompts.strip()
		sub_extend_prompts = sub_extend_prompts.strip()
		
		# output prompts txt
		if save_prompts:
			params = {}
			params["raw_wave_list"] = raw_wave_list
			params["raw_sub_wave_list"] = raw_sub_wave_list
			params["project_dir"] = project_dir
			params["sound_file_path"] = org_sound_file_path
			params["mode_setting"] = mode_setting
			params["use_video_frame_for_controlnet_in_loopback_mode"] = use_video_frame_for_controlnet_in_loopback_mode
			params["op_mask_blur"] = op_mask_blur
			params["op_inpainting_fill"] = op_inpainting_fill
			params["op_str"] = op_str
			params["inner_lb_count"] = inner_lb_count
			params["inner_lb_str"] = inner_lb_str
			params["denoising_strength_change_amplitude"] = denoising_strength_change_amplitude
			params["initial_image_number"] = initial_image_number
			params["common_prompts"] = common_prompts
			params["extend_prompts"] = extend_prompts
			params["sub_extend_prompts"] = sub_extend_prompts
			params["save_prompts"] = save_prompts
			params["save_video"] = save_video
			params["output_name"] = output_name
			params["fps"] = fps
			params["video_quality"] = video_quality
			params["video_encoding"] = video_encoding
			params["ffmpeg_path"] = ffmpeg_path
			params["segment_video"] = segment_video
			params["video_segment_duration"] = video_segment_duration
			params["use_controlnet_for_lb"] = use_controlnet_for_lb
			params["use_controlnet_for_img2img"] = use_controlnet_for_img2img
			params["use_controlnet_for_inpaint"] = use_controlnet_for_inpaint
			params["use_controlnet_for_outpaint"] = use_controlnet_for_outpaint
			params["us_width"] = us_width
			params["us_height"] = us_height
			params["us_method"] = us_method
			params["us_denoising_strength"] = us_denoising_strength
			
			params["p_denoising_strength"] = p.denoising_strength
			params["p_prompt"] = p.prompt
			params["p_negative_prompt"] = p.negative_prompt
			params["p_seed"] = p.seed
			params["p_sampler_name"] = p.sampler_name
			params["p_cfg_scale"] = p.cfg_scale
			params["p_width"] = p.width
			params["p_height"] = p.height

			save_param_file(loopback_wave_images_path + "-inputs.txt", params)


			with open(loopback_wave_images_path + "-prompts.txt", "w") as f:
				generation_settings = [
					"Generation Settings",
					f"Wave List: ",
					f"{raw_wave_list}",
					"",
					f"Sub Wave List: ",
					f"{raw_sub_wave_list}",
					"",
					f"FPS: {fps}",
					f"Base Denoising Strength: {initial_denoising_strength}",
					f"Max Additional Denoise: {denoising_strength_change_amplitude}",
					f"Project Directory: {project_dir}",
					f"Sound File: {org_sound_file_path}",
					f"Initial Image Number: {initial_image_number}",
					"",
					f"Mode: {mode_setting}",
					f"Use Video Frame for Controlnet in Loopback mode: {use_video_frame_for_controlnet_in_loopback_mode}",
					"",
					f"OutPainting Mask blur: {op_mask_blur}",
					f"OutPainting Masked content: {op_inpainting_fill}",
					f"OutPainting Denoising Strength: {op_str}",
					"",
					f"Inner Loopback Count: {inner_lb_count}",
					f"Denoising Strength for Inner Loopback: {inner_lb_str}",
					"",
					"Video Encoding Settings",
					f"Save Video: {save_video}",
					"",
					"Upscale Settings",
					f"Width: {us_width}",
					f"Height: {us_height}",
					f"Upscale Method: {us_method}",
					f"Denoising strength for latent: {us_denoising_strength}",
				]
				
				if save_video:
					generation_settings = generation_settings + [
						f"Framerate: {fps}",
						f"Quality: {video_quality}",
						f"Encoding: {video_encoding}",
						f"Create Segmented Video: {segment_video}"
					]

					if segment_video:
						generation_settings = generation_settings + [f"Segment Duration: {video_segment_duration}"]
				
				generation_settings = generation_settings + [
					"",
					"Prompt Details",
					"Initial Prompt:",
					p.prompt,
					"",
					"Negative Prompt:",
					p.negative_prompt,
					"",
					"Prompt Changes:",
					common_prompts,
					"",
					"Extend Prompts:",
					extend_prompts,

					"",
					"Sub Extend Prompts:",
					sub_extend_prompts
				]

				f.write('\n'.join(generation_settings))

		# create maps

		def create_init_image_per_wave_map(root_path):
			result = {}
			init_image_per_wave_path = os.path.join(root_path, "video_frame_per_wave")
			if os.path.isdir( init_image_per_wave_path ):
				pngs = glob.glob( os.path.join(init_image_per_wave_path ,"[0-9]*.png"), recursive=False)
				for png in pngs:
					basename_without_ext = os.path.splitext(os.path.basename(png))[0]
					result[int(basename_without_ext)] = png
			return result
		
		init_image_per_wave_map = create_init_image_per_wave_map(loopback_wave_path)
		print("init_image_per_wave_map", init_image_per_wave_map)

		def create_common_prompt_map(prompts):
			result = {}
			if prompts:
				lines = prompts.split("\n")
				for prompt_line in lines:
					# wave_index::prompt
					# wave_index::seed::prompt
					params = prompt_line.split("::")
					if len(params) == 2:
						result[int(params[0])] = { "prompt": params[1] }
#					elif len(params) == 3:
#						result[int(params[0])] = { "seed": params[1] , "prompt": params[2] }
					else:
						raise IOError(f"Invalid input in common prompt line: {prompt_line}")
			return result

		common_prompt_map = create_common_prompt_map(common_prompts)
		print("common_prompt_map", common_prompt_map)

		
		extend_prompt_map = create_extend_prompt_map(extend_prompts, wave_list)
		print("extend_prompt_map", extend_prompt_map)

		sub_extend_prompt_map = create_extend_prompt_map(sub_extend_prompts, sub_wave_list)
		print("sub_extend_prompt_map", sub_extend_prompt_map)

		wild_card_dir = get_wild_card_dir()
		print("wild_card_dir : ", wild_card_dir)

		wild_card_map = create_wild_card_map(wild_card_dir)
		print("wild_card_map", wild_card_map)


		history = []

		# Reset to original init image at the start of each batch
		p.init_images = original_init_image
		
		seed_state = "adding"
		initial_seed = p.seed

		i = 0
		
		main_wave_status = {
			"wave_index" : 0,
			"prompt_changed" : False,
			"current_common_prompt" : p.prompt,
			"current_extend_prompt" : "",
			# output
			"init_image" : None,
			"denoising_strength" : 0,
			"new_prompt" : "",
		}

		sub_wave_status = {
			"wave_index" : 0,
			"prompt_changed" : False,
			"current_extend_prompt" : "",
			# output
			"new_prompt" : "",
		}

		effects = scripts.util_sd_loopback_music_sync_wave.sync_effect.SyncEffect(fps)
		bpm_event = scripts.util_sd_loopback_music_sync_wave.bpm.BpmEvent(fps, total_length)

		seed_for_img2img_outpainting = int(random.randrange(4294967294))

		use_controlnet_for_main_generation = use_controlnet_for_img2img if mode_setting == "img2img" else use_controlnet_for_lb

		us_map = {}

		# generation loop
		while True:

			# cancelled.
			if state.interrupted:
				print("Generation cancelled.")
				raise Exception("Generation cancelled.")
			
			state.job = ""
			current_time = 1000 * i / fps
			total_progress = i/frames

			if main_wave_loop(p, wave_list, current_time, total_progress, mode_setting, initial_denoising_strength, denoising_strength_change_amplitude, fps,
		     					main_wave_status, common_prompt_map, extend_prompt_map, init_image_per_wave_map, wild_card_map, effects, bpm_event) == False:
				break

			if main_wave_status["init_image"]:
				p.init_images = [main_wave_status["init_image"]]
			
			p.denoising_strength = main_wave_status["denoising_strength"]
			new_prompt = main_wave_status["new_prompt"]

			if sub_wave_list and sub_extend_prompt_map:
				sub_wave_loop(p, sub_wave_list, current_time, total_progress, sub_wave_status, sub_extend_prompt_map, wild_card_map, effects, bpm_event)
				
				new_prompt += "," + sub_wave_status["new_prompt"]
			

			# override init_image for img2img
			if mode_setting == "img2img":
				frame_path = get_video_frame_path(project_dir, i)
				if frame_path and os.path.isfile(frame_path):
					org_frame = image_open_and_resize(frame_path, p.width, p.height)
					p.init_images = [org_frame]
				else:
					print("Warning! File not found : ",frame_path)

			p.n_iter = 1
			p.batch_size = 1
			p.do_not_save_grid = True

			if opts.img2img_color_correction:
				p.color_corrections = initial_color_corrections
			
			# bpm_event
			bpm_prompt = bpm_event.get_current_prompt(current_time)
			if bpm_prompt:
				# wild card
				bpm_prompt = re.sub(wild_card_regex, lambda x: replace_wild_card_token(x, wild_card_map ), bpm_prompt)

				# random
				bpm_prompt = re.sub(random_regex, lambda x: get_random_value(x ), bpm_prompt)

				# effect
				bpm_prompt = effects.parse_prompt(bpm_prompt)

				new_prompt += "," + bpm_prompt


			ef_prompt = effects.get_current_prompt()
			if ef_prompt:
				new_prompt += "," + ef_prompt

			# parse #func
			# affine
			is_affine_need, affine_input, new_prompt = parse_sharp_func(new_prompt,fps)
			print("affine_input : ", affine_input)

			# inpaint
			inpaint_mask_prompt = []
			inpaint_inpaint_prompt = []
			new_prompt = re.sub(inpaint_regex, lambda x: get_weights_text(x, inpaint_mask_prompt, inpaint_inpaint_prompt), new_prompt)
			print("inpaint_input : ", ( inpaint_mask_prompt, inpaint_inpaint_prompt))

			
			# slide/blind
			is_slide_need, slide_inputs, new_prompt = parse_slide_func(new_prompt, fps)
			print("slide_inputs : ", slide_inputs)


			# other
			blur_str=[]
			hue_type=[]
			hue_angle=[]
			post_process=[]

			new_prompt = re.sub(blur_regex, lambda x: get_weights(x, blur_str), new_prompt)
			new_prompt = re.sub(hue_regex, lambda x: get_weights(x, hue_type, hue_angle), new_prompt)

			new_prompt = re.sub(postprocess_regex, lambda x: get_weights(x, post_process), new_prompt)

			_blur_str = 0 if not blur_str else blur_str[0]
			_hue_type = -1 if not hue_type else hue_type[0]
			_hue_angle = 0 if not hue_angle else hue_angle[0]
			_post_process = 0 if not post_process else post_process[0]

			other_effect_input = [_blur_str,_hue_type,_hue_angle]
			print("other_effect_input : ", other_effect_input)


			p.prompt = new_prompt
			print(new_prompt)

			state.job += f"Iteration {i + 1}/{frames}. Denoising Strength: {p.denoising_strength}"

			control_net_input_image = None

			if mode_setting == "loopback":
				if use_video_frame_for_controlnet_in_loopback_mode:
					frame_path = get_video_frame_path(project_dir, i)
					if frame_path and os.path.isfile(frame_path):
						org_frame = image_open_and_resize(frame_path, p.width, p.height)
						control_net_input_image = org_frame
					else:
						print("!!!!!!!!!!!!! Warning! File not found : ",frame_path)
						print("use_video_frame_for_controlnet_in_loopback_mode failed.")

			op_seed = p.seed if mode_setting == "loopback" else seed_for_img2img_outpainting

			# affine
			if is_affine_need:

				state.job_count += 1

				p.init_images = [ affine_image(p, op_mask_blur, op_inpainting_fill, op_str, op_seed, affine_input, use_controlnet_for_outpaint, control_net_input_image)]
			
			# inpaint
			if inpaint_mask_prompt:
				if not inpaint_inpaint_prompt:
					inpaint_inpaint_prompt.append(p.prompt)

				state.job_count += 1

				p.init_images = [ apply_inpaint(p, inpaint_mask_prompt[0], inpaint_inpaint_prompt[0], op_mask_blur, op_inpainting_fill, op_str, op_seed, use_controlnet_for_inpaint, control_net_input_image ) ]
			
			# slide/blind
			if is_slide_need:

				state.job_count += 1

				p.init_images = [ apply_slide(p, op_mask_blur, op_inpainting_fill, op_str, op_seed, slide_inputs, use_controlnet_for_outpaint, control_net_input_image) ]


			# other
			if _post_process == 0:
				if _blur_str != 0 or _hue_type != -1:
					print("apply_other_effect")
					p.init_images = [ scripts.util_sd_loopback_music_sync_wave.other_effect.apply_other_effect( p.init_images[0], *other_effect_input ) ]
			
			if mode_setting == "img2img":
				p.seed = initial_seed

			processed = process_image(p, inner_lb_count, inner_lb_str, use_controlnet_for_main_generation, control_net_input_image)

			if initial_info is None:
				initial_info = processed.info
			
			processed_img = processed.images[0]

			# set init_image
			if mode_setting == "loopback":
				p.init_images = [processed_img]
			else:
				# Replace at the beginning of loop
				pass

			# post process
			if _post_process != 0:
				if _blur_str != 0 or _hue_type != -1:
					print("post apply_other_effect")
					processed_img = scripts.util_sd_loopback_music_sync_wave.other_effect.apply_other_effect(processed_img, *other_effect_input )
			
			image_number = int(initial_image_number + i)

			us_map[image_number] = {
				"seed" : p.seed,
				"prompt" : p.prompt,
				"info" : processed.info,
			}

			
			if seed_state == "adding":
				p.seed = processed.seed + 1
			elif seed_state == "subtracting":
				p.seed = processed.seed - 1
				
			images.save_image(processed_img, p.outpath_samples, "", processed.seed, processed.prompt, info=processed.info, save_to_dirs=False, forced_filename=str(image_number), p=p)

			history.append(processed_img)

			i+=1

		grid = images.image_grid(history, rows=1)
		if opts.grid_save:
			images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=initial_info, save_to_dirs=False, short_filename=not opts.grid_extended_filename, grid=True, p=p)
		grids.append(grid)

		all_images += history

		if opts.return_grid:
			all_images = grids + all_images

		out_video_path = loopback_wave_images_path
		# upscale
		if us_width != -1 or us_height != -1:
			if us_method != 'None':
				loopback_wave_images_path = os.path.join(loopback_wave_images_path, "upscale")
				scripts.util_sd_loopback_music_sync_wave.upscale.upscale(p, us_map, loopback_wave_images_path, us_width, us_height, us_method, us_denoising_strength)

		if save_video:
			input_pattern = os.path.join(loopback_wave_images_path, "%d.png")
			encode_video(input_pattern, initial_image_number, out_video_path, fps, video_quality, video_encoding, segment_video, video_segment_duration, ffmpeg_path, sound_file_path)

		processed = Processed(p, all_images, initial_seed, initial_info)

		calc_time_end = time.perf_counter()

		print("elapsed_time (sec) : ", calc_time_end - calc_time_start)

		return processed


def fake_run(raw_wave_list, extend_prompts, fps, initial_denoising_strength, denoising_strength_change_amplitude):

	#input validation
	raw_wave_list = raw_wave_list.strip()
	wave_list = str_to_wave_list(raw_wave_list)

	#calc frames
	total_length = wave_list[-1]["end_msec"]
	frames = total_length * fps / 1000
	print( "end_time = ",total_length )
	print( "fps = ",fps )
	print( "frames = ",frames )
	frames = int(frames)

	extend_prompts = extend_prompts.strip()
	
	# create maps
	extend_prompt_map = create_extend_prompt_map(extend_prompts, wave_list)
	print("extend_prompt_map", extend_prompt_map)
	
	wild_card_dir = get_wild_card_dir()
	print("wild_card_dir : ", wild_card_dir)

	wild_card_map = create_wild_card_map(wild_card_dir)
	print("wild_card_map", wild_card_map)

	i = 0
	
	main_wave_status = {
		"wave_index" : 0,
		"prompt_changed" : False,
		"current_common_prompt" : "",
		"current_extend_prompt" : "",
		# output
		"init_image" : None,
		"denoising_strength" : 0,
		"new_prompt" : "",
	}

	effects = scripts.util_sd_loopback_music_sync_wave.sync_effect.SyncEffect(fps)
	bpm_event = scripts.util_sd_loopback_music_sync_wave.bpm.BpmEvent(fps, total_length)

	stat_map = {}

	# generation loop
	while True:
		current_time = 1000 * i / fps
		total_progress = i/frames

		if main_wave_loop(None, wave_list, current_time, total_progress, "loop_back", initial_denoising_strength, denoising_strength_change_amplitude, fps,
							main_wave_status, None, extend_prompt_map, None, wild_card_map, effects, bpm_event) == False:
			break
		
		denoising_strength = main_wave_status["denoising_strength"]
		new_prompt = main_wave_status["new_prompt"]

		# bpm_event
		bpm_prompt = bpm_event.get_current_prompt(current_time)
		if bpm_prompt:
			# wild card
			bpm_prompt = re.sub(wild_card_regex, lambda x: replace_wild_card_token(x, wild_card_map ), bpm_prompt)
			
			# random
			bpm_prompt = re.sub(random_regex, lambda x: get_random_value(x ), bpm_prompt)

			# effect
			bpm_prompt = effects.parse_prompt(bpm_prompt)

			new_prompt += "," + bpm_prompt

		ef_prompt = effects.get_current_prompt()
		if ef_prompt:
			new_prompt += "," + ef_prompt
		

		# parse #func
		# affine
		is_affine_need, affine_input, new_prompt = parse_sharp_func(new_prompt,fps)
		print("affine_input : ", affine_input)

		# slide/blind
		is_slide_need, slide_inputs, new_prompt = parse_slide_func(new_prompt, fps)
		print("slide_inputs : ", slide_inputs)

		# other
		blur_str=[]
		hue_type=[]
		hue_angle=[]
		post_process=[]

		new_prompt = re.sub(blur_regex, lambda x: get_weights(x, blur_str), new_prompt)
		new_prompt = re.sub(hue_regex, lambda x: get_weights(x, hue_type, hue_angle), new_prompt)

		new_prompt = re.sub(postprocess_regex, lambda x: get_weights(x, post_process), new_prompt)

		_blur_str = 0 if not blur_str else blur_str[0]
		_hue_type = -1 if not hue_type else hue_type[0]
		_hue_angle = 0 if not hue_angle else hue_angle[0]
		_post_process = 0 if not post_process else post_process[0]

		other_effect_input = [_blur_str,_hue_type,_hue_angle]
		print("other_effect_input : ", other_effect_input)


		#new_prompt
		#denoising_strength
		print(new_prompt)

		stat_map[current_time/1000] = {
			"prompt":new_prompt,
			"denoising_strength":denoising_strength,
			"affine_input":affine_input,
			"slide_inputs":slide_inputs,
			"other_effect_input":other_effect_input,
		}

		i+=1


	return stat_map

