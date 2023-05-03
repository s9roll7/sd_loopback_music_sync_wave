
import statistics
from PIL import Image

import scripts.util_sd_loopback_music_sync_wave.audio_analyzer
import scripts.loopback_music_sync_wave


def prompt_test_process(wave_list_str:str, extend_prompts:str):

	wave_list_str = wave_list_str.strip()
	if not wave_list_str:
		print("Error wave_list_str empty")
		return None, " "
	extend_prompts = extend_prompts.strip()
	if not extend_prompts:
		print("Error extend_prompts empty")
		return None, " "
	
	fps = 24
	initial_denoising_strength=0
	denoising_strength_change_amplitude=1.0

	stat_map = scripts.loopback_music_sync_wave.fake_run( wave_list_str, extend_prompts, fps, initial_denoising_strength, denoising_strength_change_amplitude )

	times = []
	dstrs = []
	affine_velx = []
	affine_vely = []
	affine_rot = []
	affine_zoom = []
	affine_cx = []
	affine_cy = []
	affine_rot_x = []
	affine_rot_y = []

	slide_x_spd = []
	slide_x_pos = []
	slide_y_spd = []
	slide_y_pos = []

	other_blur = []
	other_hue = []

	'''
	"prompt":new_prompt,
	"denoising_strength":denoising_strength,
	"affine_input":affine_input,		#affine_input = [_velx,_vely,_rot,_zoom,_cx,_cy, _rot_x, _rot_y]
	"slide_inputs":slide_inputs,		#slide_inputs = [(int(slide_x_type[-1]), slide_x_speed[-1]/ fps, slide_x_border[-1]), (int(slide_y_type[-1]), slide_y_speed[-1]/ fps, slide_y_border[-1]) ]
	"other_effect_input":other_effect_input,	#other_effect_input = [_blur_str,_hue_type,_hue_angle]
	'''

	for t in stat_map:
		times.append(t)
		item = stat_map[t]
		dstrs.append(item["denoising_strength"])
		affine_velx.append(item["affine_input"][0] * fps)
		affine_vely.append(item["affine_input"][1] * fps)
		affine_rot.append(item["affine_input"][2] * fps)
		affine_zoom.append( 1.0 +  (item["affine_input"][3] - 1.0) * fps)
		affine_cx.append(item["affine_input"][4])
		affine_cy.append(item["affine_input"][5])
		affine_rot_x.append(item["affine_input"][6] * fps)
		affine_rot_y.append(item["affine_input"][7] * fps)

		slide_x_spd.append(item["slide_inputs"][0][1] * fps)
		slide_x_pos.append(item["slide_inputs"][0][2])
		slide_y_spd.append(item["slide_inputs"][1][1] * fps)
		slide_y_pos.append(item["slide_inputs"][1][2])

		other_blur.append(item["other_effect_input"][0])
		other_hue.append(item["other_effect_input"][2])
	
	def standardization(l):
		l_mean = statistics.mean(l)
		l_stdev = statistics.stdev(l)
		if l_stdev == 0:
			return [0 for i in l]
		return [(i - l_mean) / l_stdev for i in l]
	def normalization(l):
		l_min = min(l)
		l_max = max(l)
		if l_max == l_min:
			return [0 for i in l]
		return [(i - l_min) / (l_max - l_min) for i in l]	
	
	affine_velx = normalization(affine_velx)
	affine_vely = normalization(affine_vely)
	affine_rot = normalization(affine_rot)
	affine_zoom = normalization(affine_zoom)
	affine_cx = normalization(affine_cx)
	affine_cy = normalization(affine_cy)
	affine_rot_x = normalization(affine_rot_x)
	affine_rot_y = normalization(affine_rot_y)

	slide_x_spd = normalization(slide_x_spd)
	slide_x_pos = normalization(slide_x_pos)
	slide_y_spd = normalization(slide_y_spd)
	slide_y_pos = normalization(slide_y_pos)

	other_blur = normalization(other_blur)
	other_hue = normalization(other_hue)

	wave_list = scripts.loopback_music_sync_wave.str_to_wave_list(wave_list_str)

	plot_data = {
		"time" : times,
		"wave" : [x["start_msec"]/1000 for x in wave_list],
		"data" : {
			"denoising_strength" : dstrs, 
			"vel x" : affine_velx,
			"vel y" : affine_vely,
			"rotate" : affine_rot,
			"zoom" : affine_zoom,
			"center x" : affine_cx,
			"center y" : affine_cy,
			"rotate_x" : affine_rot_x,
			"rotate_y" : affine_rot_y,

			"slide_x_vel" : slide_x_spd,
			"slide_x_pos" : slide_x_pos,
			"slide_y_vel" : slide_y_spd,
			"slide_y_pos" : slide_y_pos,

			"other_blur_str" : other_blur,
			"other_hue_angle" : other_hue,
		}
	}

	fig = scripts.util_sd_loopback_music_sync_wave.audio_analyzer.create_prompt_figure(plot_data)

	return fig, " "



