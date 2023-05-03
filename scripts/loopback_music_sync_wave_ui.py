
import gradio as gr

from modules import script_callbacks
from modules.call_queue import wrap_gradio_gpu_call

from scripts.loopback_music_sync_wave import get_wave_type_list,str_to_wave_list,wave_list_to_str,merge_wave_list
from scripts.util_sd_loopback_music_sync_wave.wave_generator import wave_generator_process
from scripts.util_sd_loopback_music_sync_wave.audio_analyzer import audio_analyzer_process
from scripts.util_sd_loopback_music_sync_wave.wave_list_test import wave_list_test_process
from scripts.util_sd_loopback_music_sync_wave.frame_extractor import frame_extract_all,frame_extract_per_wave
from scripts.util_sd_loopback_music_sync_wave.prompt_test import prompt_test_process

def on_ui_tabs():

	with gr.Blocks(analytics_enabled=False) as wave_generation_interface:
		with gr.Tabs():
			with gr.TabItem('Wave List Generate'):
				with gr.Row().style(equal_height=True):
					with gr.Column(variant='panel'):
						with gr.Accordion(label="Common Input", open=True):
							input_audio_path = gr.Textbox(label='Audio File Path', lines=1)

							input_audio = gr.Audio(interactive=True, mirror_webcam=False, type="filepath")
							def fn_upload_org_audio(a):
								return a
							input_audio.upload(fn_upload_org_audio, input_audio, input_audio_path)
							gr.HTML(value="<p style='margin-bottom: 1.2em'>\
									If you have trouble entering the audio file path manually, you can also use drag and drop.\
									</p>")

						with gr.Row():
							with gr.Tabs(elem_id="lmsw_settings"):
								with gr.TabItem('Wave List Generator'):

									with gr.Accordion(label="Input", open=True):
										wave_bpm = gr.Slider(minimum=1, maximum=300, step=0.01, label='BPM', value=120.0)

										wave_beat_per_wave = gr.Slider(minimum=1, maximum=32, step=1, label='Beat Per Wave', value=4)

										wave_start_msec = gr.Number(value=0, label="Start Time (millisecond)", precision=0, interactive=True)
										wave_end_msec = gr.Number(value=5000, label="End Time (millisecond)", precision=0, interactive=True)

										sels = get_wave_type_list()
										wave_default_type = gr.Radio(label='Default Wave Type', choices=sels, value=sels[2], type="value")
										wave_default_strength = gr.Slider(minimum=0, maximum=3.0, step=0.1, label='Default Wave Strength', value=1.0)

									with gr.Row():
										wave_generate_btn = gr.Button('Generate', variant='primary')
									
									with gr.Accordion(label="Result", open=True):
										with gr.Row():
											wave_list_txt = gr.Textbox(label='Wave List', lines=30, interactive=True)
										with gr.Row():
											test_generate_btn = gr.Button('Generate Test Audio', variant='primary')
											send_to_extract_btn = gr.Button('Send to Frame Extract', variant='primary')
										with gr.Row():
											prompt_test_txt = gr.Textbox(label='Input Prompt you want to test(Extend Prompt format)', lines=5, interactive=True)
										with gr.Row():
											prompt_test_btn = gr.Button('Prompt Test', variant='primary')

							with gr.Tabs(elem_id="lmsw_settings2"):
								with gr.TabItem('Audio Analyzer'):

									with gr.Accordion(label="Input", open=True):
										aa_offset = gr.Slider(minimum=-1000, maximum=1000, step=1, label='Offset Time', value=0)

										
										with gr.Accordion(label="Advanced Settings", open=False):
											aa_band_min = gr.Slider(minimum=-1, maximum=128, step=1, label='Min Band', value=-1)
											aa_band_max = gr.Slider(minimum=-1, maximum=128, step=1, label='Max Band', value=-1)
											aa_hpss_type = gr.Radio(label='HPSS', choices=["none","H","P"], value="none", type="index")
											aa_onset_th = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Onset Threshold', value=0.07)
											aa_is_backtrack = gr.Checkbox(label='Get Backtracked Onsets', value=True)

									with gr.Row():
										aa_generate_btn = gr.Button('Run', variant='primary')
									
									with gr.Accordion(label="Result", open=True):
										aa_bpm = gr.Slider(minimum=1, maximum=300, step=0.5, label='BPM', value=1, interactive=False)
										aa_length_msec = gr.Number(value=0, label="End Time (millisecond)", precision=0, interactive=False)

										with gr.Row():
											aa_list_txt = gr.Textbox(label='Wave List(Generated from Onset Time)', lines=30, interactive=True)
										with gr.Row():
											merge_start = gr.Number(value=0, label="Merge Start Time (millisecond)", precision=0, interactive=True)
											merge_end = gr.Number(value=0, label="Merge End Time (millisecond)", precision=0, interactive=True)
											merge_wave_btn = gr.Button('Merge Wave List', variant='primary')
										with gr.Row():
											test_generate_btn2 = gr.Button('Generate Test Audio', variant='primary')


						with gr.Column(variant='panel'):
							test_result_audio = gr.Audio(interactive=False, mirror_webcam=False, type="filepath")

							wave_plt = gr.Plot(elem_id='lmsw_wave_plot')
							html_info = gr.HTML(visible=False)

			with gr.TabItem('Frame Extract'):
				with gr.Accordion(label="Input", open=True):
					fe_project_dir = gr.Textbox(label='Project directory', lines=1)
					fe_movie_path = gr.Textbox(label='Movie Path', lines=1)

					fe_video = gr.Video(interactive=True, mirror_webcam=False)
					def fn_upload_org_video(video):
						return video
					fe_video.upload(fn_upload_org_video, fe_video, fe_movie_path)
					gr.HTML(value="<p style='margin-bottom: 1.2em'>\
							If you have trouble entering the video path manually, you can also use drag and drop.For large videos, please enter the path manually. \
							</p>")
					
					fe_ffmpeg_path = gr.Textbox(label="ffmpeg binary.	Only set this if it fails otherwise.", lines=1, value="")
					
				with gr.Tabs():
					with gr.TabItem('Frame Extract For Controlnet or img2img'):	
						all_extract_fps = gr.Slider(minimum=1, maximum=120, step=1, label='Frames per second(This must exactly match the fps setting in img2img script setting)', value=24)
						with gr.Row():
							all_extract_btn = gr.Button('Extract', variant='primary')

					with gr.TabItem('Frame Extract For Initial image switching per wave'):
						with gr.Row():
							per_wave_extract_list_txt = gr.Textbox(label='Wave List', lines=30, interactive=True)
						with gr.Row():
							per_wave_extract_btn = gr.Button('Extract', variant='primary')


			def send_to_extract(list_txt):
				return list_txt
				
			send_to_extract_btn.click(fn=send_to_extract, inputs=wave_list_txt, outputs=per_wave_extract_list_txt)

			def merge_btn_func(org_list,add_list,start,end):
				start,end = min(start,end),max(start,end)
				a = str_to_wave_list(org_list)
				b = str_to_wave_list(add_list)
				c = merge_wave_list(a,b,start,end)
				return wave_list_to_str(c)

			merge_wave_btn.click(fn=merge_btn_func, inputs=[wave_list_txt, aa_list_txt, merge_start, merge_end], outputs=wave_list_txt)

			wave_gen_args = dict(
				fn=wrap_gradio_gpu_call(wave_generator_process),
				inputs=[
					wave_bpm,
					wave_beat_per_wave,
					wave_start_msec,
					wave_end_msec,
					wave_default_type,
					wave_default_strength,

				],
				outputs=[
					wave_list_txt,
					wave_plt,
					html_info
				],
				show_progress=False,
			)
			wave_generate_btn.click(**wave_gen_args)

			aa_gen_args = dict(
				fn=wrap_gradio_gpu_call(audio_analyzer_process),
				inputs=[
					input_audio_path,
					aa_offset,
					aa_band_min,
					aa_band_max,
					aa_hpss_type,
					aa_onset_th,
					wave_default_type,
					wave_default_strength,
					aa_is_backtrack,
				],
				outputs=[
					aa_bpm,
					aa_length_msec,
					aa_list_txt,
					wave_plt,
					html_info
				],
				show_progress=False,
			)
			aa_generate_btn.click(**aa_gen_args)

			test_gen_args = dict(
				fn=wrap_gradio_gpu_call(wave_list_test_process),
				inputs=[
					input_audio_path,
					wave_list_txt,
				],
				outputs=[
					test_result_audio,
					html_info
				],
				show_progress=False,
			)
			test_generate_btn.click(**test_gen_args)

			test_gen2_args = dict(
				fn=wrap_gradio_gpu_call(wave_list_test_process),
				inputs=[
					input_audio_path,
					aa_list_txt,
				],
				outputs=[
					test_result_audio,
					html_info
				],
				show_progress=False,
			)
			test_generate_btn2.click(**test_gen2_args)

			fe_all_gen_args = dict(
				fn=wrap_gradio_gpu_call(frame_extract_all),
				inputs=[
					fe_project_dir,
					fe_movie_path,
					fe_ffmpeg_path,
					all_extract_fps,
				],
				outputs=[
					html_info
				],
				show_progress=False,
			)
			all_extract_btn.click(**fe_all_gen_args)

			fe_per_wave_gen_args = dict(
				fn=wrap_gradio_gpu_call(frame_extract_per_wave),
				inputs=[
					fe_project_dir,
					fe_movie_path,
					fe_ffmpeg_path,
					per_wave_extract_list_txt,
				],
				outputs=[
					html_info
				],
				show_progress=False,
			)
			per_wave_extract_btn.click(**fe_per_wave_gen_args)

			prompt_test_args = dict(
				fn=wrap_gradio_gpu_call(prompt_test_process),
				inputs=[
					wave_list_txt,
					prompt_test_txt,
				],
				outputs=[
					wave_plt,
					html_info
				],
				show_progress=False,
			)
			prompt_test_btn.click(**prompt_test_args)
		   
	return (wave_generation_interface, "Loopback Music Sync Wave", "wave_generation_interface"),


script_callbacks.on_ui_tabs(on_ui_tabs)

