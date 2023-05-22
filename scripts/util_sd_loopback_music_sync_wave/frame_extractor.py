import os
import time
import glob
import platform
import shutil

from scripts.loopback_music_sync_wave import str_to_wave_list,run_cmd
import scripts.util_sd_loopback_music_sync_wave.raft

def remove_pngs_in_dir(path):
    if not os.path.isdir(path):
        return
    pngs = glob.glob( os.path.join(path, "*.png") )
    for png in pngs:
        os.remove(png)

def frame_extract_all(fe_project_dir:str, fe_movie_path:str, fe_ffmpeg_path:str, all_extract_fps:int):

	if (not fe_project_dir) or (not os.path.isdir(fe_project_dir)):
		print("Directory not found : ", fe_project_dir)
		return " "
	
	if (not fe_movie_path) or (not os.path.isfile(fe_movie_path)):
		print("File not found : ", fe_movie_path)
		return " "
	
	extract_dir = os.path.join(os.path.join(fe_project_dir, "video_frame"), f"{all_extract_fps}")

	os.makedirs(extract_dir, exist_ok=True)

	remove_pngs_in_dir(extract_dir)

	args = [
		"-i", fe_movie_path,
		"-start_number", 0,
		"-vf",
		f"fps={all_extract_fps}",
		os.path.join(extract_dir, "%05d.png")
	]

	if(fe_ffmpeg_path == ""):
		fe_ffmpeg_path = "ffmpeg"
		if(platform.system == "Windows"):
			fe_ffmpeg_path += ".exe"

	run_cmd([fe_ffmpeg_path] + args)

	return " "

def frame_extract_one(fe_project_dir:str, fe_movie_path:str, fe_ffmpeg_path:str, fe_fps:int):

	if (not fe_project_dir) or (not os.path.isdir(fe_project_dir)):
		print("Directory not found : ", fe_project_dir)
		return " "
	
	if (not fe_movie_path) or (not os.path.isfile(fe_movie_path)):
		print("File not found : ", fe_movie_path)
		return " "
	
	extract_dir = fe_project_dir

	args = [
		"-i", fe_movie_path,
		"-frames:v",1,
		os.path.join(extract_dir, "00000.png")
	]

	if(fe_ffmpeg_path == ""):
		fe_ffmpeg_path = "ffmpeg"
		if(platform.system == "Windows"):
			fe_ffmpeg_path += ".exe"

	run_cmd([fe_ffmpeg_path] + args)

	if fe_fps > 0:
		#extract all frame
		return frame_extract_all(fe_project_dir, fe_movie_path, fe_ffmpeg_path, fe_fps)

	return " "

def frame_extract_per_wave(fe_project_dir:str, fe_movie_path:str, fe_ffmpeg_path:str, wave_list_str:str):

	if (not fe_project_dir) or (not os.path.isdir(fe_project_dir)):
		print("Directory not found : ", fe_project_dir)
		return " "
	
	if (not fe_movie_path) or (not os.path.isfile(fe_movie_path)):
		print("File not found : ", fe_movie_path)
		return " "
	
	wave_list = str_to_wave_list(wave_list_str)

	time_list=[]
	for w in wave_list:
		if w["type"] == "end":
			continue
		time_list.append(w["start_msec"]/1000)
	
	extract_dir = os.path.join(fe_project_dir , "video_frame_per_wave")
	os.makedirs(extract_dir, exist_ok=True)

	remove_pngs_in_dir(extract_dir)

	if(fe_ffmpeg_path == ""):
		fe_ffmpeg_path = "ffmpeg"
		if(platform.system == "Windows"):
			fe_ffmpeg_path += ".exe"

	for i, t in enumerate( time_list ):
		args = [
			"-ss",t,
			"-i", fe_movie_path,
			"-frames:v",1,
			os.path.join(extract_dir, f"{str(i).zfill(5)}.png")
		]
		run_cmd([fe_ffmpeg_path] + args, True)


	return " "

def frame_extract_scene_change(fe_project_dir:str, fe_movie_path:str, fe_ffmpeg_path:str, sc_fe_fps:int, sc_use_optical_flow_cache:bool, sc_flow_occ_detect_th:float, sc_sd_threshold:float):

	if (not fe_project_dir) or (not os.path.isdir(fe_project_dir)):
		print("Directory not found : ", fe_project_dir)
		return " "
	
	if (not fe_movie_path) or (not os.path.isfile(fe_movie_path)):
		print("File not found : ", fe_movie_path)
		return " "
	
	print("create video frame")
	frame_extract_all(fe_project_dir, fe_movie_path, fe_ffmpeg_path, sc_fe_fps)

	def get_video_frame_path(project_dir, i, fps):
		path = os.path.join(os.path.join(project_dir, "video_frame"), f"{fps}")
		path = os.path.join(path, f"{str(i).zfill(5)}.png")
		return path
	
	print("create optical flow")
	sample_frame_path = get_video_frame_path(fe_project_dir, 0, sc_fe_fps)
	if sample_frame_path and os.path.isfile(sample_frame_path):
		v_path = os.path.join(os.path.join(fe_project_dir, "video_frame"), f"{sc_fe_fps}")
		o_path = os.path.join(os.path.join(fe_project_dir, "optical_flow"), f"{sc_fe_fps}")
		m_path = os.path.join(os.path.join(fe_project_dir, "occ_mask"), f"{sc_fe_fps}")
		scripts.util_sd_loopback_music_sync_wave.raft.create_optical_flow(v_path, o_path, m_path, sc_use_optical_flow_cache, None, sc_flow_occ_detect_th)
	else:
		print("video frame not found")
		return " "
	
	print("scene detection list")
	m_path = os.path.join(os.path.join(fe_project_dir, "occ_mask"), f"{sc_fe_fps}")
	mask_path_list = sorted(glob.glob( os.path.join(m_path ,"[0-9]*.png"), recursive=False))
	scene_detection_list = scripts.util_sd_loopback_music_sync_wave.raft.get_scene_detection_list(sc_sd_threshold, 1, mask_path_list)

	dst_dir_path = os.path.join(os.path.join(fe_project_dir, "scene_change_frame"), f"{sc_fe_fps}")
	os.makedirs(dst_dir_path, exist_ok=True)
	remove_pngs_in_dir(dst_dir_path)

	src_dir_path = os.path.join(os.path.join(fe_project_dir, "video_frame"), f"{sc_fe_fps}")

	scene_detection_list[0] = True

	for i, sd in enumerate(scene_detection_list):
		if sd:
			src_path = os.path.join(src_dir_path, f"{str(i).zfill(5)}.png")
			dst_path = os.path.join(dst_dir_path, f"{str(i).zfill(5)}.png")
			shutil.copyfile(src_path, dst_path)
	

	print("finished")
	return " "





