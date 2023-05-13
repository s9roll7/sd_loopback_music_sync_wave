import os
import time
import glob
import platform

from scripts.loopback_music_sync_wave import str_to_wave_list,run_cmd

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
	
	extract_dir = os.path.join(fe_project_dir , "video_frame")
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

def frame_extract_one(fe_project_dir:str, fe_movie_path:str, fe_ffmpeg_path:str):

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


