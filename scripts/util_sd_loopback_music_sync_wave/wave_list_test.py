import os
import time
from pydub import AudioSegment

from scripts.loopback_music_sync_wave import str_to_wave_list


def get_test_wav_dir():
	t= os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
	t= os.path.join(t, "..")
	t = os.path.join(t, "wav")
	return os.path.normpath(t)

def wave_list_test_process(audio_file:str, wave_list_str:str):
	#start_time,type,(strength)

	if (not audio_file) or (not os.path.isfile(audio_file)):
		print("File not found : ", audio_file)
		return None, " "
	
	wave_list = str_to_wave_list(wave_list_str)

	audio_name = os.path.splitext(os.path.basename(audio_file))[0]

	base_audio = AudioSegment.from_file(audio_file)
	test_audio = AudioSegment.from_file(os.path.join( get_test_wav_dir(), "metronome.wav" ))

	for w in wave_list:
		if w["start_msec"] == 0 or w["type"] == "end":
			continue
		base_audio = base_audio.overlay(test_audio, position=w["start_msec"])
	
	audio_tmp_dir = os.path.join(get_test_wav_dir(), "tmp")
	os.makedirs(audio_tmp_dir, exist_ok=True)

	audio_tmp_file_path = os.path.join(audio_tmp_dir, audio_name + "_" + time.strftime("%Y%m%d-%H%M%S")+".mp3")

	base_audio.export(audio_tmp_file_path, format="mp3")


	return audio_tmp_file_path, " "



