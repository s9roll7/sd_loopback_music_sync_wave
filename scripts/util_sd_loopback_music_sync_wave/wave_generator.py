
from PIL import Image

import scripts.util_sd_loopback_music_sync_wave.audio_analyzer

def wave_generator_process(bpm: float, beat_per_wave:int, start_msec:int, end_msec:int, default_type:str, default_strength:float):
	#start_time,type,(strength)

	if start_msec >= end_msec:
		print("Error start_msec >= end_msec")
		return "",None," "

	wave_list = []

	msec_per_beat = 60 * 1000 / bpm
	msec_per_wave = msec_per_beat * beat_per_wave

	print("msec_per_beat : ", msec_per_beat)
	print("msec_per_wave : ", msec_per_wave)

	cur = 0

	if cur < start_msec:
		wave_list.append( ( cur, "zero", 1.0 ) )

	cur = start_msec

	while True:
		if cur + msec_per_wave >= end_msec:
			if cur + msec_per_wave > end_msec:
				wave_list.append( ( int(cur), "zero", 1.0 ) )
			else:
				wave_list.append( ( int(cur), default_type, default_strength ) )

			wave_list.append( ( end_msec, "end", 1.0 ) )
			break

		wave_list.append( ( int(cur), default_type, default_strength ) )

		cur += msec_per_wave
	
	wave_str_list=[]

	for w in wave_list:
		if w[1] in ("zero", "end") or w[2] == 1.0:
			wave_str_list.append( f"{w[0]},{w[1]}" )
		else:
			wave_str_list.append( f"{w[0]},{w[1]},{w[2]}" )
	
	print(wave_str_list)

	fig = scripts.util_sd_loopback_music_sync_wave.audio_analyzer.create_figure([x[0]/1000 for x in wave_list])

	return "\n".join(wave_str_list), fig, " "



