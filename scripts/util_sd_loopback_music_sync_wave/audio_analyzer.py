import librosa
import numpy as np
import os
from PIL import Image


class AudioAnalyzer:
	def __init__(self, path, band_min, band_max, hpss_type, onset_th):
		
		aggregate_is_median = True
		hop_length = 512
		hpss_margin = 3.0
		max_size = 1
		onset_detect_normalize = True

		self.path = path
		self.band_min = band_min
		self.band_max = band_max
		self.hpss_type = hpss_type	 # none(0) , H(1) , P(2)
		self.hop_length = hop_length
		self.hpss_margin = hpss_margin
		self.max_size = max_size
		self.aggregate = np.median if aggregate_is_median else np.mean
		self.onset_detect_normalize = onset_detect_normalize
		self.onset_th = onset_th
		self.is_backtrack_reqested = False

		self.onset_env = None
		self.wave = None
		self.sr = None
		self.onset_backtrack = None

		self.result_onset = None
		self.result_beat = None
		self.result_beat_plp = None
		self.times = None
		self.result_bpm = -1
		self.length = -1

		print("hop_length = {}".format(self.hop_length))
		print("hpss_margin = {}".format(self.hpss_margin))
		print("max_size = {}".format(self.max_size))
		print("aggregate = {}".format(self.aggregate))
		print("onset_detect_normalize = {}".format(self.onset_detect_normalize))
		
		self.is_loaded = False
		
		self._analyze()
		

	def GetResult(self, is_backtrack):
		if not self.is_loaded:
			return None
		
		if is_backtrack:
			self.is_backtrack_reqested = True
			return self.times[self.onset_backtrack]
		else:
			self.is_backtrack_reqested = False
			return self.times[self.result_beat]

	def GetBPM(self):
		return self.result_bpm
	
	def GetLength(self):
		return self.length
	
	def IsSuccess(self):
		return self.is_loaded
	
	def CreateFig(self, wave_list):
		import matplotlib.pyplot as plt

		fig = plt.Figure(dpi=100, figsize=( (self.length/1000)*4 ,3*2))
		ax1 = fig.add_subplot(2, 1, 1)
		ax1.plot(self.times, self.onset_env/self.onset_env.max(), label='onset envelope')

		if self.is_backtrack_reqested:
			ax1.vlines(self.times[self.onset_backtrack], 0, 1, color='r', linestyle='--', label='backtrack')
		else:
			ax1.vlines(self.times[self.result_onset], 0, 1, color='r', linestyle='--', label='onsets')
		ax1.legend(frameon=True, framealpha=0.75)

		ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
		librosa.display.waveshow(self.wave, sr=self.sr, ax=ax2)

		if wave_list:
			ax2.vlines( wave_list, -1,1,color='r', label='wave list')
		
		ax1.set_xlim(0, self.length/1000 + 1)
		x_labels = np.arange(0,self.length/1000 + 1,0.5)
		ax1.set_xticks(x_labels,x_labels)		
		ax1.minorticks_on()
		ax1.grid(which="major", color="gray", linestyle=":", axis="x")
		ax1.grid(which="minor", color="gray", linestyle=":", axis="x")

		ax2.legend(frameon=True, framealpha=0.75)
		ax2.minorticks_on()
		ax2.grid(which="major", color="gray", linestyle=":", axis="x")
		ax2.grid(which="minor", color="gray", linestyle=":", axis="x")

		plt.tight_layout()
		#fig.savefig(img_path)
		return fig

	def CreatePromptFig(self, plot_data):
		import matplotlib.pyplot as plt
		import matplotlib.gridspec as gridspec

		plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("tab20").colors)

		fig = plt.Figure(dpi=100, figsize=( (self.length/1000)*4 ,3*2))
		gs = gridspec.GridSpec(3, 1, figure=fig)
		ax1 = fig.add_subplot(gs[0, :])
#		ax1 = fig.add_subplot(2, 1, 1)
		ax1.plot(self.times, self.onset_env/self.onset_env.max(), label='onset envelope')

		if self.is_backtrack_reqested:
			ax1.vlines(self.times[self.onset_backtrack], 0, 1, color='r', linestyle='--', label='backtrack')
		else:
			ax1.vlines(self.times[self.result_onset], 0, 1, color='r', linestyle='--', label='onsets')
		ax1.legend(frameon=True, framealpha=0.75)

		ax2 = fig.add_subplot(gs[1:, :],sharex=ax1)
#		ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
		librosa.display.waveshow(self.wave, sr=self.sr, ax=ax2)

		ax2.vlines( plot_data["wave"], -1,1,color='r', label='wave list')

		# plot data
		for d in plot_data["data"]:
			ax2.plot(plot_data["time"], plot_data["data"][d], label=d)

		ax1.set_xlim(0, self.length/1000 + 1)
		x_labels = np.arange(0,self.length/1000 + 1,0.5)
		ax1.set_xticks(x_labels,x_labels)
		ax1.minorticks_on()
		ax1.grid(which="major", color="gray", linestyle=":", axis="x")
		ax1.grid(which="minor", color="gray", linestyle=":", axis="x")

		ax2.legend(frameon=True, framealpha=0.75)
		ax2.minorticks_on()
		ax2.grid(which="major", color="gray", linestyle=":", axis="x")
		ax2.grid(which="minor", color="gray", linestyle=":", axis="x")

		plt.tight_layout()
		return fig


	def _get_onset_env_multi(self, wave, sr):
		channels = [self.band_min, self.band_max]
		onset_envelope_multi = librosa.onset.onset_strength_multi(y=wave, sr=sr, channels=channels, hop_length=self.hop_length, aggregate=self.aggregate, max_size=self.max_size) 
		return onset_envelope_multi[0]
	
	def _get_onset_env(self, wave, sr):
		return librosa.onset.onset_strength(y=wave, sr=sr, hop_length=self.hop_length, aggregate=self.aggregate, max_size=self.max_size)
	
	def _get_wave(self):
		wave, sr = librosa.load(self.path)

		self.length = librosa.get_duration(y=wave, sr=sr)
		self.length = int(self.length * 1000)

		if self.hpss_type == 1:
			y_harm, y_perc = librosa.effects.hpss(wave, margin=self.hpss_margin)
			return y_harm,sr
		elif self.hpss_type == 2:
			y_harm, y_perc = librosa.effects.hpss(wave, margin=self.hpss_margin)
			return y_perc,sr
		else:
			return wave,sr



	def _analyze(self):
		if (not self.path) or (not os.path.isfile(self.path)):
			print("File not found : ", self.path)
			return

		wave, sr = self.wave, self.sr = self._get_wave()
		

		if self.band_min == -1 or self.band_max == -1:
			self.onset_env = self._get_onset_env(wave, sr)
		else:
			self.onset_env = self._get_onset_env_multi(wave, sr)
		
		self.times = librosa.times_like(self.onset_env, sr=sr, hop_length=self.hop_length)
		
		self.result_onset = librosa.onset.onset_detect(onset_envelope=self.onset_env, sr=sr, hop_length=self.hop_length, normalize = self.onset_detect_normalize, delta=self.onset_th)

		self.onset_backtrack = librosa.onset.onset_backtrack(self.result_onset, self.onset_env)

		tempo, self.result_beat = librosa.beat.beat_track(onset_envelope=self.onset_env, sr=sr)

		pulse = librosa.beat.plp(onset_envelope=self.onset_env, sr=sr, hop_length = self.hop_length)

		self.result_beat_plp = np.flatnonzero(librosa.util.localmax(pulse))

		self.result_bpm = tempo

		self.is_loaded = True



def create_onset_wave_list(onset_timing, length, default_type, default_strength, offset_time):
	#start_time,type,(strength)

	onset_timing = [int(i*1000) for i in onset_timing]
	print("onset_timing : ",onset_timing)
	onset_timing = [ i + offset_time for i in onset_timing if 0 < (i + offset_time) < length ]
	print("onset_timing + offset : ",onset_timing)

	wave_list = []

	if 0 < onset_timing[0]:
		wave_list.append( ( 0, "zero", 1.0 ) )

	for cur in onset_timing:
		wave_list.append( ( cur, default_type, default_strength ) )

	wave_list.append( ( length, "end", 1.0 ) )
	
	
	wave_str_list=[]

	for w in wave_list:
		if w[1] in ("zero", "end") or w[2] == 1.0:
			wave_str_list.append( f"{w[0]},{w[1]}" )
		else:
			wave_str_list.append( f"{w[0]},{w[1]},{w[2]}" )
	
	print(wave_str_list)

	return "\n".join(wave_str_list)

def create_figure(wave_list):
	if not _aa_cache:
		return None
	return _aa_cache.CreateFig(wave_list)

def create_prompt_figure(plot_data):
	if not _aa_cache:
		return None
	return _aa_cache.CreatePromptFig(plot_data)

_aa_cache = None

def audio_analyzer_process(audio_file:str, offset:int, band_min:int, band_max:int, hpss_type:int, onset_th:float, default_type:str, default_strength:float, is_backtrack:bool):
	global _aa_cache

	print("audio_file : ",audio_file)
	
	aa = None

	if _aa_cache:
		if _aa_cache.path == audio_file and\
			_aa_cache.band_min == band_min and\
			_aa_cache.band_max == band_max and\
			_aa_cache.hpss_type == hpss_type and\
			_aa_cache.onset_th == onset_th:
			print("use cache")
			aa = _aa_cache

	if not aa:
		aa = AudioAnalyzer(audio_file, band_min, band_max, hpss_type, onset_th)
	
	if not aa.IsSuccess():
		return -1,-1,"",None, " "

	bpm = aa.GetBPM()
	length_msec = aa.GetLength()
	list_txt = create_onset_wave_list(aa.GetResult(is_backtrack), length_msec, default_type, default_strength, offset)

	_aa_cache = aa

	fig = create_figure(None)

	return bpm, length_msec, list_txt, fig, " "









