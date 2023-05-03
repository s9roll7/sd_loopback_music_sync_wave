import re
import numpy as np

# @@bpm123@12345[]
bpm_regex = r'@@bpm([0-9\.]+)@([0-9]+)\[(.*?)\]'


class BpmEvent:
	def __init__(self, fps, total_time):
		self.fps = fps
		self.total_time = int(total_time)
		self.events = {}

	def add_event(self, match_obj, start_time):
		bpm = 0
		prompt = ""
		#start_time = int(start_time)
		last_msec = self.total_time

		if match_obj.group(1) is not None:
			bpm = float(match_obj.group(1))
		if match_obj.group(2) is not None:
			last_msec = min( start_time + int(match_obj.group(2)) , last_msec)
		if match_obj.group(3) is not None:
			prompt = match_obj.group(3)
		
		if bpm < 1 or not prompt:
			return ""
		
		msec_per_beat = (60 * 1000 / bpm)

		frames = [int(t * self.fps / 1000) for t in np.arange(start_time, last_msec, msec_per_beat)]

		print("bpm event : ",frames)

		for i in frames:
			if i in self.events:
				self.events[i] = self.events[i] + "," + prompt
			else:
				self.events[i] = prompt
		
		return ""
	
	def get_current_prompt(self, cur_time):

		cur_frame = int(cur_time * self.fps / 1000)
		prompt = ""
		if cur_frame in self.events:
			prompt = self.events[cur_frame]
			print("bpm prompt : ", prompt)

		return prompt
	
	def parse_prompt(self, prompt, cur_time):
		prompt = re.sub(bpm_regex, lambda x: self.add_event(x, cur_time), prompt)
		return prompt

