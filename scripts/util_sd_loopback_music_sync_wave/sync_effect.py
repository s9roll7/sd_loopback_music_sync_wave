import re
from scripts.util_sd_loopback_music_sync_wave import perlin
from scripts.util_sd_loopback_music_sync_wave.regex import create_regex, create_regex_text

def quadratic_generator(init,top,count):
	count = int(count)
	b = 4*(top - init)/count
	a = -1*b / count
	c = init

	i = 0
	while count > i:
		i += 1
		# y = ax^2 + bx + c
		v = a * i*i + b*i + c

		yield v

def warped_quadratic_generator(init,top,count,ratio):
    first_half = max(1, int(count * ratio))
    second_half = max(1, int(count - first_half))

    lst = list(quadratic_generator(init, top, first_half*2))
    for i in range(first_half):
        yield lst[i]

    lst = list(quadratic_generator(top, init, second_half*2))
    for i in range(second_half):
        yield lst[i]


def pendulum_generator(init,first,second,count):
	first_half = max(1, int(count * 0.5))
	second_half = max(1, int(count - first_half))

	lst = list(quadratic_generator(init, first, first_half))
	for i in range(first_half):
		yield lst[i]

	lst = list(quadratic_generator(init, second, second_half))
	for i in range(second_half):
		yield lst[i]

def shake_generator(init, top, count, attenuation_rate=0.7):
	count = int(count)
	cur = top
	for i in range(count-1):
		yield cur + init
		cur *= -1 * attenuation_rate
	
	yield init

def pos_2_vel(pos_list, init):
	vel_list = []

	cur = init
	for pos in pos_list:
		vel = pos - cur
		vel_list.append(vel)
		cur = pos

	return vel_list

def norm_pnf(pnf):
	return abs( (pnf + 0.866) / (0.866*2) )
def norm_pnf2(pnf):
	return pnf / 0.866


#########################

def shake_x_generator(frames, fps, _, amp):
	if frames < 2:
		yield -1
	
	amp *= fps
	
	gen = shake_generator(0,amp,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("shake_x : ",result_vel)

	for r in result_vel:
		yield f"#vel_x({r:.3f})"
	yield -1


def shake_y_generator(frames, fps, _, amp):
	if frames < 2:
		yield -1
	
	amp *= fps

	gen = shake_generator(0,amp,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("shake_y : ",result_vel)

	for r in result_vel:
		yield f"#vel_y({r:.3f})"
	yield -1

def shake_rot_generator(frames, fps, _, amp):
	if frames < 2:
		yield -1
	
	amp *= fps
	
	gen = shake_generator(0,amp,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("shake_rot : ",result_vel)

	for r in result_vel:
		yield f"#rot({r:.3f})"
	yield -1

def shake_rot_x_generator(frames, fps, _, amp):
	if frames < 2:
		yield -1
	
	amp *= fps
	
	gen = shake_generator(0,amp,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("shake_rot_x : ",result_vel)

	for r in result_vel:
		yield f"#rot_x({r:.3f})"
	yield -1

def shake_rot_y_generator(frames, fps, _, amp):
	if frames < 2:
		yield -1
	
	amp *= fps
	
	gen = shake_generator(0,amp,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("shake_rot_Y : ",result_vel)

	for r in result_vel:
		yield f"#rot_y({r:.3f})"
	yield -1

def shake_zoom_generator(frames, fps, _, amp):
	if frames < 2:
		yield -1
	
	amp = (amp-1)*fps + 1

	gen = shake_generator(0,amp-1.0,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	result_vel = [x+1.0 for x in result_vel]

	print("shake_zoom : ",result_vel)

	for r in result_vel:
		yield f"#zoom({r:.3f})"
	yield -1

def vibration_generator(frames, fps, _, max):
	if frames < 2:
		yield -1
	
	max = (max-1)*fps + 1

	gen = quadratic_generator(0, max - 1.0, frames)

	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)
	result_vel = [x+1.0 for x in result_vel]

	print("vibration : ",result_vel)

	for r in result_vel:
		yield f"#zoom({r:.3f})"
	yield -1

def random_xy_generator(frames, fps, _, amp_x, amp_y, resolution_msec):
	amp_x *= fps
	amp_y *= fps

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(2, octaves=4, tile=(frames//resolution, frames//resolution), unbias = True)

	result_pos = [norm_pnf2(pnf(0, x/resolution)) for x in range(frames)]
	result_vel = pos_2_vel(result_pos, 0)
	result_pos2 = [norm_pnf2(pnf(x/resolution,0)) for x in range(frames)]
	result_vel2 = pos_2_vel(result_pos2, 0)

	print("random_xy : ",result_vel)
	print("random_xy : ",result_vel2)

	for x, y in zip( result_vel, result_vel2):
		yield f"#vel_x({x * amp_x:.3f}),#vel_y({y * amp_y:.3f})"
	yield -1

def random_z_generator(frames, fps, _, amp_z, resolution_msec):
	amp_z = (amp_z-1)*fps + 1

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(1, octaves=4, tile=(frames//resolution,), unbias = True)

	result_pos = [norm_pnf2(pnf(x/resolution)) for x in range(frames)]
	result_vel = pos_2_vel(result_pos, 0)

	print("random_z : ",result_vel)

	for z in result_vel:
		yield f"#zoom({z * (amp_z-1) + 1:.3f})"
	yield -1

def random_rot_generator(frames, fps, _, amp_r, resolution_msec):
	amp_r *= fps

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(1, octaves=4, tile=(frames//resolution,), unbias = True)

	result_pos = [norm_pnf2(pnf(x/resolution)) for x in range(frames)]
	result_vel = pos_2_vel(result_pos, 0)

	print("random_rot : ",result_vel)

	for r in result_vel:
		yield f"#rot({amp_r * r:.3f})"
	yield -1

def random_rot_x_generator(frames, fps, _, amp_r, resolution_msec):
	amp_r *= fps

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(1, octaves=4, tile=(frames//resolution,), unbias = True)

	result_pos = [norm_pnf2(pnf(x/resolution)) for x in range(frames)]
	result_vel = pos_2_vel(result_pos, 0)

	print("random_rot_x : ",result_vel)

	for r in result_vel:
		yield f"#rot_x({amp_r * r:.3f})"
	yield -1

def random_rot_y_generator(frames, fps, _, amp_r, resolution_msec):
	amp_r *= fps

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(1, octaves=4, tile=(frames//resolution,), unbias = True)

	result_pos = [norm_pnf2(pnf(x/resolution)) for x in range(frames)]
	result_vel = pos_2_vel(result_pos, 0)

	print("random_rot_y : ",result_vel)

	for r in result_vel:
		yield f"#rot_y({amp_r * r:.3f})"
	yield -1


def random_c_generator(frames, fps, _, amp_x, amp_y, cx, cy, resolution_msec):
	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(2, octaves=4, tile=(frames//resolution, frames//resolution), unbias = True)

	result_pos = [norm_pnf2(pnf(0, x/resolution)) for x in range(frames)]
#	result_vel = pos_2_vel(result_pos, 0)
	result_pos2 = [norm_pnf2(pnf(x/resolution,0)) for x in range(frames)]
#	result_vel2 = pos_2_vel(result_pos2, 0)

#	print("random_c : ",result_vel)
#	print("random_c : ",result_vel2)
	print("random_c : ",result_pos)
	print("random_c : ",result_pos2)

	for x, y in zip( result_pos, result_pos2):
		yield f"#center({cx + x * amp_x:.3f},{cy + y * amp_y:.3f})"
	yield -1


def pendulum_xy_generator(frames, fps, _, x1, x2, y1, y2):
	x1 *= fps
	x2 *= fps
	y1 *= fps
	y2 *= fps

	gen = pendulum_generator( 0 ,x1,x2,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)
	gen = pendulum_generator( 0, y1,y2,frames)
	result_pos2 = list(gen)
	result_vel2 = pos_2_vel(result_pos2, 0)

	print("pendulum_xy : ",result_vel)
	print("pendulum_xy : ",result_vel2)

	yield ""	# first frame

	for x, y in zip( result_vel, result_vel2):
		yield f"#vel_x({x:.3f}),#vel_y({y:.3f})"
	yield -1

def pendulum_rot_generator(frames, fps, _, angle1, angle2):
	angle1 *= fps
	angle2 *= fps

	gen = pendulum_generator(0,angle1,angle2,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("pendulum_rot : ",result_vel)

	yield ""	# first frame
	
	for r in result_vel:
		yield f"#rot({r:.3f})"
	yield -1

def pendulum_rot_x_generator(frames, fps, _, angle1, angle2):
	angle1 *= fps
	angle2 *= fps

	gen = pendulum_generator(0,angle1,angle2,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("pendulum_rot_x : ",result_vel)

	yield ""	# first frame
	
	for r in result_vel:
		yield f"#rot_x({r:.3f})"
	yield -1

def pendulum_rot_y_generator(frames, fps, _, angle1, angle2):
	angle1 *= fps
	angle2 *= fps

	gen = pendulum_generator(0,angle1,angle2,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	print("pendulum_rot_y : ",result_vel)

	yield ""	# first frame
	
	for r in result_vel:
		yield f"#rot_y({r:.3f})"
	yield -1

def pendulum_zoom_generator(frames, fps, _, z1, z2):
	z1 = (z1-1)*fps + 1
	z2 = (z2-1)*fps + 1

	gen = pendulum_generator(0,z1-1.0,z2-1.0,frames)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	result_vel = [x+1.0 for x in result_vel]

	print("pendulum_zoom : ",result_vel)

	yield ""	# first frame

	for r in result_vel:
		yield f"#zoom({r:.3f})"
	yield -1

def pendulum_center_generator(frames, fps, _, cx1, cx2, cy1, cy2):

	gen = pendulum_generator((cx1+cx2)/2,cx1,cx2,frames)
	result_pos = list(gen)
	gen = pendulum_generator((cy1+cy2)/2,cy1,cy2,frames)
	result_pos2 = list(gen)

	print("pendulum_center : ",result_pos)
	print("pendulum_center : ",result_pos2)

	yield ""	# first frame

	for cx, cy in zip( result_pos, result_pos2):
		yield f"#center({cx:.3f},{cy:.3f})"
	yield -1


def beat_blur_generator(frames, fps, _, amp_str):

	gen = warped_quadratic_generator(0, amp_str, frames, 0.1)
	result_pos = list(gen)

	print("beat_blur : ",result_pos)

	yield ""	# first frame

	for s in result_pos:
		yield f"#blur({s:.3f})"
	yield -1

def random_blur_generator(frames, fps, _, amp_str, resolution_msec):

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(1, octaves=4, tile=(frames//resolution,), unbias = True)

	result_pos = [norm_pnf2(pnf(x/resolution)) for x in range(frames)]

	print("random_blur : ",result_pos)

	for r in result_pos:
		yield f"#blur({amp_str * r:.3f})"
	yield -1

def pendulum_hue_generator(frames, fps, _,type,angle1,angle2):

	gen = pendulum_generator((angle1+angle2)/2, angle1, angle2, frames)
	result_pos = list(gen)

	print("pendulum_hue : ",result_pos)

	yield ""	# first frame

	for s in result_pos:
		yield f"#hue({type:.3f},{s:.3f})"
	yield -1

def random_hue_generator(frames, fps, _, type, start_angle, amp_angle, resolution_msec):

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(1, octaves=4, tile=(frames//resolution,), unbias = True)

	result_pos = [norm_pnf2(pnf(x/resolution)) for x in range(frames)]

	print("random_hue : ",result_pos)

	for s in result_pos:
		yield f"#hue({type:.3f},{start_angle + s * amp_angle:.3f})"
	yield -1

def beat_slide_x_generator(frames, fps, _, type, amp_slide_val, border_pos, amp_border):

	amp_slide_val *= fps

	gen = warped_quadratic_generator(0, amp_slide_val, frames, 0.1)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	gen = warped_quadratic_generator(0, amp_border, frames, 0.1)
	result_pos2 = list(gen)

	print("beat_slide_x : ",result_vel)
	print("beat_slide_x : ",result_pos2)

	for x, y in zip( result_vel, result_pos2):
		yield f"#slide_x({type:.3f},{x:.3f},{border_pos + y:.3f})"
	yield -1

def beat_slide_y_generator(frames, fps, _, type, amp_slide_val, border_pos, amp_border):

	amp_slide_val *= fps

	gen = warped_quadratic_generator(0, amp_slide_val, frames, 0.1)
	result_pos = list(gen)
	result_vel = pos_2_vel(result_pos, 0)

	gen = warped_quadratic_generator(0, amp_border, frames, 0.1)
	result_pos2 = list(gen)

	print("beat_slide_y : ",result_vel)
	print("beat_slide_y : ",result_pos2)

	for x, y in zip( result_vel, result_pos2):
		yield f"#slide_y({type:.3f},{x:.3f},{border_pos + y:.3f})"
	yield -1


def random_slide_x_generator(frames, fps, _, type, amp_slide_val, border_pos, amp_border, resolution_msec):
	amp_slide_val *= fps

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(2, octaves=4, tile=(frames//resolution, frames//resolution), unbias = True)

	result_pos = [norm_pnf2(pnf(0, x/resolution)) for x in range(frames)]
	result_vel = pos_2_vel(result_pos, 0)
	result_pos2 = [norm_pnf2(pnf(x/resolution,0)) for x in range(frames)]

	print("random_slide_x : ",result_vel)
	print("random_slide_x : ",result_pos2)

	for x, y in zip( result_vel, result_pos2):
		yield f"#slide_x({type:.3f},{x * amp_slide_val:.3f},{border_pos + y*amp_border:.3f})"
	yield -1

def random_slide_y_generator(frames, fps, _, type, amp_slide_val, border_pos, amp_border, resolution_msec):
	amp_slide_val *= fps

	resolution = int(resolution_msec * fps / 1000)
	pnf = perlin.PerlinNoiseFactory(2, octaves=4, tile=(frames//resolution, frames//resolution), unbias = True)

	result_pos = [norm_pnf2(pnf(0, x/resolution)) for x in range(frames)]
	result_vel = pos_2_vel(result_pos, 0)
	result_pos2 = [norm_pnf2(pnf(x/resolution,0)) for x in range(frames)]

	print("random_slide_y : ",result_vel)
	print("random_slide_y : ",result_pos2)

	for x, y in zip( result_vel, result_pos2):
		yield f"#slide_y({type:.3f},{x * amp_slide_val:.3f},{border_pos + y*amp_border:.3f})"
	yield -1

def inpaint_generator(frames, fps, _, mask_prompt, inpaint_prompt):

	print("inpaint mask_prompt : ",mask_prompt)
	print("inpaint inpaint_prompt : ",inpaint_prompt)

	if inpaint_prompt:
		yield f"#__inpaint(\"{mask_prompt}\",\"{inpaint_prompt}\")"
	else:
		yield f"#__inpaint(\"{mask_prompt}\")"

	yield -1


effect_map={
	"shake_x":shake_x_generator,
	"shake_y":shake_y_generator,
	"shake_rot":shake_rot_generator,
	"shake_rot_x":shake_rot_x_generator,
	"shake_rot_y":shake_rot_y_generator,
	"shake_zoom":shake_zoom_generator,
	"vibration":vibration_generator,
	"random_xy":random_xy_generator,
	"random_z":random_z_generator,
	"random_rot":random_rot_generator,
	"random_rot_x":random_rot_x_generator,
	"random_rot_y":random_rot_y_generator,
	"random_c":random_c_generator,
	"pendulum_xy":pendulum_xy_generator,
	"pendulum_rot":pendulum_rot_generator,
	"pendulum_rot_x":pendulum_rot_x_generator,
	"pendulum_rot_y":pendulum_rot_y_generator,
	"pendulum_zoom":pendulum_zoom_generator,
	"pendulum_center":pendulum_center_generator,

	"beat_blur":beat_blur_generator,
	"random_blur":random_blur_generator,

	"pendulum_hue":pendulum_hue_generator,
	"random_hue":random_hue_generator,

	"beat_slide_x":beat_slide_x_generator,
	"beat_slide_y":beat_slide_y_generator,
	"random_slide_x":random_slide_x_generator,
	"random_slide_y":random_slide_y_generator,

	"inpaint":inpaint_generator,
}



# $func
shake_x_regex = create_regex(r'\$','shake_x', 2)
shake_y_regex = create_regex(r'\$','shake_y', 2)
shake_rot_regex = create_regex(r'\$','shake_rot', 2)
shake_rot_x_regex = create_regex(r'\$','shake_rot_x', 2)
shake_rot_y_regex = create_regex(r'\$','shake_rot_y', 2)
shake_zoom_regex = create_regex(r'\$','shake_zoom', 2)

vibration_regex = create_regex(r'\$','vibration', 2)

random_xy_regex = create_regex(r'\$','random_xy', 3,1)
random_z_regex = create_regex(r'\$','random_zoom', 2,1)
random_rot_regex = create_regex(r'\$','random_rot', 2,1)
random_rot_x_regex = create_regex(r'\$','random_rot_x', 2,1)
random_rot_y_regex = create_regex(r'\$','random_rot_y', 2,1)
random_c_regex = create_regex(r'\$','random_center', 3,3)

pendulum_xy_regex = create_regex(r'\$','pendulum_xy', 5)
pendulum_rot_regex = create_regex(r'\$','pendulum_rot', 3)
pendulum_rot_x_regex = create_regex(r'\$','pendulum_rot_x', 3)
pendulum_rot_y_regex = create_regex(r'\$','pendulum_rot_y', 3)
pendulum_zoom_regex = create_regex(r'\$','pendulum_zoom', 3)
pendulum_center_regex = create_regex(r'\$','pendulum_center', 5)

beat_blur_regex = create_regex(r'\$','beat_blur', 2)
random_blur_regex = create_regex(r'\$','random_blur', 2,1)

pendulum_hue_regex = create_regex(r'\$','pendulum_hue', 4)
random_hue_regex = create_regex(r'\$','random_hue', 4,1)

beat_slide_x_regex = create_regex(r'\$','beat_slide_x', 3,2)
beat_slide_y_regex = create_regex(r'\$','beat_slide_y', 3,2)

random_slide_x_regex = create_regex(r'\$','random_slide_x', 3,3)
random_slide_y_regex = create_regex(r'\$','random_slide_y', 3,3)


inpaint_regex = create_regex_text(r'\$',"inpaint",1,1)


def effect_create(match_obj, eff, eff_name, default_vals):
	dur = 0
	# first token is duration
	if match_obj.group(1) is not None:
		dur = int(match_obj.group(1))
	
	for i in range(1, len(match_obj.groups())):
		if match_obj.group(i+1) is not None:
			default_vals[i-1] = float(match_obj.group(i+1))

	eff.add_effect( dur, eff_name, *default_vals)

	return ""


def effect_create2(match_obj, eff, eff_name, default_vals):
	for i in range(0, len(match_obj.groups())):
		if match_obj.group(i+1) is not None:
			default_vals[i] = str(match_obj.group(i+1))

	eff.add_effect2( eff_name, *default_vals)

	return ""


# $shake_x(dur, amp)
# $shake_y(dur, amp)
# $shake_rot(dur, amp)
# $shake_zoom(dur, amp)
# $vibration(dur, amp)
# $random_xy(dur, x_amp, y_amp)
# $random_zoom(dur, z_amp)
# $random_rot(dur, r_amp)
# $random_center(dur, amp_x, amp_y, cx, cy)




class SyncEffect:
	def __init__(self, fps):
		self.fps = fps
		self.effect_list = []

	def add_effect(self, duration, *effect_param):
		frames = self.fps * duration / 1000
		self.effect_list.append(effect_map[effect_param[0]](int(frames), self.fps, *effect_param))
	
	def add_effect2(self, *effect_param):
		self.effect_list.append(effect_map[effect_param[0]](int(1), self.fps, *effect_param))

	def get_current_prompt(self):
		remove_index = []
		prompt = ""

		for i, ef in enumerate( self.effect_list ):
			result = next(ef)
			if result == -1:
				remove_index.append(i)
			else:
				if result:
					if prompt:
						prompt += "," + result
					else:
						prompt += result
		
		for i in reversed(remove_index):
			self.effect_list.pop(i)
		
		print("effect prompt : ", prompt)
		return prompt
	



	def parse_prompt(self, prompt):

		# $shake_x(dur, amp)
		prompt = re.sub(shake_x_regex, lambda x: effect_create(x, self, "shake_x", [-1]), prompt)
		# $shake_y(dur, amp)
		prompt = re.sub(shake_y_regex, lambda x: effect_create(x, self, "shake_y", [-1]), prompt)
		# $shake_rot(dur, amp)
		prompt = re.sub(shake_rot_regex, lambda x: effect_create(x, self, "shake_rot", [-1]), prompt)
		# $shake_rot_x(dur, amp)
		prompt = re.sub(shake_rot_x_regex, lambda x: effect_create(x, self, "shake_rot_x", [-1]), prompt)
		# $shake_rot_y(dur, amp)
		prompt = re.sub(shake_rot_y_regex, lambda x: effect_create(x, self, "shake_rot_y", [-1]), prompt)
		# $shake_zoom(dur, amp)
		prompt = re.sub(shake_zoom_regex, lambda x: effect_create(x, self, "shake_zoom", [-1]), prompt)
		# $vibration(dur, amp)
		prompt = re.sub(vibration_regex, lambda x: effect_create(x, self, "vibration", [-1]), prompt)

		# $random_xy(dur, x_amp, y_amp, resolution_msec=1000)
		prompt = re.sub(random_xy_regex, lambda x: effect_create(x, self, "random_xy", [-1,-1,1000]), prompt)
		# $random_zoom(dur, z_amp, resolution_msec=1000)
		prompt = re.sub(random_z_regex, lambda x: effect_create(x, self, "random_z", [-1,1000]), prompt)
		# $random_rot(dur, r_amp, resolution_msec=1000)
		prompt = re.sub(random_rot_regex, lambda x: effect_create(x, self, "random_rot", [-1,1000]), prompt)
		# $random_rot_x(dur, r_amp, resolution_msec=1000)
		prompt = re.sub(random_rot_x_regex, lambda x: effect_create(x, self, "random_rot_x", [-1,1000]), prompt)
		# $random_rot_y(dur, r_amp, resolution_msec=1000)
		prompt = re.sub(random_rot_y_regex, lambda x: effect_create(x, self, "random_rot_y", [-1,1000]), prompt)
		# $random_center(dur, amp_x, amp_y, cx=0.5, cy=0.5, resolution_msec=1000 )
		prompt = re.sub(random_c_regex, lambda x: effect_create(x, self, "random_c", [-1,-1,0.5,0.5,1000]), prompt)

		# $pendulum_xy(dur, x1, x2, y1, y2 )
		prompt = re.sub(pendulum_xy_regex, lambda x: effect_create(x, self, "pendulum_xy", [-1,-1,-1,-1]), prompt)
		# $pendulum_rot(dur, angle1, angle2 )
		prompt = re.sub(pendulum_rot_regex, lambda x: effect_create(x, self, "pendulum_rot", [-1,-1]), prompt)
		# $pendulum_rot_x(dur, angle1, angle2 )
		prompt = re.sub(pendulum_rot_x_regex, lambda x: effect_create(x, self, "pendulum_rot_x", [-1,-1]), prompt)
		# $pendulum_rot_y(dur, angle1, angle2 )
		prompt = re.sub(pendulum_rot_y_regex, lambda x: effect_create(x, self, "pendulum_rot_y", [-1,-1]), prompt)
		# $pendulum_zoom(dur, z1, z2 )
		prompt = re.sub(pendulum_zoom_regex, lambda x: effect_create(x, self, "pendulum_zoom", [-1,-1]), prompt)
		# $pendulum_center(dur, cx1, cx2, cy1, cy2 )
		prompt = re.sub(pendulum_center_regex, lambda x: effect_create(x, self, "pendulum_center", [-1,-1,-1,-1]), prompt)

		# $beat_blur(dur, amp)
		prompt = re.sub(beat_blur_regex, lambda x: effect_create(x, self, "beat_blur", [-1]), prompt)
		# $random_blur(dur, amp, resolution_msec=1000)
		prompt = re.sub(random_blur_regex, lambda x: effect_create(x, self, "random_blur", [-1,1000]), prompt)
		# $pendulum_hue(dur, type, angle1, angle2)
		prompt = re.sub(pendulum_hue_regex, lambda x: effect_create(x, self, "pendulum_hue", [-1,-1,-1]), prompt)
		# $random_hue(dur, type, start_angle, amp_angle, resolution_msec=1000)
		prompt = re.sub(random_hue_regex, lambda x: effect_create(x, self, "random_hue", [-1,-1,-1,1000]), prompt)

		# $beat_slide_x(dur, type, amp_slide_val, border_pos=0.5, amp_border=0)
		prompt = re.sub(beat_slide_x_regex, lambda x: effect_create(x, self, "beat_slide_x", [-1,-1,0.5,0]), prompt)
		# $beat_slide_y(dur, type, amp_slide_val, border_pos=0.5, amp_border=0)
		prompt = re.sub(beat_slide_y_regex, lambda x: effect_create(x, self, "beat_slide_y", [-1,-1,0.5,0]), prompt)
		# $random_slide_x(dur, type, amp_slide_val, border_pos=0.5, amp_border=0, resolution_msec=1000)
		prompt = re.sub(random_slide_x_regex, lambda x: effect_create(x, self, "random_slide_x", [-1,-1,0.5,0,1000]), prompt)
		# $random_slide_y(dur, type, amp_slide_val, border_pos=0.5, amp_border=0, resolution_msec=1000)
		prompt = re.sub(random_slide_y_regex, lambda x: effect_create(x, self, "random_slide_y", [-1,-1,0.5,0,1000]), prompt)

		# $inpaint(mask_prompt, inpaint_prompt)
		prompt = re.sub(inpaint_regex, lambda x: effect_create2(x, self, "inpaint", [-1,""]), prompt)


		return prompt







