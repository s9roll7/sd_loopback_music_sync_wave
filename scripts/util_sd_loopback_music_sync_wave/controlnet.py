import importlib
import numpy as np
import json
import os
from PIL import Image

from scripts.util_sd_loopback_music_sync_wave.controlnet_web import get_detectmap

cn_stat={
	"initialized" : False,
	"external_code" : None,
	"controlnet_units" : None,
	"current_stat" : True,
	"cache_dir" : "",
	"controlnet_modules" : []
}

def get_cache(module_name, img_path):
	if not cn_stat["cache_dir"]:
		return None
	
	if module_name == "none":
		return None
	
	basename = os.path.basename(img_path)
	module_path = os.path.join(cn_stat["cache_dir"], module_name)
	cache_path = os.path.join(module_path, basename)

	if not os.path.isfile(cache_path):
		os.makedirs(module_path, exist_ok=True)
		det = get_detectmap( module_name, Image.open(img_path) )
		det.save(cache_path)

	return Image.open(cache_path)


def get_external_code():
	if cn_stat["external_code"]:
		return cn_stat["external_code"]
	try:
		if importlib.util.find_spec('extensions.sd-webui-controlnet.scripts.external_code'):
			cn_stat["external_code"] = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
	except Exception as e:
		print(e)
		print("import controlnet failed.")
	return cn_stat["external_code"]


def initialize(p, cache_dir):
	cn_stat["current_stat"] = True

	external_code = get_external_code()
	if not external_code:
		return
	
	cn_stat["controlnet_units"] = external_code.get_all_units_in_processing(p)
	cn_stat["cache_dir"] = cache_dir

	if cache_dir:
		os.makedirs(cache_dir, exist_ok=True)

	if cn_stat["controlnet_units"]:
		cn_stat["controlnet_modules"] = [i.module for i in cn_stat["controlnet_units"]]
		cn_stat["initialized"] = True
	
	print("controlnet found : ", cn_stat["initialized"])

def dump(path):
	if not cn_stat["initialized"]:
		return
	
	d = {}
	for i, c in enumerate(cn_stat["controlnet_units"]):
		d[i] = vars(c)

	def default_func(o):
		return str(o)
	
	with open(path, 'w') as f:
		json.dump(d, f, indent=4, default=default_func)


def enable_controlnet(p, img_path):
	if not cn_stat["initialized"]:
		return
	
	external_code = get_external_code()
	if not external_code:
		return
	
	for i,c in enumerate(cn_stat["controlnet_units"]):
		if c.enabled:
			if img_path is not None:
				cache = get_cache( cn_stat["controlnet_modules"][i], img_path)

				if cache:
					img = cache
					c.module = "none"
				else:
					img = Image.open(img_path)
					c.module = cn_stat["controlnet_modules"][i]

				c.image = np.array(img)
				c.resize_mode = 0
			else:
				c.image = None
				c.module = cn_stat["controlnet_modules"][i]
	
	print("enable_controlnet")

	external_code.update_cn_script_in_processing(p, cn_stat["controlnet_units"])

	cn_stat["current_stat"] = True

def disable_controlnet(p):
	if not cn_stat["initialized"]:
		return
	
	if cn_stat["current_stat"] == False:
		return
	
	external_code = get_external_code()
	if not external_code:
		return
	
	print("disable_controlnet")

	external_code.update_cn_script_in_processing(p, [])

	cn_stat["current_stat"] = False
