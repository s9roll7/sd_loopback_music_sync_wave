import importlib
import numpy as np

_initialized = False
_external_code = None
_controlnet_units = None
_current_stat = None

def get_external_code():
	global _external_code
	if _external_code:
		return _external_code
	try:
		if importlib.util.find_spec('extensions.sd-webui-controlnet.scripts.external_code'):
			_external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
	except Exception as e:
		print(e)
		print("import controlnet failed.")
	return _external_code

def initialize(p):
	global _initialized, _controlnet_units

	external_code = get_external_code()
	if not external_code:
		return
	
	_controlnet_units = external_code.get_all_units_in_processing(p)

	if _controlnet_units:
		_initialized = True
	
	print("controlnet found : ", _initialized)


def enable_controlnet(p, img):
	global _initialized, _controlnet_units, _current_stat

	if not _initialized:
		return
	
	external_code = get_external_code()
	if not external_code:
		return
	
	for c in _controlnet_units:
		if img is not None:
			c.image = np.array(img)
		else:
			c.image = None
	
	print("enable_controlnet")

	external_code.update_cn_script_in_processing(p, _controlnet_units)

	_current_stat = True

def disable_controlnet(p):
	global _initialized, _controlnet_units, _current_stat

	if not _initialized:
		return
	
	if _current_stat is not None:
		if _current_stat == False:
			return
	
	external_code = get_external_code()
	if not external_code:
		return
	
	print("disable_controlnet")

	external_code.update_cn_script_in_processing(p, [])

	_current_stat = False
