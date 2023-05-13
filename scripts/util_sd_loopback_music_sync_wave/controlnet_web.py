import base64
import requests
from PIL import Image
from io import BytesIO

cn_detect_url = "http://127.0.0.1:7860/controlnet/detect"





debug_c = 0
def debug_save_img(img,comment):
	global debug_c
	img.save( f"scripts/testpngs/{debug_c}_{comment}.png")

	debug_c += 1



def image_to_base64(img_path: str) -> str:
	with open(img_path, "rb") as img_file:
		img_base64 = base64.b64encode(img_file.read()).decode()
	return img_base64

def pil_to_base64(img:Image, format="png") -> str:
	buffer = BytesIO()
	img.save(buffer, format)
	img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
	
	return img_str

def get_detectmap(preprocess_module_name, img:Image):
    
	payload = {
		"controlnet_module": preprocess_module_name,
		"controlnet_input_images": [pil_to_base64(img)],
#		"controlnet_processor_res": res,
#		"controlnet_threshold_a": th_a,
#		"controlnet_threshold_b:": th_b,
	}
	res = requests.post(cn_detect_url, json=payload)
	
#	print("res from cn : ",res)
	
	reply = res.json()
	
	if res.status_code == 200:
		print(reply["info"])
		img64 = reply["images"][0]
		image_data = base64.b64decode(img64)
		image = Image.open(BytesIO(image_data))
		return image
	else:
		return None


