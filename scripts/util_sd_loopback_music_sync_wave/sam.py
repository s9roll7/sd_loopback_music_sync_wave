import base64
import requests
from PIL import Image
from io import BytesIO

sam_url = "http://127.0.0.1:7860/sam/sam-predict"





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

def get_mask_from_sam(img:Image, prompt, box_th=0.3, padding=30):
	payload = {
		"input_image": pil_to_base64(img),
		"dino_enabled": True,
		"dino_text_prompt": prompt,
		"dino_preview_checkbox": False,
		"dino_box_threshold:": box_th,
	}
	res = requests.post(sam_url, json=payload)
	
	print("res from sam : ",res)
	
	reply = res.json()
	
	print(reply["msg"])
	
	if res.status_code == 200:
		masks = []
		for img64 in reply["masks"]:
			image_data = base64.b64decode(img64)
			image = Image.open(BytesIO(image_data))
			masks.append(image.convert("L"))
			#debug_save_img(image, "sam")
		return masks
	else:
		return None


