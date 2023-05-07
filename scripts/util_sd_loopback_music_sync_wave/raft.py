import os
import glob
from PIL import Image
import numpy as np
import cv2
import torch

from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

img_batch_size = 1


debug_c = 0
def debug_save_img(img,comment):
	global debug_c
	img.save( f"scripts/testpngs/{debug_c}_{comment}.png")

	debug_c += 1


def resize_img_array(img_array, w, h):
	if img_array.shape[0] + img_array.shape[1] < h + w:
		interpolation = interpolation=cv2.INTER_CUBIC
	else:
		interpolation = interpolation=cv2.INTER_AREA
	return cv2.resize(img_array, (w, h), interpolation=interpolation)


def pathlist_to_stack(img_path_list):
	img_list = [np.array(Image.open(f)) for f in img_path_list]
	vframes = torch.as_tensor(np.stack(img_list))
	vframes = vframes.permute(0, 3, 1, 2)
	return vframes


def create_optical_flow(v_path, o_path):
	from modules import devices

	os.makedirs(o_path, exist_ok=True)

	npys = glob.glob( os.path.join(o_path ,"[0-9]*.npy"), recursive=False)
	if npys:
		print("npy file found. skip create optical flow")
		return

	pngs = glob.glob( os.path.join(v_path ,"[0-9]*.png"), recursive=False)

	weights = Raft_Large_Weights.DEFAULT
	transforms = weights.transforms()

	print("create_optical_flow")
	
	def preprocess(img1_path, img2_path):
		h, w, _ = np.array(Image.open(img1_path[0])).shape

		img1_batch = pathlist_to_stack(img1_path)
		img2_batch = pathlist_to_stack(img2_path)

		img1_batch = F.resize(img1_batch, size=[h//8*8, w//8*8], antialias=False)
		img2_batch = F.resize(img2_batch, size=[h//8*8, w//8*8], antialias=False)

		return transforms(img1_batch, img2_batch)

	# If you can, run this example on a GPU, it will be a lot faster.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
	model = model.eval()
	
	for i in range(0, len(pngs)-1 ,img_batch_size):
		print("i = ",i)
		img_path_1 = pngs[i:i+img_batch_size]
		img_path_2 = pngs[i+1:i+1+img_batch_size]

		img1_batch, img2_batch = preprocess(img_path_1, img_path_2)

		with torch.no_grad():
			list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

		predicted_flows = list_of_flows[-1]
		
		'''
		predicted_flows = predicted_flows.to("cpu").detach()

		for j in range(len(predicted_flows)):
			fl = predicted_flows[j]

			im = flow_to_image(fl)

			out_path = os.path.join(o_path, f"{str(i+j + 1).zfill(5)}.png")

			F.to_pil_image(im).save( out_path )

			print("output : ", out_path)
		'''

		predicted_flows = predicted_flows.to("cpu").detach().numpy()

		for j in range(len(predicted_flows)):
			fl = predicted_flows[j]
			out_path = os.path.join(o_path, f"{str(i+j + 1).zfill(5)}.npy")
			#torch.save(fl, out_path)
			np.save(out_path, fl)
			print("output : ", out_path)

	devices.torch_gc()


def apply_flow1(processed_img, flow):
	flow = flow.transpose(1,2,0)

	h = flow.shape[0]
	w = flow.shape[1]
	flow = -flow
	flow[:,:,0] += np.arange(w)
	flow[:,:,1] += np.arange(h)[:,np.newaxis]

	processed_array = np.array(processed_img)

	org_h,org_w,_ = processed_array.shape

	processed_array = resize_img_array(processed_array, w, h)


	result = cv2.remap(processed_array, flow, None, cv2.INTER_LINEAR)


	result = Image.fromarray(resize_img_array(result, org_w, org_h))
	return result

# https://github.com/MCG-NKU/AMT/blob/main/utils/flow_utils.py
def _warp(img, flow):
	B, _, H, W = flow.shape
	xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
	yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
	grid = torch.cat([xx, yy], 1).to(img)
	flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
	grid_ = (grid + flow_).permute(0, 2, 3, 1)
	output = torch.nn.functional.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
	return output

def apply_flow2(processed_img, flow):
	_, H, W = flow.shape

	x = np.array(processed_img)
	ORG_H,ORG_W,_ = x.shape
	x = resize_img_array(x, W, H)

	torch_x = torch.as_tensor(x.astype("float32")).div(255).permute(2, 1, 0).unsqueeze(0).contiguous()
	torch_flow = torch.as_tensor(flow).permute(0, 2, 1).unsqueeze(0).contiguous()

	warped = _warp(torch_x, torch_flow)

	result = warped.squeeze(0).mul(255).permute(2, 1, 0).numpy().astype("uint8")

	result = resize_img_array(result, ORG_W, ORG_H)

	return Image.fromarray(result)


def apply_flow(processed_img, o_path):
	img = processed_img.convert('RGBA')
	img.putalpha(255)

	flow = np.load(o_path)

#	debug_save_img(processed_img, "pre")
	a = apply_flow1(img, flow)

	org_mask_array = np.array(a)[:, :, 3]

	if org_mask_array.min() == 255:
		print("skip inpainting")
		return a.convert("RGB")
	
	b = apply_flow2(processed_img, flow)
#	debug_save_img(b, "back")
	
	org_mask_array = 255 - org_mask_array
	mask_img = Image.fromarray(org_mask_array)

	img = a.convert("RGB")
#	debug_save_img(img, "fwd")

	img = Image.composite(b, img, mask_img)

#	debug_save_img(mask_img, "mask")
#	debug_save_img(img, "result")

	return img

