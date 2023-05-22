import os
import glob
from PIL import Image
import numpy as np
import cv2
import torch
import time

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

def remove_files_in_dir(path, pattern):
    if not os.path.isdir(path):
        return
    pngs = glob.glob( os.path.join(path, pattern) )
    for png in pngs:
        os.remove(png)


#############
# from https://github.com/princeton-vl/RAFT/issues/57
#############
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device="cpu"):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def create_occ_mask(v1,v2,size,device,thresh = 2.0):
    H,W = size
    
    coords0 = coords_grid(1, H, W, device)
    coords1 = coords0 + v1
    coords2 = coords1 + bilinear_sampler(v2, coords1.permute(0,2,3,1))
    
    err = (coords0 - coords2).norm(dim=1)
    occ = (err[0] > thresh).float().cpu().numpy()
    
    return occ * 255
#############
    


def create_optical_flow(v_path, o_path, m_path, use_cache, size_hw=None, occ_th=2.0):
	from modules import devices

	os.makedirs(o_path, exist_ok=True)

	if m_path:
		os.makedirs(m_path, exist_ok=True)

	if use_cache:
		npys = glob.glob( os.path.join(o_path ,"[0-9]*.npy"), recursive=False)
		if npys:
			print("npy file found. skip create optical flow")
			return
	else:
		remove_files_in_dir(o_path, "*.npy")
		if m_path:
			remove_files_in_dir(m_path, "*.png")

	pngs = glob.glob( os.path.join(v_path ,"[0-9]*.png"), recursive=False)

	devices.torch_gc()

	weights = Raft_Large_Weights.DEFAULT
	transforms = weights.transforms()

	print("create_optical_flow")
	
	if size_hw:
		h, w= size_hw
	else:
		h, w, _ = np.array(Image.open(pngs[0])).shape
	H = h//8*8
	W = w//8*8

	def preprocess(img1_path, img2_path):
		img1_batch = pathlist_to_stack(img1_path)
		img2_batch = pathlist_to_stack(img2_path)

		img1_batch = F.resize(img1_batch, size=[H, W], antialias=False)
		img2_batch = F.resize(img2_batch, size=[H, W], antialias=False)

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

		if m_path:
			with torch.no_grad():
				rev_list_of_flows = model(img2_batch.to(device), img1_batch.to(device))

			rev_predicted_flows = rev_list_of_flows[-1]

			for j in range(len(predicted_flows)):
				fl = predicted_flows[j]
				rev_fl = rev_predicted_flows[j]

				fl = fl.permute(0, 1, 2).unsqueeze(0).contiguous()
				rev_fl = rev_fl.permute(0, 1, 2).unsqueeze(0).contiguous()

				occ = create_occ_mask(fl,rev_fl,(H,W), device, occ_th)
				out_path = os.path.join(m_path, f"{str(i+j + 1).zfill(5)}.png")
				Image.fromarray(occ).convert("L").save(out_path)

		predicted_flows = predicted_flows.to("cpu").detach().numpy()

		for j in range(len(predicted_flows)):
			fl = predicted_flows[j]
			out_path = os.path.join(o_path, f"{str(i+j + 1).zfill(5)}.npy")
			np.save(out_path, fl)
			print("output : ", out_path)

	devices.torch_gc()


def warp_img(processed_array, flow):
	flow = flow.transpose(1,2,0)

	h = flow.shape[0]
	w = flow.shape[1]
	flow = -flow
	flow[:,:,0] += np.arange(w)
	flow[:,:,1] += np.arange(h)[:,np.newaxis]

	org_h,org_w,_ = processed_array.shape

	processed_array = resize_img_array(processed_array, w, h)

	result = cv2.remap(processed_array, flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

	return resize_img_array(result, org_w, org_h)

def apply_flow_single(processed_img, o_path, is_reverse = False, add_flow_path = None, rate = 0.0, use_inpaint = True):
	if use_inpaint:
		img = processed_img.convert('RGBA')
		img.putalpha(255)
	else:
		img = processed_img

	if not o_path:
		print("o_path is empty")
		return processed_img
	
	flow = np.load(o_path)

	if is_reverse:
		flow = -flow
	
	if add_flow_path:
		add_flow = np.load(add_flow_path)
		if is_reverse:
			add_flow = -add_flow
		add_flow = add_flow * rate

#		flow = flow * 0.5 + add_flow * 0.5
		th = np.mean(np.abs(flow)) / 2
		mask = (np.abs(flow) < th)
		np.putmask( flow, mask, add_flow)

#	debug_save_img(img, "pre")
	a_array = warp_img(np.array(img), flow)

	if use_inpaint:
		org_mask_array = a_array[:, :, 3]
		org_mask_array = 255 - org_mask_array

		a_array = cv2.inpaint(a_array[:, :, 0:3],org_mask_array,3,cv2.INPAINT_TELEA)

	return Image.fromarray(a_array)


def apply_flow(base_img, flow_path_list, mask_path_list):

	img = base_img
	W, H = img.size
	mask_array = None

	for f,m in zip(flow_path_list, mask_path_list):
		if not os.path.isfile(f):
			return img, cv2.resize( mask_array, (W,H), interpolation = cv2.INTER_CUBIC) if mask_array is not None else None
		img = apply_flow_single(img, f, False, None, 0, False)
		
		if mask_array is None:
			mask_array = np.array(Image.open(m))
		else:
			mask_array = mask_array + np.array(Image.open(m))

	return img, cv2.resize( mask_array, (W,H), interpolation = cv2.INTER_CUBIC) if mask_array is not None else None

def get_scene_detection_list(detection_th, interpolation_multi, mask_path_list):
	result = [False]
	all_pixels = -1
	value_list = [0]

	for i in range(0, len(mask_path_list), interpolation_multi):
		v = 0
		for m in mask_path_list[i:i+interpolation_multi]:
			mask_array = np.array(Image.open(m))
			bad_pixels = np.count_nonzero(mask_array > 0)
			if all_pixels == -1:
				all_pixels = (mask_array.shape[0] * mask_array.shape[1])
			bad_rate = bad_pixels / all_pixels
			if bad_rate > v:
				v = bad_rate
		result.append( v > detection_th )
		value_list.append(v)
	

	for i, r in enumerate(result):
		if r:
			print(f"{i} : {r} ({value_list[i]})")

	return result



def interpolate_frame(head_img, tail_img, cur_flow, add_flow):
	list_a = []
	list_b = []

	img = head_img
	i = 0
	for f in cur_flow[:-1]:
		img = apply_flow_single(img, f, False, add_flow, (i+1)/ len(cur_flow))
		list_a.append(img)

	img = tail_img
	i = 0
	for f in cur_flow[:0:-1]:
		img = apply_flow_single(img, f, True, add_flow, (i+1)/ len(cur_flow))
		list_b.append(img)
	
	result = [head_img]
	i = 0

	for h,t in zip(list_a, list_b[::-1]):
		i_frame = Image.blend(h,t, (i+1)/ len(cur_flow) )
		result.append( i_frame )
		i+=1

	return result

def interpolate_frame2(head_img, tail_img, num_of_frames):
	result = []

	for i in range(num_of_frames):
		i_frame = Image.blend(head_img,tail_img, (i)/ num_of_frames )
		result.append( i_frame )
		i+=1

	return result

def interpolate(org_frame_path, out_frame_path, flow_interpolation_multi, flow_path, scene_changed_list ):

	print("interpolate start")

	calc_time_start = time.perf_counter()

	org_frames = sorted(glob.glob( os.path.join(org_frame_path ,"[0-9]*.png"), recursive=False))
	flows = sorted(glob.glob( os.path.join(flow_path ,"[0-9]*.npy"), recursive=False))
	_, FLOW_H, FLOW_W = np.load(flows[0]).shape	# 2, H, W

	flows = iter(flows)

	tmp_flow_path = os.path.join(org_frame_path ,"tmp_flow")
	create_optical_flow(org_frame_path, tmp_flow_path, None, False, (FLOW_H, FLOW_W) )

	tmp_flows = sorted(glob.glob( os.path.join(tmp_flow_path ,"[0-9]*.npy"), recursive=False))

	tmp_flows = iter(tmp_flows)

	os.makedirs(out_frame_path, exist_ok=True)

	i = 0
	tail_img=None

	for org_i, (head, tail) in enumerate(zip(org_frames, org_frames[1:])):
		cur_flow = [next(flows, None) for x in range(flow_interpolation_multi)]

		cur_flow = [ x for x in cur_flow if i is not None]

		tmp_flow = next(tmp_flows, None)

		head_img = Image.open( head )
		tail_img = Image.open( tail )
		
		if scene_changed_list[org_i+1]:
#			result_imgs = [ (head_img if f<(flow_interpolation_multi/2) else tail_img) for f in range(flow_interpolation_multi)]
			result_imgs = interpolate_frame2(head_img, tail_img, flow_interpolation_multi)
		else:
			result_imgs = interpolate_frame(head_img, tail_img, cur_flow, tmp_flow)

		for f in result_imgs:
			output_img_path = os.path.join(out_frame_path, f"{str(i).zfill(5)}.png" )
			f.save( output_img_path )

			print("output : ",i)

			i += 1
	
	output_img_path = os.path.join(out_frame_path, f"{str(i).zfill(5)}.png" )
	tail_img.save( output_img_path )

	print("output : ",i)

	calc_time_end = time.perf_counter()
	print("interpolate elapsed_time (sec) : ", calc_time_end - calc_time_start)	



