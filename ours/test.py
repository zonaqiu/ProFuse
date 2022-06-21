# test phase
import torch
from torch.autograd import Variable

import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
from network import R2U_Net as unet
from fusion_strategy import addition_fusion,attention_fusion_weight_1,attention_fusion_weight_2,channel_attention_fusion,spatial_fusion_3
from channel_fusion import channel_f

def load_model(path, input_nc, output_nc):

	nest_model = unet(input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model

def _generate_fusion_image(model, strategy_type, img1, img2):
	a = 0.5
	# encoder
	x1_r,x2_r,x3_r,x4_r = model.encoder(img1)
	# vision_features(en_r, 'ir')
	x1_v,x2_v,x3_v,x4_v = model.encoder(img2)
	# # vision_features(en_v, 'vi')
	# x1_f = (channel_attention_fusion(x1_r,x1_v) + attention_fusion_weight_1(x1_r,x1_v))*0.5
	# x2_f = (channel_attention_fusion(x2_r,x2_v) + attention_fusion_weight_1(x2_r,x2_v))*0.5
	# x3_f = (channel_attention_fusion(x3_r,x3_v) + attention_fusion_weight_1(x3_r,x3_v))*0.5
	# x4_f = (channel_attention_fusion(x4_r,x4_v) + attention_fusion_weight_1(x4_r,x4_v))*0.5
	# x5_f = (channel_attention_fusion(x5_r,x5_v) + attention_fusion_weight_1(x5_r,x5_v))*0.5
	x1_f = attention_fusion_weight_1(x1_r, x1_v)
	x2_f = attention_fusion_weight_1(x2_r, x2_v)
	x3_f = attention_fusion_weight_1(x3_r, x3_v)
	x4_f = attention_fusion_weight_1(x4_r, x4_v)
	# x5_f = attention_fusion_weight_1(x5_r, x5_v)
	# decoder
	img_fusion = model.decoder(x1_f,x2_f,x3_f,x4_f)
	return img_fusion[0]


def run_demo(model, infrared_path, visible_path, output_path_root, name, fusion_type, network_type, strategy_type, ssim_weight_str, mode):

	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)

	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)
	dimension = ir_img.size()

	img_fusion = _generate_fusion_image(model, strategy_type, ir_img, vis_img)
	############################ multi outputs ##############################################
	file_name =  str(name) +  '.png'
	output_path = output_path_root + file_name
	# # save images
	# utils.save_image_test(img_fusion, output_path)
	# utils.tensor_save_rgbimage(img_fusion, output_path)
	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)

	print(output_path)

def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)

def main():
	# run demo
	# test_path = "images/test-RGB/"
	os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	# test_path = "/data/data_c/zona/U-net/crop-images/"
	# test_path = "/data/disk_c/zona/ResNetFusion-master/TNO/"
	test_path = "/data_d/zona/profuse/train/"
	network_type = 'densefuse'
	fusion_type = 'auto'  # auto, fusion_layer, fusion_all
	strategy_type_list = ['addition', 'attention_weight']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

	output_path = '/data_d/zona/profuse/output/ino_1/'
	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	# in_c = 3 for RGB images; in_c = 1 for gray images
	in_c = 1
	if in_c == 1:
		out_c = in_c
		mode = 'L'
		model_path = args.model_path_gray
	else:
		out_c = in_c
		mode = 'RGB'
		model_path = args.model_path_rgb

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[0])
		ssim_weight_str = args.ssim_path[0]
		model = load_model(model_path, in_c, out_c)
		houzhui = '.bmp'
		index=0
		# for infrared_name in os.listdir(test_path+'ir_crop/'):
		# 	# print('V'+infrared_path[1:-4])
		# 	# exit(0)
		# 	visible_name = 'V'+infrared_name[1:]
		#
		# 	infrared_path=test_path+'ir_crop/'+infrared_name
		# 	visible_path=test_path+'vi_crop/'+visible_name
		# 	run_demo(model, infrared_path, visible_path, output_path, infrared_name[1:-4], fusion_type, network_type, strategy_type,
		# 		 ssim_weight_str, mode)
		# 	# index+=1
		# 	# exit(0)
		for i in range(1026):
			index = i + 1
			infrared_path = test_path + 'ir_crop/' + str(index) + houzhui
			visible_path = test_path + 'vi_crop/' + str(index) + houzhui
			run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
	print('Done......')

if __name__ == '__main__':
	main()
