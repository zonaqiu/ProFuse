# Training DenseFuse network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import DenseFuse_net
from args_fusion import args
import pytorch_msssim
# from unet import UNet as unet
from network import R2U_Net as unet
from tensorboardX import SummaryWriter
from torchvision import utils as vutils
from perceptual_loss import PerceptualLoss
from fusion_strategy import addition_fusion,attention_fusion_weight_1,attention_fusion_weight_2


def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = "6"
	# original_imgs_path = utils.list_images(args.dataset)
	original_imgs_path_ir = utils.list_images('/data/disk_c/zona/ResNetFusion-master/TNO/C_Test_ir/')
	# original_imgs_path_vi = utils.list_images('/data/disk_c/zona/ResNetFusion-master/TNO/C_Test_vi/')

	train_num = 41
	original_imgs_path_ir = original_imgs_path_ir[:train_num]
	# original_imgs_path_vi = original_imgs_path_vi[:train_num]
	random.shuffle(original_imgs_path_ir)
	# random.shuffle(original_imgs_path_vi)
	# for i in range():

	train(6, original_imgs_path_ir)

def train(i, original_imgs_path_ir):
	writer = SummaryWriter('runs/re_train/att_4-2')

	batch_size = args.batch_size

	# load network model
	in_c = 1 #
	img_model = 'L'
	input_nc = in_c
	output_nc = in_c
	unet_model=unet(input_nc,output_nc)
	unet_model.load_state_dict(torch.load(args.model_path_gray))

	# print("unet_model")
	optimizer = Adam(unet_model.parameters(),args.lr_re)
	mse_loss = torch.nn.MSELoss()
	perceptual_loss = PerceptualLoss([0,1,2],[1,1,1])
	# ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		unet_model.cuda()
		perceptual_loss.cuda()
		mse_loss.cuda()
	#trange 同python中的range,区别在于trange在循环执行的时候会输出打印进度条
	tbar = trange(args.epochs)
	print('Start training.....')

	# 创建保存路径
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_per = []
	Loss_all = []
	all_per_loss = 0.
	all_pixel_loss = 0.
	count = 0
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path_ir, batch_size)
		# image_set_vi, batches = utils.load_dataset(original_imgs_path_vi, batch_size)

		unet_model.train()

		for batch in range(batches):
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			# image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]
			#读取图片[N,C,H,W]
			img_ir ,img_vi = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
			# img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode=img_model)

			count += 1
			optimizer.zero_grad()
			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)

			if args.cuda:
				img_ir = img_ir.cuda()
				img_vi = img_vi.cuda()

			x1_ir,x2_ir,x3_ir,x4_ir,x5_ir = unet_model.encoder(img_ir)
			x1_vi,x2_vi,x3_vi,x4_vi,x5_vi = unet_model.encoder(img_vi)

			x1 = attention_fusion_weight_1(x1_ir,x1_vi)
			x2 = attention_fusion_weight_1(x2_ir,x2_vi)
			x3 = attention_fusion_weight_1(x3_ir,x3_vi)
			x4 = attention_fusion_weight_1(x4_ir,x4_vi)
			x5 = attention_fusion_weight_1(x5_ir,x5_vi)

			outputs = unet_model.decoder(x1,x2,x3,x4,x5)
			# resolution loss
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)

			# 对一个batch计算
			perceptual_loss_value = 0.
			pixel_loss_value = 0.
			for output in outputs:
				pixel_loss_temp = mse_loss(output.repeat(1,3,1,1), x_ir.repeat(1,3,1,1))
				perceptual_loss_temp = perceptual_loss(output.repeat(1,3,1,1), x_vi.repeat(1,3,1,1))
				perceptual_loss_value += perceptual_loss_temp
				pixel_loss_value += pixel_loss_temp
			#对这一个batch的结果求平均值
			perceptual_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss 这一个batch的最后损失，然后backward
			total_loss = pixel_loss_value + args.ssim_weight[i] * perceptual_loss_value
			total_loss.backward()
			optimizer.step()

			#总体损失，不只是一个batch
			all_per_loss += perceptual_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			#log_interval = 5 ，5个batch之后输出损失函数的信息
			if (count /batch_size + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t perc loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e , count, batches,
								  all_pixel_loss / args.log_interval,
								  all_per_loss / args.log_interval,
								  (args.ssim_weight[i] * all_per_loss + all_pixel_loss) / args.log_interval
				)
				writer.add_scalar('pixel_loss',all_pixel_loss/args.log_interval,count)
				writer.add_scalar('perceptual_loss',all_per_loss/args.log_interval,count)
				writer.add_scalar('total_loss', (args.ssim_weight[i] * all_per_loss + all_pixel_loss) / args.log_interval,count)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval) #列表
				Loss_per.append(all_per_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_per_loss + all_pixel_loss) / args.log_interval)

				all_per_loss = 0.
				all_pixel_loss = 0.
			##每1000个batch，保存模型，
			if (count/batch_size + 1) % (20 * args.log_interval) == 0:
				img_dir = '/data/disk_c/zona/U-net/test7/output/re-train/att/'+args.ssim_path[i] + '/'
				if os.path.exists(img_dir) is False:
					os.mkdir(img_dir)

				img_name = "Epoch_" + str(e) + "_iters_" + str(count) +  "_" + args.ssim_path[i] + ".png"
				img_path = os.path.join(img_dir,img_name)
				vutils.save_image(outputs[0][0], img_path, normalize=True)


				# save model
				unet_model.eval()
				unet_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)


				torch.save(unet_model.state_dict(),save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_perceptual = np.array(Loss_per)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_per_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_perceptual})
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				unet_model.train()
				unet_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_per)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	# densefuse_model.eval()
	# densefuse_model.cpu()
	unet_model.eval()
	unet_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	# torch.save(densefuse_model.state_dict(), save_model_path)
	torch.save(unet_model.state_dict(),save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
