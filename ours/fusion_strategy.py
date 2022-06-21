import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from guided_filter import guided_filter
EPSILON = 1e-10


# addition fusion strategy
def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2
def adp_addtion_fusion(tensor1,tensor2,w1,w2):
    return w1*tensor1 + w2*tensor2


# attention fusion strategy, average based on weight maps
def attention_fusion_weight_1(tensor1, tensor2):
    # avg, max, nuclear
    f_spatial = spatial_fusion_1(tensor1, tensor2,'mean')
    tensor_f = f_spatial
    return tensor_f

def attention_fusion_weight_2(tensor1,tensor2):
    f_spatial = spatial_fusion_2(tensor1,tensor2,'sum');
    tensor_f=f_spatial
    return tensor_f


def spatial_fusion_2(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()

    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)  #C_i
    spatial2 = spatial_attention(tensor2, spatial_type)

    w = torch.ones(1, 1, 3, 3)
    b = torch.zeros(1)
    w=w.cuda()
    b=b.cuda()
    C_1_hat = F.conv2d(spatial1, w, b, stride=1, padding=1)
    C_2_hat = F.conv2d(spatial2, w, b, stride=1, padding=1)


    C_1_hat = C_1_hat / 9.0
    C_2_hat = C_2_hat / 9.0
    # spatial_w1 = spatial1/(spatial1+spatial2+EPSILON)
    # spatial_w2 = spatial1/(spatial1+spatial2+EPSILON)

    spatial_w1 = C_1_hat/(C_1_hat+C_2_hat+EPSILON)
    spatial_w2 = C_2_hat / (C_1_hat + C_2_hat + EPSILON)
    # # get weight map, soft-max

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
    # print(spatial_w1.shape)
    # exit(9)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f

def spatial_fusion_1(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size() #[1,16,h,w]
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type) # [1,1,h,w]
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    # print(spatial1.shape)
    # exit(9)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f

# spatial attention
def spatial_attention(tensor, spatial_type='mean'):
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial #C_i

def channel_attention_fusion(f1,f2):
    fp1 = F.avg_pool2d(f1, f1.size(2))
    fp2 = F.avg_pool2d(f2, f2.size(2))

    # print(fp1.sgape);

    # fp1 = F.adaptive_avg_pool2d(f1,(1,1))
    # fp2 = F.adaptive_avg_pool2d(f2,(1,1))
    mask1 = fp1 / (fp1 + fp2)
    mask2 = 1 - mask1
    return f1 * mask1 + f2 * mask2

def spatial_fusion_3(tensor1, tensor2):

    tensor1_c = tensor1.cpu().numpy()
    tensor2_c = tensor2.cpu().numpy()
    tensor1_c = (tensor1_c * 255).astype(np.uint8)
    tensor2_c = (tensor2_c * 255).astype(np.uint8)

    histo1 = np.histogram(tensor1_c, bins=256)
    histo2 = np.histogram(tensor2_c, bins=256)
    shape = tensor1.shape

    S1 = np.zeros(shape)
    S2 = np.zeros(shape)

    for i in range(256):
        # print(histo1[0][i])
        # exit(0)
        t1 = np.full(shape, i, dtype=np.uint8)
        t2 = np.full(shape, histo1[0][i])

        t3 = np.full(shape, histo2[0][i])
        S1 += t2 * np.abs(tensor1_c - i * t1)
        S2 += t3 * np.abs(tensor2_c - i * t1)

    w1_hat = (S1 / (S1 + S2 + EPSILON)).astype(np.float32)
    w2_hat = (S2 / (S1 + S2 + EPSILON)).astype(np.float32)
    #
    # w1_hat = guided_filter(tensor1, w1_hat, 7, 1e-6)
    # w2_hat = guided_filter(tensor2, w2_hat, 7, 1e-6)

    w1 = w1_hat #/ (w1_hat + w2_hat + EPSILON)
    w2 = w2_hat #/ (w1_hat + w2_hat + EPSILON)

    w1 = torch.from_numpy(w1).cuda()
    w2 = torch.from_numpy(w2).cuda()




    tensor_f = w1 * tensor1 + w2 * tensor2

    return tensor_f