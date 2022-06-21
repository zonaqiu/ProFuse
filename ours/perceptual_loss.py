import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import vgg16_bn


class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights):
        super().__init__()
        self.feature_loss = loss
        self.weights = weights
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        # self.weights = torch.tensor(weights).to(device)
        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        # vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        inputs = F.normalize(inputs, 2, 1)
        targets = F.normalize(targets, 2, 1)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0

        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            # loss += self.feature_loss(lhs, rhs)
            mse_loss = torch.nn.MSELoss()
            loss += mse_loss(lhs,rhs)

        return loss


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


def perceptual_loss(x, y):
    F.mse_loss(x, y)


def PerceptualLoss(blocks, weights):
    return FeatureLoss(perceptual_loss, blocks, weights)
