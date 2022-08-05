"""
Implements manual equivariance to known transforms in convolutional layers
E.g. if we have a know 90 degree clockwise rotation, each convolutional filter is accompanied with a 90 degree
rotation of the filter.
"""
import torch
import numpy as np
import torch.nn.functional as F
from skimage import transform


# Todo: torch docs says about hooks:
#  "This adds global state to the nn.module module and it is only intended for debugging/profiling purposes."
#   We are not using hooks for debugging purposes - may need to think/understand the global state stuff, the gradient
#   flows etc. We can and should visualise the weights learned with equivariant hooks to check they do what we want.

class RotationHook:
    def __init__(self, module, rotation_angle=90):
        assert rotation_angle in [90, 180, 270]
        self.rotation_angle = rotation_angle
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        weight = module.weight.data
        bias = module.bias.data  # as bias is constant over locations it is fine to use for equivariance to rotation
        # Rotate the weights counter-clockwise
        if self.rotation_angle == 90:
            transformed_weight = weight.transpose(2, 3).flip(2)
        elif self.rotation_angle == 180:
            transformed_weight = weight.flip(3).flip(2)
        elif self.rotation_angle == 270:
            transformed_weight = weight.transpose(2, 3).flip(3)

        # From the _conv_forward method in the torch.nn.Conv2d class
        if module.padding_mode != 'zeros':
            raise NotImplementedError('Only zero padding is supported for now')
        transformed_filters_output = F.conv2d(input[0], transformed_weight, bias, module.stride, module.padding,
                                              module.dilation, module.groups)
        return torch.cat((output, transformed_filters_output), dim=1)

    def close(self):
        self.hook.remove()


class SimpleInverseHook:
    """
    A simple test inverse hook.
    Flips the sign of all weights (since images are normalised to -1,1)
    Not sure this is a general solution to inverse or not (particularly if we are deep in the network)
    May not even be correct???
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        weight = module.weight.data
        bias = module.bias.data

        # Invert the weights
        transformed_weight = -weight

        # From the _conv_forward method in the torch.nn.Conv2d class
        transformed_filters_output = F.conv2d(input[0], transformed_weight, bias, module.stride, module.padding,
                                              module.dilation, module.groups)

        return torch.cat((output, transformed_filters_output), dim=1)

    def close(self):
        self.hook.remove()


# Todo: not sure this is right at all. Does scaling even make sense for equivariance?
class ScaleHook:
    def __init__(self, module, severity=5):
        self.severity = severity
        self.hook = module.register_forward_hook(self.hook_fn)

    def scale(self, x, severity=5):  # From data.data_transforms.py
        c = [(1 / .9, 1 / .9), (1 / .8, 1 / .8), (1 / .7, 1 / .7), (1 / .6, 1 / .6), (1 / .5, 1 / .5)][severity - 1]

        aff = transform.AffineTransform(scale=c)

        a1, a2 = aff.params[0, :2]
        b1, b2 = aff.params[1, :2]
        a3 = 13.5 * (1 - a1 - a2)
        b3 = 13.5 * (1 - b1 - b2)
        aff = transform.AffineTransform(scale=c, translation=[a3, b3])

        print(aff)  # a transformation matrix - can I then apply this with torch?
                    # Also why does applying this to the 5x5 patch make zeros? Is scale too severe?

                    # Other options: from torchvision.transforms.functional import affine
                    # Use resize then pad.
        x = x.detach().cpu().numpy()
        print(x.shape)
        x = x[0, 0, :, :]
        print(x.shape)

        x = np.array(x) / 255.
        x = transform.warp(x, inverse_map=aff)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)


    def torch_scale(self, x, severity=5):
        c = [(1 / .9, 1 / .9), (1 / .8, 1 / .8), (1 / .7, 1 / .7), (1 / .6, 1 / .6), (1 / .5, 1 / .5)][severity - 1]

        from torchvision.transforms.functional import affine
        import PIL.Image

        x = affine(x, angle=0, translate=[0, 0], scale=1/5., shear=0, resample=PIL.Image.NEAREST)
        return x

    def hook_fn(self, module, input, output):
        weight = module.weight.data
        bias = module.bias.data

        # # Todo: this doesn't work yet - need to figure out how to scale the weights
        # print(weight[0, 0, :, :])
        # # Scale the weights
        # transformed_weight = self.scale(weight, severity=self.severity)
        # # print(transformed_weight[0, 0, :, :])
        # print(transformed_weight[:, :])

        # Todo: not at all confident this is doing what I want but it should work to test
        # print(weight[0, 0, :, :])
        transformed_weight = self.torch_scale(weight, severity=self.severity)
        # print(transformed_weight[0, 0, :, :])
        # print(2/0)


        # From the _conv_forward method in the torch.nn.Conv2d class
        if module.padding_mode != 'zeros':
            raise NotImplementedError('Only zero padding is supported for now')
        transformed_filters_output = F.conv2d(input[0], transformed_weight, bias, module.stride, module.padding,
                                              module.dilation, module.groups)
        return torch.cat((output, transformed_filters_output), dim=1)

    def close(self):
        self.hook.remove()
