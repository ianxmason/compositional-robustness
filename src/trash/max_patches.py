"""
Collect max patches for specific corruptions and compositions to perform intuitive analysis.
As always with max patches, take care not to over-interpret.
"""
import argparse
import os
import torch
import torch.nn as nn
from data.data_transforms import denormalize
from data.data_loaders import get_multi_static_emnist_dataloaders
from lib.networks import DTN
from lib.receptive_field import receptive_field, receptive_field_for_unit
from lib.utils import *

# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility
    # Todo - set input shape from data. For now is hardcoded (3, 28, 28).
    # Todo - set the number of units from the network. For now is hardcoded as 64.


def unravel_index(index, shape):
    """
    https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/2
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class MaxConvHook:
    def __init__(self, module, unit, topk):
        self.unit = unit
        self.topk = topk
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        feature_map = output[:, self.unit, :, :]
        self.max_values, self.ravelled_max_locations = torch.topk(torch.flatten(feature_map), k=self.topk)
        self.feature_map_shape = feature_map.shape
        assert feature_map[unravel_index(self.ravelled_max_locations[0], feature_map.shape)] == self.max_values[0]

    def close(self):
        self.hook.remove()


class MaxLinHook:
    def __init__(self, module, unit, topk):
        self.unit = unit
        self.topk = topk
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        feature = output[:, self.unit]
        self.max_values, self.max_locations = torch.topk(feature, k=self.topk)
        assert feature[self.max_locations[0]] == self.max_values[0]

    def close(self):
        self.hook.remove()


def main(network_corruption, data_corruptions, topk, data_root, ckpt_path, save_path, total_n_classes, batch_size,
         n_workers, pin_mem, dev, check_if_run):
    # Over all the corruptions in data_corruptions which are the max activating patches
    ckpt_name = "{}.pt".format('-'.join(network_corruption))
    network = DTN(total_n_classes).to(dev)
    receptive_field_dict = receptive_field(network.conv_params, (3, 28, 28))
    network.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name)))
    network.eval()

    # Load data
    corruption_paths = [os.path.join(data_root, corruption_name) for corruption_name in data_corruptions]
    data_name = '_'.join(data_corruptions)
    train_classes = list(range(total_n_classes))
    _, _, tst_dl = get_multi_static_emnist_dataloaders(corruption_paths, train_classes, batch_size, True, n_workers,
                                                       pin_mem)

    # For now, we only care to hook the first conv, and we hardcode the number of units in conv1
    layer = 0
    num_units = 64
    for unit in range(num_units):
        fname = "max_patches_layer_{}_unit_{}_network_{}_data_{}.png".format(layer, unit, ckpt_name[:-3], data_name)
        if check_if_run and os.path.exists(os.path.join(save_path, fname)):
            print("Patches file already exists at {}. \n Skipping calculation for network: {} with data: {}".format(
                os.path.join(save_path, fname), ckpt_name[:-3], data_name))
            continue

        for module in network.modules():
            if isinstance(module, nn.Conv2d):
                print("Hooked module: {}".format(module))
                # num_units = module.weight.shape[0]
                hooked_conv = MaxConvHook(module, unit, topk)
                break

        with torch.no_grad():
            conv_maxs_so_far = None
            for data_tuple in tst_dl:
                x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
                _ = network(x_tst)
                conv_maxs = hooked_conv.max_values.detach().cpu().numpy()

                if conv_maxs_so_far is None:
                    conv_maxs_so_far = conv_maxs
                    patches = []
                    for i in range(topk):
                        unravelled = unravel_index(hooked_conv.ravelled_max_locations[i], hooked_conv.feature_map_shape)
                        patch_location = (int(unravelled[0]), int(unravelled[1]), int(unravelled[2]))
                        rf = receptive_field_for_unit(receptive_field_dict, layer="1", unit_position=patch_location[1:])
                        patches.append(x_tst[patch_location[0], :, int(rf[0][0]):int(rf[0][1]),
                                             int(rf[1][0]):int(rf[1][1])].detach().cpu().numpy())
                    continue

                for i, cm in enumerate(conv_maxs):
                    if cm > np.min(conv_maxs_so_far):
                        print("Old conv maxs {}".format(conv_maxs_so_far))
                        min_idx = np.argmin(conv_maxs_so_far)
                        conv_maxs_so_far[min_idx] = cm
                        unravelled = unravel_index(hooked_conv.ravelled_max_locations[i], hooked_conv.feature_map_shape)
                        patch_location = (int(unravelled[0]), int(unravelled[1]), int(unravelled[2]))
                        rf = receptive_field_for_unit(receptive_field_dict, layer="1", unit_position=patch_location[1:])
                        patches[min_idx] = x_tst[patch_location[0], :, int(rf[0][0]):int(rf[0][1]),
                                                 int(rf[1][0]):int(rf[1][1])].detach().cpu().numpy()
                        print("New conv max {}".format(conv_maxs_so_far))

        fig, axs = plt.subplots(int(np.sqrt(topk)), int(np.sqrt(topk)))
        for i in range(int(np.sqrt(topk))):
            for j in range(int(np.sqrt(topk))):
                patch = np.transpose(patches[i * int(np.sqrt(topk)) + j], (1, 2, 0))
                patch = denormalize(patch).astype(np.uint8)
                axs[i, j].imshow(patch)
                axs[i, j].axis('off')
        fig.savefig(os.path.join(save_path, fname), bbox_inches='tight')
        print("Saved fig to: {}".format(os.path.join(save_path, fname)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect activations for different corruptions and compositions')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/max_patches/EMNIST/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--topk', type=int, default=25, help="Top k patches (square number)")
    parser.add_argument('--n-workers', type=int, default=4, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which max patches"
                                                                    " have already been calculated. Useful for slurm.")
    args = parser.parse_args()

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Create unmade directories
    mkdir_p(args.save_path)

    # network_corruption = ['identity', 'inverse', 'stripe']  # Network trained on these 3 elemental corruptions
    # data_corruptions = ['identity', 'inverse', 'stripe']  # Use data from these 3 elemental corruptions

    # network_corruption = ['identity', 'inverse', 'gaussian_blur']
    # data_corruptions = ['identity', 'inverse', 'gaussian_blur']

    # network_corruption = ['identity', 'canny_edges', 'inverse']
    # data_corruptions = ['identity', 'canny_edges', 'inverse']

    network_corruption = ['identity', 'rotate_fixed', 'scale']
    data_corruptions = ['identity', 'rotate_fixed', 'scale']

main(network_corruption, data_corruptions, args.topk, args.data_root, args.ckpt_path, args.save_path,
         args.total_n_classes, args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run)
