"""
Collect per-neuron firing rates for different corruptions and compositions to perform analysis
"""
import argparse
import os
import torch
import torchvision
import torch.nn as nn
import pickle
from copy import deepcopy
import data.data_transforms as dt
from data.data_loaders import get_transformed_static_dataloaders
from lib.networks import create_emnist_network, create_emnist_modules, create_emnist_autoencoder, \
                         create_cifar_network, create_cifar_modules, create_cifar_autoencoder, \
                         create_facescrub_network, create_facescrub_modules, create_facescrub_autoencoder
from lib.utils import *


class FiringConvHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # output - batchsize, num_units/channels, width, height (height and width may be other way round - not checked)
        max_spatial_firing, _ = torch.max(output.reshape(output.shape[0], output.shape[1], -1), dim=2)
        self.output = max_spatial_firing  # batch_size, num_units

    def close(self):
        self.hook.remove()


class FiringLinHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output  # batch_size, num_units

    def close(self):
        self.hook.remove()

def main(corruptions, experiment, dataset, data_root, ckpt_path, save_path, total_n_classes, batch_size,
         n_workers, pin_mem, dev, check_if_run):

    if experiment != "Modules":
        raise NotImplementedError("Only implemented for Modules experiment")

    if dataset == "EMNIST":
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, "Modules",
                                                                         ["Identity"], dev)
    elif dataset == "CIFAR":
        network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, "Modules",
                                                                        ["Identity"], dev)
    elif dataset == "FACESCRUB":
        network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, "Modules",
                                                                            ["Identity"], dev)

    corruptions += [["Identity"]]  # Want to include identity activations to compare against

    for test_corruption in corruptions:
        fname = "avg_firing_rates_{}_{}.pkl".format(experiment, '-'.join(test_corruption))
        if check_if_run and os.path.exists(os.path.join(save_path, fname)):
            print("Pickle file already exists at {}. \n Skipping analysis for experiment: {} on data: {}".format(
                os.path.join(save_path, fname), experiment, '-'.join(test_corruption)))
            continue

        print("Collecting activations for {}".format(test_corruption))
        sys.stdout.flush()
        trained_classes = list(range(total_n_classes))
        identity_path = os.path.join(data_root, "Identity")
        if dataset == "EMNIST":
            transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
            transforms += [getattr(dt, c)() for c in test_corruption]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
            transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
        else:
            transforms = [getattr(dt, c)() for c in test_corruption]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='RGB'))]
        trn_dl, val_dl, _ = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                       batch_size, False, n_workers, pin_mem)


        # Pass through data once to get activations
        with torch.no_grad():
            class_cumsum = [None for _ in range(len(network_blocks))]  # Per class. Element size: num_classes, num_units.
            class_cumsum_sq = [None for _ in range(len(network_blocks))]  # Per class. Element size: num_classes, num_units.
            max_activations = [None for _ in range(len(network_blocks))]  # Over all classes
            class_dpoints = [None for _ in range(len(network_blocks))]  # Per class. Class counts.
            min_activations = None  # Over all classes, only needed for output layer

            for data_tuple in trn_dl:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                for i, block in enumerate(network_blocks):
                    x_val = block(x_val)
                    activations = x_val  # in conv layers 4D, in linear layers 2D
                    if len(activations.shape) == 4:
                        max_spatial_firing, _ = torch.max(activations.reshape(activations.shape[0],
                                                                              activations.shape[1], -1), dim=2)
                        activations = max_spatial_firing  # batch_size, num_units

                    batch_maxs, _ = torch.max(activations, dim=0)  # num_units
                    batch_mins, _ = torch.min(activations, dim=0)  # num_units

                    if class_cumsum[i] is None:
                        class_cumsum[i] = torch.zeros(total_n_classes, activations.shape[1]).to(dev)
                    class_cumsum[i].index_add_(dim=0, index=y_val, source=activations)

                    if class_cumsum_sq[i] is None:
                        class_cumsum_sq[i] = torch.zeros(total_n_classes, activations.shape[1]).to(dev)
                    class_cumsum_sq[i].index_add_(dim=0, index=y_val, source=activations ** 2)

                    if class_dpoints[i] is None:
                        class_dpoints[i] = torch.zeros(total_n_classes).to(dev)
                    class_dpoints[i].index_add_(dim=0, index=y_val, source=torch.ones(activations.shape[0]).to(dev))

                    if max_activations[i] is not None:
                        max_activations[i] = torch.where(batch_maxs > max_activations[i], batch_maxs,
                                                         max_activations[i])
                    else:
                        max_activations[i] = batch_maxs

                    if i == len(network_blocks) - 1:  # output layer
                        if min_activations is not None:
                            min_activations = torch.where(batch_mins < min_activations, batch_mins, min_activations)
                        else:
                            min_activations = batch_mins

            # Calculate normalized (between 0 and 1) average firing over whole dataset
            class_avg_firings = []
            class_std_firings = []
            class_max_firings = []
            for i in range(len(class_cumsum)):
                # Except for last layer hooks are after relu so activations are >= 0 - so no min for normalizing
                if i < len(class_cumsum) - 1:
                    class_max_firings.append(max_activations[i])
                    # Avoid division by 0 if max_activations is 0 (neuron never fires)
                    max_activations[i] = torch.where(max_activations[i] == 0,
                                                     torch.ones(max_activations[i].shape).to(dev),
                                                     max_activations[i])
                    # class_dpoints[i] = torch.where(class_dpoints[i] == 0,
                    #                                # It is possible the val set has no examples of a class
                    #                                torch.ones(class_dpoints[i].shape).to(dev),
                    #                                class_dpoints[i])

                    # Normalize between 0 and 1
                    # normalized_cumsum = class_cumsum[i] / max_activations[i]
                    # class_avg_firings.append(normalized_cumsum / class_dpoints[i][:, None])
                    # assert torch.all(class_avg_firings[i] <= 1 + 1e-3)  # isclose
                    # assert torch.all(class_avg_firings[i] >= 0 - 1e-3)

                    # Todo: check this more carefully with prints? I think it is good but not sure
                    class_avg_firings.append(class_cumsum[i] / class_dpoints[i][:, None])
                    class_std_firings.append(torch.sqrt(class_cumsum_sq[i] / class_dpoints[i][:, None] -
                                                        class_avg_firings[i] ** 2))

                    # Todo: check this more carefully with prints? I think it is good but not sure
                    # # Rather than normalising by the max, get relative change from average firing for a neuron in layer i
                    # average_firing_per_class_per_neuron = class_cumsum[i] / class_dpoints[i][:, None]  # num_classes, num_units
                    # average_firing_per_layer = torch.mean(average_firing_per_class_per_neuron)  # scalar
                    # class_avg_firings.append(average_firing_per_class_per_neuron / average_firing_per_layer)  # ratio to average firing
                    # # Get relative change from the average firing for a specific neuron over all classes
                    # average_firing_per_class_per_neuron = class_cumsum[i] / class_dpoints[i][:, None]  # num_classes, num_units
                    # average_firing_per_neuron = torch.mean(average_firing_per_class_per_neuron, dim=0)  # num_units
                    # class_avg_firings.append(average_firing_per_class_per_neuron / average_firing_per_neuron)  # ratio to average firing



                else:  # Output layer
                    class_max_firings.append(max_activations[i])
                    max_activations[i] = torch.where(max_activations[i] == 0,
                                                     torch.ones(max_activations[i].shape).to(dev),
                                                     max_activations[i])
                    # class_dpoints[i] = torch.where(class_dpoints[i] == 0,
                    #                                # It is possible the val set has no examples of a class
                    #                                torch.ones(class_dpoints[i].shape).to(dev),
                    #                                class_dpoints[i])

                    # Normalize between 0 and 1
                    # normalized_cumsum = (class_cumsum[i] - torch.outer(class_dpoints[i], min_activations)) / (
                    #     max_activations[i] - min_activations)
                    # class_avg_firings.append(normalized_cumsum / class_dpoints[i][:, None])
                    # assert torch.all(class_avg_firings[i] <= 1 + 1e-3)
                    # assert torch.all(class_avg_firings[i] >= 0 - 1e-3)

                    # Todo: check this more carefully with prints? I think it is good but not sure
                    class_avg_firings.append(class_cumsum[i] / class_dpoints[i][:, None])
                    class_std_firings.append(torch.sqrt(class_cumsum_sq[i] / class_dpoints[i][:, None] -
                                                        class_avg_firings[i] ** 2))

                    # Todo: check this more carefully with prints? I think it is good but not sure
                    # # Rather than normalising by the max, get relative change from average firing for a neuron in layer i
                    # average_firing_per_class_per_neuron = class_cumsum[i] / class_dpoints[i][:, None]  # num_classes, num_units
                    # average_firing_per_layer = torch.mean(average_firing_per_class_per_neuron)  # scalar
                    # class_avg_firings.append(average_firing_per_class_per_neuron / average_firing_per_layer)  # ratio to average firing
                    # # Get relative change from the average firing for a specific neuron over all classes
                    # average_firing_per_class_per_neuron = class_cumsum[i] / class_dpoints[i][:, None]  # num_classes, num_units
                    # average_firing_per_neuron = torch.mean(average_firing_per_class_per_neuron, dim=0)  # num_units
                    # class_avg_firings.append(average_firing_per_class_per_neuron / average_firing_per_neuron)  # ratio to average firing



        # Save
        print("Collected per class average firing rates.")
        layer_firing_rates = [tens.detach().cpu().numpy() for tens in class_avg_firings]
        layer_std_firings = [tens.detach().cpu().numpy() for tens in class_std_firings]
        layer_max_firings = [tens.detach().cpu().numpy() for tens in class_max_firings]
        with open(os.path.join(save_path, fname), "wb") as f:
            # pickle.dump(layer_firing_rates, f)
            # pickle.dump((layer_firing_rates, layer_std_firings), f)
            # pickle.dump(layer_max_firings, f)
            pickle.dump((layer_firing_rates, layer_std_firings, layer_max_firings), f)
        print("Firing rates saved to {}".format(os.path.join(save_path, fname)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect activations for different corruptions and compositions')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/analysis/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--experiment', type=str, default='CrossEntropy',
                        help="which method to use. CrossEntropy or Contrastive or Modules.")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which activations"
                                                                    " have already been collected. Useful for slurm.")
    args = parser.parse_args()

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Create unmade directories
    args.data_root = os.path.join(args.data_root, args.dataset)
    args.ckpt_path = os.path.join(args.ckpt_path, args.dataset)
    args.save_path = os.path.join(args.save_path, args.dataset)
    mkdir_p(args.save_path)

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)

    corruptions = []
    # Training on only the corruptions in the composition. Always include identity at the end, remove permutations
    for corr in all_corruptions:
        if "Identity" not in corr:
            corr.sort()
            corr += ["Identity"]
            if corr not in corruptions:
                corruptions.append(corr)
        elif len(corr) == 1:  # identity only
            if corr not in corruptions:
                corruptions.append(corr)
        else:
            raise ValueError("Only expect the identity to appear as its own corruption")

    assert len(corruptions) == 64  # 64 for EMNIST 5, 128 for EMNIST4

    if "Modules" in args.experiment or "ImgSpace" in args.experiment:  # For the Modules approach, we want all elemental corruptions
        corruptions = [["Contrast"], ["GaussianBlur"], ["ImpulseNoise"], ["Invert"], ["Rotate90"], ["Swirl"]]
        # corruptions = [corr for corr in corruptions if len(corr) == 2]
        # assert len(corruptions) == 6  # hardcoded for EMNIST5
        # corruptions = corruptions[args.corruption_ID:args.corruption_ID+1]
    elif "Contrastive" in args.experiment or "CrossEntropy" in args.experiment:
        # For contrastive and cross entropy we only want the case with all corruptions together
        max_corr_count = max([len(corr) for corr in corruptions])
        corruptions = [corr for corr in corruptions if len(corr) == max_corr_count]
        assert len(corruptions) == 1

    """
    CUDA_VISIBLE_DEVICES=4 python collect_activations.py --dataset EMNIST --experiment Modules --total-n-classes 47 --pin-mem --check-if-run
    CUDA_VISIBLE_DEVICES=5 python collect_activations.py --dataset CIFAR --experiment Modules --total-n-classes 10 --pin-mem --check-if-run
    CUDA_VISIBLE_DEVICES=6 python collect_activations.py --dataset FACESCRUB --experiment Modules --total-n-classes 388 --pin-mem --check-if-run
    """

    # Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility

    # Todo:
    # Todo 1. Get firing rate changes for modular backbone to see where modules should go
    # Todo 2. Get firing rates for CE and contrastive data to recreate the invariance plots

    main(corruptions, args.experiment, args.dataset, args.data_root, args.ckpt_path, args.save_path,
         args.total_n_classes, args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run)

"""
Below is old code, it may be useful for when we want to get the firing rates for jointly trained networks
"""

# def main(network_corruptions, data_corruptions, data_root, ckpt_path, save_path, total_n_classes, batch_size,
#          experiment_name, n_workers, pin_mem, dev, check_if_run):
#     # Loop over networks being analysed
#     for network_corruption in network_corruptions:
#         # Load each ckpt into the network
#         if experiment_name != '':
#             ckpt_name = "{}_{}.pt".format('-'.join(network_corruption), experiment_name)
#         else:
#             ckpt_name = "{}.pt".format('-'.join(network_corruption))
#         network = DTN(total_n_classes).to(dev)
#         network.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name)))
#         network.eval()
#
#         # Count linear layers
#         num_linear_layers = 0
#         for name, module in network.named_modules():
#             if isinstance(module, nn.Linear):
#                 num_linear_layers += 1
#
#         # Hook each relu and output layer
#         hooks = []
#         in_feature_extractor = True
#         linear_count = 0
#         for module in network.modules():
#             if isinstance(module, nn.Linear):
#                 in_feature_extractor = False
#                 linear_count += 1
#             if isinstance(module, nn.ReLU) and in_feature_extractor:
#                 hooks.append(FiringConvHook(module))
#                 # print("Hooked {}".format(module))
#             elif isinstance(module, nn.ReLU) and not in_feature_extractor:
#                 hooks.append(FiringLinHook(module))
#                 # print("Hooked {}".format(module))
#             elif isinstance(module, nn.Linear) and linear_count == num_linear_layers:  # output layer
#                 hooks.append(FiringLinHook(module))
#                 # print("Hooked {}".format(module))
#
#         # Loop over data corruptions
#         for data_corruption in data_corruptions:
#             data_name = "{}".format('-'.join(data_corruption))
#             fname = "avg_firing_rates_network_{}_data_{}.pkl".format(ckpt_name[:-3], data_name)
#             corruption_path = os.path.join(data_root, data_name)
#             trained_classes = list(range(total_n_classes))
#             if check_if_run and os.path.exists(os.path.join(save_path, fname)):
#                 print("Pickle file already exists at {}. \n Skipping analysis for network: {} with data: {}".format(
#                     os.path.join(save_path, fname), ckpt_name[:-3], data_name))
#                 continue
#             print("Collecting activations for {}".format(ckpt_name))
#             print("Using data {}".format(data_name))
#
#             # Load data - use test set as this is the same data used to make heatmaps
#             _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, True,
#                                                          n_workers, pin_mem)
#
#             # Pass through data once to get activations
#             with torch.no_grad():
#                 class_cumsum = [None for _ in range(len(hooks))]  # Per class. Element size: num_classes, num_units.
#                 max_activations = [None for _ in range(len(hooks))]  # Over all classes
#                 class_dpoints = [None for _ in range(len(hooks))]  # Per class. Class counts.
#                 min_activations = None  # Over all classes, only needed for output layer
#
#                 for data_tuple in tst_dl:
#                     x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
#                     _ = network(x_tst)
#                     for i, hook in enumerate(hooks):
#                         output = hook.output
#                         batch_maxs, _ = torch.max(output, dim=0)  # num_units
#                         batch_mins, _ = torch.min(output, dim=0)  # num_units
#
#                         if class_cumsum[i] is None:
#                             class_cumsum[i] = torch.zeros(total_n_classes, output.shape[1]).to(dev)
#                         class_cumsum[i].index_add_(dim=0, index=y_tst, source=output)
#
#                         if class_dpoints[i] is None:
#                             class_dpoints[i] = torch.zeros(total_n_classes).to(dev)
#                         class_dpoints[i].index_add_(dim=0, index=y_tst, source=torch.ones(output.shape[0]).to(dev))
#
#                         if max_activations[i] is not None:
#                             max_activations[i] = torch.where(batch_maxs > max_activations[i], batch_maxs,
#                                                              max_activations[i])
#                         else:
#                             max_activations[i] = batch_maxs
#
#                         if i == len(hooks) - 1:  # output layer
#                             if min_activations is not None:
#                                 min_activations = torch.where(batch_mins < min_activations, batch_mins, min_activations)
#                             else:
#                                 min_activations = batch_mins
#
#                 # Calculate normalized (between 0 and 1) average firing over whole dataset
#                 class_avg_firings = []
#                 for i in range(len(class_cumsum)):
#                     # Except for last layer hooks are after relu so activations are >= 0 - so no min for normalizing
#                     if i < len(class_cumsum) - 1:
#                         # Avoid division by 0 if max_activations is 0 (neuron never fires)
#                         max_activations[i] = torch.where(max_activations[i] == 0,
#                                                          torch.ones(max_activations[i].shape).to(dev),
#                                                          max_activations[i])
#                         normalized_cumsum = class_cumsum[i] / max_activations[i]
#                         class_avg_firings.append(normalized_cumsum / class_dpoints[i][:, None])
#
#                         assert torch.all(class_avg_firings[i] <= 1 + 1e-3)  # isclose
#                         assert torch.all(class_avg_firings[i] >= 0 - 1e-3)
#                     else:  # Output layer
#                         max_activations[i] = torch.where(max_activations[i] == 0,
#                                                          torch.ones(max_activations[i].shape).to(dev),
#                                                          max_activations[i])
#                         normalized_cumsum = (class_cumsum[i] - torch.outer(class_dpoints[i], min_activations)) / (
#                             max_activations[i] - min_activations)
#                         class_avg_firings.append(normalized_cumsum / class_dpoints[i][:, None])
#                         assert torch.all(class_avg_firings[i] <= 1 + 1e-3)
#                         assert torch.all(class_avg_firings[i] >= 0 - 1e-3)
#
#             # Save
#             print("Collected per class average firing rates.")
#             layer_firing_rates = [tens.detach().cpu().numpy() for tens in class_avg_firings]
#             with open(os.path.join(save_path, fname), "wb") as f:
#                 pickle.dump(layer_firing_rates, f)
#             print("Firing rates saved to {}".format(os.path.join(save_path, fname)))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Collect activations for different corruptions and compositions')
#     parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST3/',
#                         help="path to directory containing directories of different corruptions")
#     parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST3/',
#                         help="path to directory to save checkpoints")
#     parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/activations/EMNIST3/',
#                         help="path to directory to save test accuracies and losses")
#     parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
#     parser.add_argument('--batch-size', type=int, default=128, help="batch size")
#     parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
#     parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
#     parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
#     parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which activations"
#                                                                     " have already been collected. Useful for slurm.")
#     parser.add_argument('--experiment-name', type=str, default='',
#                         help="name of experiment - used to load the checkpoint files")
#     args = parser.parse_args()
#
#     # Set device
#     if args.cpu:
#         dev = torch.device('cpu')
#     else:
#         if not torch.cuda.is_available():
#             raise RuntimeError("GPU unavailable.")
#         dev = torch.device('cuda')
#
#     # Create unmade directories
#     mkdir_p(args.save_path)
#
#     # network_corruptions = [['impulse_noise', 'stripe'], ['stripe'], ['impulse_noise']]
#     # data_corruptions = [['impulse_noise'],  # Elemental corruption 1
#     #                     ['stripe'],  # Elemental corruption 2
#     #                     ['impulse_noise', 'stripe'],  # Composition of corruptions
#     #                     ['gaussian_blur', 'stripe'],  # Composition that 'kind of' works (includes one elemental)
#     #                     ['impulse_noise', 'inverse'],  # Composition that doesn't work (includes one elemental)
#     #                     ['inverse']]  # Elemental corruption not trained on
#
#     # network_corruptions = [['identity', 'inverse', 'gaussian_blur']]
#     # data_corruptions = [['identity'],
#     #                     ['inverse'],
#     #                     ['gaussian_blur'],
#     #                     ['inverse', 'gaussian_blur']]
#
#     # network_corruptions = [['identity', 'canny_edges', 'inverse']]
#     # data_corruptions = [['identity'],
#     #                     ['canny_edges'],
#     #                     ['inverse'],
#     #                     ['canny_edges', 'inverse']]
#
#     # main(network_corruptions, data_corruptions, args.data_root, args.ckpt_path, args.save_path, args.total_n_classes,
#     #      args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run)
#
#
#     # For violin plots across corruptions. --check-if-run flag will avoid duplicate calculations
#     # networks = [['identity', 'gaussian_blur', 'stripe'],
#     #             ['identity', 'impulse_noise', 'gaussian_blur'],
#     #             ['identity', 'impulse_noise', 'inverse'],
#     #             ['identity', 'impulse_noise', 'stripe'],
#     #             ['identity', 'inverse', 'gaussian_blur'],
#     #             ['identity', 'inverse', 'stripe']]
#     #
#     # networks = [['identity', 'rotate_fixed', 'scale']]
#     #
#     # for network in networks:
#     #     network_corruptions = [network, [network[0]] + [network[1]], [network[0]] + [network[2]]]
#     #     data_corruptions = [[network_corruptions[0][0]],
#     #                         [network_corruptions[0][1]],
#     #                         [network_corruptions[0][2]],
#     #                         [network_corruptions[0][1], network_corruptions[0][2]],
#     #                         [network_corruptions[0][2], network_corruptions[0][1]]]
#     #
#     #     main(network_corruptions, data_corruptions, args.data_root, args.ckpt_path, args.save_path,
#     #          args.total_n_classes, args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run)
#
#     # For violin plots across compositions with new data. --check-if-run flag will avoid duplicate calculations
#     # In the long run may wish to parallelize using slurm - currently just run each experiment-name separately
#     # CUDA_VISIBLE_DEVICES=2 python collect_activations.py --pin-mem --check-if-run --experiment_name SEE BELOW
#
#     # EMNIST2
#     # settings for --experiment-name: default_arg=='', invariance-loss-lr-0.001-w-1.0, L1-L2-all-invariance-loss-lr-0.001, L1-L2-invariance-loss-lr-0.001
#     # networks = [['gaussian_blur', 'identity', 'impulse_noise'],
#     #             ['gaussian_blur', 'identity', 'inverse'],
#     #             ['gaussian_blur', 'identity', 'rotate_fixed'],
#     #             ['gaussian_blur', 'identity', 'scale'],
#     #             ['gaussian_blur', 'identity', 'thinning'],
#     #             ['identity', 'impulse_noise', 'inverse'],
#     #             ['identity', 'impulse_noise', 'rotate_fixed'],
#     #             ['identity', 'impulse_noise', 'scale'],
#     #             ['identity', 'impulse_noise', 'thinning'],
#     #             ['identity', 'inverse', 'rotate_fixed'],
#     #             ['identity', 'inverse', 'scale'],
#     #             ['identity', 'inverse', 'thinning'],
#     #             ['identity', 'rotate_fixed', 'scale'],
#     #             ['identity', 'rotate_fixed', 'thinning'],
#     #             ['identity', 'scale', 'thinning']]
#
#     # EMNIST3
#     # settings for --experiment-name: default_arg=='', invariance-loss-lr-0.001, L1-L2-all-invariance-loss-lr-0.001, L1-L2-bwd-invariance-loss-lr-0.001, L1-L2-fwd-invariance-loss-lr-0.001
#     networks = [['gaussian_blur', 'impulse_noise', 'identity'],
#                 ['gaussian_blur', 'inverse', 'identity'],
#                 ['gaussian_blur', 'rotate_fixed', 'identity'],
#                 ['gaussian_blur', 'scale', 'identity'],
#                 ['gaussian_blur', 'shear_fixed', 'identity'],
#                 ['impulse_noise', 'inverse', 'identity'],
#                 ['impulse_noise', 'rotate_fixed', 'identity'],
#                 ['impulse_noise', 'scale', 'identity'],
#                 ['impulse_noise', 'shear_fixed', 'identity'],
#                 ['inverse', 'rotate_fixed', 'identity'],
#                 ['inverse', 'scale', 'identity'],
#                 ['inverse', 'shear_fixed', 'identity'],
#                 ['rotate_fixed', 'scale', 'identity'],
#                 ['rotate_fixed', 'shear_fixed', 'identity'],
#                 ['scale', 'shear_fixed', 'identity']]
#
#     for network in networks:
#         network_corruptions = [network]
#         data = deepcopy(network)
#         data.remove('identity')
#         data_corruptions = [['identity'],
#                             [data[0]],
#                             [data[1]],
#                             [data[0], data[1]],
#                             [data[1], data[0]]]
#
#         main(network_corruptions, data_corruptions, args.data_root, args.ckpt_path, args.save_path,
#              args.total_n_classes, args.batch_size, args.experiment_name, args.n_workers, args.pin_mem, dev,
#              args.check_if_run)
