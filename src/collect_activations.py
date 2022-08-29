"""
Collect per-neuron firing rates for different corruptions and compositions to perform analysis
"""
import argparse
import os
import torch
import torch.nn as nn
import pickle
from copy import deepcopy
from data.data_loaders import get_static_emnist_dataloaders
from lib.networks import DTN
from lib.utils import *


# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility

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


def main(network_corruptions, data_corruptions, data_root, ckpt_path, save_path, total_n_classes, batch_size,
         experiment_name, n_workers, pin_mem, dev, check_if_run):
    # Loop over networks being analysed
    for network_corruption in network_corruptions:
        # Load each ckpt into the network
        if experiment_name != '':
            ckpt_name = "{}_{}.pt".format('-'.join(network_corruption), experiment_name)
        else:
            ckpt_name = "{}.pt".format('-'.join(network_corruption))
        network = DTN(total_n_classes).to(dev)
        network.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name)))
        network.eval()

        # Count linear layers
        num_linear_layers = 0
        for name, module in network.named_modules():
            if isinstance(module, nn.Linear):
                num_linear_layers += 1

        # Hook each relu and output layer
        hooks = []
        in_feature_extractor = True
        linear_count = 0
        for module in network.modules():
            if isinstance(module, nn.Linear):
                in_feature_extractor = False
                linear_count += 1
            if isinstance(module, nn.ReLU) and in_feature_extractor:
                hooks.append(FiringConvHook(module))
                # print("Hooked {}".format(module))
            elif isinstance(module, nn.ReLU) and not in_feature_extractor:
                hooks.append(FiringLinHook(module))
                # print("Hooked {}".format(module))
            elif isinstance(module, nn.Linear) and linear_count == num_linear_layers:  # output layer
                hooks.append(FiringLinHook(module))
                # print("Hooked {}".format(module))

        # Loop over data corruptions
        for data_corruption in data_corruptions:
            data_name = "{}".format('-'.join(data_corruption))
            fname = "avg_firing_rates_network_{}_data_{}.pkl".format(ckpt_name[:-3], data_name)
            corruption_path = os.path.join(data_root, data_name)
            trained_classes = list(range(total_n_classes))
            if check_if_run and os.path.exists(os.path.join(save_path, fname)):
                print("Pickle file already exists at {}. \n Skipping analysis for network: {} with data: {}".format(
                    os.path.join(save_path, fname), ckpt_name[:-3], data_name))
                continue
            print("Collecting activations for {}".format(ckpt_name))
            print("Using data {}".format(data_name))

            # Load data - use test set as this is the same data used to make heatmaps
            _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, True,
                                                         n_workers, pin_mem)

            # Pass through data once to get activations
            with torch.no_grad():
                class_cumsum = [None for _ in range(len(hooks))]  # Per class. Element size: num_classes, num_units.
                max_activations = [None for _ in range(len(hooks))]  # Over all classes
                class_dpoints = [None for _ in range(len(hooks))]  # Per class. Class counts.
                min_activations = None  # Over all classes, only needed for output layer

                for data_tuple in tst_dl:
                    x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    _ = network(x_tst)
                    for i, hook in enumerate(hooks):
                        output = hook.output
                        batch_maxs, _ = torch.max(output, dim=0)  # num_units
                        batch_mins, _ = torch.min(output, dim=0)  # num_units

                        if class_cumsum[i] is None:
                            class_cumsum[i] = torch.zeros(total_n_classes, output.shape[1]).to(dev)
                        class_cumsum[i].index_add_(dim=0, index=y_tst, source=output)

                        if class_dpoints[i] is None:
                            class_dpoints[i] = torch.zeros(total_n_classes).to(dev)
                        class_dpoints[i].index_add_(dim=0, index=y_tst, source=torch.ones(output.shape[0]).to(dev))

                        if max_activations[i] is not None:
                            max_activations[i] = torch.where(batch_maxs > max_activations[i], batch_maxs,
                                                             max_activations[i])
                        else:
                            max_activations[i] = batch_maxs

                        if i == len(hooks) - 1:  # output layer
                            if min_activations is not None:
                                min_activations = torch.where(batch_mins < min_activations, batch_mins, min_activations)
                            else:
                                min_activations = batch_mins

                # Calculate normalized (between 0 and 1) average firing over whole dataset
                class_avg_firings = []
                for i in range(len(class_cumsum)):
                    # Except for last layer hooks are after relu so activations are >= 0 - so no min for normalizing
                    if i < len(class_cumsum) - 1:
                        # Avoid division by 0 if max_activations is 0 (neuron never fires)
                        max_activations[i] = torch.where(max_activations[i] == 0,
                                                         torch.ones(max_activations[i].shape).to(dev),
                                                         max_activations[i])
                        normalized_cumsum = class_cumsum[i] / max_activations[i]
                        class_avg_firings.append(normalized_cumsum / class_dpoints[i][:, None])

                        assert torch.all(class_avg_firings[i] <= 1 + 1e-3)  # isclose
                        assert torch.all(class_avg_firings[i] >= 0 - 1e-3)
                    else:  # Output layer
                        max_activations[i] = torch.where(max_activations[i] == 0,
                                                         torch.ones(max_activations[i].shape).to(dev),
                                                         max_activations[i])
                        normalized_cumsum = (class_cumsum[i] - torch.outer(class_dpoints[i], min_activations)) / (
                            max_activations[i] - min_activations)
                        class_avg_firings.append(normalized_cumsum / class_dpoints[i][:, None])
                        assert torch.all(class_avg_firings[i] <= 1 + 1e-3)
                        assert torch.all(class_avg_firings[i] >= 0 - 1e-3)

            # Save
            print("Collected per class average firing rates.")
            layer_firing_rates = [tens.detach().cpu().numpy() for tens in class_avg_firings]
            with open(os.path.join(save_path, fname), "wb") as f:
                pickle.dump(layer_firing_rates, f)
            print("Firing rates saved to {}".format(os.path.join(save_path, fname)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect activations for different corruptions and compositions')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST3/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST3/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/activations/EMNIST3/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which activations"
                                                                    " have already been collected. Useful for slurm.")
    parser.add_argument('--experiment-name', type=str, default='',
                        help="name of experiment - used to load the checkpoint files")
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

    # network_corruptions = [['impulse_noise', 'stripe'], ['stripe'], ['impulse_noise']]
    # data_corruptions = [['impulse_noise'],  # Elemental corruption 1
    #                     ['stripe'],  # Elemental corruption 2
    #                     ['impulse_noise', 'stripe'],  # Composition of corruptions
    #                     ['gaussian_blur', 'stripe'],  # Composition that 'kind of' works (includes one elemental)
    #                     ['impulse_noise', 'inverse'],  # Composition that doesn't work (includes one elemental)
    #                     ['inverse']]  # Elemental corruption not trained on

    # network_corruptions = [['identity', 'inverse', 'gaussian_blur']]
    # data_corruptions = [['identity'],
    #                     ['inverse'],
    #                     ['gaussian_blur'],
    #                     ['inverse', 'gaussian_blur']]

    # network_corruptions = [['identity', 'canny_edges', 'inverse']]
    # data_corruptions = [['identity'],
    #                     ['canny_edges'],
    #                     ['inverse'],
    #                     ['canny_edges', 'inverse']]

    # main(network_corruptions, data_corruptions, args.data_root, args.ckpt_path, args.save_path, args.total_n_classes,
    #      args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run)


    # For violin plots across corruptions. --check-if-run flag will avoid duplicate calculations
    # networks = [['identity', 'gaussian_blur', 'stripe'],
    #             ['identity', 'impulse_noise', 'gaussian_blur'],
    #             ['identity', 'impulse_noise', 'inverse'],
    #             ['identity', 'impulse_noise', 'stripe'],
    #             ['identity', 'inverse', 'gaussian_blur'],
    #             ['identity', 'inverse', 'stripe']]
    #
    # networks = [['identity', 'rotate_fixed', 'scale']]
    #
    # for network in networks:
    #     network_corruptions = [network, [network[0]] + [network[1]], [network[0]] + [network[2]]]
    #     data_corruptions = [[network_corruptions[0][0]],
    #                         [network_corruptions[0][1]],
    #                         [network_corruptions[0][2]],
    #                         [network_corruptions[0][1], network_corruptions[0][2]],
    #                         [network_corruptions[0][2], network_corruptions[0][1]]]
    #
    #     main(network_corruptions, data_corruptions, args.data_root, args.ckpt_path, args.save_path,
    #          args.total_n_classes, args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run)

    # For violin plots across compositions with new data. --check-if-run flag will avoid duplicate calculations
    # In the long run may wish to parallelize using slurm - currently just run each experiment-name separately
    # CUDA_VISIBLE_DEVICES=2 python collect_activations.py --pin-mem --check-if-run --experiment_name SEE BELOW

    # EMNIST2
    # settings for --experiment-name: default_arg=='', invariance-loss-lr-0.001-w-1.0, L1-L2-all-invariance-loss-lr-0.001, L1-L2-invariance-loss-lr-0.001
    # networks = [['gaussian_blur', 'identity', 'impulse_noise'],
    #             ['gaussian_blur', 'identity', 'inverse'],
    #             ['gaussian_blur', 'identity', 'rotate_fixed'],
    #             ['gaussian_blur', 'identity', 'scale'],
    #             ['gaussian_blur', 'identity', 'thinning'],
    #             ['identity', 'impulse_noise', 'inverse'],
    #             ['identity', 'impulse_noise', 'rotate_fixed'],
    #             ['identity', 'impulse_noise', 'scale'],
    #             ['identity', 'impulse_noise', 'thinning'],
    #             ['identity', 'inverse', 'rotate_fixed'],
    #             ['identity', 'inverse', 'scale'],
    #             ['identity', 'inverse', 'thinning'],
    #             ['identity', 'rotate_fixed', 'scale'],
    #             ['identity', 'rotate_fixed', 'thinning'],
    #             ['identity', 'scale', 'thinning']]

    # EMNIST3
    # settings for --experiment-name: default_arg=='', invariance-loss-lr-0.001, L1-L2-all-invariance-loss-lr-0.001, L1-L2-bwd-invariance-loss-lr-0.001, L1-L2-fwd-invariance-loss-lr-0.001
    networks = [['gaussian_blur', 'impulse_noise', 'identity'],
                ['gaussian_blur', 'inverse', 'identity'],
                ['gaussian_blur', 'rotate_fixed', 'identity'],
                ['gaussian_blur', 'scale', 'identity'],
                ['gaussian_blur', 'shear_fixed', 'identity'],
                ['impulse_noise', 'inverse', 'identity'],
                ['impulse_noise', 'rotate_fixed', 'identity'],
                ['impulse_noise', 'scale', 'identity'],
                ['impulse_noise', 'shear_fixed', 'identity'],
                ['inverse', 'rotate_fixed', 'identity'],
                ['inverse', 'scale', 'identity'],
                ['inverse', 'shear_fixed', 'identity'],
                ['rotate_fixed', 'scale', 'identity'],
                ['rotate_fixed', 'shear_fixed', 'identity'],
                ['scale', 'shear_fixed', 'identity']]

    for network in networks:
        network_corruptions = [network]
        data = deepcopy(network)
        data.remove('identity')
        data_corruptions = [['identity'],
                            [data[0]],
                            [data[1]],
                            [data[0], data[1]],
                            [data[1], data[0]]]

        main(network_corruptions, data_corruptions, args.data_root, args.ckpt_path, args.save_path,
             args.total_n_classes, args.batch_size, args.experiment_name, args.n_workers, args.pin_mem, dev,
             args.check_if_run)