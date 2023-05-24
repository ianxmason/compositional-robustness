"""
Collects neural activity and images and saves them in a pickle file for use with Deephys. https://deephys.org/
Currently set up only for EMNIST modular networks on elemental corruptions.

To run without slurm: CUDA_VISIBLE_DEVICES=0 python deephys_collect.py --pin-mem --dataset EMNIST
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision
import pickle
sys.path.append("../")
import data.data_transforms as dt
from data.data_loaders import get_transformed_static_dataloaders, get_static_dataloaders
from data.emnist import EMNIST_MEAN, EMNIST_STD
from data.cifar import CIFAR10_MEAN, CIFAR10_STD
from data.facescrub import FACESCRUB_MEAN, FACESCRUB_STD
from lib.networks import create_emnist_network, create_emnist_modules, \
                         create_cifar_network, create_cifar_modules, \
                         create_facescrub_network, create_facescrub_modules
from lib.utils import *


def modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, dataloader, dev):
    """
    Calculates average loss and accuracy of a modular network on the given dataloader.

    Collects layer activations for Deephys:
    [all_activs, all_outputs] :: Lists containing neural activity for intermediate and output layers
                                 each is a multidimensional list of dimensions  [#neurons, #images].
                                 For 4D convolutional feature maps take the maximum spatial firing.
                                 The output layer is always mandatory to be present.
    all_images :: List containing images resized to 32x32 pixels, it h    as size [#images,#channels,32,32].
    all_cats :: Labels is a 1-dimensional list of ground-truth label number
    """
    assert len(network_blocks) >= 2  # assumed when the network is called
    criterion = nn.CrossEntropyLoss()
    all_pre_mods, all_post_mods, all_outputs, all_images, all_cats = None, None, None, None, None
    for block in network_blocks:
        block.eval()
    for mod in test_modules:
        mod.eval()

    # Modules are applied as follows:
    # 1. the earlier the level the earlier the module is used
    # 2. the earlier the corruption is in the test corruption name the earlier it is applied
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            features = x_tst
            for j, block in enumerate(network_blocks):
                if j in test_module_levels:
                    if all_pre_mods is None:
                        if len(features.shape) == 4:
                            max_spatial_firing, _ = torch.max(features.reshape(features.shape[0], features.shape[1], -1), dim=2)
                            all_pre_mods = [max_spatial_firing.cpu().numpy()]
                        elif len(features.shape) == 2:
                            all_pre_mods = [features.cpu().numpy()]
                        else:
                            raise RuntimeError("Unexpected shape of features: {}".format(features.shape))
                    else:
                        if len(features.shape) == 4:
                            max_spatial_firing, _ = torch.max(features.reshape(features.shape[0], features.shape[1], -1), dim=2)
                            all_pre_mods.append(max_spatial_firing.cpu().numpy())
                        elif len(features.shape) == 2:
                            all_pre_mods.append(features.cpu().numpy())
                        else:
                            raise RuntimeError("Unexpected shape of features: {}".format(features.shape))
                    for m, l in zip(test_modules, test_module_levels):
                        if l == j:
                            features = m(features)
                    if all_post_mods is None:
                        if len(features.shape) == 4:
                            max_spatial_firing, _ = torch.max(features.reshape(features.shape[0], features.shape[1], -1), dim=2)
                            all_post_mods = [max_spatial_firing.cpu().numpy()]
                        elif len(features.shape) == 2:
                            all_post_mods = [features.cpu().numpy()]
                        else:
                            raise RuntimeError("Unexpected shape of features: {}".format(features.shape))
                    else:
                        if len(features.shape) == 4:
                            max_spatial_firing, _ = torch.max(features.reshape(features.shape[0], features.shape[1], -1), dim=2)
                            all_post_mods.append(max_spatial_firing.cpu().numpy())
                        elif len(features.shape) == 2:
                            all_post_mods.append(features.cpu().numpy())
                        else:
                            raise RuntimeError("Unexpected shape of features: {}".format(features.shape))
                if j != len(network_blocks) - 1:
                    features = block(features)
                else:
                    output = block(features)
                    if all_outputs is None:
                        all_outputs = [output.cpu().numpy()]
                    else:
                        all_outputs.append(output.cpu().numpy())
                    if all_cats is None:
                        all_cats = [y_tst.cpu().numpy()]
                    else:
                        all_cats.append(y_tst.cpu().numpy())
                    if all_images is None:
                        x_tst = dt.denormalize_255(x_tst, EMNIST_MEAN, EMNIST_STD)
                        all_images = [x_tst.cpu().numpy()]
                    else:
                        x_tst = dt.denormalize_255(x_tst, EMNIST_MEAN, EMNIST_STD)
                        all_images.append(x_tst.cpu().numpy())

            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

        all_pre_mods = np.concatenate(all_pre_mods, axis=0)
        all_post_mods = np.concatenate(all_post_mods, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        all_cats = np.concatenate(all_cats, axis=0)
        print(all_pre_mods.shape)
        print(all_post_mods.shape)
        print(all_outputs.shape)
        print(all_images.shape)
        print(all_cats.shape)

    return test_loss / len(dataloader), test_acc / len(dataloader), all_pre_mods, all_post_mods, all_outputs, \
           all_images, all_cats


def test_modules(experiment, validate, dataset, data_root, ckpt_path, save_path, total_n_classes, batch_size,
                 n_workers, pin_mem, dev):
    files = os.listdir(ckpt_path)
    for f in files:
        if "es_" == f[:3]:
            raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(f))
    files = [f for f in files if f.split('_')[0] == experiment]
    module_ckpts = [f for f in files if len(f.split('-')) == 2]
    assert len(module_ckpts) == 6
    if dataset == "EMNIST":
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, "Modules",
                                                                         ["Identity"], dev)
    elif dataset == "CIFAR":
        network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, "Modules",
                                                                        ["Identity"], dev)
    elif dataset == "FACESCRUB":
        network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, "Modules",
                                                                            ["Identity"], dev)
    for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
        block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

    modules = []
    module_levels = []
    for module_ckpt in module_ckpts:
        if dataset == "EMNIST":
            all_modules, all_module_ckpt_names = create_emnist_modules(experiment,
                                                                       module_ckpt.split('_')[-1][:-3].split('-'),
                                                                       dev)
        elif dataset == "CIFAR":
            all_modules, all_module_ckpt_names = create_cifar_modules(experiment,
                                                                      module_ckpt.split('_')[-1][:-3].split('-'),
                                                                      dev)
        elif dataset == "FACESCRUB":
            all_modules, all_module_ckpt_names = create_facescrub_modules(experiment,
                                                                       module_ckpt.split('_')[-1][:-3].split('-'),
                                                                       dev)
        module_level = int(module_ckpt.split('_')[-2].split("Module")[-1])
        module_levels.append(module_level)
        modules.append(all_modules[module_level])
        modules[-1].load_state_dict(torch.load(os.path.join(ckpt_path, module_ckpt)))
        print("Loaded {}".format(module_ckpt))
        print("From {}".format(os.path.join(ckpt_path, module_ckpt)))
        print("At Abstraction Level {}".format(module_level))
        sys.stdout.flush()

    # Get deephys data for all elemental corruptions
    corruption_accs = {}
    corruption_losses = {}
    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        corruptions = pickle.load(f)
    corruptions.sort()
    corruptions = [c for c in corruptions if len(c) == 1 and c != ["Identity"]]
    assert len(corruptions) == 6

    for test_corruption in corruptions:
        print("Testing on {}".format(test_corruption))
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
        if validate:
            _, tst_dl, _ = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                              batch_size, False, n_workers, pin_mem)
        else:
            _, _, tst_dl = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                              batch_size, False, n_workers, pin_mem)

        test_modules = []
        test_module_levels = []
        for c in test_corruption:
            test_ckpt = [m for m in module_ckpts if c in m][0]
            test_modules.append(modules[module_ckpts.index(test_ckpt)])
            test_module_levels.append(module_levels[module_ckpts.index(test_ckpt)])
            print("Selected module {}".format(test_ckpt))
        tst_loss, tst_acc, all_pre_mods, all_post_mods, all_outputs, all_images, all_cats = \
            modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, tst_dl, dev)
        corruption_accs['-'.join(test_corruption)] = tst_acc
        corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, '-'.join(test_corruption), tst_loss,
                                                                   tst_acc))
        sys.stdout.flush()
        # Save numpy arrays to use deephys on a local machine
        np.savez(os.path.join(save_path, f"{dataset}-{test_corruption[0]}.npz"), all_cats=all_cats,
                 all_images=all_images / 255.0, all_pre_mods=all_pre_mods, all_post_mods=all_post_mods,
                 all_outputs=all_outputs)
        print("Saved deephys data to {}".format(os.path.join(save_path, f"{dataset}-{test_corruption[0]}.npz")))

        # Now use the same module on the Identity data
        if dataset == "EMNIST":
            transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
            transforms += [getattr(dt, "Identity")()]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
            transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
        else:
            transforms = [getattr(dt, "Identity")()]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='RGB'))]
        if validate:
            _, tst_dl, _ = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                              batch_size, False, n_workers, pin_mem)
        else:
            _, _, tst_dl = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                              batch_size, False, n_workers, pin_mem)

        tst_loss, tst_acc, all_pre_mods, all_post_mods, all_outputs, all_images, all_cats = \
            modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, tst_dl, dev)
        corruption_accs['-'.join(test_corruption)] = tst_acc
        corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, "Identity", tst_loss,
                                                                   tst_acc))
        sys.stdout.flush()
        # Save numpy arrays to use deephys on a local machine
        np.savez(os.path.join(save_path, f"{dataset}-Identity-With-{test_corruption[0]}-Module.npz"), all_cats=all_cats,
                 all_images=all_images / 255.0, all_pre_mods=all_pre_mods, all_post_mods=all_post_mods,
                 all_outputs=all_outputs)
        print("Saved deephys data to {}".format(os.path.join(save_path,
                                                f"{dataset}-Identity-With-{test_corruption[0]}-Module.npz")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str,
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str,
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str,
                        help="path to directory to save deephys data to")
    parser.add_argument('--seed', type=int, default=38164641, help="random seed")
    parser.add_argument('--validate', action='store_true', help="If set, uses the validation rather than the test set")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=256, help="batch size")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    args = parser.parse_args()

    if args.dataset not in ["EMNIST", "CIFAR", "FACESCRUB"]:
        raise ValueError("Dataset {} not implemented".format(args.dataset))

    # Set seeding
    reset_rngs(seed=args.seed, deterministic=True)

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Set up and create unmade directories
    variance_dir_name = f"seed-{args.seed}"  # f"lr-0.01_weight-1.0"
    args.data_root = os.path.join(args.data_root, args.dataset)
    args.ckpt_path = os.path.join(args.ckpt_path, args.dataset, variance_dir_name)
    args.save_path = os.path.join(args.save_path, args.dataset, variance_dir_name)
    mkdir_p(args.save_path)

    test_modules("AutoModules", args.validate, args.dataset, args.data_root, args.ckpt_path, args.save_path,
                 args.total_n_classes, args.batch_size, args.n_workers, args.pin_mem, dev)

