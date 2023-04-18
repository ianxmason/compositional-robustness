"""
Test models trained with different combinations of data on all available compositions
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision
import pickle
import data.data_transforms as dt
from data.data_loaders import get_transformed_static_dataloaders, get_static_dataloaders
from data.emnist import EMNIST_MEAN, EMNIST_STD
from data.cifar import CIFAR10_MEAN, CIFAR10_STD
from data.facescrub import FACESCRUB_MEAN, FACESCRUB_STD
from lib.networks import create_emnist_network, create_emnist_modules, create_emnist_autoencoder, \
                         create_cifar_network, create_cifar_modules, create_cifar_autoencoder, \
                         create_facescrub_network, create_facescrub_modules, create_facescrub_autoencoder
from lib.utils import *


def accumulate_activations(activations, labels, total_n_classes, class_cumsum, class_cumsum_sq, class_dpoints,
                           max_activations, min_activations, raw_activations):
    """
    Takes activations of shape (batch_size, num_units) and labels of shape (batch_size,) and
    accumulates them for later averaging.
    """
    if class_cumsum is None:
        class_cumsum = torch.zeros(total_n_classes, activations.shape[1]).to(dev)
    class_cumsum.index_add_(dim=0, index=labels, source=activations)

    if class_cumsum_sq is None:
        class_cumsum_sq = torch.zeros(total_n_classes, activations.shape[1]).to(dev)
    class_cumsum_sq.index_add_(dim=0, index=labels, source=activations ** 2)

    if class_dpoints is None:
        class_dpoints = torch.zeros(total_n_classes).to(dev)
    class_dpoints.index_add_(dim=0, index=labels, source=torch.ones(activations.shape[0]).to(dev))

    batch_maxs, _ = torch.max(activations, dim=0)  # num_units
    if max_activations is not None:
        max_activations = torch.where(batch_maxs > max_activations, batch_maxs, max_activations)
    else:
        max_activations = batch_maxs

    batch_mins, _ = torch.min(activations, dim=0)  # num_units
    if min_activations is not None:
        min_activations = torch.where(batch_mins < min_activations, batch_mins, min_activations)
    else:
        min_activations = batch_mins

    if raw_activations is not None:
        raw_activations = torch.cat((raw_activations, activations), dim=0)
    else:
        raw_activations = activations

    return class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations, raw_activations


def process_activations(class_cumsum, class_cumsum_sq, class_dpoints, min_activations):
    """
    Take activations and calculate average, max and std
    """
    # All activations should be after relu so activations are >= 0.
    # If this is not true, need to return min_activations for later normalization of activations between 0 and 1
    if not torch.min(min_activations) >= 0:
        raise ValueError("Min activations are not >= 0")

    class_avg_firings = class_cumsum / class_dpoints[:, None]

    class_std_firings = torch.sqrt(class_cumsum_sq / class_dpoints[:, None] - class_avg_firings ** 2)

    return class_avg_firings, class_std_firings


def loss_and_accuracy(network_blocks, dataloader, dev, collect_activations=False, total_n_classes=None):
    """
    Calculates average loss and accuracy of a monolithic network on the given dataloader.

    Optionally collects penultimate layer activations. total_n_classes is only required if this is True.
    """
    assert len(network_blocks) >= 2  # assumed when the network is called
    if collect_activations:
        class_cumsum, class_cumsum_sq, class_dpoints = None, None, None
        max_activations, min_activations, raw_activations = None, None, None
        assert total_n_classes is not None
    criterion = nn.CrossEntropyLoss()
    for block in network_blocks:
        block.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            for j, block in enumerate(network_blocks):
                if j == 0:
                    features = block(x_tst)
                elif j == len(network_blocks) - 1:
                    if collect_activations:
                        class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations, raw_activations\
                            = accumulate_activations(features, y_tst, total_n_classes, class_cumsum, class_cumsum_sq,
                                                     class_dpoints, max_activations, min_activations, raw_activations)
                    output = block(features)
                else:
                    features = block(features)
            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    if not collect_activations:
        return test_loss / len(dataloader), test_acc / len(dataloader)
    else:
        return test_loss / len(dataloader), test_acc / len(dataloader), \
               class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations, raw_activations


def modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, dataloader, dev,
                              collect_activations=False, total_n_classes=None):
    """
    Calculates average loss and accuracy of a modular network on the given dataloader.

    Optionally collects penultimate layer activations. total_n_classes is only required if this is True.
    """
    assert len(network_blocks) >= 2  # assumed when the network is called
    if collect_activations:
        class_cumsum, class_cumsum_sq, class_dpoints = None, None, None
        max_activations, min_activations, raw_activations = None, None, None
        assert total_n_classes is not None
    criterion = nn.CrossEntropyLoss()
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
                    for m, l in zip(test_modules, test_module_levels):
                        if l == j:
                            features = m(features)
                if j != len(network_blocks) - 1:
                    features = block(features)
                else:
                    if collect_activations:
                        class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations, raw_activations\
                            = accumulate_activations(features, y_tst, total_n_classes, class_cumsum, class_cumsum_sq,
                                                     class_dpoints, max_activations, min_activations, raw_activations)
                    output = block(features)

            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    if not collect_activations:
        return test_loss / len(dataloader), test_acc / len(dataloader)
    else:
        return test_loss / len(dataloader), test_acc / len(dataloader), \
               class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations, raw_activations


def autoencoders_loss_and_accuracy(all_ae_blocks, clsf_blocks, dataloader, dev, collect_activations=False,
                                   total_n_classes=None):
    """
    Calculates average loss and accuracy of autoencoders followed by a classifying network on the given dataloader.

    Optionally collects penultimate layer activations. total_n_classes is only required if this is True.
    """
    assert len(clsf_blocks) >= 2  # assumed when the network is called
    if collect_activations:
        class_cumsum, class_cumsum_sq, class_dpoints = None, None, None
        max_activations, min_activations, raw_activations = None, None, None
        assert total_n_classes is not None
    criterion = nn.CrossEntropyLoss()
    for ae_blocks in all_ae_blocks:
        for block in ae_blocks:
            block.eval()
    for block in clsf_blocks:
        block.eval()

    # Autoencoders are applied as follows:
    # 1. the earlier the corruption is in the test corruption name the earlier the autoencoder is applied
    test_loss = 0.0
    test_acc = 0.0
    collected_imgs = False
    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            features = x_tst

            # Visualise images before autoencoding
            if not collected_imgs:
                pre_ae_imgs = features.detach().cpu().numpy()
                pre_ae_lbls = y_tst.detach().cpu().numpy()

            # Apply autoencoders
            for ae_blocks in all_ae_blocks:
                for block in ae_blocks:
                    features = block(features)

            # Visualise images after autoencoding
            if not collected_imgs:
                post_ae_imgs = features.detach().cpu().numpy()
                post_ae_lbls = y_tst.detach().cpu().numpy()
                collected_imgs = True

            # Classify autoencoder outputs
            for j, block in enumerate(clsf_blocks):
                if j != len(clsf_blocks) - 1:
                    features = block(features)
                else:
                    if collect_activations:
                        class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations, raw_activations\
                            = accumulate_activations(features, y_tst, total_n_classes, class_cumsum, class_cumsum_sq,
                                                     class_dpoints, max_activations, min_activations, raw_activations)
                    output = block(features)

            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    if not collect_activations:
        return test_loss / len(dataloader), test_acc / len(dataloader), pre_ae_imgs, pre_ae_lbls, post_ae_imgs,\
               post_ae_lbls
    else:
        return test_loss / len(dataloader), test_acc / len(dataloader), pre_ae_imgs, pre_ae_lbls, post_ae_imgs,\
               post_ae_lbls, \
               class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations, raw_activations


def test_monolithic(experiment, validate, dataset, data_root, ckpt_path, save_path, activations_path, total_n_classes,
                    batch_size, collect_activations, n_workers, pin_mem, dev, check_if_run, total_processes, process):
    """
    Get the specific checkpoint trained on all corruptions and test on every composition

    Parallelise by testing different compositions in different processes
    """
    files = os.listdir(ckpt_path)
    for f in files:
        if "es_" == f[:3]:
            raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(f))
    files = ['_'.join(f.split('_')[2:]) for f in files if f.split('_')[0] == experiment]
    files.sort(key=lambda x: len(x.split('-')))
    ckpt = files[-1]
    assert len(ckpt.split('-')) == 7  # hardcoded for EMNIST. 8 EMNIST4. 7 EMNIST5.
    # Load the ckpt
    if check_if_run and os.path.exists(os.path.join(save_path, "{}_{}_all_losses_process_{}_of_{}.pkl".format(
                                                    experiment, ckpt[:-3], process, total_processes))):
        raise RuntimeError("Pickle file already exists at {}. \n Skipping testing for {}".format(
            os.path.join(save_path, "{}_{}_all_losses_process_{}_of_{}.pkl".format(experiment, ckpt[:-3], process,
                                                                                   total_processes)), ckpt))
    else:
        if dataset == "EMNIST":
            network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment,
                                                                             ckpt[:-3].split('-'), dev)
        elif dataset == "CIFAR":
            network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, experiment,
                                                                            ckpt[:-3].split('-'), dev)
        elif dataset == "FACESCRUB":
            network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, experiment,
                                                                                ckpt[:-3].split('-'), dev)
        for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

    # Test the model trained on all corruptions on all compositions
    corruption_accs = {}
    corruption_losses = {}

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        corruptions = pickle.load(f)
    corruptions.sort()
    assert len(corruptions) == 167  # hardcoded for EMNIST. 149 EMNIST4. 167 EMNIST5.
    assert total_processes <= len(corruptions)
    assert process < total_processes

    corruptions_per_process = len(corruptions) // total_processes
    if process == total_processes - 1:
        corruptions = corruptions[corruptions_per_process * process:]
    else:
        corruptions = corruptions[corruptions_per_process * process:corruptions_per_process * (process + 1)]

    for test_corruption in corruptions:
        print("Testing {} on {}".format(ckpt, test_corruption))
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

        if not collect_activations:
            tst_loss, tst_acc = loss_and_accuracy(network_blocks, tst_dl, dev)
        else:
            tst_loss, tst_acc, class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations,\
            raw_activations = loss_and_accuracy(network_blocks, tst_dl, dev, collect_activations, total_n_classes)
            class_avg_firings, class_std_firings = process_activations(class_cumsum, class_cumsum_sq, class_dpoints,
                                                                       min_activations)

        corruption_accs['-'.join(test_corruption)] = tst_acc
        corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, '-'.join(test_corruption), tst_loss,
                                                                   tst_acc))
        sys.stdout.flush()

    # Save the results
    with open(os.path.join(save_path, "{}_{}_all_accs_process_{}_of_{}.pkl".format(experiment, ckpt[:-3], process,
                                                                                   total_processes)), "wb") as f:
        pickle.dump(corruption_accs, f)
    with open(os.path.join(save_path, "{}_{}_all_losses_process_{}_of_{}.pkl".format(experiment, ckpt[:-3], process,
                                                                                     total_processes)), "wb") as f:
        pickle.dump(corruption_losses, f)
    if collect_activations:
        class_avg_firings = class_avg_firings.detach().cpu().numpy()
        class_std_firings = class_std_firings.detach().cpu().numpy()
        max_activations = max_activations.detach().cpu().numpy()
        raw_activations = raw_activations.detach().cpu().numpy()
        with open(os.path.join(activations_path, "{}_{}_all_activations_process_{}_of_{}.pkl".format(experiment,
                               ckpt[:-3], process, total_processes)), "wb") as f:
            pickle.dump((class_avg_firings, class_std_firings, max_activations, raw_activations), f)


def test_modules(experiment, validate, dataset, data_root, ckpt_path, save_path, activations_path, total_n_classes,
                 batch_size, collect_activations, n_workers, pin_mem, dev, check_if_run, total_processes, process):
    """
    Get the specific checkpoint trained on all corruptions and test on every composition

    Parallelise by testing different compositions in different processes
    """
    files = os.listdir(ckpt_path)
    for f in files:
        if "es_" == f[:3]:
            raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(f))
    files = [f for f in files if f.split('_')[0] == experiment]
    module_ckpts = [f for f in files if len(f.split('-')) == 2]
    assert len(module_ckpts) == 6  # hardcoded for EMNIST5.
    # Load the ckpt
    if check_if_run and os.path.exists(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(
                                                    experiment, process, total_processes))):
        raise RuntimeError("Pickle file already exists at {}. \n Skipping testing.".format(
            os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment, process, total_processes))))
    else:
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

    # Test the model on all compositions
    corruption_accs = {}
    corruption_losses = {}

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        corruptions = pickle.load(f)
    corruptions.sort()
    assert len(corruptions) == 167
    assert total_processes <= len(corruptions)
    assert process < total_processes

    corruptions_per_process = len(corruptions) // total_processes
    if process == total_processes - 1:
        corruptions = corruptions[corruptions_per_process * process:]
    else:
        corruptions = corruptions[corruptions_per_process * process:corruptions_per_process * (process + 1)]

    for test_corruption in corruptions:
        print("Testing on {}".format(test_corruption))
        sys.stdout.flush()
        trained_classes = list(range(total_n_classes))

        # _, _, tst_dl = get_static_dataloaders(dataset,os.path.join(data_root, "Contrast"), trained_classes, batch_size,
        #                                       False, n_workers, pin_mem)
        # vis_path = '/om2/user/imason/compositions/figs/EMNIST_TEMP/'
        # x, y = next(iter(tst_dl))
        # fig_name = "{}_old.png".format('-'.join(test_corruption))
        # fig_path = os.path.join(vis_path, fig_name)
        # # Denormalise Images
        # x = x.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        # x = dt.denormalize_255(x, np.array(CIFAR10_MEAN).astype(np.float32), np.array(CIFAR10_STD).astype(np.float32)).astype(np.uint8)
        # # And visualise
        # visualise_data(x[:25], y[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)

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

        # vis_path = '/om2/user/imason/compositions/figs/EMNIST_TEMP/'
        # x, y = next(iter(tst_dl))
        # fig_name = "{}_new.png".format('-'.join(test_corruption))
        # fig_path = os.path.join(vis_path, fig_name)
        # # Denormalise Images
        # x = x.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        # x = dt.denormalize_255(x, np.array(CIFAR10_MEAN).astype(np.float32), np.array(CIFAR10_STD).astype(np.float32)).astype(np.uint8)
        # # And visualise
        # visualise_data(x[:25], y[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)


        test_modules = []
        test_module_levels = []
        for c in test_corruption:
            if c == "Identity":
                continue
            else:
                test_ckpt = [m for m in module_ckpts if c in m][0]
                test_modules.append(modules[module_ckpts.index(test_ckpt)])
                test_module_levels.append(module_levels[module_ckpts.index(test_ckpt)])
                print("Selected module {}".format(test_ckpt))
        if not collect_activations:
            tst_loss, tst_acc = modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, tst_dl, dev)
        else:
            tst_loss, tst_acc, class_cumsum, class_cumsum_sq, class_dpoints, max_activations, min_activations,\
            raw_activations = modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, tst_dl, dev,
                                                        collect_activations, total_n_classes)
            class_avg_firings, class_std_firings = process_activations(class_cumsum, class_cumsum_sq, class_dpoints,
                                                                       min_activations)
        corruption_accs['-'.join(test_corruption)] = tst_acc
        corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, '-'.join(test_corruption), tst_loss,
                                                                   tst_acc))
        sys.stdout.flush()

    # Save the results
    with open(os.path.join(save_path, "{}_all_accs_process_{}_of_{}.pkl".format(experiment, process,
                                                                                   total_processes)), "wb") as f:
        pickle.dump(corruption_accs, f)
    with open(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment, process,
                                                                                  total_processes)), "wb") as f:
        pickle.dump(corruption_losses, f)
    if collect_activations:
        class_avg_firings = class_avg_firings.detach().cpu().numpy()
        class_std_firings = class_std_firings.detach().cpu().numpy()
        max_activations = max_activations.detach().cpu().numpy()
        raw_activations = raw_activations.detach().cpu().numpy()
        with open(os.path.join(activations_path, "{}_all_activations_process_{}_of_{}.pkl".format(experiment,
                               process, total_processes)), "wb") as f:
            pickle.dump((class_avg_firings, class_std_firings, max_activations, raw_activations), f)


def test_autoencoders(experiment, validate, dataset, data_root, ckpt_path, save_path, activations_path, vis_path,
                      total_n_classes, batch_size, collect_activations, n_workers, pin_mem, dev, check_if_run,
                      total_processes, process):
    """
    Get the specific checkpoint trained on all corruptions and test on every composition

    Parallelise by testing different compositions in different processes
    """
    files = os.listdir(ckpt_path)
    for f in files:
        if "es_" == f[:3]:
            raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(f))
    files = [f for f in files if f.split('_')[0] == experiment]

    ae_ckpts = [f for f in files if len(f.split('-')) == 2]
    ae_corrs = list(set([f.split('_')[-1][:-3] for f in ae_ckpts]))
    ae_corrs.sort()
    assert len(ae_corrs) == 6  # hardcoded for EMNIST5. Encoder and Decoder.

    # Load the ckpt
    if check_if_run and os.path.exists(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(
                                                    experiment + "IdentityClassifier", process, total_processes))):
        raise RuntimeError("Pickle file already exists at {}. \n Skipping testing.".format(
            os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment, process, total_processes))))
    else:
        # Load the autoencoders
        ae_blocks = []
        ae_block_ckpt_names = []
        for corr in ae_corrs:
            if dataset == "EMNIST":
                blocks, block_ckpt_names = create_emnist_autoencoder(experiment, corr.split('-'), dev)
            elif dataset == "CIFAR":
                blocks, block_ckpt_names = create_cifar_autoencoder(experiment, corr.split('-'), dev)
            elif dataset == "FACESCRUB":
                blocks, block_ckpt_names = create_facescrub_autoencoder(experiment, corr.split('-'), dev)
            ae_blocks.append(blocks)
            ae_block_ckpt_names.append(block_ckpt_names)
        for blocks, block_ckpt_names in zip(ae_blocks, ae_block_ckpt_names):
            for block, block_ckpt_name in zip(blocks, block_ckpt_names):
                block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
                print("Loaded {}".format(block_ckpt_name))
                print("From {}".format(os.path.join(ckpt_path, block_ckpt_name)))
                sys.stdout.flush()

        # Load the identity trained classifier
        if dataset == "EMNIST":
            id_clsf_blocks, id_clsf_block_ckpt_names = create_emnist_network(total_n_classes, experiment + "Classifier",
                                                                             ["Identity"], dev)
            denorm_mean, denorm_std = EMNIST_MEAN, EMNIST_STD
        elif dataset == "CIFAR":
            id_clsf_blocks, id_clsf_block_ckpt_names = create_cifar_network(total_n_classes, experiment + "Classifier",
                                                                             ["Identity"], dev)
            denorm_mean, denorm_std = CIFAR10_MEAN, CIFAR10_STD
        elif dataset == "FACESCRUB":
            id_clsf_blocks, id_clsf_block_ckpt_names = create_facescrub_network(total_n_classes,
                                                                                experiment + "Classifier",
                                                                                ["Identity"], dev)
            denorm_mean, denorm_std = FACESCRUB_MEAN, FACESCRUB_STD
        for block, block_ckpt_name in zip(id_clsf_blocks, id_clsf_block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
            print("Loaded {}".format(block_ckpt_name))
            print("From {}".format(os.path.join(ckpt_path, block_ckpt_name)))

        # Load the classifier trained on autoencoder outputs
        if dataset == "EMNIST":
            all_clsf_blocks, all_clsf_block_ckpt_names = create_emnist_network(total_n_classes,
                                                                               experiment + "Classifier",
                                                                               [corr.split('-')[0] for corr in
                                                                                ae_corrs] + ["Identity"], dev)
        elif dataset == "CIFAR":
            all_clsf_blocks, all_clsf_block_ckpt_names = create_cifar_network(total_n_classes,
                                                                              experiment + "Classifier",
                                                                              [corr.split('-')[0] for corr in
                                                                               ae_corrs] + ["Identity"], dev)
        elif dataset == "FACESCRUB":
            all_clsf_blocks, all_clsf_block_ckpt_names = create_facescrub_network(total_n_classes,
                                                                                  experiment + "Classifier",
                                                                                  [corr.split('-')[0] for corr in
                                                                                   ae_corrs] + ["Identity"], dev)
        for block, block_ckpt_name in zip(all_clsf_blocks, all_clsf_block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
            print("Loaded {}".format(block_ckpt_name))
            print("From {}".format(os.path.join(ckpt_path, block_ckpt_name)))

    # Test the model on all compositions
    id_clsf_corruption_accs = {}
    id_clsf_corruption_losses = {}
    all_clsf_corruption_accs = {}
    all_clsf_corruption_losses = {}

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        corruptions = pickle.load(f)
    corruptions.sort()
    assert len(corruptions) == 167  # hardcoded for EMNIST. 149 EMNIST4. 167 EMNIST5.
    assert total_processes <= len(corruptions)
    assert process < total_processes

    corruptions_per_process = len(corruptions) // total_processes
    if process == total_processes - 1:
        corruptions = corruptions[corruptions_per_process * process:]
    else:
        corruptions = corruptions[corruptions_per_process * process:corruptions_per_process * (process + 1)]

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

        test_ae_blocks = []
        for c in test_corruption:
            if c == "Identity":
                continue
            else:
                for blocks, block_ckpt_names in zip(ae_blocks, ae_block_ckpt_names):
                    if c in block_ckpt_names[0]:
                        print("Selected autoencoder {}".format(block_ckpt_names))
                        test_ae_blocks.append(blocks)

        if not collect_activations:
            tst_loss, tst_acc, pre_ae_imgs, pre_ae_lbls, post_ae_imgs, post_ae_lbls = \
                autoencoders_loss_and_accuracy(test_ae_blocks, id_clsf_blocks, tst_dl, dev)
        else:
            tst_loss, tst_acc, pre_ae_imgs, pre_ae_lbls, post_ae_imgs, post_ae_lbls, class_cumsum, class_cumsum_sq, \
            class_dpoints, id_max_activations, min_activations, id_raw_activations = \
                autoencoders_loss_and_accuracy(test_ae_blocks, id_clsf_blocks, tst_dl, dev, collect_activations,
                                               total_n_classes)
            id_class_avg_firings, id_class_std_firings = process_activations(class_cumsum, class_cumsum_sq,
                                                                             class_dpoints, min_activations)
        id_clsf_corruption_accs['-'.join(test_corruption)] = tst_acc
        id_clsf_corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{} Identity Classifier, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment,
                                                                                       '-'.join(test_corruption),
                                                                                       tst_loss, tst_acc))
        sys.stdout.flush()

        if not collect_activations:
            tst_loss, tst_acc, _, _, _, _ = autoencoders_loss_and_accuracy(test_ae_blocks, all_clsf_blocks, tst_dl, dev)
        else:
            tst_loss, tst_acc, _, _, _, _, class_cumsum, class_cumsum_sq, class_dpoints, all_max_activations, \
            min_activations, all_raw_activations = autoencoders_loss_and_accuracy(test_ae_blocks, all_clsf_blocks,
                                                                                  tst_dl, dev, collect_activations,
                                                                                  total_n_classes)
            all_class_avg_firings, all_class_std_firings = process_activations(class_cumsum, class_cumsum_sq,
                                                                               class_dpoints, min_activations)
        all_clsf_corruption_accs['-'.join(test_corruption)] = tst_acc
        all_clsf_corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{} Joint Classifier, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment,
                                                                                    '-'.join(test_corruption),
                                                                                    tst_loss, tst_acc))
        sys.stdout.flush()

        # Visualise the autoencoder input and output
        fig_name = "before_ae_{}.png".format('-'.join(test_corruption))
        fig_path = os.path.join(vis_path, fig_name)
        pre_ae_imgs = dt.denormalize_255(pre_ae_imgs, np.array(denorm_mean).astype(np.float32),
                                         np.array(denorm_std).astype(np.float32)).astype(np.uint8)
        visualise_data(pre_ae_imgs[:25], pre_ae_lbls[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)

        fig_name = "after_ae_{}.png".format('-'.join(test_corruption))
        fig_path = os.path.join(vis_path, fig_name)
        post_ae_imgs = dt.denormalize_255(post_ae_imgs, np.array(denorm_mean).astype(np.float32),
                                          np.array(denorm_std).astype(np.float32)).astype(np.uint8)
        visualise_data(post_ae_imgs[:25], post_ae_lbls[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5,
                       n_cols=5)

    # Save the results
    with open(os.path.join(save_path, "{}_all_accs_process_{}_of_{}.pkl".format(experiment + "IdentityClassifier",
                                                                                process, total_processes)), "wb") as f:
        pickle.dump(id_clsf_corruption_accs, f)
    with open(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment + "IdentityClassifier",
                                                                                  process, total_processes)), "wb") as f:
        pickle.dump(id_clsf_corruption_losses, f)
    with open(os.path.join(save_path, "{}_all_accs_process_{}_of_{}.pkl".format(experiment + "JointClassifier",
                                                                                process, total_processes)), "wb") as f:
        pickle.dump(all_clsf_corruption_accs, f)
    with open(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment + "JointClassifier",
                                                                                  process, total_processes)), "wb") as f:
        pickle.dump(all_clsf_corruption_losses, f)
    if collect_activations:
        id_class_avg_firings = id_class_avg_firings.detach().cpu().numpy()
        id_class_std_firings = id_class_std_firings.detach().cpu().numpy()
        id_max_activations = id_max_activations.detach().cpu().numpy()
        id_raw_activations = id_raw_activations.detach().cpu().numpy()
        with open(os.path.join(activations_path, "{}_all_activations_process_{}_of_{}.pkl".format(
                               experiment + "IdentityClassifier", process, total_processes)), "wb") as f:
            pickle.dump((id_class_avg_firings, id_class_std_firings, id_max_activations, id_raw_activations), f)

        all_class_avg_firings = all_class_avg_firings.detach().cpu().numpy()
        all_class_std_firings = all_class_std_firings.detach().cpu().numpy()
        all_max_activations = all_max_activations.detach().cpu().numpy()
        all_raw_activations = all_raw_activations.detach().cpu().numpy()
        with open(os.path.join(activations_path, "{}_all_activations_process_{}_of_{}.pkl".format(
                                experiment + "JointClassifier", process, total_processes)), "wb") as f:
            pickle.dump((all_class_avg_firings, all_class_std_firings, all_max_activations, all_raw_activations), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--activations-path', type=str, default='/om2/user/imason/compositions/activations/',
                        help="path to directory to save neural activations")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/',
                        help="path to directory to save autoencoder visualisations")
    parser.add_argument('--experiment', type=str, default='CrossEntropy',
                        help="which method to use. CrossEntropy or Contrastive or Modules.")
    parser.add_argument('--validate', action='store_true', help="If set, uses the validation rather than the test set")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=256, help="batch size")
    parser.add_argument('--collect-activations', action='store_true', help="Collects penultimate layer activations")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    parser.add_argument('--num-processes', type=int, default=20, help="total processes to split into = SLURM_ARRAY_TASK_COUNT")
    parser.add_argument('--process', type=int, default=0, help="which process is running = SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()

    if args.dataset not in ["EMNIST", "CIFAR", "FACESCRUB"]:
        raise ValueError("Dataset {} not implemented".format(args.dataset))

    # Set seeding
    seed = 48121620
    reset_rngs(seed=seed, deterministic=True)

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Set up and create unmade directories
    variance_dir_name = f"lr-0.01_weight-1.0"  # f"seed-{seed}"
    args.data_root = os.path.join(args.data_root, args.dataset)
    args.ckpt_path = os.path.join(args.ckpt_path, args.dataset, variance_dir_name)
    args.save_path = os.path.join(args.save_path, args.dataset, variance_dir_name)
    args.activations_path = os.path.join(args.activations_path, args.dataset, variance_dir_name)
    args.vis_path = os.path.join(args.vis_path, args.dataset, "autoencoder_visualisations", variance_dir_name)
    mkdir_p(args.save_path)
    mkdir_p(args.activations_path)
    if "ImgSpace" in args.experiment:
        mkdir_p(args.vis_path)

    print("Running process {} of {}".format(args.process + 1, args.num_processes))
    sys.stdout.flush()

    if "Modules" in args.experiment:
        test_modules(args.experiment, args.validate, args.dataset, args.data_root, args.ckpt_path, args.save_path,
                     args.activations_path, args.total_n_classes, args.batch_size, args.collect_activations,
                     args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes, args.process)
    elif "ImgSpace" in args.experiment:
        test_autoencoders(args.experiment, args.validate, args.dataset, args.data_root, args.ckpt_path, args.save_path,
                          args.activations_path, args.vis_path, args.total_n_classes, args.batch_size,
                          args.collect_activations, args.n_workers, args.pin_mem, dev, args.check_if_run,
                          args.num_processes, args.process)
    else:
        test_monolithic(args.experiment, args.validate, args.dataset, args.data_root, args.ckpt_path, args.save_path,
                        args.activations_path, args.total_n_classes, args.batch_size, args.collect_activations,
                        args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes, args.process)
