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
from data.data_loaders import get_static_emnist_dataloaders, get_transformed_static_emnist_dataloaders
from lib.networks import create_emnist_network, create_emnist_modules, create_emnist_autoencoder
from lib.utils import *
from lib.equivariant_hooks import *


def loss_and_accuracy(network_blocks, dataloader, dev):
    assert len(network_blocks) >= 2  # assumed when the network is called
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
                    output = block(features)
                else:
                    features = block(features)
            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, dataloader, dev):
    assert len(network_blocks) >= 2  # assumed when the network is called
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
                    output = block(features)

            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def autoencoders_loss_and_accuracy(all_ae_blocks, clsf_blocks, dataloader, dev):
    assert len(clsf_blocks) >= 2  # assumed when the network is called
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
    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            features = x_tst

            # Sanity check to comment in for visualising before and after autoencoding
            # vis_path = '/om2/user/imason/compositions/figs/EMNIST5/visualisations/'
            # fig_name = "before_ae.png"
            # fig_path = os.path.join(vis_path, fig_name)
            # # Denormalise Images
            # x = features.detach().cpu().numpy()
            # y = y_tst.detach().cpu().numpy()
            # x = denormalize(x).astype(np.uint8)
            # # And visualise
            # visualise_data(x[:25], y[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)

            for ae_blocks in all_ae_blocks:
                for block in ae_blocks:
                    features = block(features)

            # vis_path = '/om2/user/imason/compositions/figs/EMNIST5/visualisations/'
            # fig_name = "after_ae.png"
            # fig_path = os.path.join(vis_path, fig_name)
            # # Denormalise Images
            # x = features.detach().cpu().numpy()
            # y = y_tst.detach().cpu().numpy()
            # x = denormalize(x).astype(np.uint8)
            # # And visualise
            # visualise_data(x[:25], y[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)
            # print(2/0)  # break execution - only need one batch to check

            for j, block in enumerate(clsf_blocks):
                if j != len(clsf_blocks) - 1:
                    features = block(features)
                else:
                    output = block(features)

            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def test_specific(experiment, data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev,
                  check_if_run, total_processes, process):
    """
    Take every checkpoint and test on the composition of the corruptions trained on

    Parallelise by testing different checkpoints in different processes
    """
    files = os.listdir(ckpt_path)
    for f in files:
        if "es_" == f[:3]:
            raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(f))
    # First select only the files that use the correct experiment and remove the experiment name and network block name
    files = ['_'.join(f.split('_')[2:]) for f in files if f.split('_')[0] == experiment]
    # Then remove duplicates that come from the network blocks
    files = list(set(files))
    files.sort()
    assert len(files) == 64  # hardcoded for EMNIST. 128 EMNIST4. 64 EMNIST5.
    assert total_processes <= len(files)
    assert process < total_processes

    files_per_process = len(files) // total_processes
    if process == total_processes - 1:
        files = files[files_per_process * process:]
    else:
        files = files[files_per_process * process:files_per_process * (process + 1)]

    for ckpt in files:
        # Load each ckpt
        if check_if_run and os.path.exists(os.path.join(save_path, "{}_{}_losses.pkl".format(experiment, ckpt[:-3]))):
            print("Pickle file already exists at {}. \n Skipping testing for {}".format(
                os.path.join(save_path, "{}_{}_losses.pkl".format(experiment, ckpt[:-3])), ckpt))
            sys.stdout.flush()
            continue
        else:
            network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment,
                                                                             ckpt[:-3].split('-'), dev)
            for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
                block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

        # Test the trained model on all permutations of the composition of the corruptions it was trained on
        corruptions = ckpt[:-3].split('-')
        if len(corruptions) > 1:
            corruptions.remove('Identity')
        corruption_accs = {}
        corruption_losses = {}
        for test_corruption in os.listdir(data_root):
            if test_corruption == "raw" or test_corruption == "corruption_names.pkl":
                continue
            if set(test_corruption.split('-')) != set(corruptions):
                continue

            print("Testing {} on composition {}".format(ckpt, test_corruption))
            sys.stdout.flush()
            trained_classes = list(range(total_n_classes))

            # Old Version
            # corruption_path = os.path.join(data_root, test_corruption)
            # _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
            #                                              n_workers, pin_mem)

            identity_path = os.path.join(data_root, "Identity")
            transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
            transforms += [getattr(dt, c)() for c in test_corruption.split('-')]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
            transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
            _, _, tst_dl = get_transformed_static_emnist_dataloaders(identity_path, transforms, trained_classes,
                                                                     batch_size, False, n_workers, pin_mem)

            tst_loss, tst_acc = loss_and_accuracy(network_blocks, tst_dl, dev)
            corruption_accs[test_corruption] = tst_acc
            corruption_losses[test_corruption] = tst_loss
            print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, test_corruption, tst_loss, tst_acc))
            sys.stdout.flush()

        # Save the results
        with open(os.path.join(save_path, "{}_{}_accs.pkl".format(experiment, ckpt[:-3])), "wb") as f:
            pickle.dump(corruption_accs, f)
        with open(os.path.join(save_path, "{}_{}_losses.pkl".format(experiment, ckpt[:-3])), "wb") as f:
            pickle.dump(corruption_losses, f)


def test_all(experiment, data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev,
             check_if_run, total_processes, process):
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
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment,
                                                                         ckpt[:-3].split('-'), dev)
        for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

    # Test the model trained on all corruptions on all compositions
    corruption_accs = {}
    corruption_losses = {}

    corruptions = os.listdir(data_root)
    corruptions = [c for c in corruptions if c != "raw" and c != "corruption_names.pkl"]
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

        # Old Version
        # corruption_path = os.path.join(data_root, test_corruption)
        # _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
        #                                              n_workers, pin_mem)

        identity_path = os.path.join(data_root, "Identity")
        transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
        transforms += [getattr(dt, c)() for c in test_corruption.split('-')]
        transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
        transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
        _, _, tst_dl = get_transformed_static_emnist_dataloaders(identity_path, transforms, trained_classes, batch_size,
                                                                 False, n_workers, pin_mem)

        tst_loss, tst_acc = loss_and_accuracy(network_blocks, tst_dl, dev)
        corruption_accs[test_corruption] = tst_acc
        corruption_losses[test_corruption] = tst_loss
        print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, test_corruption, tst_loss, tst_acc))
        sys.stdout.flush()

    # Save the results
    with open(os.path.join(save_path, "{}_{}_all_accs_process_{}_of_{}.pkl".format(experiment, ckpt[:-3], process,
                                                                                   total_processes)), "wb") as f:
        pickle.dump(corruption_accs, f)
    with open(os.path.join(save_path, "{}_{}_all_losses_process_{}_of_{}.pkl".format(experiment, ckpt[:-3], process,
                                                                                     total_processes)), "wb") as f:
        pickle.dump(corruption_losses, f)


def test_modules(experiment, data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev,
                 check_if_run, total_processes, process):
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
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, "Modules",
                                                                         ["Identity"], dev)
        for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

        modules = []
        module_levels = []
        for module_ckpt in module_ckpts:
            all_modules, all_module_ckpt_names = create_emnist_modules(experiment,
                                                                       module_ckpt.split('_')[-1][:-3].split('-'), dev)
            module_level = int(module_ckpt.split('_')[-2][-1])

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

    corruptions = os.listdir(data_root)
    corruptions = [c for c in corruptions if c != "raw" and c != "corruption_names.pkl"]
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

        # Old Version
        # corruption_path = os.path.join(data_root, test_corruption)
        # _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
        #                                              n_workers, pin_mem)

        # Todo: will need to check again the visualisation with RGB images as we use mode 'L' for grayscale for EMNIST
        # vis_path = '/om2/user/imason/compositions/figs/EMNIST_TEMP/'
        # x, y = next(iter(tst_dl))
        # fig_name = "{}_old.png".format(test_corruption)
        # fig_path = os.path.join(vis_path, fig_name)
        # # Denormalise Images
        # x = x.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        # x = dt.denormalize(x).astype(np.uint8)
        # # And visualise
        # visualise_data(x[:25], y[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)

        identity_path = os.path.join(data_root, "Identity")
        transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
        transforms += [getattr(dt, c)() for c in test_corruption.split('-')]
        transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
        transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
        _, _, tst_dl = get_transformed_static_emnist_dataloaders(identity_path, transforms, trained_classes, batch_size,
                                                                 False, n_workers, pin_mem)

        # Todo: will need to check again the visualisation with RGB images as we use mode 'L' for grayscale for EMNIST
        # vis_path = '/om2/user/imason/compositions/figs/EMNIST_TEMP/'
        # x, y = next(iter(tst_dl))
        # fig_name = "{}_new.png".format(test_corruption)
        # fig_path = os.path.join(vis_path, fig_name)
        # # Denormalise Images
        # x = x.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        # x = dt.denormalize(x).astype(np.uint8)
        # # And visualise
        # visualise_data(x[:25], y[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)

        test_modules = []
        test_module_levels = []
        for c in test_corruption.split('-'):
            if c == "Identity":
                continue
            else:
                test_ckpt = [m for m in module_ckpts if c in m][0]
                test_modules.append(modules[module_ckpts.index(test_ckpt)])
                test_module_levels.append(module_levels[module_ckpts.index(test_ckpt)])
                print("Selected module {}".format(test_ckpt))
        tst_loss, tst_acc = modules_loss_and_accuracy(network_blocks, test_modules, test_module_levels, tst_dl, dev)
        corruption_accs[test_corruption] = tst_acc
        corruption_losses[test_corruption] = tst_loss
        print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, test_corruption, tst_loss, tst_acc))
        sys.stdout.flush()

    # Save the results
    with open(os.path.join(save_path, "{}_all_accs_process_{}_of_{}.pkl".format(experiment, process,
                                                                                   total_processes)), "wb") as f:
        pickle.dump(corruption_accs, f)
    with open(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment, process,
                                                                                     total_processes)), "wb") as f:
        pickle.dump(corruption_losses, f)


def test_autoencoders(experiment, data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev,
                 check_if_run, total_processes, process):
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
            blocks, block_ckpt_names = create_emnist_autoencoder(experiment, corr.split('-'), dev)
            ae_blocks.append(blocks)
            ae_block_ckpt_names.append(block_ckpt_names)
        for blocks, block_ckpt_names in zip(ae_blocks, ae_block_ckpt_names):
            for block, block_ckpt_name in zip(blocks, block_ckpt_names):
                block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
                print("Loaded {}".format(block_ckpt_name))
                print("From {}".format(os.path.join(ckpt_path, block_ckpt_name)))
                sys.stdout.flush()

        # Load the identity trained classifier
        id_clsf_blocks, id_clsf_block_ckpt_names = create_emnist_network(total_n_classes, experiment + "Classifier",
                                                                         ["Identity"], dev)
        for block, block_ckpt_name in zip(id_clsf_blocks, id_clsf_block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
            print("Loaded {}".format(block_ckpt_name))
            print("From {}".format(os.path.join(ckpt_path, block_ckpt_name)))

        # Load the classifier trained on autoencoder outputs
        all_clsf_blocks, all_clsf_block_ckpt_names = create_emnist_network(total_n_classes, experiment + "Classifier",
                                                        [corr.split('-')[0] for corr in ae_corrs] + ["Identity"], dev)
        for block, block_ckpt_name in zip(all_clsf_blocks, all_clsf_block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
            print("Loaded {}".format(block_ckpt_name))
            print("From {}".format(os.path.join(ckpt_path, block_ckpt_name)))

    # Test the model on all compositions
    id_clsf_corruption_accs = {}
    id_clsf_corruption_losses = {}
    all_clsf_corruption_accs = {}
    all_clsf_corruption_losses = {}

    corruptions = os.listdir(data_root)
    corruptions = [c for c in corruptions if c != "raw" and c != "corruption_names.pkl"]
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

        # Old Version
        # corruption_path = os.path.join(data_root, test_corruption)
        # _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
        #                                              n_workers, pin_mem)

        identity_path = os.path.join(data_root, "Identity")
        transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
        transforms += [getattr(dt, c)() for c in test_corruption.split('-')]
        transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
        transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
        _, _, tst_dl = get_transformed_static_emnist_dataloaders(identity_path, transforms, trained_classes, batch_size,
                                                                 False, n_workers, pin_mem)

        test_ae_blocks = []
        for c in test_corruption.split('-'):
            if c == "Identity":
                continue
            else:
                for blocks, block_ckpt_names in zip(ae_blocks, ae_block_ckpt_names):
                    if c in block_ckpt_names[0]:
                        print("Selected autoencoder {}".format(block_ckpt_names))
                        test_ae_blocks.append(blocks)

        tst_loss, tst_acc = autoencoders_loss_and_accuracy(test_ae_blocks, id_clsf_blocks, tst_dl, dev)
        id_clsf_corruption_accs[test_corruption] = tst_acc
        id_clsf_corruption_losses[test_corruption] = tst_loss
        print("{} Identity Classifier, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, test_corruption,
                                                                                       tst_loss, tst_acc))
        sys.stdout.flush()

        tst_loss, tst_acc = autoencoders_loss_and_accuracy(test_ae_blocks, all_clsf_blocks, tst_dl, dev)
        all_clsf_corruption_accs[test_corruption] = tst_acc
        all_clsf_corruption_losses[test_corruption] = tst_loss
        print("{} Joint Classifier, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, test_corruption,
                                                                                       tst_loss, tst_acc))
        sys.stdout.flush()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST5/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST5/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/EMNIST5/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--experiment', type=str, default='CrossEntropy',
                        help="which method to use. CrossEntropy or Contrastive or Modules.")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=256, help="batch size")
    parser.add_argument('--test-all', action='store_true', help="if true tests the model trained on all corruptions on"
                                                                " all compositions, if false tests the model trained on"
                                                                " specific corruptions on the available compositions of"
                                                                " those corruptions")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    parser.add_argument('--num-processes', type=int, default=20, help="total processes to split into = SLURM_ARRAY_TASK_COUNT")
    parser.add_argument('--process', type=int, default=0, help="which process is running = SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()

    # Set seeding
    reset_rngs(seed=246810, deterministic=True)

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Create unmade directories
    mkdir_p(args.save_path)

    print("Running process {} of {}".format(args.process + 1, args.num_processes))
    sys.stdout.flush()

    """
    If running on polestar
    CUDA_VISIBLE_DEVICES=4 python test_emnist.py --pin-mem --check-if-run --num-processes 10 --process 0
    """

    if "Modules" in args.experiment:
        test_modules(args.experiment, args.data_root, args.ckpt_path, args.save_path, args.total_n_classes,
                     args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes,
                     args.process)
    elif "ImgSpace" in args.experiment:
        test_autoencoders(args.experiment, args.data_root, args.ckpt_path, args.save_path, args.total_n_classes,
                          args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes,
                          args.process)
    elif args.test_all:
        test_all(args.experiment, args.data_root, args.ckpt_path, args.save_path, args.total_n_classes, args.batch_size,
                 args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes, args.process)
    else:
        test_specific(args.experiment, args.data_root, args.ckpt_path, args.save_path, args.total_n_classes,
                      args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes,
                      args.process)

