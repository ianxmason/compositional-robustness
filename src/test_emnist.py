"""
Test models trained with different combinations of data on all available compositions
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import pickle
from data.data_loaders import get_static_emnist_dataloaders
from lib.networks import SimpleConvBlock, SimpleFullyConnectedBlock, SimpleClassifier
from lib.utils import *
from lib.equivariant_hooks import *


def create_and_load_network_blocks(ckpt_path, ckpt_name, total_n_classes, dev):
    network_blocks = []
    network_block_ckpt_names = []

    network_blocks.append(
        nn.Sequential(
            SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.1)  # 0.1
        ).to(dev)
    )
    network_block_ckpt_names.append("ConvBlock1_{}".format(ckpt_name))

    network_blocks.append(
        nn.Sequential(
            SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.3)  # 0.3
        ).to(dev)
    )
    network_block_ckpt_names.append("ConvBlock2_{}".format(ckpt_name))

    network_blocks.append(
        nn.Sequential(
            SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5)  # 0.5
        ).to(dev)
    )
    network_block_ckpt_names.append("ConvBlock3_{}".format(ckpt_name))

    network_blocks.append(
        nn.Sequential(
            nn.Flatten(),  # Flattens everything except the batch dimension by default
            SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=False, dropout=0.5)
        ).to(dev)
    )
    network_block_ckpt_names.append("FullyConnected_{}".format(ckpt_name))

    network_blocks.append(
        nn.Sequential(
            SimpleClassifier(512, total_n_classes)
        ).to(dev)
    )
    network_block_ckpt_names.append("Classifier_{}".format(ckpt_name))

    assert len(network_blocks) == len(network_block_ckpt_names)
    for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
        block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

    return network_blocks


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


def test_specific(data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev, check_if_run,
                  total_processes, process):
    """
    Take every checkpoint and test on the composition of the corruptions trained on

    Parallelise by testing different checkpoints in different processes
    """
    files = os.listdir(ckpt_path)
    files = list(set(['_'.join(f.split('_')[1:]) for f in files]))  # removes duplicated corruptions
    files.sort()
    assert len(files) == 128  # hardcoded for EMNIST
    assert total_processes <= len(files)
    assert process < total_processes

    files_per_process = len(files) // total_processes
    if process == total_processes - 1:
        files = files[files_per_process * process:]
    else:
        files = files[files_per_process * process:files_per_process * (process + 1)]

    for ckpt in files:
        # Load each ckpt
        if check_if_run and os.path.exists(os.path.join(save_path, "{}_losses.pkl".format(ckpt[:-3]))):
            print("Pickle file already exists at {}. \n Skipping testing for {}".format(
                os.path.join(save_path, "{}_losses.pkl".format(ckpt[:-3])), ckpt))
            sys.stdout.flush()
            continue
        if "ckpt" == ckpt[:4]:  # don't want to test early stopping ckpts
            print("Skipped early stopping ckpt es_{}".format(ckpt))
            sys.stdout.flush()
            continue
        else:
            network_blocks = create_and_load_network_blocks(ckpt_path, ckpt, total_n_classes, dev)

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

            corruption_path = os.path.join(data_root, test_corruption)
            trained_classes = list(range(total_n_classes))
            # Shuffle=False should give identical results for symmetric shifts
            _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
                                                         n_workers, pin_mem)
            tst_loss, tst_acc = loss_and_accuracy(network_blocks, tst_dl, dev)
            corruption_accs[test_corruption] = tst_acc
            corruption_losses[test_corruption] = tst_loss
            print("{}. test loss: {:.4f}, test acc: {:.4f}".format(test_corruption, tst_loss, tst_acc))
            sys.stdout.flush()

        # Save the results
        with open(os.path.join(save_path, "{}_accs.pkl".format(ckpt[:-3])), "wb") as f:
            pickle.dump(corruption_accs, f)
        with open(os.path.join(save_path, "{}_losses.pkl".format(ckpt[:-3])), "wb") as f:
            pickle.dump(corruption_losses, f)


def test_all(data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev, check_if_run,
             total_processes, process):
    """
    Get the specific checkpoint trained on all corruptions and test on every composition

    Parallelise by testing different compositions in different processes
    """
    files = os.listdir(ckpt_path)
    files.sort(key=lambda x: len(x.split('-')))
    ckpt = files[-1]
    if "es_" == ckpt[:3]:
        raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(ckpt))
    assert len(ckpt.split('-')) == 8  # hardcoded for EMNIST
    ckpt = ckpt.split('_')[1]  # remove the block name at the start

    # Load the ckpt
    if check_if_run and os.path.exists(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(ckpt[:-3],
                                                    process, total_processes))):
        raise RuntimeError("Pickle file already exists at {}. \n Skipping testing for {}".format(
            os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(ckpt[:-3], process, total_processes)),
            ckpt))
    else:
        network_blocks = create_and_load_network_blocks(ckpt_path, ckpt, total_n_classes, dev)

    # Test the model trained on all corruptions on all compositions
    corruption_accs = {}
    corruption_losses = {}

    corruptions = os.listdir(data_root)
    corruptions = [c for c in corruptions if c != "raw" and c != "corruption_names.pkl"]
    corruptions.sort()
    assert len(corruptions) == 149  # hardcoded for EMNIST
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
        corruption_path = os.path.join(data_root, test_corruption)
        trained_classes = list(range(total_n_classes))
        # Shuffle=False should give identical results for symmetric shifts
        _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
                                                     n_workers, pin_mem)
        tst_loss, tst_acc = loss_and_accuracy(network_blocks, tst_dl, dev)
        corruption_accs[test_corruption] = tst_acc
        corruption_losses[test_corruption] = tst_loss
        print("{}. test loss: {:.4f}, test acc: {:.4f}".format(test_corruption, tst_loss, tst_acc))
        sys.stdout.flush()

    # Save the results
    with open(os.path.join(save_path, "{}_all_accs_process_{}_of_{}.pkl".format(ckpt[:-3], process, total_processes)),
              "wb") as f:
        pickle.dump(corruption_accs, f)
    with open(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(ckpt[:-3], process, total_processes)),
              "wb") as f:
        pickle.dump(corruption_losses, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST4/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST4/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/EMNIST4/',
                        help="path to directory to save test accuracies and losses")
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
    reset_rngs(seed=13579, deterministic=True)

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

    if args.test_all:
        test_all(args.data_root, args.ckpt_path, args.save_path, args.total_n_classes, args.batch_size, args.n_workers,
             args.pin_mem, dev, args.check_if_run, args.num_processes, args.process)
    else:
        test_specific(args.data_root, args.ckpt_path, args.save_path, args.total_n_classes, args.batch_size,
                      args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes, args.process)

