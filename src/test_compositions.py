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
from lib.networks import DTN, DTN_half
from lib.utils import *
from lib.equivariant_hooks import *

# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility


def loss_and_accuracy(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            output = model(x_tst)
            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def main(data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev, check_if_run,
         total_processes, process):
    files = os.listdir(ckpt_path)
    files.sort()
    assert total_processes <= len(files)
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
        if "es_" == ckpt[:3]:  # don't want to test early stopping ckpts
            print("Skipped early stopping ckpt {}".format(ckpt))
            sys.stdout.flush()
            continue
        if "equivariant" in ckpt:
            network = DTN_half(total_n_classes).to(dev)
            for module in network.modules():
                if isinstance(module, nn.Conv2d):
                    rot_hook = RotationHook(module)
                    print("Hooked module: {}".format(module))
                    sys.stdout.flush()
                    break
        else:
            network = DTN(total_n_classes).to(dev)
        network.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt)))
        print("Testing {}".format(ckpt))
        sys.stdout.flush()

        # Test the trained models on all existing corruptions and compositions
        corruption_accs = {}
        corruption_losses = {}
        for test_corruption in os.listdir(data_root):
            if test_corruption == "raw" or test_corruption == "corruption_names.pkl":
                continue
            corruption_path = os.path.join(data_root, test_corruption)
            trained_classes = list(range(total_n_classes))
            # Shuffle=False should give identical results for symmetric shifts
            _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
                                                         n_workers, pin_mem)
            tst_loss, tst_acc = loss_and_accuracy(network, tst_dl)
            corruption_accs[test_corruption] = tst_acc
            corruption_losses[test_corruption] = tst_loss
            print("{}. test loss: {:.4f}, test acc: {:.4f}".format(test_corruption, tst_loss, tst_acc))
            sys.stdout.flush()

        # Save the results
        with open(os.path.join(save_path, "{}_accs.pkl".format(ckpt[:-3])), "wb") as f:
            pickle.dump(corruption_accs, f)
        with open(os.path.join(save_path, "{}_losses.pkl".format(ckpt[:-3])), "wb") as f:
            pickle.dump(corruption_losses, f)
        if "equivariant" in ckpt:
            rot_hook.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST3/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST3/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/EMNIST3/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    parser.add_argument('--num-processes', type=int, default=20, help="total processes to split into = SLURM_ARRAY_TASK_COUNT")
    parser.add_argument('--process', type=int, default=0, help="which process is running = SLURM_ARRAY_TASK_ID")
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

    print("Running process {} of {}".format(args.process + 1, args.num_processes))
    sys.stdout.flush()

    """
    If running on polestar
    CUDA_VISIBLE_DEVICES=4 python test_compositions.py --pin-mem --check-if-run --num-processes 10 --process 0
    """

    main(args.data_root, args.ckpt_path, args.save_path, args.total_n_classes, args.batch_size, args.n_workers,
         args.pin_mem, dev, args.check_if_run, args.num_processes, args.process)
