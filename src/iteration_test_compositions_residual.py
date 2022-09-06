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
from data.data_transforms import denormalize
from lib.networks import DTN_Part_One, DTN_Part_Two, Filter_Bank, SimpleConvBlock, SimpleFullyConnectedBlock, \
                         SimpleClassifier, Filter_Bank_Before_Net, Filter_Bank_Resad
from lib.utils import *
from lib.equivariant_hooks import *

# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility


def single_batch_distances(network_one, network_two, filter_banks, id_dataloader, corr_dataloader, vis_path):
    # Vis data checks the batch is the same datapoints for both identity and corruption
    x_id, y_id = next(iter(id_dataloader))
    x_corr, y_corr = next(iter(corr_dataloader))

    if vis_path is not None:
        x_vis = torch.cat((x_id[:50], x_corr[:50]), dim=0)
        y_vis = torch.cat((y_id[:50], y_corr[:50]), dim=0)

        fig_name = "features_distances_check.png"
        fig_path = os.path.join(vis_path, fig_name)
        # Denormalise Images
        x_vis = x_vis.detach().cpu().numpy()
        y_vis = y_vis.detach().cpu().numpy()
        x_vis = denormalize(x_vis).astype(np.uint8)
        # And visualise
        visualise_data(x_vis[:100], y_vis[:100], save_path=fig_path, title=fig_name[:-4], n_rows=10, n_cols=10)

    network_one.eval()
    network_two.eval()
    if filter_banks is not None:
        for filter_bank in filter_banks:
            filter_bank.eval()
    with torch.no_grad():
        x_id, y_id = x_id.to(dev), y_id.to(dev)
        x_corr, y_corr = x_corr.to(dev), y_corr.to(dev)
        features_id = network_one(x_id)
        features_corr = network_one(x_corr)
        pre_distances = torch.sqrt(torch.sum((features_id - features_corr) ** 2, dim=(1, 2, 3)))
        if filter_banks is not None:
            for filter_bank in filter_banks:
                features_corr = filter_bank(features_corr)
        output_id = network_two(features_id)
        output_corr = network_two(features_corr)
        acc_id = accuracy(output_id, y_id)
        acc_corr = accuracy(output_corr, y_corr)
        post_distances = torch.sqrt(torch.sum((features_id - features_corr) ** 2, dim=(1, 2, 3)))

    return torch.mean(pre_distances), torch.mean(post_distances), acc_id, acc_corr


def feature_distances(data_root, ckpt_path, vis_path, total_n_classes, batch_size, experiment_name, n_workers, pin_mem,
                      dev):
    """
    Take identical datapoints under different corruptions. Compare the identity features with the corruption features
    after using the relevant filter bank.
    """
    network_one = DTN_Part_One().to(dev)
    network_two = DTN_Part_Two(total_n_classes).to(dev)
    id_one_ckpt_name = "identity_part_one.pt"
    id_two_ckpt_name = "identity_part_two.pt"
    network_one.load_state_dict(torch.load(os.path.join(ckpt_path, id_one_ckpt_name)))
    print("Loaded identity network part one from checkpoint")
    network_two.load_state_dict(torch.load(os.path.join(ckpt_path, id_two_ckpt_name)))
    print("Loaded identity network part two from checkpoint")
    sys.stdout.flush()

    for test_corruption in os.listdir(data_root):
        if test_corruption == "raw" or test_corruption == "corruption_names.pkl":
            continue
        if len(test_corruption.split("-")) != 1 or test_corruption == "identity":
            continue
        print("Testing on {}".format(test_corruption))

        trained_classes = list(range(total_n_classes))
        identity_generator = torch.Generator(device='cpu').manual_seed(2147483647)
        corruption_generator = torch.Generator(device='cpu').manual_seed(2147483647)
        identity_path = os.path.join(data_root, "identity")
        corruption_path = os.path.join(data_root, test_corruption)
        _, _, id_dl = get_static_emnist_dataloaders(identity_path, trained_classes, batch_size, False,
                                                    n_workers, pin_mem, fixed_generator=identity_generator)
        _, _, corr_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
                                                      n_workers, pin_mem, fixed_generator=corruption_generator)

        # Assume a single filter bank
        filter_bank = Filter_Bank().to(dev)
        filter_bank.load_state_dict(torch.load(os.path.join(ckpt_path, "{}_{}.pt".format(test_corruption, experiment_name))))
        print("Loaded filter bank " + os.path.join(ckpt_path, "{}_{}.pt".format(test_corruption, experiment_name)))
        pre_distances, post_distances, acc_id, acc_corr = single_batch_distances(network_one, network_two,
                                                                                 [filter_bank], id_dl, corr_dl, None)

        print("Corruption: {}.".format(test_corruption))
        print("Before filter bank: {}".format(pre_distances))
        print("After filter bank: {}".format(post_distances))
        print("Identity accuracy: {}".format(acc_id))
        print("Corruption accuracy: {}".format(acc_corr))
        print("-------------")
        sys.stdout.flush()


def loss_and_accuracy(network_one, network_two, filter_banks, dataloader):
    criterion = nn.CrossEntropyLoss()
    network_one.eval()
    network_two.eval()
    if filter_banks is not None:
        for filter_bank in filter_banks:
            filter_bank.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            features = network_one(x_tst)

            # Most cases
            # if filter_banks is not None:
            #     for filter_bank in filter_banks:
            #         features = filter_bank(features)

            # Parallel/residual adapter case
            if filter_banks is not None:
                for filter_bank in filter_banks:
                    features += filter_bank(x_tst)

            output = network_two(features)
            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def loss_and_accuracy_resads(network_one, network_two, network_three, test_corruption,
                             filter_banks, dataloader):
    test_corruptions = test_corruption.split("-")
    if filter_banks is not None:
        assert len(test_corruptions) == len(filter_banks)
    criterion = nn.CrossEntropyLoss()
    network_one.eval()
    network_two.eval()
    network_three.eval()
    # network_four.eval()
    if filter_banks is not None:
        for filter_bank in filter_banks:
            filter_bank.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            features = network_one(x_tst)

            # Most cases
            # if filter_banks is not None:
            #     for filter_bank in filter_banks:
            #         features = filter_bank(features)

            # Parallel/residual adapter case
            # if filter_banks is not None:
            #     for filter_bank in filter_banks:
            #         features += filter_bank(x_tst)

            # preserves order as much as possible whilst always doing blur and noise first
            # if filter_banks is not None:
            #     for tc, fb in zip(test_corruptions, filter_banks):
            #         if tc == "impulse_noise":
            #             features += fb(x_tst)
            #
            # pre_adapt_features = network_two(features)
            # features = network_three(pre_adapt_features)
            #
            # if filter_banks is not None:
            #     for tc, fb in zip(test_corruptions, filter_banks):
            #         if tc != "impulse_noise":
            #             features += fb(pre_adapt_features)
            #
            # output = network_four(features)

            # Different location serial filter bank case
            if filter_banks is not None:
                for tc, fb in zip(test_corruptions, filter_banks):
                    if tc == "impulse_noise" or tc == "gaussian_blur" or tc == "inverse":
                        features = fb(features)

            features = network_two(features)

            if filter_banks is not None:
                for tc, fb in zip(test_corruptions, filter_banks):
                    if tc == "rotate_fixed" or tc == "scale" or tc == "shear_fixed":
                        features = fb(features)

            output = network_three(features)

            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def main(data_root, ckpt_path, save_path, total_n_classes, batch_size, experiment_name, n_workers, pin_mem, dev,
         check_if_run):

    # Load each ckpt
    if check_if_run and os.path.exists(os.path.join(save_path, "{}_losses.pkl".format(experiment_name))):
        print("Pickle file already exists at {}. \n Not testing".format(
            os.path.join(save_path, "{}_losses.pkl".format(experiment_name))))
        sys.stdout.flush()
    else:
        # network_one = nn.Sequential(
        #                 SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.1),  # 0.1
        #                 SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.3)  # 0.3
        #               ).to(dev)
        # network_two = nn.Sequential(
        #                 SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.5),  # 0.5
        #                 nn.Flatten(),  # Flattens everything except the batch dimension by default
        #                 SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=True, dropout=0.5),  # 0.5
        #                 SimpleClassifier(512, total_n_classes)
        #               ).to(dev)

        # Old set up (check for comparison)
        # network_one = DTN_Part_One().to(dev)
        # network_two = DTN_Part_Two(total_n_classes).to(dev)

        # To put modules before the first layer
        # network_one = nn.Sequential(
        #                 nn.Identity()
        #               ).to(dev)
        # network_two = nn.Sequential(
        #                 SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.1),  # 0.1
        #                 SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.3),  # 0.3
        #                 SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.5),  # 0.5
        #                 nn.Flatten(),  # Flattens everything except the batch dimension by default
        #                 SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=True, dropout=0.5),  # 0.5
        #                 SimpleClassifier(512, total_n_classes)
        #               ).to(dev)

        # For residual adapters
        # network_one = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        # ).to(dev)
        # network_two = nn.Sequential(
        #     nn.BatchNorm2d(64),
        #     nn.Dropout2d(0.1),
        #     nn.ReLU(),
        #     SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.3),
        #     SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.5),  # 0.5
        #     nn.Flatten(),  # Flattens everything except the batch dimension by default
        #     SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=True, dropout=0.5),  # 0.5
        #     SimpleClassifier(512, total_n_classes)
        # ).to(dev)

        # For residual adapters at different layers - also now BN is off
        # network_one = nn.Sequential(
        #                 nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        #               ).to(dev)
        # network_two = nn.Sequential(
        #                 nn.Dropout2d(0.1),
        #                 nn.ReLU(),
        #                 SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.3)
        #               ).to(dev)
        # network_three = nn.Sequential(
        #                   nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        #                 ).to(dev)
        # network_four = nn.Sequential(
        #                  nn.Dropout2d(0.5),
        #                  nn.ReLU(),
        #                  nn.Flatten(),  # Flattens everything except the batch dimension by default
        #                  SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=False, dropout=0.5),  # 0.5
        #                  SimpleClassifier(512, total_n_classes)
        #                ).to(dev)

        # For in series adapters at different layers - again BN is now off
        network_one = nn.Sequential(
                        SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.1),  # 0.1
                      ).to(dev)
        network_two = nn.Sequential(
                        SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.3)  # 0.3
                      ).to(dev)
        network_three = nn.Sequential(
                          SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5),  # 0.5
                          nn.Flatten(),  # Flattens everything except the batch dimension by default
                          SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=False, dropout=0.5),  # 0.5
                          SimpleClassifier(512, total_n_classes)
                        ).to(dev)

        id_one_ckpt_name = "identity_part_one_{}.pt".format(experiment_name)
        id_two_ckpt_name = "identity_part_two_{}.pt".format(experiment_name)
        id_three_ckpt_name = "identity_part_three_{}.pt".format(experiment_name)
        # id_four_ckpt_name = "identity_part_four_{}.pt".format(experiment_name)

        network_one.load_state_dict(torch.load(os.path.join(ckpt_path, id_one_ckpt_name)))
        print("Loaded identity network part one from checkpoint")
        network_two.load_state_dict(torch.load(os.path.join(ckpt_path, id_two_ckpt_name)))
        print("Loaded identity network part two from checkpoint")
        network_three.load_state_dict(torch.load(os.path.join(ckpt_path, id_three_ckpt_name)))
        # print("Loaded identity network part three from checkpoint")
        # network_four.load_state_dict(torch.load(os.path.join(ckpt_path, id_four_ckpt_name)))
        print("Loaded identity network part four from checkpoint")
        sys.stdout.flush()


        # Test the filter banks on all existing corruptions and compositions
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
            if test_corruption == "identity":
                filter_banks = None
            else:
                test_corruptions = test_corruption.split("-")

                # filter_banks = [Filter_Bank().to(dev) for _ in range(len(test_corruptions))]  # usual case
                # filter_banks = [Filter_Bank_Before_Net().to(dev) for _ in range(len(test_corruptions))]  # modules before network
                # filter_banks = [Filter_Bank_Resad().to(dev) for _ in range(len(test_corruptions))]  # residual adapters

                # filter_banks = []
                # for tc in test_corruptions:
                #     if tc == "impulse_noise":
                #         filter_banks.append(nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                #                                           nn.BatchNorm2d(64),
                #                                           nn.Dropout2d(0.1),
                #                                           nn.ReLU(),
                #                                           nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                #                                           nn.BatchNorm2d(64),
                #                                           nn.Dropout2d(0.3),
                #                                           nn.ReLU(),
                #                                           nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                #                                           nn.BatchNorm2d(64),
                #                                           nn.Dropout2d(0.3),
                #                                           nn.ReLU()
                #                                           ).to(dev)
                #                             )
                #     else:
                #         filter_banks.append(nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                #                                           nn.BatchNorm2d(256),
                #                                           nn.Dropout2d(0.3),
                #                                           nn.ReLU(),
                #                                           nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                #                                           nn.BatchNorm2d(256),
                #                                           nn.Dropout2d(0.3),
                #                                           nn.ReLU(),
                #                                           nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                #                                           nn.BatchNorm2d(256),
                #                                           nn.Dropout2d(0.3),
                #                                           nn.ReLU()
                #                                           ).to(dev)
                #                             )

                filter_banks = []
                for tc in test_corruptions:
                    if tc == "impulse_noise" or tc == "gaussian_blur" or tc == "inverse":
                        filter_banks.append(nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                                                          # nn.BatchNorm2d(64),
                                                          nn.Dropout2d(0.3),
                                                          nn.ReLU(),
                                                          nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                                                          # nn.BatchNorm2d(64),
                                                          nn.Dropout2d(0.3),
                                                          nn.ReLU()
                                                          ).to(dev)
                                            )
                    elif tc == "rotate_fixed" or tc == "scale" or tc == "shear_fixed":
                        filter_banks.append(nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
                                                         # nn.BatchNorm2d(128),
                                                         nn.Dropout2d(0.3),
                                                         nn.ReLU(),
                                                         nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
                                                         # nn.BatchNorm2d(128),
                                                         nn.Dropout2d(0.3),
                                                         nn.ReLU()
                                                         ).to(dev)
                                            )
                    else:
                        raise ValueError("Corruption name not recognised")

                # if "augmented" not in experiment_name:
                #     filter_banks = [Filter_Bank().to(dev) for _ in range(len(test_corruptions))]
                #     # Load filter banks from ckpts e.g. shear_fixed_filter-bank.pt
                #     # [fb.load_state_dict(torch.load(os.path.join(ckpt_path, "{}_{}.pt".format(tc, experiment_name))))
                #     #  for fb, tc in zip(filter_banks, test_corruptions)]
                #     # print("Loaded filter bank(s) from checkpoint(s)")
                #     # sys.stdout.flush()
                # else:
                #     if "gaussian_blur" in test_corruptions and "impulse_noise" in test_corruptions:
                #         filter_banks = [Filter_Bank().to(dev) for _ in range(len(test_corruptions) - 2)]
                #         test_corruptions.remove("gaussian_blur")
                #         test_corruptions.remove("impulse_noise")
                #     elif "gaussian_blur" in test_corruptions:
                #         filter_banks = [Filter_Bank().to(dev) for _ in range(len(test_corruptions) - 1)]
                #         test_corruptions.remove("gaussian_blur")
                #     elif "impulse_noise" in test_corruptions:
                #         filter_banks = [Filter_Bank().to(dev) for _ in range(len(test_corruptions) - 1)]
                #         test_corruptions.remove("impulse_noise")
                #     else:
                #         filter_banks = [Filter_Bank().to(dev) for _ in range(len(test_corruptions))]

                # Checking each ckpt to be more sure
                for fb, tc in zip(filter_banks, test_corruptions):
                    fb.load_state_dict(torch.load(os.path.join(ckpt_path, "{}_{}.pt".format(tc, experiment_name))))
                    print("Loaded filter bank " + os.path.join(ckpt_path, "{}_{}.pt".format(tc, experiment_name)))
                    sys.stdout.flush()

            # tst_loss, tst_acc = loss_and_accuracy(network_one, network_two, filter_banks, tst_dl)
            tst_loss, tst_acc = loss_and_accuracy_resads(network_one, network_two, network_three, test_corruption, filter_banks, tst_dl)
            corruption_accs[test_corruption] = tst_acc
            corruption_losses[test_corruption] = tst_loss
            print("{}. test loss: {:.4f}, test acc: {:.4f}".format(test_corruption, tst_loss, tst_acc))
            sys.stdout.flush()

        # Save the results
        with open(os.path.join(save_path, "{}_accs.pkl".format(experiment_name)), "wb") as f:
            pickle.dump(corruption_accs, f)
        with open(os.path.join(save_path, "{}_losses.pkl".format(experiment_name)), "wb") as f:
            pickle.dump(corruption_losses, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST3/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST_TEMP/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/EMNIST_TEMP/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/EMNIST_TEMP/visualisations/',
                        help="path to directory to save data visualisations")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    parser.add_argument('--experiment-name', type=str, default='filter-bank-invariance-10',
                        help="name of experiment - used to name the checkpoint and log files")
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
    mkdir_p(args.vis_path)

    # print("Running process {} of {}".format(args.process + 1, args.num_processes))
    # sys.stdout.flush()

    """
    If running on polestar
    CUDA_VISIBLE_DEVICES=5 python iteration_test_compositions_residual.py --pin-mem --check-if-run --experiment-name split-net-augmented
    """

    main(args.data_root, args.ckpt_path, args.save_path, args.total_n_classes, args.batch_size, args.experiment_name,
         args.n_workers, args.pin_mem, dev, args.check_if_run)

    # feature_distances(args.data_root, args.ckpt_path, args.vis_path, args.total_n_classes, args.batch_size,
    #                   args.experiment_name, args.n_workers, args.pin_mem, dev)
