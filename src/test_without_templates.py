"""
Right now - hardcoded test, dropping specific neurons vs random neurons
"""
import argparse
import os
import torch
import torch.nn as nn
import pickle
from data.data_loaders import get_static_emnist_dataloaders
from lib.networks import DTN
from lib.utils import *

# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility

# Todo: if makes sense, make much more general and less hardcode. Else delete.


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


def main(data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev, check_if_run):
    ckpt = "identity-rotate_fixed-scale_invariance-loss.pt"  # "identity-gaussian_blur-stripe.pt"  # "identity-rotate_fixed-scale.pt"  # "identity-canny_edges-inverse.pt"
    network = DTN(total_n_classes).to(dev)
    print("Testing {}".format(ckpt))

    # Hardcode number of units for now, can get from network if required later
    all_units = {"conv1": list(range(64)),
                 "conv2": []}

    no_units = {"conv1": [],
                "conv2": []}

    # Templates found using candidate_templates.py
    template_units = {"conv1": [2, 5, 9, 12, 21, 23, 25, 26, 27, 32, 34, 36, 40, 47, 50, 53, 59, 60, 62],
                      "conv2": []}

    non_template_units = {"conv1": [x for x in all_units["conv1"] if x not in template_units["conv1"]],
                          "conv2": []}

    # Do random runs of same size as template units from non template units?
    # Do random 50/50 splits?
    # Do turning only on vs random?
    num_random_runs = 100
    random_results = []

    # for turn_off_units in [template_units, non_template_units, all_units, no_units]:
    for turn_off_units in [no_units]:
    # for i in range(num_random_runs):
        network.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt)))

        # For random runs choosing from all possible units (including template units)
        # turn_off_units = {"conv1": np.random.choice(all_units["conv1"], size=len(template_units["conv1"]), replace=False),
        #                   "conv2": []}

        conv_count = 0
        for module in network.modules():
            # get first layer params
            if isinstance(module, nn.Conv2d):
                conv_count += 1
                for unit in turn_off_units["conv{}".format(conv_count)]:
                    module.weight.data[unit, :, :, :] = 0
                    module.bias.data[unit] = 0
                # if conv_count == 1:
                #     from lib.equivariant_hooks import RotationHook
                #     rotation_hook = RotationHook(module)
                if conv_count == 2:
                    break

        # Test the trained models on all existing corruptions and compositions
        corruption_accs = {}
        corruption_losses = {}
        test_corruption = "rotate_fixed-scale"  # "gaussian_blur-stripe"  # "rotate_fixed-scale"  # "canny_edges-inverse"
        corruption_path = os.path.join(data_root, test_corruption)
        trained_classes = list(range(total_n_classes))
        # Shuffle=False should give identical results for symmetric shifts
        _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
                                                     n_workers, pin_mem)
        tst_loss, tst_acc = loss_and_accuracy(network, tst_dl)
        corruption_accs[test_corruption] = tst_acc
        corruption_losses[test_corruption] = tst_loss
        print("{}. test loss: {:.4f}, test acc: {:.4f}".format(test_corruption, tst_loss, tst_acc))
        random_results.append(tst_acc)

    print("Random runs mean {}, std {}".format(np.mean(random_results), np.std(random_results)))
    print("Worst random run {}".format(np.min(random_results)))

    # turn_off_units = []
    # Most data selective in layer 2
    # Uses selective to elemental corruptions - I think what is needed is selectivity to the composition (by the second layer)
    # template_units = [0, 6, 17, 20, 28, 33, 36, 38, 51, 52, 58, 61, 66, 70, 75, 79, 82, 83, 87, 88, 89, 95, 99, 100, 101, 113, 114, 119, 123, 127]
    # turn_off_units = template_units  # 83.5456 -> Goes up! Combining with layer 1 template units. If use w/ random l1 units seems acc is 85 very often.
    # possible_units = list(np.arange(128))
    # for tu in template_units:
    #     possible_units.remove(tu)
    # turn_off_units = np.random.choice(possible_units, size=len(template_units), replace=False)
    # 10 runs w/ l1 template units: 76.3003, 69.6811, 80.1958, 73.0469, 79.8908, 77.1351, 78.8581, 78.7725, 76.7605, 80.1156
    # 10 runs w/ l1 random 15 units: 82.4219, 83.8238, 79.4735, 82.3684, 84.8673, 81.4640, 81.8493, 81.9617, 83.8345, 84.3429

    # This uses selective to composition in layer 2
    # template_units = [1, 8, 14, 16, 17, 22, 43, 49, 51, 64, 88, 94, 103, 108, 120, 125]
    # template_units = [0, 4, 5, 8, 10, 11, 16, 20, 21, 24, 25, 27, 30, 31, 32, 33, 34, 35, 41, 47, 50, 53, 54, 55, 56, 61, 62, 63, 65, 66, 70, 73, 74, 75, 76, 77, 79, 80, 83, 84, 88, 89, 91, 95, 97, 99, 100, 101, 104, 110, 111, 112, 113, 114, 117, 118, 123, 125, 126]
    # print(len(template_units))
    # turn_off_units = template_units  # 70.5693. w/ l1 template units.
    # 10 runs w/ l1 random 15 units: 76.7284, 75.6047, 77.3705, 76.0060, 78.1785, 77.1351, 77.9270, 76.3485, 77.9217, 77.2528
    # w/ l1 selective to composition 15 units: 67.2303 ?? (Cannot currently explain this - some hybrid stuff?)
    # possible_units = list(np.arange(128))
    # for tu in template_units:
    #     possible_units.remove(tu)
    # turn_off_units = np.random.choice(possible_units, size=len(template_units), replace=False)
    # 10 runs w/ l1 template units: 83.7489, 84.2359, 82.8339, 83.2406, 84.3804, 81.9456, 83.7061, 83.6205, 83.5295, 83.0747
    # 10 runs w/ l1 random 15 units: 86.0873, 86.1622, 85.9429, 85.6057, 86.0713, 86.6064, 85.5362, 86.0927, 85.2526, 85.3382, 85.6325

    # To check - what is select selective to composition in first layer - have not checked this performance?


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST/',
                        help="path to directory to load checkpoints from")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/EMNIST/',
                        help="path to directory to save test accuracies and losses")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--n-workers', type=int, default=4, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
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

    main(args.data_root, args.ckpt_path, args.save_path, args.total_n_classes, args.batch_size, args.n_workers,
         args.pin_mem, dev, args.check_if_run)

