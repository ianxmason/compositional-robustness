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
from lib.equivariant_hooks import *

# import deephys as dp


def loss_and_accuracy(network_blocks, dataloader, dev):
    assert len(network_blocks) >= 2  # assumed when the network is called
    criterion = nn.CrossEntropyLoss()
    for block in network_blocks:
        block.eval()
    test_loss = 0.0
    test_acc = 0.0

    deephys_init = False
    all_activs, all_outputs, all_images, all_cats = None, None, None, None
    """
    [all_activs,all_outputs] #Lists containing neural activity for intermediate and output layer
          # each is a multidimensional list of dimensions  [#neurons, #images]. 
          #The output layer is always mandatory to be present.
    all_images #List containing images resized to 32x32 pixels, it h    as size [#images,#channels,32,32].
    all_cats #Labels is a 1-dimensional list of ground-truth label number
    """

    with torch.no_grad():
        for data_tuple in dataloader:
            x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
            for j, block in enumerate(network_blocks):
                if j == 0:
                    features = block(x_tst)
                elif j == len(network_blocks) - 1:  # Classification layer

                    if all_activs is None:
                        all_activs = [features.cpu().numpy()]
                    else:
                        all_activs.append(features.cpu().numpy())

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


                    dp_model = None
                    # if not deephys_init:  # https://colab.research.google.com/github/mjgroth/deephys-aio/blob/master/Python_Tutorial.ipynb#scrollTo=bcEBrF-WYOKI
                    #     dp_model = dp.model(
                    #         name="emnist_simple_net",
                    #         layers={  # include any additional layers
                    #             "penultimate_layer": np.shape(features)[1],
                    #             "output": np.shape(output)[1],
                    #         },
                    #         classification_layer="output"
                    #     )
                    #     dp_model.save()
                    #
                    #     deephys_init = True
                else:
                    features = block(features)
            loss = criterion(output, y_tst)
            acc = accuracy(output, y_tst)
            test_loss += loss.item()
            test_acc += acc

        all_activs = np.concatenate(all_activs, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        all_cats = np.concatenate(all_cats, axis=0)
        print(all_activs.shape)
        print(all_outputs.shape)
        print(all_images.shape)
        print(all_cats.shape)
        print(np.max(all_images))

    return test_loss / len(dataloader), test_acc / len(dataloader), all_activs, all_outputs, all_images, all_cats, dp_model


def test_all(experiment, dataset, data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev,
             check_if_run, total_processes, process):
    """
    Get the specific checkpoint trained on all corruptions and test on every composition

    Parallelise by testing different compositions in different processes
    """
    files = os.listdir(ckpt_path)
    # for f in files:
    #     if "es_" == f[:3]:
    #         raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(f))
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
        test_corruption = ["Contrast"]
        print("Testing {} on {}".format(ckpt, test_corruption))
        sys.stdout.flush()
        trained_classes = list(range(total_n_classes))

        # Old Version
        # corruption_path = os.path.join(data_root, test_corruption)
        # _, _, tst_dl = get_static_emnist_dataloaders(corruption_path, trained_classes, batch_size, False,
        #                                              n_workers, pin_mem)

        identity_path = os.path.join(data_root, "Identity")
        if dataset == "EMNIST":
            transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
            transforms += [getattr(dt, c)() for c in test_corruption]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
            transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
        else:
            transforms = [getattr(dt, c)() for c in test_corruption]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='RGB'))]
        _, _, tst_dl = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                          batch_size, False, n_workers, pin_mem)

        tst_loss, tst_acc, all_activs, all_outputs, all_images, all_cats, dp_model = loss_and_accuracy(network_blocks, tst_dl, dev)
        corruption_accs['-'.join(test_corruption)] = tst_acc
        corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{}, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment, '-'.join(test_corruption), tst_loss,
                                                                   tst_acc))
        sys.stdout.flush()

        classes = tuple([str(i) for i in range(total_n_classes)])


        print(type(all_cats))  # np.array
        print(all_cats.shape)  # (18688,)
        print(type(all_images))  # np.array
        print(all_images.shape)  # (18688, 3, 28, 28)
        print(type(all_activs))  # np.array
        print(all_activs.shape)  # (18688, 512)
        print(type(all_outputs))  # np.array
        print(all_outputs.shape)  # (18688, 47)
        print(type(classes))  # tuple
        print(classes)  # 47

        # Save numpy arrays to test deephys locally if not working on cluster
        np.savez(os.path.join("/om2/user/imason/compositions/slurm/EMNIST5/deephys/",
                              f"EMNIST-{test_corruption[0]}.npz"), all_cats=all_cats, all_images=all_images / 255.0,
                              all_activs=all_activs, all_outputs=all_outputs)

        # all_images = torch.tensor(all_images / 255.0)  # Needs to be in [0,1]
        # test = dp.import_test_data(
        #     name=f"EMNIST-{test_corruption[0]}", # Ugly, uses test_corruption from last iteration of loop
        #     pixel_data=all_images,  # Images resized to 32x32 pixels
        #     ground_truths=all_cats.tolist(),  # Labels
        #     classes=classes,  # List with all category names
        #     state=[torch.tensor(all_activs), torch.tensor(all_outputs)],  # List with neural activity
        #     model=dp_model
        # )
        # test.save()
        print(2/0)  # Don't overwrite existing saved accs

    # Save the results
    # with open(os.path.join(save_path, "{}_{}_all_accs_process_{}_of_{}.pkl".format(experiment, ckpt[:-3], process,
    #                                                                                total_processes)), "wb") as f:
    #     pickle.dump(corruption_accs, f)
    # with open(os.path.join(save_path, "{}_{}_all_losses_process_{}_of_{}.pkl".format(experiment, ckpt[:-3], process,
    #                                                                                  total_processes)), "wb") as f:
    #     pickle.dump(corruption_losses, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/',
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

    if args.dataset not in ["EMNIST", "CIFAR", "FACESCRUB"]:
        raise ValueError("Dataset {} not implemented".format(args.dataset))

    # Set seeding
    reset_rngs(seed=246810, deterministic=True)

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Set up and create unmade directories
    args.data_root = os.path.join(args.data_root, args.dataset)
    args.ckpt_path = os.path.join(args.ckpt_path, args.dataset)
    args.save_path = os.path.join(args.save_path, args.dataset)
    mkdir_p(args.save_path)

    print("Running process {} of {}".format(args.process + 1, args.num_processes))
    sys.stdout.flush()

    """
    If running on polestar
    CUDA_VISIBLE_DEVICES=4 python deephys_test.py --pin-mem --check-if-run --dataset EMNIST --experiment CrossEntropy --num-processes 10 --process 0
    """
    test_all(args.experiment, args.dataset, args.data_root, args.ckpt_path, args.save_path, args.total_n_classes,
                 args.batch_size, args.n_workers, args.pin_mem, dev, args.check_if_run, args.num_processes,
                 args.process)

