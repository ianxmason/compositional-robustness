"""
Train a model on individual data transformations, pairs of transformations and all transformations. .
"""
import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from data.data_transforms import denormalize
from data.data_loaders import get_static_emnist_dataloaders
from lib.networks import DTN_Part_One, DTN_Part_Two, Filter_Bank
from lib.early_stopping import EarlyStopping
from lib.utils import *
from lib.equivariant_hooks import *


def train_identity_network(network_one, network_two, id_one_ckpt_name, id_two_ckpt_name, data_root, ckpt_path,
                           logging_path, total_n_classes, min_epochs, max_epochs, batch_size, lr, experiment_name,
                           n_workers, pin_mem, dev):
    log_name = "identity_{}.log".format(experiment_name)
    logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
    print("Logging file created to train on identity")

    # Data Set Up
    corruption_path = os.path.join(data_root, "identity")
    train_classes = list(range(total_n_classes))
    trn_dl, val_dl, tst_dl = get_static_emnist_dataloaders(corruption_path, train_classes, batch_size, True,
                                                           n_workers, pin_mem)

    optim = torch.optim.Adam(list(network_one.parameters()) + list(network_two.parameters()), lr)
    criterion = nn.CrossEntropyLoss()
    es_ckpt_path_one = os.path.join(ckpt_path, "es_ckpt_{}.pt".format(id_one_ckpt_name))
    es_ckpt_path_two = os.path.join(ckpt_path, "es_ckpt_{}.pt".format(id_two_ckpt_name))
    early_stopping_one = EarlyStopping(patience=25, verbose=True, path=es_ckpt_path_one)
    early_stopping_two = EarlyStopping(patience=25, verbose=True, path=es_ckpt_path_two)

    # Training
    for epoch in range(max_epochs):
        network_one.train()
        network_two.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for i, data_tuple in enumerate(trn_dl, 1):
            x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
            optim.zero_grad()
            features = network_one(x_trn)
            output = network_two(features)
            loss = criterion(output, y_trn)
            acc = accuracy(output, y_trn)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_acc += acc
        results = [epoch,
                   epoch_loss / len(trn_dl),
                   epoch_acc / len(trn_dl)]
        logger.info("Identity Network. Epoch {}. Avg CE train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

        # Validation
        network_one.eval()
        network_two.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuple in val_dl:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                features = network_one(x_val)
                output = network_two(features)
                loss = criterion(output, y_val)
                acc = accuracy(output, y_val)
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Identity Network. Validation loss {:6.4f}".format(valid_loss / len(val_dl)))
        logger.info("Identity Network. Validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))

        # Early Stopping
        if epoch > min_epochs:
            early_stopping_one(valid_loss / len(val_dl), network_one)  # ES on loss
            early_stopping_two(valid_loss / len(val_dl), network_two)
            # early_stopping(100. - (valid_acc / len(val_dl), network_one)  # ES on acc
            if early_stopping_one.early_stop:
                logger.info("Early stopping")
                break

    # Save model
    logger.info("Loading early stopped checkpoints")
    early_stopping_one.load_from_checkpoint(network_one)
    early_stopping_two.load_from_checkpoint(network_two)
    network_one.eval()
    network_two.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        for data_tuple in val_dl:
            x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
            features = network_one(x_val)
            output = network_two(features)
            loss = criterion(output, y_val)
            acc = accuracy(output, y_val)
            valid_loss += loss.item()
            valid_acc += acc
    logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dl)))
    logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
    early_stopping_one.delete_checkpoint()  # Removes from disk
    early_stopping_two.delete_checkpoint()
    torch.save(network_one.state_dict(), os.path.join(ckpt_path, id_one_ckpt_name))
    torch.save(network_two.state_dict(), os.path.join(ckpt_path, id_two_ckpt_name))
    logger.info("Saved best identity network one to {}".format(os.path.join(ckpt_path, id_one_ckpt_name)))
    logger.info("Saved best identity network two to {}".format(os.path.join(ckpt_path, id_two_ckpt_name)))

    return network_one, network_two


def main(corruptions, data_root, ckpt_path, logging_path, vis_path, total_n_classes, min_epochs, max_epochs, batch_size,
         lr, weights, temperature, compare_corrs, experiment_name, n_workers, pin_mem, dev, vis_data, check_if_run):
    assert max_epochs > min_epochs

    # Check if identity networks are already trained
    network_one = DTN_Part_One().to(dev)
    network_two = DTN_Part_Two(total_n_classes).to(dev)
    id_one_ckpt_name = "identity_part_one.pt"
    id_two_ckpt_name = "identity_part_two.pt"
    if os.path.exists(os.path.join(ckpt_path, id_one_ckpt_name)):
        network_one.load_state_dict(torch.load(os.path.join(ckpt_path, id_one_ckpt_name)))
        print("Loaded identity network part one from checkpoint")
        network_two.load_state_dict(torch.load(os.path.join(ckpt_path, id_two_ckpt_name)))
        print("Loaded identity network part two from checkpoint")
    else:
        # Todo: (?) Careful with parralelisation here, don't want to train identity multiple times.
        print("No identity network checkpoint found. Training identity network")
        network_one, network_two = train_identity_network(network_one, network_two, id_one_ckpt_name,
                                                          id_two_ckpt_name, data_root, ckpt_path, logging_path,
                                                          total_n_classes, min_epochs, max_epochs, batch_size, lr,
                                                          experiment_name, n_workers, pin_mem, dev)

    # Todo: do we want the identity network in train or eval mode (batch norm, dropout etc)? Try both
    network_one.eval()
    network_two.eval()

    # Train all models
    for corruption_names in corruptions:
        assert len(corruption_names) == 1  # This code designed for training on single corruptions
        corruption_name = corruption_names[0]
        assert corruption_name != "identity"
        ckpt_name = "{}_{}.pt".format(corruption_name, experiment_name)
        # Check if training has already completed for the corruption in question.
        if check_if_run and os.path.exists(os.path.join(ckpt_path, ckpt_name)):
            print("Checkpoint already exists at {}. \n Skipping training on corruption: {}".format(
                os.path.join(ckpt_path, ckpt_name), corruption_name))
            continue

        # Log File set up
        log_name = "{}_{}.log".format(corruption_name, experiment_name)
        logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
        print("Logging file created to train on corruption: {}".format(corruption_names))
        # Data Set Up
        corruption_path = os.path.join(data_root, corruption_name)
        train_classes = list(range(total_n_classes))
        trn_dl, val_dl, tst_dl = get_static_emnist_dataloaders(corruption_path, train_classes, batch_size, True,
                                                               n_workers, pin_mem)

        if vis_data:
            x, y = next(iter(trn_dl))
            fig_name = "{}_{}.png".format(corruption_name, experiment_name)
            fig_path = os.path.join(vis_path, fig_name)
            # Denormalise Images
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = denormalize(x).astype(np.uint8)
            # And visualise
            visualise_data(x[:36], y[:36], save_path=fig_path, title=fig_name[:-4], n_rows=6, n_cols=6)

        # Network & Optimizer Set Up
        filter_bank = Filter_Bank().to(dev)
        optim = torch.optim.Adam(filter_bank.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        es_ckpt_path = os.path.join(ckpt_path, "es_ckpt_{}_{}.pt".format('-'.join(corruption_names), experiment_name))
        early_stopping = EarlyStopping(patience=25, verbose=True, path=es_ckpt_path)

        # Training
        for epoch in range(max_epochs):
            # Training
            filter_bank.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, data_tuple in enumerate(trn_dl, 1):
                x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
                optim.zero_grad()
                features = network_one(x_trn)
                features = filter_bank(features)
                output = network_two(features)
                loss = criterion(output, y_trn)
                acc = accuracy(output, y_trn)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                epoch_acc += acc
            results = [epoch,
                       epoch_loss / len(trn_dl),
                       epoch_acc / len(trn_dl)]
            logger.info("Epoch {}. Avg CE train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

            # Validation
            filter_bank.eval()
            valid_loss = 0.0
            valid_acc = 0.0
            with torch.no_grad():
                for data_tuple in val_dl:
                    x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    features = network_one(x_val)
                    features = filter_bank(features)
                    output = network_two(features)
                    loss = criterion(output, y_val)
                    acc = accuracy(output, y_val)
                    valid_loss += loss.item()
                    valid_acc += acc
            logger.info("Validation loss {:6.4f}".format(valid_loss / len(val_dl)))
            logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))

            # Early Stopping
            if epoch > min_epochs:
                early_stopping(valid_loss / len(val_dl), filter_bank)  # ES on loss
                # early_stopping(100. - (valid_acc / len(val_dl), filter_bank)  # ES on acc
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

        # Save model
        logger.info("Loading early stopped checkpoint")
        early_stopping.load_from_checkpoint(filter_bank)
        filter_bank.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuple in val_dl:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                features = network_one(x_val)
                features = filter_bank(features)
                output = network_two(features)
                loss = criterion(output, y_val)
                acc = accuracy(output, y_val)
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dl)))
        logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
        early_stopping.delete_checkpoint()  # Removes from disk
        torch.save(filter_bank.state_dict(), os.path.join(ckpt_path, ckpt_name))
        logger.info("Saved best model to {}".format(os.path.join(ckpt_path, ckpt_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to train networks on different corruptions.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST3/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST3/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--logging-path', type=str, default='/om2/user/imason/compositions/logs/EMNIST3/',
                        help="path to directory to save logs")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/EMNIST3/visualisations/',
                        help="path to directory to save data visualisations")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--min-epochs', type=int, default=10, help="min number of training epochs")
    parser.add_argument('--max-epochs', type=int, default=50, help="max number of training epochs")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--temperature', type=float, default=0.15, help="contrastive loss temperature")
    parser.add_argument('--n-workers', type=int, default=1, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--vis-data', action='store_true', help="set to save a png of one batch of data")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    parser.add_argument('--experiment-name', type=str, default='filter-bank',
                        help="name of experiment - used to name the checkpoint and log files")
    parser.add_argument('--weights', type=str, help="comma delimited string of weights for the loss functions")
    parser.add_argument('--compare-corrs', type=str,
                        help="comma delimited string of which corrs to use as postive pairs")
    parser.add_argument('--corruption-ID', type=int, default=0, help="which corruption to generate")
    args = parser.parse_args()

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Create unmade directories
    mkdir_p(args.ckpt_path)
    mkdir_p(args.logging_path)
    if args.vis_data:
        mkdir_p(args.vis_path)

    # Convert comma delimited strings to lists
    if args.weights is not None:
        weights = [float(x) for x in args.weights.split(',')]
    else:
        weights = None
    if args.compare_corrs is not None:
        compare_corrs = [tuple([int(x) for x in list(corr)]) for corr in args.compare_corrs.split(',')]
    else:
        compare_corrs = None

    # Get all individual corruptions (excluding identity)
    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)
    corruptions = []
    # Always include identity, remove permutations
    for corr in all_corruptions:
        if 'identity' in corr:
            continue
        elif len(corr) == 1:
            corruptions.append(corr)

    # Using slurm to parallelise the training
    corruptions = corruptions[args.corruption_ID:args.corruption_ID + 1]

    """
    6 possible experiments. Run this with corruption ID 0 to 5.
    CARE - make sure identity is trained before running with multiple corruptions.
    CUDA_VISIBLE_DEVICES=3 python train_compositions_residual.py --pin-mem --check-if-run --corruption-ID 0
    """

    assert args.n_workers != 0  # Seems this is a problem - needs investigating
    main(corruptions, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.total_n_classes,
         args.min_epochs, args.max_epochs, args.batch_size, args.lr, weights, args.temperature, compare_corrs,
         args.experiment_name, args.n_workers, args.pin_mem, dev, args.vis_data, args.check_if_run)
