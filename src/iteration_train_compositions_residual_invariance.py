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
from train_compositions_residual import train_identity_network
from data.data_transforms import denormalize
from data.data_loaders import get_static_emnist_dataloaders
from lib.networks import DTN_Part_One, DTN_Part_Two, Filter_Bank, SimpleConvBlock, SimpleFullyConnectedBlock, \
                         SimpleClassifier, Filter_Bank_Before_Net, Filter_Bank_Resad
from lib.early_stopping import EarlyStopping
from lib.utils import *
from lib.equivariant_hooks import *


"""
Want to set up code so can rapidly test
    - Module location
    - Module capacity
    - BN/dropout on/off
    - Identity net in train or eval mode?
    - Joint training feed-forward part with noise+blur, then adding the others
        - Invariance loss in first layer or in penultimate layer?
        - Easiest (and I think most sensible) is in the same layer as the modules.
    - Supervised contrastive
    
How to do this
    - Train only 3-5 epochs - train on each corruption in parallel on openmind
        - Also need some testing code for a few corruptions to check
        - Those that give some signal we can train for longer
    - Change network structure to give a conv block with args for capacity, BN, dropout
        - Set up the whole network so that the architecture is defined in this file and we can easily move the modules around
    - Supervised contrastive - need to make it a lot faster - some way to vectorize?
        - Check the comments in contrastive_layer_loss and be sure we are happy with it
"""


def contrastive_layer_loss(features, single_corr_bs, dev, temperature=0.15, compare=(1, 2)):
    if compare == ():
        return torch.tensor(0.0).to(dev)
    assert len(features) % single_corr_bs == 0
    features = F.normalize(features, dim=1)
    features = torch.cat([features[(i - 1) * single_corr_bs:i * single_corr_bs] for i in compare], dim=0)

    labels = torch.cat([torch.arange(single_corr_bs) for _ in range(len(compare))], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dev)

    similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(dev)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    contrastive_criterion = torch.nn.CrossEntropyLoss()
    if len(compare) == 2:
        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dev)
        loss = contrastive_criterion(logits, labels)
    elif len(compare) == 3:
        # Explanation: https://github.com/sthalles/SimCLR/issues/16  https://github.com/sthalles/SimCLR/issues/33
        logits_1 = torch.cat([positives[:, 0:1], negatives], dim=1)
        logits_1 = logits_1 / temperature
        labels_1 = torch.zeros(logits_1.shape[0], dtype=torch.long).to(dev)

        logits_2 = torch.cat([positives[:, 1:2], negatives], dim=1)
        logits_2 = logits_2 / temperature
        labels_2 = torch.zeros(logits_2.shape[0], dtype=torch.long).to(dev)

        # Todo: there is weirdness here. All positive pairs are captured but not 'considered' in the same way
        # i.e. we don't take every possible pair of f1/f2/f3 and get all positive and negatives
        # rather we have some positives and some negatives and make things close/far
        # Overall I think it should be okay but can think a bit more about it
        # Todo: Try larger batch size/different temperature may help.
        loss = contrastive_criterion(logits_1, labels_1)
        loss += contrastive_criterion(logits_2, labels_2)
    else:
        raise NotImplementedError("Only currently handles 2 or 3 corruptions")

    return loss


def train_identity_networks(networks, id_ckpt_names, data_root, ckpt_path,
                           logging_path, total_n_classes, min_epochs, max_epochs, batch_size, lr, experiment_name,
                           n_workers, pin_mem, dev):
    assert len(id_ckpt_names) == len(networks)
    log_name = "identity_{}.log".format(experiment_name)
    logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
    print("Logging file created to train on identity")

    # Data Set Up
    corruption_path = os.path.join(data_root, "identity")
    train_classes = list(range(total_n_classes))
    trn_dl, val_dl, tst_dl = get_static_emnist_dataloaders(corruption_path, train_classes, batch_size, True,
                                                           n_workers, pin_mem)

    all_parameters = None
    for network in networks:
        if all_parameters is None:
            all_parameters = list(network.parameters())
        else:
            all_parameters += list(network.parameters())
    optim = torch.optim.Adam(all_parameters, lr)

    criterion = nn.CrossEntropyLoss()
    es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(id_ckpt_name)) for id_ckpt_name in id_ckpt_names]
    early_stoppings = [EarlyStopping(patience=25, verbose=True, path=es_ckpt_path) for es_ckpt_path in es_ckpt_paths]
    assert len(early_stoppings) == len(networks)

    # Training
    for epoch in range(max_epochs):
        for network in networks:
            network.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for i, data_tuple in enumerate(trn_dl, 1):
            x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
            optim.zero_grad()
            assert len(networks) >= 2
            for j, network in enumerate(networks):
                if j == 0:
                    features = network(x_trn)
                elif j == len(networks) - 1:
                    output = network(features)
                else:
                    features = network(features)
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
        for network in networks:
            network.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuple in val_dl:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                assert len(networks) >= 2
                for j, network in enumerate(networks):
                    if j == 0:
                        features = network(x_val)
                    elif j == len(networks) - 1:
                        output = network(features)
                    else:
                        features = network(features)
                loss = criterion(output, y_val)
                acc = accuracy(output, y_val)
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Identity Network. Validation loss {:6.4f}".format(valid_loss / len(val_dl)))
        logger.info("Identity Network. Validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))

        # Early Stopping
        if epoch > min_epochs:
            for es, network in zip(early_stoppings, networks):
                es(valid_loss / len(val_dl), network)  # ES on loss
            if early_stoppings[0].early_stop:
                logger.info("Early stopping")
                break

    # Save model
    logger.info("Loading early stopped checkpoints")
    for es, network in zip(early_stoppings, networks):
        es.load_from_checkpoint(network)
    for network in networks:
        network.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        for data_tuple in val_dl:
            x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
            assert len(networks) >= 2
            for j, network in enumerate(networks):
                if j == 0:
                    features = network(x_val)
                elif j == len(networks) - 1:
                    output = network(features)
                else:
                    features = network(features)
            loss = criterion(output, y_val)
            acc = accuracy(output, y_val)
            valid_loss += loss.item()
            valid_acc += acc
    logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dl)))
    logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
    for es in early_stoppings:
        es.delete_checkpoint()  # Removes from disk
    for id_ckpt_name, network in zip(id_ckpt_names, networks):
        torch.save(network.state_dict(), os.path.join(ckpt_path, id_ckpt_name))
        logger.info("Saved best identity network part to {}".format(os.path.join(ckpt_path, id_ckpt_name)))

    return networks


def train_augmented_identity_network(network_one, network_two, id_one_ckpt_name, id_two_ckpt_name, data_root, ckpt_path,
                                     logging_path, total_n_classes, min_epochs, max_epochs, batch_size, lr,
                                     experiment_name, n_workers, pin_mem, dev):
    """
    Can generalise as needed
        - include temperature and compare_corrs in args
        - include the corruptions to augment with in the args
    """
    log_name = "identity_{}.log".format(experiment_name)
    logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
    print("Logging file created to train on identity")

    # Data Set Up - hardcode to include blur and noise as well as identity
    corruption_paths = [os.path.join(data_root, "identity"),
                        os.path.join(data_root, "impulse_noise"),
                        os.path.join(data_root, "gaussian_blur")]
    train_classes = list(range(total_n_classes))
    generators = [torch.Generator(device='cpu').manual_seed(2147483647) for _ in range(len(corruption_paths))]
    single_corr_bs = batch_size // len(corruption_paths)
    trn_dls, val_dls, tst_dls = [], [], []
    for corruption_path, generator in zip(corruption_paths, generators):
        trn_dl, val_dl, tst_dl = get_static_emnist_dataloaders(corruption_path, train_classes, single_corr_bs,
                                                               True, n_workers, pin_mem, fixed_generator=generator)
        trn_dls.append(trn_dl)
        val_dls.append(val_dl)
        tst_dls.append(tst_dl)

    _, comp_dl_1, _ = get_static_emnist_dataloaders(os.path.join(data_root, 'gaussian_blur-impulse_noise'),
                                                    train_classes, batch_size, False, n_workers, pin_mem)
    _, comp_dl_2, _ = get_static_emnist_dataloaders(os.path.join(data_root, 'impulse_noise-gaussian_blur'),
                                                    train_classes, batch_size, False, n_workers, pin_mem)

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
        epoch_ce_loss = 0.0
        epoch_inv_loss = 0.0
        epoch_acc = 0.0
        for data_tuples in zip(*trn_dls):
            for i, data_tuple in enumerate(data_tuples):
                if i == 0:
                    x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
                else:
                    x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    x_trn = torch.cat((x_trn, x_temp), dim=0)
                    y_trn = torch.cat((y_trn, y_temp), dim=0)
            optim.zero_grad()
            inv_loss = 0
            features = network_one(x_trn)
            flat_features = features.reshape(features.shape[0], -1)  # Flatten spatial dimensions
            inv_loss += 1 * contrastive_layer_loss(flat_features, single_corr_bs, dev, temperature=0.15, compare=(1, 2, 3))
            output = network_two(features)
            ce_loss = criterion(output, y_trn)
            loss = ce_loss + inv_loss
            acc = accuracy(output, y_trn)
            loss.backward()
            optim.step()
            epoch_ce_loss += ce_loss.item()
            epoch_inv_loss += inv_loss.item()
            epoch_acc += acc
        results = [epoch,
                   epoch_ce_loss / len(trn_dls[0]),
                   epoch_inv_loss / len(trn_dls[0]),
                   epoch_acc / len(trn_dls[0])]
        logger.info("Identity Network. Epoch {}. Avg CE train loss {:6.4f}. Avg inv train loss {:6.4f}. "
                    "Avg train acc {:6.3f}.".format(*results))

        # Validation
        network_one.eval()
        network_two.eval()
        valid_ce_loss = 0.0
        valid_inv_loss = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuples in zip(*val_dls):
                for i, data_tuple in enumerate(data_tuples):
                    if i == 0:
                        x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    else:
                        x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
                        x_val = torch.cat((x_val, x_temp), dim=0)
                        y_val = torch.cat((y_val, y_temp), dim=0)
                inv_loss = 0
                features = network_one(x_val)
                flat_features = features.reshape(features.shape[0], -1)  # Flatten spatial dimensions
                inv_loss += 1 * contrastive_layer_loss(flat_features, single_corr_bs, dev, temperature=0.15,
                                                       compare=(1, 2, 3))
                output = network_two(features)
                ce_loss = criterion(output, y_val)
                loss = ce_loss + inv_loss
                acc = accuracy(output, y_val)
                valid_ce_loss += ce_loss.item()
                valid_inv_loss += inv_loss.item()
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Identity Network. Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
        logger.info("Identity Network. Validation inv loss {:6.4f}".format(valid_inv_loss / len(val_dls[0])))
        logger.info("Identity Network. Validation total loss {:6.4f}".format(valid_loss / len(val_dls[0])))
        logger.info("Identity Network. Validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))

        comp_loss = 0.0
        comp_acc = 0.0
        with torch.no_grad():
            for data_tuple in comp_dl_1:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                features = network_one(x_val)
                output = network_two(features)
                loss = criterion(output, y_val)
                acc = accuracy(output, y_val)
                comp_loss += loss.item()
                comp_acc += acc
        logger.info("Composition (gb-im) CE loss {:6.4f}".format(comp_loss / len(comp_dl_1)))
        logger.info("Composition (gb-im) accuracy {:6.3f}".format(comp_acc / len(comp_dl_1)))

        comp_loss = 0.0
        comp_acc = 0.0
        with torch.no_grad():
            for data_tuple in comp_dl_2:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                features = network_one(x_val)
                output = network_two(features)
                loss = criterion(output, y_val)
                acc = accuracy(output, y_val)
                comp_loss += loss.item()
                comp_acc += acc
        logger.info("Composition (im-gb) CE loss {:6.4f}".format(comp_loss / len(comp_dl_2)))
        logger.info("Composition (im-gb) accuracy {:6.3f}".format(comp_acc / len(comp_dl_2)))

        # Early Stopping
        if epoch > min_epochs:
            early_stopping_one(valid_loss / len(val_dls[0]), network_one)  # ES on loss
            early_stopping_two(valid_loss / len(val_dls[0]), network_two)  # Does nothing
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
    valid_ce_loss = 0.0
    valid_inv_loss = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        for data_tuples in zip(*val_dls):
            for i, data_tuple in enumerate(data_tuples):
                if i == 0:
                    x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                else:
                    x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    x_val = torch.cat((x_val, x_temp), dim=0)
                    y_val = torch.cat((y_val, y_temp), dim=0)
            inv_loss = 0
            features = network_one(x_val)
            flat_features = features.reshape(features.shape[0], -1)  # Flatten spatial dimensions
            inv_loss += 1 * contrastive_layer_loss(flat_features, single_corr_bs, dev, temperature=0.15,
                                                   compare=(1, 2, 3))
            output = network_two(features)
            ce_loss = criterion(output, y_val)
            loss = ce_loss + inv_loss
            acc = accuracy(output, y_val)
            valid_ce_loss += ce_loss.item()
            valid_inv_loss += inv_loss.item()
            valid_loss += loss.item()
            valid_acc += acc
    logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
    logger.info("Early Stopped validation inv loss {:6.4f}".format(valid_inv_loss / len(val_dls[0])))
    logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dls[0])))
    logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
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

    # Set up architecture

    # Split up network - main network to use
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
    #                 nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
    #               ).to(dev)
    # network_two = nn.Sequential(
    #                 nn.BatchNorm2d(64),
    #                 nn.Dropout2d(0.1),
    #                 nn.ReLU(),
    #                 SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.3),
    #                 SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=True, dropout=0.5),  # 0.5
    #                 nn.Flatten(),  # Flattens everything except the batch dimension by default
    #                 SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=True, dropout=0.5),  # 0.5
    #                 SimpleClassifier(512, total_n_classes)
    #               ).to(dev)

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
    # network_one = nn.Sequential(
    #                 SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.1),  # 0.1
    #               ).to(dev)
    # network_two = nn.Sequential(
    #                 SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.3),  # 0.3
    #                 SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5),  # 0.5
    #               ).to(dev)
    # network_three = nn.Sequential(
    #                 nn.Flatten(),  # Flattens everything except the batch dimension by default
    #                 SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=False, dropout=0.5),  # 0.5
    #                 SimpleClassifier(512, total_n_classes)
    #               ).to(dev)

    # For in series adapters at all possible layers - autoselecting the best layer - again BN is now off
    network_one = nn.Sequential(
        SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.1)  # 0.1
    ).to(dev)
    network_two = nn.Sequential(
        SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.3)  # 0.3
    ).to(dev)
    network_three = nn.Sequential(
        SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5)  # 0.5
    ).to(dev)
    network_four = nn.Sequential(
        nn.Flatten(),  # Flattens everything except the batch dimension by default
        SimpleFullyConnectedBlock(256 * 4 * 4, 512, batch_norm=False, dropout=0.5)
    ).to(dev)
    network_five = nn.Sequential(
        SimpleClassifier(512, total_n_classes)
    ).to(dev)


    print("##########")
    print(network_one)
    print("##########")
    print(network_two)
    print("##########")

    # Check if identity networks are already trained
    id_one_ckpt_name = "identity_part_one_{}.pt".format(experiment_name)
    id_two_ckpt_name = "identity_part_two_{}.pt".format(experiment_name)
    id_three_ckpt_name = "identity_part_three_{}.pt".format(experiment_name)
    id_four_ckpt_name = "identity_part_four_{}.pt".format(experiment_name)
    id_five_ckpt_name = "identity_part_five_{}.pt".format(experiment_name)
    if os.path.exists(os.path.join(ckpt_path, id_one_ckpt_name)):
        network_one.load_state_dict(torch.load(os.path.join(ckpt_path, id_one_ckpt_name)))
        print("Loaded identity network part one from checkpoint")
        network_two.load_state_dict(torch.load(os.path.join(ckpt_path, id_two_ckpt_name)))
        print("Loaded identity network part two from checkpoint")
        network_three.load_state_dict(torch.load(os.path.join(ckpt_path, id_three_ckpt_name)))
        print("Loaded identity network part three from checkpoint")
        network_four.load_state_dict(torch.load(os.path.join(ckpt_path, id_four_ckpt_name)))
        print("Loaded identity network part four from checkpoint")
        network_five.load_state_dict(torch.load(os.path.join(ckpt_path, id_five_ckpt_name)))
        print("Loaded identity network part five from checkpoint")
    else:
        # Todo: (?) Careful with parralelisation here, don't want to train identity multiple times.
        print("No identity network checkpoint found. Training identity network")
        # network_one, network_two = train_identity_network(network_one, network_two, id_one_ckpt_name,
        #                                                   id_two_ckpt_name, data_root, ckpt_path, logging_path,
        #                                                   total_n_classes, min_epochs, max_epochs, batch_size, lr,
        #                                                   experiment_name, n_workers, pin_mem, dev)

        # network_one, network_two = train_augmented_identity_network(network_one, network_two, id_one_ckpt_name,
        #                                                   id_two_ckpt_name, data_root, ckpt_path, logging_path,
        #                                                   total_n_classes, min_epochs, max_epochs, batch_size, lr,
        #                                                   experiment_name, n_workers, pin_mem, dev)

        # Should now handle arbitrary number of networks
        network_one, network_two, network_three, network_four, network_five = train_identity_networks(
                                            [network_one, network_two, network_three, network_four, network_five],
                                            [id_one_ckpt_name, id_two_ckpt_name, id_three_ckpt_name, id_four_ckpt_name, id_five_ckpt_name],
                                            data_root, ckpt_path, logging_path, total_n_classes, min_epochs, max_epochs,
                                            batch_size, lr, experiment_name, n_workers, pin_mem, dev)
        # A hack:
        raise(RuntimeError("Identity network training completed - rerun for corruption experiments"))

    # Todo: do we want the identity network in train or eval mode (batch norm, dropout etc)? Try both
    network_one.eval()
    network_two.eval()
    network_three.eval()
    network_four.eval()
    network_five.eval()

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
        corruption_paths = [os.path.join(data_root, "identity"), os.path.join(data_root, corruption_name)]
        train_classes = list(range(total_n_classes))
        generators = [torch.Generator(device='cpu').manual_seed(2147483647) for _ in range(len(corruption_paths))]
        single_corr_bs = batch_size // len(corruption_paths)
        # Use static dataloader for the potential to change the classes being trained on
        trn_dls, val_dls, tst_dls = [], [], []
        for corruption_path, generator in zip(corruption_paths, generators):
            trn_dl, val_dl, tst_dl = get_static_emnist_dataloaders(corruption_path, train_classes, single_corr_bs,
                                                                   True, n_workers, pin_mem, fixed_generator=generator)
            trn_dls.append(trn_dl)
            val_dls.append(val_dl)
            tst_dls.append(tst_dl)

        if vis_data:
            for i, trn_dl in enumerate(trn_dls):
                if i == 0:
                    x, y = next(iter(trn_dl))
                else:
                    x_temp, y_temp = next(iter(trn_dl))
                    x = torch.cat((x, x_temp), dim=0)
                    y = torch.cat((y, y_temp), dim=0)
            fig_name = "{}_{}.png".format(corruption_name, experiment_name)
            fig_path = os.path.join(vis_path, fig_name)
            # Denormalise Images
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = denormalize(x).astype(np.uint8)
            # And visualise
            visualise_data(x[:100], y[:100], save_path=fig_path, title=fig_name[:-4], n_rows=10, n_cols=10)

        # Network & Optimizer Set Up
        # filter_bank = Filter_Bank().to(dev)  # Most cases
        # filter_bank = Filter_Bank_Before_Net().to(dev) # For the case with modules before the net
        # filter_bank = Filter_Bank_Resad().to(dev)
        # if corruption_name == "impulse_noise":
        #     filter_bank = nn.Sequential(
        #                     nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
        #                     nn.BatchNorm2d(64),
        #                     nn.Dropout2d(0.1),
        #                     nn.ReLU(),
        #                     nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        #                     nn.BatchNorm2d(64),
        #                     nn.Dropout2d(0.3),
        #                     nn.ReLU(),
        #                     nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        #                     nn.BatchNorm2d(64),
        #                     nn.Dropout2d(0.3),
        #                     nn.ReLU()
        #                   ).to(dev)
        # else:
        #     filter_bank = nn.Sequential(
        #                     nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
        #                     nn.BatchNorm2d(256),
        #                     nn.Dropout2d(0.3),
        #                     nn.ReLU(),
        #                     nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
        #                     nn.BatchNorm2d(256),
        #                     nn.Dropout2d(0.3),
        #                     nn.ReLU(),
        #                     nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
        #                     nn.BatchNorm2d(256),
        #                     nn.Dropout2d(0.3),
        #                     nn.ReLU()
        #                   ).to(dev)

        # if corruption_name == "impulse_noise" or corruption_name == "gaussian_blur" or corruption_name == "inverse":
        #     filter_bank = nn.Sequential(
        #                     nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        #                     # nn.BatchNorm2d(64),
        #                     nn.Dropout2d(0.3),
        #                     nn.ReLU(),
        #                     nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        #                     # nn.BatchNorm2d(64),
        #                     nn.Dropout2d(0.3),
        #                     nn.ReLU()
        #                   ).to(dev)
        # elif corruption_name == "rotate_fixed" or corruption_name == "scale" or corruption_name == "shear_fixed":
        #     filter_bank = nn.Sequential(
        #                     nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
        #                     # nn.BatchNorm2d(128),
        #                     nn.Dropout2d(0.5),
        #                     nn.ReLU(),
        #                     nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
        #                     # nn.BatchNorm2d(128),
        #                     nn.Dropout2d(0.5),
        #                     nn.ReLU()
        #                   ).to(dev)
        # else:
        #     raise ValueError("Corruption name not recognised")

        filter_bank_one = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.Dropout2d(0.3),
            nn.ReLU()
        ).to(dev)

        filter_bank_two = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU()
        ).to(dev)

        filter_bank_three = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.ReLU()
        ).to(dev)

        filter_bank_four = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        ).to(dev)



        """
        Experiment with automating the level of abstraction 
        """
        # networks = [network_one, network_two, network_three, network_four, network_five]
        # filter_banks = [filter_bank_one, filter_bank_two, filter_bank_three, filter_bank_four]
        # Todo: is there an issue where the gradients are different magnitudes at different levels of abstraction?
        # Todo: do the levels of abstraction match with the engineering ones
        # criterion = nn.CrossEntropyLoss()
        # filter_bank_results = []
        # for i, filter_bank in enumerate(filter_banks):
        #     optim = torch.optim.Adam(filter_bank.parameters(), lr)
        #     filter_bank.train()
        #     epoch_ce_loss = 0.0
        #     epoch_inv_loss = 0.0
        #     epoch_acc = 0.0
        #     for j, data_tuples in enumerate(zip(*trn_dls)):
        #         for k, data_tuple in enumerate(data_tuples):
        #             if k == 0:
        #                 x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
        #             else:
        #                 x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
        #                 x_trn = torch.cat((x_trn, x_temp), dim=0)
        #                 y_trn = torch.cat((y_trn, y_temp), dim=0)
        #
        #         optim.zero_grad()
        #
        #         # Different location serial filter bank case
        #         features = network_one(x_trn)
        #
        #         # Todo: if looks promising this can definitely be made a lot more beautiful
        #         # Todo: experiment with dropout, batch norm, capacity, invariance loss, etc.
        #         # Todo: major weirdness in the implementation - we calculate the CE loss on clean and corrupted data!!!!
        #         #   this means we are asking that the module "does nothing" on the clean data. This is not what we want.
        #         #   this is another thing to try with and without and see if it improves performance over contrastive losstraining
        #         # Todo: Monday - careful implementation of all these methods and flexibility - include contrastive loss, supervised contrastive loss, maybe indep. mechanisms etc.
        #         # Todo: With auto selection may need to check the variance of the accuracy over runs. If it is too high different levels of abstraction may be selected.
        #         if i == 0:
        #             id_features = features[:single_corr_bs, :, :, :]
        #             id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
        #             features = filter_bank(features)
        #             corr_features = features[single_corr_bs:, :, :, :]
        #             corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
        #             inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                   single_corr_bs,
        #                                                   dev, temperature=temperature, compare=(1, 2))
        #             output = network_five(network_four(network_three(network_two(features))))
        #         elif i == 1:
        #             features = network_two(features)
        #             id_features = features[:single_corr_bs, :, :, :]
        #             id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
        #             features = filter_bank(features)
        #             corr_features = features[single_corr_bs:, :, :, :]
        #             corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
        #             inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                   single_corr_bs,
        #                                                   dev, temperature=temperature, compare=(1, 2))
        #             output = network_five(network_four(network_three(features)))
        #         elif i == 2:
        #             features = network_three(network_two(features))
        #             id_features = features[:single_corr_bs, :, :, :]
        #             id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
        #             features = filter_bank(features)
        #             corr_features = features[single_corr_bs:, :, :, :]
        #             corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
        #             inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                   single_corr_bs,
        #                                                   dev, temperature=temperature, compare=(1, 2))
        #             output = network_five(network_four(features))
        #         elif i == 3:
        #             features = network_four(network_three(network_two(features)))
        #             # Todo: check this with some print statements - not thought carefully about the modules in the fully connected case
        #             id_features = features[:single_corr_bs, :]
        #             id_features = id_features.reshape(id_features.shape[0], -1)
        #             features = filter_bank(features)
        #             corr_features = features[single_corr_bs:, :]
        #             corr_features = corr_features.reshape(corr_features.shape[0], -1)
        #             inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                   single_corr_bs,
        #                                                   dev, temperature=temperature, compare=(1, 2))
        #             output = network_five(features)
        #         else:
        #             raise RuntimeError("Invalid filter bank index {}".format(i))
        #
        #         ce_loss = criterion(output, y_trn)
        #         loss = ce_loss + inv_loss
        #         acc = accuracy(output, y_trn)
        #         loss.backward()
        #         optim.step()
        #         epoch_ce_loss += ce_loss.item()
        #         epoch_inv_loss += inv_loss.item()
        #         epoch_acc += acc
        #
        #         if j == 49:
        #             print("###############")
        #             print("50 minibatches training done for filter bank {}".format(i))
        #             break
        #     results = [epoch_ce_loss / 50,
        #                epoch_inv_loss / 50,
        #                epoch_acc / 50]
        #     filter_bank_results.append(results)
        #
        #     print("Avg CE train loss {:6.4f}. Avg inv train loss {:6.4f}.".format(results[0], results[1]))
        #     print("Avg train acc {:6.3f}.".format(results[2]))
        #
        #     # Validation
        #     filter_bank.eval()
        #     valid_ce_loss = 0.0
        #     valid_inv_loss = 0.0
        #     valid_loss = 0.0
        #     valid_acc = 0.0
        #     with torch.no_grad():
        #         for j, data_tuples in enumerate(zip(*val_dls)):
        #             for k, data_tuple in enumerate(data_tuples):
        #                 if k == 0:
        #                     x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
        #                 else:
        #                     x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
        #                     x_val = torch.cat((x_val, x_temp), dim=0)
        #                     y_val = torch.cat((y_val, y_temp), dim=0)
        #
        #             features = network_one(x_val)
        #             if i == 0:
        #                 id_features = features[:single_corr_bs, :, :, :]
        #                 id_features = id_features.reshape(id_features.shape[0],
        #                                                   -1)  # Flatten spatial dimensions
        #                 features = filter_bank(features)
        #                 corr_features = features[single_corr_bs:, :, :, :]
        #                 corr_features = corr_features.reshape(corr_features.shape[0],
        #                                                       -1)  # Flatten spatial dimensions
        #                 inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                       single_corr_bs,
        #                                                       dev, temperature=temperature, compare=(1, 2))
        #                 output = network_five(network_four(network_three(network_two(features))))
        #             elif i == 1:
        #                 features = network_two(features)
        #                 id_features = features[:single_corr_bs, :, :, :]
        #                 id_features = id_features.reshape(id_features.shape[0],
        #                                                   -1)  # Flatten spatial dimensions
        #                 features = filter_bank(features)
        #                 corr_features = features[single_corr_bs:, :, :, :]
        #                 corr_features = corr_features.reshape(corr_features.shape[0],
        #                                                       -1)  # Flatten spatial dimensions
        #                 inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                       single_corr_bs,
        #                                                       dev, temperature=temperature, compare=(1, 2))
        #                 output = network_five(network_four(network_three(features)))
        #             elif i == 2:
        #                 features = network_three(network_two(features))
        #                 id_features = features[:single_corr_bs, :, :, :]
        #                 id_features = id_features.reshape(id_features.shape[0],
        #                                                   -1)  # Flatten spatial dimensions
        #                 features = filter_bank(features)
        #                 corr_features = features[single_corr_bs:, :, :, :]
        #                 corr_features = corr_features.reshape(corr_features.shape[0],
        #                                                       -1)  # Flatten spatial dimensions
        #                 inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                       single_corr_bs,
        #                                                       dev, temperature=temperature, compare=(1, 2))
        #                 output = network_five(network_four(features))
        #             elif i == 3:
        #                 features = network_four(network_three(network_two(features)))
        #                 # Todo: check this with some print statements - not thought carefully about the modules in the fully connected case
        #                 id_features = features[:single_corr_bs, :]
        #                 id_features = id_features.reshape(id_features.shape[0], -1)
        #                 features = filter_bank(features)
        #                 corr_features = features[single_corr_bs:, :]
        #                 corr_features = corr_features.reshape(corr_features.shape[0], -1)
        #                 inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
        #                                                       single_corr_bs,
        #                                                       dev, temperature=temperature, compare=(1, 2))
        #                 output = network_five(features)
        #             else:
        #                 raise RuntimeError("Invalid filter bank index {}".format(i))
        #
        #             ce_loss = criterion(output, y_val)
        #             loss = ce_loss + inv_loss
        #             acc = accuracy(output, y_val)
        #             valid_ce_loss += ce_loss.item()
        #             valid_inv_loss += inv_loss.item()
        #             valid_loss += loss.item()
        #             valid_acc += acc
        #
        #             if j == 49:
        #                 print("50 minibatches validation done for filter bank {}".format(i))
        #                 break
        #             # results = [epoch_ce_loss / 20,
        #             #            epoch_inv_loss / 20,
        #             #            epoch_acc / 20]
        #             # filter_bank_results.append(results)
        #         print("Validation CE loss {:6.4f}".format(valid_ce_loss / 50))
        #         print("Validation inv loss {:6.4f}".format(valid_inv_loss / 50))
        #         print("Validation loss {:6.4f}".format(valid_loss / 50))
        #         print("Validation acc {:6.3f}".format(valid_acc / 50))
        #     # Do we add a resad as an option???
        #
        #     # logger.info("Epoch {}. Avg CE train loss {:6.4f}. Avg inv train loss {:6.4f}. "
        #     #             "Avg train acc {:6.3f}.".format(*results))
        #
        # print(2/0)
        """
        End experiment with automating the level of abstraction 
        """

        if corruption_name == "impulse_noise" or corruption_name == "gaussian_blur" or corruption_name == "inverse":
            filter_bank = filter_bank_one
        elif corruption_name == "rotate_fixed" or corruption_name == "scale" or corruption_name == "shear_fixed":
            filter_bank = filter_bank_three
        else:
            raise ValueError("Corruption name not recognised")

        optim = torch.optim.Adam(filter_bank.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        es_ckpt_path = os.path.join(ckpt_path, "es_ckpt_{}_{}.pt".format('-'.join(corruption_names), experiment_name))
        early_stopping = EarlyStopping(patience=25, verbose=True, path=es_ckpt_path)

        # Training
        for epoch in range(max_epochs):
            # Training
            filter_bank.train()
            epoch_ce_loss = 0.0
            epoch_inv_loss = 0.0
            epoch_acc = 0.0
            for data_tuples in zip(*trn_dls):
                for i, data_tuple in enumerate(data_tuples):
                    if i == 0:
                        x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    else:
                        x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
                        x_trn = torch.cat((x_trn, x_temp), dim=0)
                        y_trn = torch.cat((y_trn, y_temp), dim=0)

                optim.zero_grad()

                # features = network_one(x_trn)
                # id_features = features[:single_corr_bs, :, :, :]
                # id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions

                # Most cases
                # features = filter_bank(features)
                # corr_features = features[single_corr_bs:, :, :, :]
                # corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                # inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0), single_corr_bs,
                #                                       dev, temperature=temperature, compare=(1, 2))
                # output = network_two(features)

                # Parallel/residual adapter case
                # resad_features = filter_bank(x_trn) + features
                # corr_features = resad_features[single_corr_bs:, :, :, :]
                # corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                # inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0), single_corr_bs,
                #                                       dev, temperature=temperature, compare=(1, 2))
                # output = network_two(resad_features)

                # Different location adapter case
                # if corruption_name == "impulse_noise":
                #     features = network_one(x_trn)
                #     id_features = features[:single_corr_bs, :, :, :]
                #     id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                #     resad_features = filter_bank(x_trn) + features
                #     corr_features = resad_features[single_corr_bs:, :, :, :]
                #     corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                #     inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                #                                           single_corr_bs,
                #                                           dev, temperature=temperature, compare=(1, 2))
                #     output = network_four(network_three(network_two(resad_features)))
                # else:
                #     pre_adapt_features = network_two(network_one(x_trn))
                #     features = network_three(pre_adapt_features)
                #     id_features = features[:single_corr_bs, :, :, :]
                #     id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                #     resad_features = filter_bank(pre_adapt_features) + features
                #     corr_features = resad_features[single_corr_bs:, :, :, :]
                #     corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                #     inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                #                                           single_corr_bs,
                #                                           dev, temperature=temperature, compare=(1, 2))
                #     output = network_four(resad_features)

                # Different location serial filter bank case
                features = network_one(x_trn)
                if corruption_name == "impulse_noise" or corruption_name == "gaussian_blur" or corruption_name == "inverse":
                    id_features = features[:single_corr_bs, :, :, :]
                    id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                    features = filter_bank(features)
                    corr_features = features[single_corr_bs:, :, :, :]
                    corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0), single_corr_bs,
                                                          dev, temperature=temperature, compare=(1, 2))
                    output = network_five(network_four(network_three(network_two(features))))
                else:
                    features = network_three(network_two(features))
                    id_features = features[:single_corr_bs, :, :, :]
                    id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                    features = filter_bank(features)
                    corr_features = features[single_corr_bs:, :, :, :]
                    corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                                                          single_corr_bs,
                                                          dev, temperature=temperature, compare=(1, 2))
                    output = network_five(network_four(features))

                ce_loss = criterion(output, y_trn)
                loss = ce_loss + inv_loss
                acc = accuracy(output, y_trn)
                loss.backward()
                optim.step()
                epoch_ce_loss += ce_loss.item()
                epoch_inv_loss += inv_loss.item()
                epoch_acc += acc
            results = [epoch,
                       epoch_ce_loss / len(trn_dls[0]),
                       epoch_inv_loss / len(trn_dls[0]),
                       epoch_acc / len(trn_dls[0])]
            logger.info("Epoch {}. Avg CE train loss {:6.4f}. Avg inv train loss {:6.4f}. "
                        "Avg train acc {:6.3f}.".format(*results))

            # Validation
            filter_bank.eval()
            valid_ce_loss = 0.0
            valid_inv_loss = 0.0
            valid_loss = 0.0
            valid_acc = 0.0
            with torch.no_grad():
                for data_tuples in zip(*val_dls):
                    for i, data_tuple in enumerate(data_tuples):
                        if i == 0:
                            x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                        else:
                            x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
                            x_val = torch.cat((x_val, x_temp), dim=0)
                            y_val = torch.cat((y_val, y_temp), dim=0)

                    # features = network_one(x_val)
                    # id_features = features[:single_corr_bs, :, :, :]
                    # id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions

                    # Most cases
                    # features = filter_bank(features)
                    # corr_features = features[single_corr_bs:, :, :, :]
                    # corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    # inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                    #                                       single_corr_bs, dev, temperature=temperature, compare=(1, 2))
                    # output = network_two(features)

                    # Parallel/residual adapter case
                    # resad_features = filter_bank(x_val) + features
                    # corr_features = resad_features[single_corr_bs:, :, :, :]
                    # corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    # inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                    #                                       single_corr_bs, dev, temperature=temperature, compare=(1, 2))
                    # output = network_two(resad_features)

                    # Different location adapter case
                    # if corruption_name == "impulse_noise":
                    #     features = network_one(x_val)
                    #     id_features = features[:single_corr_bs, :, :, :]
                    #     id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                    #     resad_features = filter_bank(x_val) + features
                    #     corr_features = resad_features[single_corr_bs:, :, :, :]
                    #     corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    #     inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                    #                                           single_corr_bs,
                    #                                           dev, temperature=temperature, compare=(1, 2))
                    #     output = network_four(network_three(network_two(resad_features)))
                    # else:
                    #     pre_adapt_features = network_two(network_one(x_val))
                    #     features = network_three(pre_adapt_features)
                    #     id_features = features[:single_corr_bs, :, :, :]
                    #     id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                    #     resad_features = filter_bank(pre_adapt_features) + features
                    #     corr_features = resad_features[single_corr_bs:, :, :, :]
                    #     corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    #     inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                    #                                           single_corr_bs,
                    #                                           dev, temperature=temperature, compare=(1, 2))
                    #     output = network_four(resad_features)

                    # Different location serial filter bank case
                    features = network_one(x_val)
                    if corruption_name == "impulse_noise" or corruption_name == "gaussian_blur" or corruption_name == "inverse":
                        id_features = features[:single_corr_bs, :, :, :]
                        id_features = id_features.reshape(id_features.shape[0],
                                                          -1)  # Flatten spatial dimensions
                        features = filter_bank(features)
                        corr_features = features[single_corr_bs:, :, :, :]
                        corr_features = corr_features.reshape(corr_features.shape[0],
                                                              -1)  # Flatten spatial dimensions
                        inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                                                              single_corr_bs,
                                                              dev, temperature=temperature, compare=(1, 2))
                        output = network_five(network_four(network_three(network_two(features))))
                    else:
                        features = network_three(network_two(features))
                        id_features = features[:single_corr_bs, :, :, :]
                        id_features = id_features.reshape(id_features.shape[0],
                                                          -1)  # Flatten spatial dimensions
                        features = filter_bank(features)
                        corr_features = features[single_corr_bs:, :, :, :]
                        corr_features = corr_features.reshape(corr_features.shape[0],
                                                              -1)  # Flatten spatial dimensions
                        inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                                                              single_corr_bs,
                                                              dev, temperature=temperature, compare=(1, 2))
                        output = network_five(network_four(features))

                    ce_loss = criterion(output, y_val)
                    loss = ce_loss + inv_loss
                    acc = accuracy(output, y_val)
                    valid_ce_loss += ce_loss.item()
                    valid_inv_loss += inv_loss.item()
                    valid_loss += loss.item()
                    valid_acc += acc
            logger.info("Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
            logger.info("Validation inv loss {:6.4f}".format(valid_inv_loss / len(val_dls[0])))
            logger.info("Validation total loss {:6.4f}".format(valid_loss / len(val_dls[0])))
            logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))

            # Early Stopping
            if epoch > min_epochs:
                early_stopping(valid_loss / len(val_dls[0]), filter_bank)  # ES on loss
                # early_stopping(100. - (valid_acc / len(val_dl), filter_bank)  # ES on acc
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

        # Save model
        logger.info("Loading early stopped checkpoint")
        early_stopping.load_from_checkpoint(filter_bank)
        filter_bank.eval()
        valid_ce_loss = 0.0
        valid_inv_loss = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuples in zip(*val_dls):
                for i, data_tuple in enumerate(data_tuples):
                    if i == 0:
                        x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    else:
                        x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
                        x_val = torch.cat((x_val, x_temp), dim=0)
                        y_val = torch.cat((y_val, y_temp), dim=0)

                # features = network_one(x_val)
                # id_features = features[:single_corr_bs, :, :, :]
                # id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions

                # Most cases
                # features = filter_bank(features)
                # corr_features = features[single_corr_bs:, :, :, :]
                # corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                # inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                #                                       single_corr_bs, dev, temperature=temperature, compare=(1, 2))
                # output = network_two(features)

                # Parallel/residual adapter case
                # resad_features = filter_bank(x_val) + features
                # corr_features = resad_features[single_corr_bs:, :, :, :]
                # corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                # inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                #                                       single_corr_bs, dev, temperature=temperature, compare=(1, 2))
                # output = network_two(resad_features)

                # Different location adapter case
                # if corruption_name == "impulse_noise":
                #     features = network_one(x_val)
                #     id_features = features[:single_corr_bs, :, :, :]
                #     id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                #     resad_features = filter_bank(x_val) + features
                #     corr_features = resad_features[single_corr_bs:, :, :, :]
                #     corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                #     inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                #                                           single_corr_bs,
                #                                           dev, temperature=temperature, compare=(1, 2))
                #     output = network_four(network_three(network_two(resad_features)))
                # else:
                #     pre_adapt_features = network_two(network_one(x_val))
                #     features = network_three(pre_adapt_features)
                #     id_features = features[:single_corr_bs, :, :, :]
                #     id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                #     resad_features = filter_bank(pre_adapt_features) + features
                #     corr_features = resad_features[single_corr_bs:, :, :, :]
                #     corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                #     inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                #                                           single_corr_bs,
                #                                           dev, temperature=temperature, compare=(1, 2))
                #     output = network_four(resad_features)

                # Different location serial filter bank case
                features = network_one(x_val)
                if corruption_name == "impulse_noise" or corruption_name == "gaussian_blur" or corruption_name == "inverse":
                    id_features = features[:single_corr_bs, :, :, :]
                    id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                    features = filter_bank(features)
                    corr_features = features[single_corr_bs:, :, :, :]
                    corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                                                          single_corr_bs,
                                                          dev, temperature=temperature, compare=(1, 2))
                    output = network_five(network_four(network_three(network_two(features))))
                else:
                    features = network_three(network_two(features))
                    id_features = features[:single_corr_bs, :, :, :]
                    id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions
                    features = filter_bank(features)
                    corr_features = features[single_corr_bs:, :, :, :]
                    corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions
                    inv_loss = 1 * contrastive_layer_loss(torch.cat((id_features, corr_features), dim=0),
                                                          single_corr_bs,
                                                          dev, temperature=temperature, compare=(1, 2))
                    output = network_five(network_four(features))

                ce_loss = criterion(output, y_val)
                loss = ce_loss + inv_loss
                acc = accuracy(output, y_val)
                valid_ce_loss += ce_loss.item()
                valid_inv_loss += inv_loss.item()
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
        logger.info("Early Stopped validation inv loss {:6.4f}".format(valid_inv_loss / len(val_dls[0])))
        logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dls[0])))
        logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
        early_stopping.delete_checkpoint()  # Removes from disk
        torch.save(filter_bank.state_dict(), os.path.join(ckpt_path, ckpt_name))
        logger.info("Saved best model to {}".format(os.path.join(ckpt_path, ckpt_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to train networks on different corruptions.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST3/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST_TEMP/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--logging-path', type=str, default='/om2/user/imason/compositions/logs/EMNIST_TEMP/',
                        help="path to directory to save logs")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/EMNIST_TEMP/visualisations/',
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
    parser.add_argument('--experiment-name', type=str, default='filter-bank-invariance',
                        help="name of experiment - used to name the checkpoint and log files")
    parser.add_argument('--weights', type=str, help="comma delimited string of weights for the loss functions")
    parser.add_argument('--compare-corrs', type=str,
                        help="comma delimited string of which corrs to use as postive pairs")
    parser.add_argument('--corruption-ID', type=int, default=0, help="which corruption to generate")
    args = parser.parse_args()

    """
    CUDA_VISIBLE_DEVICES=4 python iteration_train_compositions_residual_invariance.py --min-epochs 1 --max-epochs 3 --pin-mem --check-if-run --vis-data --corruption-ID 0 --experiment-name joint-train
    """

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
    CUDA_VISIBLE_DEVICES=0 python train_compositions_residual_invariance.py --pin-mem --check-if-run --vis-data --corruption-ID 0 
    CUDA_VISIBLE_DEVICES=1 python train_compositions_residual_invariance.py --pin-mem --check-if-run --vis-data --corruption-ID 1
    CUDA_VISIBLE_DEVICES=2 python train_compositions_residual_invariance.py --pin-mem --check-if-run --vis-data --corruption-ID 2
    CUDA_VISIBLE_DEVICES=3 python train_compositions_residual_invariance.py --pin-mem --check-if-run --vis-data --corruption-ID 3
    CUDA_VISIBLE_DEVICES=4 python train_compositions_residual_invariance.py --pin-mem --check-if-run --vis-data --corruption-ID 4
    CUDA_VISIBLE_DEVICES=5 python train_compositions_residual_invariance.py --pin-mem --check-if-run --vis-data --corruption-ID 5 
    """

    assert args.n_workers != 0  # Seems this is a problem - needs investigating
    main(corruptions, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.total_n_classes,
         args.min_epochs, args.max_epochs, args.batch_size, args.lr, weights, args.temperature, compare_corrs,
         args.experiment_name, args.n_workers, args.pin_mem, dev, args.vis_data, args.check_if_run)
