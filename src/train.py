"""
Train a model on individual data transformations, pairs of transformations and all transformations. .
"""
import argparse
import os
import pickle
import torch
import torch.nn as nn
import time
from data.data_transforms import denormalize
from data.data_loaders import get_multi_static_dataloaders, get_static_dataloaders
from lib.networks import create_emnist_network, create_emnist_modules, create_emnist_autoencoder, \
                         create_cifar_network, create_cifar_modules, create_cifar_autoencoder, \
                         create_facescrub_network, create_facescrub_modules, create_facescrub_autoencoder
from lib.early_stopping import EarlyStopping
from lib.contrastive_loss import ContrastiveLayerLoss
from lib.forwards_passes import *
from lib.utils import *


def train_identity_network(network_blocks, network_block_ckpt_names, dataset, data_root, ckpt_path, logging_path,
                           experiment, total_n_classes, max_epochs, batch_size, lr, n_workers, pin_mem,
                           dev):
    # Log File set up
    log_name = "{}_Identity.log".format(experiment)
    id_logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
    print("Logging file created for experiment {} to train on Identity".format(experiment))

    # Data Set Up
    corruption_path = os.path.join(data_root, "Identity")
    train_classes = list(range(total_n_classes))
    trn_dl, val_dl, _ = get_static_dataloaders(dataset, corruption_path, train_classes, batch_size, True, n_workers,
                                               pin_mem)

    # Network & Optimizer Set Up
    all_parameters = []
    for block in network_blocks:
        all_parameters += list(block.parameters())

    # optim = torch.optim.Adam(all_parameters, lr)  # Todo: used this for original EMNIST - test EMNIST with new set up
    optim = torch.optim.SGD(all_parameters, lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

    # Early Stopping Set Up
    es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(block_ckpt_name)) for block_ckpt_name in
                     network_block_ckpt_names]
    early_stoppings = [EarlyStopping(patience=max_epochs, path=es_ckpt_path, trace_func=id_logger.info) for
                       es_ckpt_path in es_ckpt_paths]
    early_stoppings[0].verbose = True
    assert len(early_stoppings) == len(network_blocks)

    # Loss Function Set Up
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = lambda x, y: accuracy(x, y)

    # Training Loop
    for epoch in range(max_epochs):
        for block in network_blocks:
            block.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for i, data_tuple in enumerate(trn_dl, 1):
            x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
            optim.zero_grad()
            loss, acc = cross_entropy_forwards_pass(network_blocks, x_trn, y_trn, cross_entropy_loss_fn, accuracy_fn)
            epoch_loss += loss.item()
            epoch_acc += acc
            loss.backward()
            optim.step()
        scheduler.step()
        # print(optim.param_groups[0]['lr'])
        results = [epoch,
                   epoch_loss / len(trn_dl),
                   epoch_acc / len(trn_dl)]
        id_logger.info("Identity Network. Epoch {}. Avg CE train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

        # Validation
        for block in network_blocks:
            block.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuple in val_dl:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val, cross_entropy_loss_fn, accuracy_fn)
                valid_loss += loss.item()
                valid_acc += acc
        id_logger.info("Validation CE loss {:6.4f}".format(valid_loss / len(val_dl)))
        id_logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
        # Early Stopping
        for es, block in zip(early_stoppings, network_blocks):
            es(valid_loss / len(val_dl), block)  # ES on loss
        if early_stoppings[0].early_stop:
            id_logger.info("Early stopping")
            break

    # Save model
    id_logger.info("Loading early stopped checkpoints")
    for es, block in zip(early_stoppings, network_blocks):
        es.load_from_checkpoint(block)
    for block in network_blocks:
        block.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        for data_tuple in val_dl:
            x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
            loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val, cross_entropy_loss_fn, accuracy_fn)
            valid_loss += loss.item()
            valid_acc += acc
    id_logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_loss / len(val_dl)))
    id_logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))

    for es in early_stoppings:
        es.delete_checkpoint()  # Removes from disk
    for block_ckpt_name, block in zip(network_block_ckpt_names, network_blocks):
        torch.save(block.state_dict(), os.path.join(ckpt_path, block_ckpt_name))
        id_logger.info("Saved best identity network block to {}".format(os.path.join(ckpt_path, block_ckpt_name)))

    return network_blocks


def train_classifiers(dataset, data_root, ckpt_path, logging_path, experiment, total_n_classes, max_epochs,
                      batch_size, lr, n_workers, pin_mem, dev, check_if_run):
    """
    Trains 2 classifiers for use with autoencoders designed to remove each corruption.

    Classifier 1 is simply trained on the identity data (assuming that all the autoencoders work perfectly)

    Classifier 2 is trained on all corruptions jointly, passing corrupted data through the corresponding autoencoder
    and then classifying the output.
    """

    # Hardcoded, a bit annoying to change the logic to make it not hardcoded
    elemental_corruptions = ["Contrast", "GaussianBlur", "ImpulseNoise", "Invert", "Rotate90", "Swirl"]
    for elem_corr in elemental_corruptions:
        if not os.path.exists(os.path.join(ckpt_path, "{}_Encoder_{}-Identity.pt".format(experiment, elem_corr))):
            raise RuntimeError("No autoencoder found for corruption: {}".format(elem_corr))

    # Train classifier on clean (Identity) data
    if dataset == "EMNIST":
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment + "Classifier",
                                                                         ["Identity"], dev)
    elif dataset == "CIFAR":
        network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, experiment + "Classifier",
                                                                        ["Identity"], dev)
    elif dataset == "FACESCRUB":
        network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, experiment + "Classifier",
                                                                            ["Identity"], dev)

    if check_if_run and os.path.exists(os.path.join(ckpt_path, network_block_ckpt_names[0])):
        print("Checkpoint already exists at {} \nSkipping training classifier on Identity".format(
            os.path.join(ckpt_path, network_block_ckpt_names[0])))
    else:
        train_identity_network(network_blocks, network_block_ckpt_names, dataset, data_root, ckpt_path, logging_path,
                               experiment + "Classifier", total_n_classes, max_epochs, batch_size, lr,
                               n_workers, pin_mem, dev)

    # Train classifier on all corruptions
    corruption_paths = [os.path.join(data_root, elem_corr) for elem_corr in elemental_corruptions]
    corruption_paths += [os.path.join(data_root, "Identity")]
    train_classes = list(range(total_n_classes))
    generators = [torch.Generator(device='cpu').manual_seed(2147483647) for _ in range(len(corruption_paths))]
    single_corr_bs = batch_size // len(corruption_paths)
    trn_dls, val_dls = [], []
    for corruption_path, generator in zip(corruption_paths, generators):
        trn_dl, val_dl, _ = get_static_dataloaders(dataset, corruption_path, train_classes, single_corr_bs,
                                                   True, n_workers, pin_mem, fixed_generator=generator)
        trn_dls.append(trn_dl)
        val_dls.append(val_dl)

    # Create Network
    if dataset == "EMNIST":
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment + "Classifier",
                                                                         elemental_corruptions + ["Identity"], dev)
    elif dataset == "CIFAR":
        network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, experiment + "Classifier",
                                                                        elemental_corruptions + ["Identity"], dev)
    elif dataset == "FACESCRUB":
        network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, experiment + "Classifier",
                                                                            elemental_corruptions + ["Identity"], dev)

    if check_if_run and os.path.exists(os.path.join(ckpt_path, network_block_ckpt_names[0])):
        print("Checkpoint already exists at {} \nSkipping training classifier on {}".format(
            os.path.join(ckpt_path, network_block_ckpt_names[0]), elemental_corruptions + ["Identity"]))
        return

    # Log File set up
    log_name = "{}_JointClassifier.log".format(experiment)
    clsf_logger = custom_logger(os.path.join(logging_path, log_name),
                                stdout=False)  # stdout True to see also in console
    print("Logging file created for experiment {} to train the joint classifier".format(experiment))

    # Network & Optimizer Set Up
    all_parameters = []
    for block in network_blocks:
        all_parameters += list(block.parameters())

    # optim = torch.optim.Adam(all_parameters, lr)  # Todo: used for EMNIST, test without
    optim = torch.optim.SGD(all_parameters, lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

    # Early Stopping Set Up
    es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(block_ckpt_name)) for block_ckpt_name in
                     network_block_ckpt_names]
    early_stoppings = [EarlyStopping(patience=max_epochs, path=es_ckpt_path, trace_func=clsf_logger.info) for
                       es_ckpt_path in es_ckpt_paths]
    early_stoppings[0].verbose = True
    assert len(early_stoppings) == len(network_blocks)
    val_freq = len(trn_dls[0]) // len(corruption_paths)
    clsf_logger.info("Validation frequency: every {} batches".format(val_freq))

    # Loss Function Set Up
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = lambda x, y: accuracy(x, y)

    # Load all trained autoencoders
    corruption_ae_blocks = []
    corruption_ae_block_ckpt_names = []
    for elem_corr in elemental_corruptions:
        if dataset == "EMNIST":
            blocks, block_ckpt_names = create_emnist_autoencoder(experiment, [elem_corr, "Identity"], dev)
        elif dataset == "CIFAR":
            blocks, block_ckpt_names = create_cifar_autoencoder(experiment, [elem_corr, "Identity"], dev)
        elif dataset == "FACESCRUB":
            blocks, block_ckpt_names = create_facescrub_autoencoder(experiment, [elem_corr, "Identity"], dev)
        corruption_ae_blocks.append(blocks)
        corruption_ae_block_ckpt_names.append(block_ckpt_names)
    for blocks, block_ckpt_names in zip(corruption_ae_blocks, corruption_ae_block_ckpt_names):
        for block, block_ckpt_name in zip(blocks, block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
            block.eval()

    # Training Loop
    for epoch in range(max_epochs):
        for block in network_blocks:
            block.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for i, trn_data_tuples in enumerate(zip(*trn_dls), 1):
            x_trn, y_trn = generate_batch(trn_data_tuples, dev)
            decoded_x = []
            for j, elem_corr in enumerate(elemental_corruptions + ["Identity"]):
                x_ae = x_trn[j * single_corr_bs:(j + 1) * single_corr_bs, :, :, :]
                if elem_corr != "Identity":
                    for block in corruption_ae_blocks[j]:
                        x_ae = block(x_ae)
                decoded_x.append(x_ae)
            x_trn = torch.cat(decoded_x, dim=0)

            optim.zero_grad()
            loss, acc = cross_entropy_forwards_pass(network_blocks, x_trn, y_trn, cross_entropy_loss_fn, accuracy_fn)
            epoch_loss += loss.item()
            epoch_acc += acc
            loss.backward()
            optim.step()

            if i % val_freq == 0:
                # Validation
                for block in network_blocks:
                    block.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                with torch.no_grad():
                    for val_data_tuples in zip(*val_dls):
                        x_val, y_val = generate_batch(val_data_tuples, dev)
                        decoded_x = []
                        for j, elem_corr in enumerate(elemental_corruptions + ["Identity"]):
                            x_ae = x_val[j * single_corr_bs:(j + 1) * single_corr_bs, :, :, :]
                            if elem_corr != "Identity":
                                for block in corruption_ae_blocks[j]:
                                    x_ae = block(x_ae)
                            decoded_x.append(x_ae)
                        x_val = torch.cat(decoded_x, dim=0)
                        loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val, cross_entropy_loss_fn,
                                                                accuracy_fn)
                        valid_loss += loss.item()
                        valid_acc += acc
                clsf_logger.info("Validation CE loss {:6.4f}".format(valid_loss / len(val_dls[0])))
                clsf_logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))

                # Early Stopping
                for es, block in zip(early_stoppings, network_blocks):
                    es(valid_loss / len(val_dls[0]), block)  # ES on loss
                if early_stoppings[0].early_stop:
                    clsf_logger.info("Early stopping")
                    break
                for block in network_blocks:
                    block.train()
        scheduler.step()
        if early_stoppings[0].early_stop:
            break
        results = [epoch,
                   epoch_loss / len(trn_dls[0]),
                   epoch_acc / len(trn_dls[0])]
        clsf_logger.info("Classifier. Epoch {}. Avg CE train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

    # Save model
    clsf_logger.info("Loading early stopped checkpoints")
    for es, block in zip(early_stoppings, network_blocks):
        es.load_from_checkpoint(block)
    for block in network_blocks:
        block.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        for val_data_tuples in zip(*val_dls):
            x_val, y_val = generate_batch(val_data_tuples, dev)
            decoded_x = []
            for j, elem_corr in enumerate(elemental_corruptions + ["Identity"]):
                x_ae = x_val[j * single_corr_bs:(j + 1) * single_corr_bs, :, :, :]
                if elem_corr != "Identity":
                    for block in corruption_ae_blocks[j]:
                        x_ae = block(x_ae)
                decoded_x.append(x_ae)
            x_val = torch.cat(decoded_x, dim=0)
            loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val, cross_entropy_loss_fn,
                                                    accuracy_fn)
            valid_loss += loss.item()
            valid_acc += acc
    clsf_logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_loss / len(val_dls[0])))
    clsf_logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
    for es in early_stoppings:
        es.delete_checkpoint()  # Removes from disk
    for block_ckpt_name, block in zip(network_block_ckpt_names, network_blocks):
        torch.save(block.state_dict(), os.path.join(ckpt_path, block_ckpt_name))
        clsf_logger.info("Saved best classifier block to {}".format(os.path.join(ckpt_path, block_ckpt_name)))


def get_contrastive_abstraction_levels(experiment, corruption_names, dataset, trn_dls, val_dls, lr,
                                       cross_entropy_loss_fn, accuracy_fn, contrastive_loss_fn, weight, total_n_classes,
                                       single_corr_bs, dev):
    if "Auto" in experiment:
        contrastive_levels = find_contrastive_abstraction_levels(corruption_names, dataset, trn_dls, val_dls,
                                                                 lr, cross_entropy_loss_fn, accuracy_fn,
                                                                 contrastive_loss_fn, weight, total_n_classes,
                                                                 single_corr_bs, dev)
    elif "ModLevel" in experiment:  # Hardcoded to match the best AutoModules levels of abstraction seen so far
        if dataset == "EMNIST":
            contrastive_levels = [1, 1, 1, 2, 3, 4]
        elif dataset == "CIFAR":  # Todo: unknown at the moment
            contrastive_levels = [1, 1, 1, 2, 3, 4]
        elif dataset == "FACESCRUB":  # Todo: unknown at the moment
            contrastive_levels = [1, 1, 1, 2, 3, 4]
    else:  # Penultimate layer
        if dataset == "EMNIST":
            contrastive_levels = [5] * (len(corruption_names) - 1)
        elif dataset == "CIFAR":
            contrastive_levels = [9] * (len(corruption_names) - 1)
        elif dataset == "FACESCRUB":
            contrastive_levels = [16] * (len(corruption_names) - 1)

    return contrastive_levels


def find_contrastive_abstraction_levels(corruption_names, dataset, trn_dls, val_dls, lr, cross_entropy_loss_fn,
                                        accuracy_fn, contrastive_loss_fn, weight, total_n_classes, single_corr_bs, dev,
                                        num_iterations=200, num_repeats=3):
    """
    Finds the best level to apply the contrastive loss at for all given corruptions.

    Returns a list of abstraction levels with the best level for each corruption in the same order as the corruptions.

    e.g. if corruption_names = ["Invert", "Rotate90", "Identity"] the return will look like [1,3]
         there is no level of abstraction for Identity

    Hardcoded assumptions
    1. the batch size is a multiple of single_corr_bs
    2. the identity data is the last corruption in the batch
    """
    if dataset == "EMNIST":
        network_blocks, _ = create_emnist_network(total_n_classes, "temp", "temp", dev)
    elif dataset == "CIFAR":
        network_blocks, _ = create_cifar_network(total_n_classes, "temp", "temp", dev)
    elif dataset == "FACESCRUB":
        network_blocks, _ = create_facescrub_network(total_n_classes, "temp", "temp", dev)

    id_idx = corruption_names.index("Identity")
    assert id_idx == len(corruption_names) - 1  # Identity must be the last corruption
    val_accs = {}
    for corr in corruption_names:
        if corr == "Identity":
            continue
        val_accs[corr] = [0.0] * (len(network_blocks) - 1)

    for _ in range(num_repeats):
        for i, corr in enumerate(corruption_names):
            if i == id_idx:
                continue

            print("Finding contrastive abstraction level for {}".format(corr))
            for j in range(1, len(network_blocks)):
                if dataset == "EMNIST":
                    network_blocks, _ = create_emnist_network(total_n_classes, "temp", "temp", dev)
                elif dataset == "CIFAR":
                    network_blocks, _ = create_cifar_network(total_n_classes, "temp", "temp", dev)
                elif dataset == "FACESCRUB":
                    network_blocks, _ = create_facescrub_network(total_n_classes, "temp", "temp", dev)

                temp_parameters = []
                for block in network_blocks:
                    temp_parameters += list(block.parameters())
                # temp_optim = torch.optim.Adam(temp_parameters, lr)
                temp_optim = torch.optim.SGD(temp_parameters, lr, momentum=0.9, weight_decay=5e-4)

                print("Abstraction level {}".format(j))
                abstraction_levels = [j]
                for block in network_blocks:
                    block.train()

                cont_ce_loss = 0.0
                cont_ctv_loss = 0.0
                cont_acc = 0.0

                for k, trn_data_tuples in enumerate(zip(*trn_dls), 1):
                    x_trn, y_trn = generate_batch(trn_data_tuples, dev)
                    # Slice out the corruption in question and the identity corruption
                    x_trn = torch.cat((x_trn[i * single_corr_bs:(i + 1) * single_corr_bs, :, :, :],
                                       x_trn[id_idx * single_corr_bs:(id_idx + 1) * single_corr_bs, :, :, :]), dim=0)
                    y_trn = torch.cat((y_trn[i * single_corr_bs:(i + 1) * single_corr_bs],
                                       y_trn[id_idx * single_corr_bs:(id_idx + 1) * single_corr_bs]), dim=0)

                    temp_optim.zero_grad()
                    # Only called when "Contrastive" in experiment
                    ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_trn, y_trn,
                                                                       cross_entropy_loss_fn, accuracy_fn,
                                                                       contrastive_loss_fn, abstraction_levels, weight,
                                                                       single_corr_bs)

                    loss = ce_loss + ctv_loss
                    cont_ce_loss += ce_loss.item()
                    cont_ctv_loss += ctv_loss.item()
                    cont_acc += acc

                    loss.backward()
                    temp_optim.step()

                    if k >= num_iterations:
                        print("Trained for {} batches.".format(k))
                        break

                if len(trn_dls[0]) < num_iterations:
                    denominator = len(trn_dls[0])
                else:
                    denominator = num_iterations

                print("CE Loss: {}".format(cont_ce_loss / denominator))
                print("CTV Loss: {}".format(cont_ctv_loss / denominator))
                print("Accuracy: {}".format(cont_acc / denominator))

                for block in network_blocks:
                    block.eval()
                valid_cont_ce_loss = 0.0
                valid_cont_ctv_loss = 0.0
                valid_cont_total_loss = 0.0
                valid_cont_acc = 0.0
                with torch.no_grad():
                    for k, val_data_tuples in enumerate(zip(*val_dls), 1):
                        x_val, y_val = generate_batch(val_data_tuples, dev)
                        ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_val, y_val,
                                                                           cross_entropy_loss_fn, accuracy_fn,
                                                                           contrastive_loss_fn,  abstraction_levels,
                                                                           weight, single_corr_bs)
                        valid_cont_ce_loss += ce_loss.item()
                        valid_cont_ctv_loss += ctv_loss.item()
                        valid_cont_total_loss += ce_loss.item() + ctv_loss.item()
                        valid_cont_acc += acc

                        if k >= num_iterations:
                            print("Validated for {} batches.".format(k))
                            break

                if len(val_dls[0]) < num_iterations:
                    denominator = len(val_dls[0])
                else:
                    denominator = num_iterations
                print("Validation CE loss {:6.4f}".format(valid_cont_ce_loss / denominator))
                print("Validation contrastive loss {:6.4f}".format(valid_cont_ctv_loss / denominator))
                print("Validation accuracy {:6.3f}".format(valid_cont_acc / denominator))
                val_accs[corr][j - 1] += valid_cont_acc / denominator

    for k, v in val_accs.items():
        val_accs[k] = [x / num_repeats for x in v]
    # print(val_accs)
    abstraction_levels = []
    for corr in corruption_names:
        if corr == "Identity":
            continue
        abstraction_levels.append(val_accs[corr].index(max(val_accs[corr])) + 1)
    assert len(abstraction_levels) == len(corruption_names) - 1
    return abstraction_levels


def get_module_abstraction_level(experiment, network_blocks, dataset, trn_dls, val_dls, logging_path, corruption_names,
                                 lr, cross_entropy_loss_fn, accuracy_fn, contrastive_loss_fn, weight, single_corr_bs, dev):
    if "Auto" in experiment:
        module_level = find_module_abstraction_level(network_blocks, dataset, trn_dls, val_dls, logging_path,
                                                     corruption_names, lr, cross_entropy_loss_fn, accuracy_fn,
                                                     contrastive_loss_fn, weight, single_corr_bs, dev)
    else:  # Manually Defined Level of Abstraction
        if dataset == "EMNIST":
            if "Contrast" in corruption_names or "GaussianBlur" in corruption_names or \
                    "ImpulseNoise" in corruption_names or "Invert" in corruption_names:
                module_level = 1  # After first conv layer for local corruptions
            else:
                module_level = 4  # After last conv layer for long range dependencies
        elif dataset == "CIFAR":
            if "Contrast" in corruption_names or "GaussianBlur" in corruption_names or \
                    "ImpulseNoise" in corruption_names or "Invert" in corruption_names:
                module_level = 1  # After first conv layer for local corruptions
            else:
                module_level = 9  # After last conv layer for long range dependencies
        elif dataset == "FACESCRUB":
            if "Contrast" in corruption_names or "GaussianBlur" in corruption_names or \
                    "ImpulseNoise" in corruption_names or "Invert" in corruption_names:
                module_level = 1  # After first conv layer for local corruptions
            else:
                module_level = 16  # After last conv layer for long range dependencies

    return module_level


def find_module_abstraction_level(network_blocks, dataset, trn_dls, val_dls, logging_path, corruption_names, lr,
                                  cross_entropy_loss_fn, accuracy_fn, contrastive_loss_fn, weight, single_corr_bs, dev,
                                  num_iterations=50, num_repeats=5):
    """
    Tries training the network with modules at every level of abstraction. Trains for num_iterations update steps.
    Repeats the experiment num_repeats times. Returns the level of abstraction that gives the best mean performance.

    Whichever level of abstraction lowers the training loss the fastest is chosen as the best level of abstraction.

    Optional ToDo: Throw a warning if means are close to each other (within 1 std).
    """
    val_accs = {}
    train_accs = {}  # For plotting learning curves over num_iterations
    for n in range(num_repeats):
        if dataset == "EMNIST":
            temp_modules, _ = create_emnist_modules("temp", "temp", dev)
        elif dataset == "CIFAR":
            temp_modules, _ = create_cifar_modules("temp", "temp", dev)
        elif dataset == "FACESCRUB":
            temp_modules, _ = create_facescrub_modules("temp", "temp", dev)

        val_accs[n] = []
        for i, module in enumerate(temp_modules):
            print("Abstraction Level {}".format(i))
            train_accs["Level-{}_Repeat-{}".format(i, n)] = []
            # temp_optim = torch.optim.Adam(module.parameters(), lr)
            temp_optim = torch.optim.SGD(module.parameters(), lr, momentum=0.9, weight_decay=5e-4)

            module_ce_loss = 0.0
            module_ctv_loss = 0.0
            module_acc = 0.0

            # Time batches
            module.train()
            for j, trn_data_tuples in enumerate(zip(*trn_dls), 1):
                x_trn, y_trn = generate_batch(trn_data_tuples, dev)
                temp_optim.zero_grad()
                # Only called when "Modules" in experiment
                ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, i, x_trn, y_trn,
                                                               cross_entropy_loss_fn, accuracy_fn, contrastive_loss_fn,
                                                               weight, single_corr_bs)
                loss = ce_loss + ctv_loss
                module_ce_loss += ce_loss.item()
                module_ctv_loss += ctv_loss.item()
                module_acc += acc
                loss.backward()
                temp_optim.step()
                train_accs["Level-{}_Repeat-{}".format(i, n)].append(acc)

                if j >= num_iterations:
                    print("Trained for {} batches.".format(j))
                    break

            if len(trn_dls[0]) < num_iterations:
                denominator = len(trn_dls[0])
            else:
                denominator = num_iterations

            print("CE Loss: {}".format(module_ce_loss / denominator))
            print("CTV Loss: {}".format(module_ctv_loss / denominator))
            print("Accuracy: {}".format(module_acc / denominator))

            module.eval()
            module_valid_ce_loss = 0.0
            module_valid_ctv_loss = 0.0
            module_valid_total_loss = 0.0
            module_valid_acc = 0.0
            with torch.no_grad():
                for j, val_data_tuples in enumerate(zip(*val_dls), 1):
                    x_val, y_val = generate_batch(val_data_tuples, dev)
                    ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, i, x_val, y_val,
                                                                   cross_entropy_loss_fn, accuracy_fn, contrastive_loss_fn,
                                                                   weight, single_corr_bs)
                    module_valid_ce_loss += ce_loss.item()
                    module_valid_ctv_loss += ctv_loss.item()
                    module_valid_total_loss += ce_loss.item() + ctv_loss.item()
                    module_valid_acc += acc

                    if j >= num_iterations:
                        print("Validated for {} batches.".format(j))
                        break

            if len(val_dls[0]) < num_iterations:
                denominator = len(val_dls[0])
            else:
                denominator = num_iterations
            print("Validation CE loss {:6.4f}".format(module_valid_ce_loss / denominator))
            print("Validation contrastive loss {:6.4f}".format(module_valid_ctv_loss / denominator))
            print("Validation accuracy {:6.3f}".format(module_valid_acc / denominator))
            val_accs[n].append(module_valid_acc / denominator)

    # Pickle train accs for plotting
    with open(os.path.join(logging_path, "module_train_accs_{}.pkl".format(corruption_names[0])), "wb") as f:
        pickle.dump(train_accs, f)

    mean_val_accs = []
    for i in range(len(val_accs[0])):
        mean_val_accs.append(np.mean([val_accs[n][i] for n in range(num_repeats)]))
    assert len(mean_val_accs) == len(temp_modules)
    # argmax of mean_val_accs
    best_level = mean_val_accs.index(max(mean_val_accs))
    return best_level


def main(corruptions, dataset, data_root, ckpt_path, logging_path, vis_path, experiment, weight, temperature,
         total_n_classes, max_epochs, batch_size, lr, n_workers, pin_mem, dev, vis_data, check_if_run):
    # Train all models
    for corruption_names in corruptions:
        # Log File set up
        log_name = "{}_{}.log".format(experiment, '-'.join(corruption_names))
        logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
        print("Logging file created for experiment {} to train on corruption(s): {}".format(experiment,
                                                                                            corruption_names))
        # Data Set Up
        corruption_paths = [os.path.join(data_root, corruption_name) for corruption_name in corruption_names]
        train_classes = list(range(total_n_classes))
        if "CrossEntropy" in experiment:  # randomly mix all corruptions
            trn_dl, val_dl, _ = get_multi_static_dataloaders(dataset, corruption_paths, train_classes, batch_size,
                                                             True, n_workers, pin_mem)
            trn_dls = [trn_dl]
            val_dls = [val_dl]
        elif "Contrastive" in experiment or "Modules" in experiment or "ImgSpace" in experiment:  # each batch contains the same images with different corruptions
            generators = [torch.Generator(device='cpu').manual_seed(2147483647) for _ in range(len(corruption_paths))]
            single_corr_bs = batch_size // len(corruption_names)
            trn_dls, val_dls = [], []
            for corruption_path, generator in zip(corruption_paths, generators):
                trn_dl, val_dl, _ = get_static_dataloaders(dataset, corruption_path, train_classes, single_corr_bs,
                                                           True, n_workers, pin_mem, fixed_generator=generator)
                trn_dls.append(trn_dl)
                val_dls.append(val_dl)

            if "Modules" in experiment:
                if len(corruption_names) != 2:
                    raise ValueError("Initial module training only uses single corruptions (plus the identity)")

                # Load identity network or create it if it doesn't exist
                if dataset == "EMNIST":
                    network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, "Modules",
                                                                                     ["Identity"], dev)
                elif dataset == "CIFAR":
                    network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, "Modules",
                                                                                    ["Identity"], dev)
                elif dataset == "FACESCRUB":
                    network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, "Modules",
                                                                                        ["Identity"], dev)
                assert len(network_blocks) >= 2  # assumed when the network is called
                if os.path.exists(os.path.join(ckpt_path, network_block_ckpt_names[0])):
                    for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
                        block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
                        block.eval()
                    logger.info("Loaded identity network from {}".format(ckpt_path))
                elif os.path.exists(os.path.join(ckpt_path, "es_ckpt_{}.pt".format(network_block_ckpt_names[0]))):
                    raise RuntimeError("Early stopping checkpoint found. Training may be running in another process.")
                else:
                    logger.info("Identity network not found. Training it now.")
                    _ = train_identity_network(network_blocks, network_block_ckpt_names, dataset, data_root,
                                               ckpt_path, logging_path, experiment, total_n_classes,
                                               max_epochs, batch_size, lr, n_workers, pin_mem, dev)
                    # A hack, not necessary but makes usage more consistent.
                    raise RuntimeError("Identity network training completed - rerun for corruption experiments")
        else:
            raise NotImplementedError("Experiment {} not implemented".format(experiment))

        if vis_data:
            for i, trn_dl in enumerate(trn_dls):
                if i == 0:
                    x, y = next(iter(trn_dl))
                else:
                    x_temp, y_temp = next(iter(trn_dl))
                    x = torch.cat((x, x_temp), dim=0)
                    y = torch.cat((y, y_temp), dim=0)
            fig_name = "{}_{}.png".format(experiment, '-'.join(corruption_names))
            fig_path = os.path.join(vis_path, fig_name)
            # Denormalise Images
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = denormalize(x).astype(np.uint8)
            # And visualise
            visualise_data(x[:225], y[:225], save_path=fig_path, title=fig_name[:-4], n_rows=15, n_cols=15)

        # Loss Function Set Up
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        accuracy_fn = lambda x, y: accuracy(x, y)
        if "Contrastive" in experiment or "Modules" in experiment:
            contrastive_loss_fn = ContrastiveLayerLoss(single_corr_bs, temperature, dev)
        if "ImgSpace" in experiment:
            mse_loss_fn = nn.MSELoss()

        # Network & Optimizer Set Up
        if "CrossEntropy" in experiment or "Contrastive" in experiment:
            if "Contrastive" in experiment:
                logger.info("Finding Contrastive Levels of Abstraction")
                contrastive_levels = get_contrastive_abstraction_levels(experiment, corruption_names, dataset, trn_dls,
                                                                        val_dls, lr, cross_entropy_loss_fn, accuracy_fn,
                                                                        contrastive_loss_fn, weight, total_n_classes,
                                                                        single_corr_bs, dev)
                logger.info("Using Levels of Abstraction {}".format(contrastive_levels))

            if dataset == "EMNIST":
                network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment,
                                                                                 corruption_names, dev)
            elif dataset == "CIFAR":
                network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, experiment,
                                                                                corruption_names, dev)
            elif dataset == "FACESCRUB":
                network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, experiment,
                                                                                    corruption_names, dev)

            assert len(network_blocks) >= 2  # assumed when the network is called
            all_parameters = []
            for block in network_blocks:
                all_parameters += list(block.parameters())
            # optim = torch.optim.Adam(all_parameters, lr) # Todo: used for EMNIST, test without
            optim = torch.optim.SGD(all_parameters, lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

            # Early Stopping Set Up
            es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(block_ckpt_name)) for block_ckpt_name in
                             network_block_ckpt_names]
            early_stoppings = [EarlyStopping(patience=max_epochs, path=es_ckpt_path, trace_func=logger.info) for
                               es_ckpt_path in es_ckpt_paths]
            early_stoppings[0].verbose = True
            assert len(early_stoppings) == len(network_blocks)
            # Check if training has already completed for the corruption(s) in question.
            if check_if_run and os.path.exists(os.path.join(ckpt_path, network_block_ckpt_names[0])):
                print("Checkpoint already exists at {} \nSkipping training on corruption(s): {}".format(
                    os.path.join(ckpt_path, network_block_ckpt_names[0]), corruption_names))
                continue
        elif "Modules" in experiment:
            logger.info("Finding Module Level of Abstraction")
            module_level = get_module_abstraction_level(experiment, network_blocks, dataset, trn_dls, val_dls,
                                                        logging_path, corruption_names, lr, cross_entropy_loss_fn,
                                                        accuracy_fn, contrastive_loss_fn, weight, single_corr_bs, dev)
            logger.info("Using Level of Abstraction {}".format(module_level))

            if dataset == "EMNIST":
                modules, module_ckpt_names = create_emnist_modules(experiment, corruption_names, dev)
            elif dataset == "CIFAR":
                modules, module_ckpt_names = create_cifar_modules(experiment, corruption_names, dev)
            elif dataset == "FACESCRUB":
                modules, module_ckpt_names = create_facescrub_modules(experiment, corruption_names, dev)

            module = modules[module_level]
            module_ckpt_name = module_ckpt_names[module_level]
            # optim = torch.optim.Adam(module.parameters(), lr)  # Todo: used for EMNIST, test without
            optim = torch.optim.SGD(module.parameters(), lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

            # Early Stopping Set Up
            es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(module_ckpt_name))]
            early_stoppings = [EarlyStopping(patience=max_epochs, path=es_ckpt_path, trace_func=logger.info) for
                               es_ckpt_path in es_ckpt_paths]
            early_stoppings[0].verbose = True
            assert len(early_stoppings) == 1
            # Check if training has already completed for the corruption(s) in question.
            if check_if_run and os.path.exists(os.path.join(ckpt_path, module_ckpt_name)):
                print("Checkpoint already exists at {} \nSkipping training on corruption(s): {}".format(
                    os.path.join(ckpt_path, module_ckpt_name), corruption_names))
                continue
        elif "ImgSpace" in experiment:
            if dataset == "EMNIST":
                network_blocks, network_block_ckpt_names = create_emnist_autoencoder(experiment, corruption_names, dev)
            elif dataset == "CIFAR":
                network_blocks, network_block_ckpt_names = create_cifar_autoencoder(experiment, corruption_names, dev)
            elif dataset == "FACESCRUB":
                network_blocks, network_block_ckpt_names = create_facescrub_autoencoder(experiment, corruption_names,
                                                                                        dev)
            assert len(network_blocks) >= 2  # assumed when the network is called
            all_parameters = []
            for block in network_blocks:
                all_parameters += list(block.parameters())
            # optim = torch.optim.Adam(all_parameters, lr)  # Todo: used for EMNIST, test without
            optim = torch.optim.SGD(all_parameters, lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

            # Early Stopping Set Up
            es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(block_ckpt_name)) for block_ckpt_name in
                             network_block_ckpt_names]
            early_stoppings = [EarlyStopping(patience=max_epochs, path=es_ckpt_path, trace_func=logger.info) for
                               es_ckpt_path in es_ckpt_paths]
            early_stoppings[0].verbose = True
            assert len(early_stoppings) == len(network_blocks)
            # Check if training has already completed for the corruption(s) in question.
            if check_if_run and os.path.exists(os.path.join(ckpt_path, network_block_ckpt_names[0])):
                print("Checkpoint already exists at {} \nSkipping training on corruption(s): {}".format(
                    os.path.join(ckpt_path, network_block_ckpt_names[0]), corruption_names))
                continue
        else:
            raise NotImplementedError("Experiment {} not implemented".format(experiment))

        # We want to check for early stopping after approximately equal number of batches:
        val_freq = len(trn_dls[0]) // len(corruption_names)
        logger.info("Validation frequency: every {} batches".format(val_freq))

        # Train Loop
        for epoch in range(max_epochs):
            # Training
            if "Modules" not in experiment:
                for block in network_blocks:
                    block.train()
            else:
                module.train()
            epoch_ce_loss = 0.0
            epoch_ctv_loss = 0.0
            epoch_acc = 0.0

            # # Time batches
            # start_time = time.time()
            for i, trn_data_tuples in enumerate(zip(*trn_dls), 1):
                x_trn, y_trn = generate_batch(trn_data_tuples, dev)
                optim.zero_grad()
                if "CrossEntropy" in experiment:
                    ce_loss, acc = cross_entropy_forwards_pass(network_blocks, x_trn, y_trn, cross_entropy_loss_fn,
                                                               accuracy_fn)
                    loss = ce_loss
                    epoch_ce_loss += ce_loss.item()
                    epoch_acc += acc
                elif "Contrastive" in experiment:
                    ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_trn, y_trn,
                                                                       cross_entropy_loss_fn, accuracy_fn,
                                                                       contrastive_loss_fn, contrastive_levels, weight,
                                                                       single_corr_bs)
                    loss = ce_loss + ctv_loss
                    epoch_ce_loss += ce_loss.item()
                    epoch_ctv_loss += ctv_loss.item()
                    epoch_acc += acc
                elif "Modules" in experiment:
                    ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, module_level, x_trn, y_trn,
                                                                   cross_entropy_loss_fn, accuracy_fn, contrastive_loss_fn,
                                                                   weight, single_corr_bs)
                    loss = ce_loss + ctv_loss
                    epoch_ce_loss += ce_loss.item()
                    epoch_ctv_loss += ctv_loss.item()
                    epoch_acc += acc
                elif "ImgSpace" in experiment:
                    mse_loss = autoencoder_forwards_pass(network_blocks, x_trn, mse_loss_fn, single_corr_bs)
                    loss = mse_loss
                    epoch_ce_loss += mse_loss.item()  # just for logging
                else:
                    raise NotImplementedError("Experiment {} not implemented".format(experiment))
                loss.backward()
                optim.step()
                # # Time batches
                # print("Batch {} of {} in epoch {} took {:.2f} seconds.".format(i, len(trn_dl), epoch,
                #                                                                time.time() - start_time))
                # logger.info("Batch {} of {} in epoch {} took {:.2f} seconds.".format(i, len(trn_dl), epoch,
                #                                                                time.time() - start_time))
                # start_time = time.time()

                if i % val_freq == 0:
                    # Validation
                    if "Modules" not in experiment:
                        for block in network_blocks:
                            block.eval()
                    else:
                        module.eval()
                    valid_ce_loss = 0.0
                    valid_ctv_loss = 0.0
                    valid_total_loss = 0.0
                    valid_acc = 0.0
                    with torch.no_grad():
                        for val_data_tuples in zip(*val_dls):
                            x_val, y_val = generate_batch(val_data_tuples, dev)
                            if "CrossEntropy" in experiment:
                                ce_loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val,
                                                                           cross_entropy_loss_fn, accuracy_fn)
                                valid_ce_loss += ce_loss.item()
                                valid_total_loss += ce_loss.item()
                                valid_acc += acc
                            elif "Contrastive" in experiment:
                                ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_val, y_val,
                                                                                   cross_entropy_loss_fn, accuracy_fn,
                                                                                   contrastive_loss_fn, contrastive_levels,
                                                                                   weight, single_corr_bs)
                                valid_ce_loss += ce_loss.item()
                                valid_ctv_loss += ctv_loss.item()
                                valid_total_loss += ce_loss.item() + ctv_loss.item()
                                valid_acc += acc
                            elif "Modules" in experiment:
                                ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, module_level,
                                                                               x_val, y_val,
                                                                               cross_entropy_loss_fn, accuracy_fn,
                                                                               contrastive_loss_fn,
                                                                               weight, single_corr_bs)
                                valid_ce_loss += ce_loss.item()
                                valid_ctv_loss += ctv_loss.item()
                                valid_total_loss += ce_loss.item() + ctv_loss.item()
                                valid_acc += acc
                            elif "ImgSpace" in experiment:
                                mse_loss = autoencoder_forwards_pass(network_blocks, x_val, mse_loss_fn, single_corr_bs)
                                valid_ce_loss += mse_loss.item()
                                valid_total_loss += mse_loss.item()
                            else:
                                raise NotImplementedError("Experiment {} not implemented".format(experiment))
                    if "CrossEntropy" in experiment:
                        logger.info("Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
                        logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
                    elif "Contrastive" in experiment or "Modules" in experiment:
                        logger.info("Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
                        logger.info("Validation contrastive loss {:6.4f}".format(valid_ctv_loss / len(val_dls[0])))
                        logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
                    elif "ImgSpace" in experiment:
                        logger.info("Validation MSE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))

                    # Early Stopping
                    if "Modules" not in experiment:
                        for es, block in zip(early_stoppings, network_blocks):
                            es(valid_total_loss / len(val_dls[0]), block)  # ES on loss
                    else:
                        early_stoppings[0](valid_total_loss / len(val_dls[0]), module)
                    if early_stoppings[0].early_stop:
                        logger.info("Early stopping")
                        break
                    if "Modules" not in experiment:
                        for block in network_blocks:
                            block.train()
                    else:
                        module.train()
            scheduler.step()
            if early_stoppings[0].early_stop:
                break
            if "CrossEntropy" in experiment:
                results = [epoch, epoch_ce_loss / len(trn_dls[0]), epoch_acc / len(trn_dls[0])]
                logger.info("Epoch {}. Avg CE train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))
            elif "Contrastive" in experiment or "Modules" in experiment:
                results = [epoch, epoch_ce_loss / len(trn_dls[0]), epoch_ctv_loss / len(trn_dls[0]),
                           epoch_acc / len(trn_dls[0])]
                logger.info("Epoch {}. Avg CE train loss {:6.4f}. Avg contrastive train loss {:6.4f}. "
                            "Avg train acc {:6.3f}.".format(*results))
            elif "ImgSpace" in experiment:
                results = [epoch, epoch_ce_loss / len(trn_dls[0]), epoch_acc / len(trn_dls[0])]
                logger.info("Epoch {}. Avg MSE train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

        # Save model
        logger.info("Loading early stopped checkpoints")
        if "Modules" not in experiment:
            for es, block in zip(early_stoppings, network_blocks):
                es.load_from_checkpoint(block)
            for block in network_blocks:
                block.eval()
        else:
            early_stoppings[0].load_from_checkpoint(module)
            module.eval()
        valid_ce_loss = 0.0
        valid_ctv_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for val_data_tuples in zip(*val_dls):
                x_val, y_val = generate_batch(val_data_tuples, dev)
                if "CrossEntropy" in experiment:
                    ce_loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val,
                                                               cross_entropy_loss_fn, accuracy_fn)
                    valid_ce_loss += ce_loss.item()
                    valid_acc += acc
                elif "Contrastive" in experiment:
                    ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_val, y_val,
                                                                       cross_entropy_loss_fn, accuracy_fn,
                                                                       contrastive_loss_fn, contrastive_levels, weight,
                                                                       single_corr_bs)
                    valid_ce_loss += ce_loss.item()
                    valid_ctv_loss += ctv_loss.item()
                    valid_acc += acc
                elif "Modules" in experiment:
                    ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, module_level, x_val, y_val,
                                                                   cross_entropy_loss_fn, accuracy_fn,
                                                                   contrastive_loss_fn, weight, single_corr_bs)
                    valid_ce_loss += ce_loss.item()
                    valid_ctv_loss += ctv_loss.item()
                    valid_acc += acc
                elif "ImgSpace" in experiment:
                    mse_loss = autoencoder_forwards_pass(network_blocks, x_val, mse_loss_fn, single_corr_bs)
                    valid_ce_loss += mse_loss.item()
                else:
                    raise NotImplementedError("Experiment {} not implemented".format(experiment))
        if "CrossEntropy" in experiment:
            logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
            logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
        elif "Contrastive" in experiment or "Modules" in experiment:
            logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
            logger.info("Early Stopped validation contrastive loss {:6.4f}".format(valid_ctv_loss / len(val_dls[0])))
            logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
        elif "ImgSpace" in experiment:
            logger.info("Early Stopped validation MSE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))

        for es in early_stoppings:
            es.delete_checkpoint()  # Removes from disk
        if "Modules" not in experiment:
            for block_ckpt_name, block in zip(network_block_ckpt_names, network_blocks):
                torch.save(block.state_dict(), os.path.join(ckpt_path, block_ckpt_name))
                logger.info("Saved best network block to {}".format(os.path.join(ckpt_path, block_ckpt_name)))
        else:
            torch.save(module.state_dict(), os.path.join(ckpt_path, module_ckpt_name))
            logger.info("Saved best module to {}".format(os.path.join(ckpt_path, module_ckpt_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to train networks on different corruptions.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--logging-path', type=str, default='/om2/user/imason/compositions/logs/',
                        help="path to directory to save logs")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/',
                        help="path to directory to save data visualisations")
    parser.add_argument('--experiment', type=str, default='CrossEntropy',
                        help="which method to use. CrossEntropy or Contrastive or Modules.")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--max-epochs', type=int, default=50, help="max number of training epochs")
    parser.add_argument('--batch-size', type=int, default=256, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--vis-data', action='store_true', help="set to save a png of one batch of data")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    parser.add_argument('--corruption-ID', type=int, default=0, help="which corruption to generate")
    parser.add_argument('--weight', type=float, default=1.0, help="weight for the contrastive loss")
    parser.add_argument('--temperature', type=float, default=0.15, help="contrastive loss temperature")
    args = parser.parse_args()

    # Check dataset
    if args.dataset not in ["EMNIST", "CIFAR", "FACESCRUB"]:
        raise ValueError("Dataset {} not implemented".format(args.dataset))

    # Set seeding
    reset_rngs(seed=1357911, deterministic=True)
    # reset_rngs(seed=246810, deterministic=True)

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
    args.logging_path = os.path.join(args.logging_path, args.dataset)
    args.vis_path = os.path.join(args.vis_path, args.dataset, "visualisations")
    mkdir_p(args.ckpt_path)
    mkdir_p(args.logging_path)
    if args.vis_data:
        mkdir_p(args.vis_path)

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)

    corruptions = []
    # Training on only the corruptions in the composition. Always include identity at the end, remove permutations
    for corr in all_corruptions:
        if "Identity" not in corr:
            corr.sort()
            corr += ["Identity"]
            if corr not in corruptions:
                corruptions.append(corr)
        elif len(corr) == 1:  # identity only
            if corr not in corruptions:
                corruptions.append(corr)
        else:
            raise ValueError("Only expect the identity to appear as its own corruption")

    assert len(corruptions) == 64  # 64 for EMNIST 5, 128 for EMNIST4

    # Using slurm to parallelise the training
    # corruptions = corruptions[args.corruption_ID:args.corruption_ID+1]

    if "Modules" in args.experiment or "ImgSpace" in args.experiment:  # For the Modules approach, we want all elemental corruptions
        corruptions = [corr for corr in corruptions if len(corr) == 2]
        assert len(corruptions) == 6  # hardcoded for EMNIST5
        corruptions = corruptions[args.corruption_ID:args.corruption_ID+1]
    elif "Contrastive" in args.experiment or "CrossEntropy" in args.experiment:
        # For contrastive and cross entropy we only want the case with all corruptions together
        max_corr_count = max([len(corr) for corr in corruptions])
        corruptions = [corr for corr in corruptions if len(corr) == max_corr_count]
        assert len(corruptions) == 1

    """
    If running on polestar
    CUDA_VISIBLE_DEVICES=4 python train.py --pin-mem --check-if-run --corruption-ID 0 -experiment CrossEntropy
    CUDA_VISIBLE_DEVICES=4 python train.py --pin-mem --check-if-run --corruption-ID 34 --vis-data --experiment Contrastive --weights  1,1,1,1,1,1
    CUDA_VISIBLE_DEVICES=4 python train.py --pin-mem --check-if-run --corruption-ID 0 --experiment Modules --weights 1,1,1,1,1,1
    CUDA_VISIBLE_DEVICES=4 python train.py --pin-mem --check-if-run --corruption-ID 0 --experiment ImgSpace
    
    CUDA_VISIBLE_DEVICES=4 python train.py --pin-mem --check-if-run --corruption-ID 0 --dataset CIFAR --total-n-classes 10 --max-epochs 200 --lr 1e-3 --experiment Modules --weights 1,1,1,1,1,1,1,1,1,1
    CUDA_VISIBLE_DEVICES=5 python train.py --pin-mem --check-if-run --corruption-ID 0 --dataset FACESCRUB --total-n-classes 388 --max-epochs 200 --experiment Modules --weights 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    
    The below does decently - about 80-82% validation (with no augmentation)
    CUDA_VISIBLE_DEVICES=5 python train.py --pin-mem --check-if-run --corruption-ID 0 --dataset FACESCRUB --total-n-classes 388 --experiment Modules --weights 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    
    For module identity training on polestar with new code 
    CUDA_VISIBLE_DEVICES=4 python train.py --pin-mem --check-if-run --corruption-ID 0 --experiment Modules --dataset EMNIST --total-n-classes 47 --max-epochs 200 --lr 1e-1 --weight 1
    CUDA_VISIBLE_DEVICES=5 python train.py --pin-mem --check-if-run --corruption-ID 0 --experiment Modules --dataset CIFAR --total-n-classes 10 --max-epochs 200 --lr 1e-1 --weight 1
    CUDA_VISIBLE_DEVICES=6 python train.py --pin-mem --check-if-run --corruption-ID 0 --experiment Modules --dataset FACESCRUB --total-n-classes 388 --max-epochs 200 --lr 1e-1 --weight 1

    """

    # Searching learning rates. Change --array=0-47 to --array=0-191
    # assert len(corruptions) == 48  # for EMNIST3
    # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    # exp_idx = args.corruption_ID // 48
    # corruptions = corruptions[(args.corruption_ID % len(corruptions)):(args.corruption_ID % len(corruptions)) + 1]
    # args.lr = lrs[exp_idx]

    main(corruptions, args.dataset, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.experiment,
         args.weight, args.temperature, args.total_n_classes, args.max_epochs, args.batch_size, args.lr,
         args.n_workers, args.pin_mem, dev, args.vis_data, args.check_if_run)

    if "ImgSpace" in args.experiment:  # Trains classifier on top of autoencoded images
        train_classifiers(args.dataset, args.data_root, args.ckpt_path, args.logging_path, args.experiment,
                          args.total_n_classes, args.max_epochs, args.batch_size, args.lr,
                          args.n_workers, args.pin_mem, dev, args.check_if_run)


