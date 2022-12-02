"""
Train a model on individual data transformations, pairs of transformations and all transformations. .
"""
import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from data.data_transforms import denormalize
from data.data_loaders import get_multi_static_emnist_dataloaders, get_static_emnist_dataloaders
from lib.networks import create_emnist_network, create_emnist_modules
from lib.early_stopping import EarlyStopping
from lib.utils import *


class ContrastiveLayerLoss:
    def __init__(self, single_corr_bs, temperature=0.15, dev=torch.device('cuda')):
        self.single_corr_bs = single_corr_bs
        self.temperature = temperature
        self.dev = dev

    def __call__(self, features, weight, n_views=2):
        if weight == 0:
            return torch.tensor(0.0).to(self.dev)
        if n_views <= 1:
            return torch.tensor(0.0).to(self.dev)

        # Flatten spatial dimensions if 4D
        features = features.reshape(features.shape[0], -1)
        assert len(features) % self.single_corr_bs == 0
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(self.single_corr_bs) for _ in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.dev)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.dev)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        contrastive_criterion = nn.CrossEntropyLoss()
        """
        This is the basic contrastive case with n_views = 2
            logits = torch.cat([positives, negatives], dim=1)
            logits = logits / self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.dev)
            loss = contrastive_criterion(logits, labels)
            
        We want a more general case with arbitrary n_views (== number of corruptions in the batch)
        As a starting point for why this works: https://github.com/sthalles/SimCLR/issues/16 (also issue 33)
        """
        for i in range(n_views - 1):
            logits = torch.cat([positives[:, i:i+1], negatives], dim=1)
            logits = logits / self.temperature
            if i == 0:
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.dev)
                loss = contrastive_criterion(logits, labels)
            else:
                loss += contrastive_criterion(logits, labels)

        return weight * loss


def generate_batch(data_tuples, dev):
    """
    Takes one or more tuples (x,y) of training data from different data loaders and concatenates them into single
    tensors for training
    """
    for j, data_tuple in enumerate(data_tuples):
        if j == 0:
            x, y = data_tuple[0].to(dev), data_tuple[1].to(dev)
        else:
            x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
            x = torch.cat((x, x_temp), dim=0)
            y = torch.cat((y, y_temp), dim=0)
    return x, y


def cross_entropy_forwards_pass(network_blocks, x, y, cross_entropy_loss, accuracy_fn):
    for i, block in enumerate(network_blocks):
        if i == 0:
            features = block(x)
        elif i == len(network_blocks) - 1:
            output = block(features)
        else:
            features = block(features)

    return cross_entropy_loss(output, y), accuracy_fn(output, y)


def contrastive_forwards_pass(network_blocks, x, y, cross_entropy_loss, accuracy_fn, contrastive_loss,
                              abstraction_levels, weights, single_corr_bs):
    """
    Network forwards pass with an option to apply the contrastive loss at any intermediate layer
    The contrastive loss is weighted by weights
    The level to apply the loss is defined by abstraction_levels
    """
    total_ctv_loss = 0.0
    id_idx = len(abstraction_levels)  # Hardcoded assumption that Identity is last in the batch
    if 0 in abstraction_levels:
        raise ValueError("Contrastive loss cannot be applied before the first layer")
    features = x
    for i, block in enumerate(network_blocks):
        if i in abstraction_levels:
            loss_features = []
            n_views = 0
            for corr_idx, a in enumerate(abstraction_levels):
                if a == i:
                    if len(features.shape) == 4:
                        loss_features.append(features[corr_idx * single_corr_bs:(corr_idx + 1) * single_corr_bs, :, :, :])
                    elif len(features.shape) == 2:
                        loss_features.append(features[corr_idx * single_corr_bs:(corr_idx + 1) * single_corr_bs, :])
                    else:
                        raise ValueError("Features must be 2d or 4d. Level {}.".format(i))
                    n_views += 1
            if len(features.shape) == 4:
                loss_features.append(features[id_idx * single_corr_bs:(id_idx + 1) * single_corr_bs, :, :, :])
            elif len(features.shape) == 2:
                loss_features.append(features[id_idx * single_corr_bs:(id_idx + 1) * single_corr_bs, :])
            else:
                raise ValueError("Features must be 2d or 4d. Level {}.".format(i))
            n_views += 1
            loss_features = torch.cat(loss_features, dim=0)
            total_ctv_loss += contrastive_loss(loss_features, weights[i], n_views=n_views)
        if i != len(network_blocks) - 1:
            features = block(features)
        else:
            output = block(features)

    return cross_entropy_loss(output, y), total_ctv_loss, accuracy_fn(output, y)


def modules_forwards_pass(network_blocks, module, module_level, x, y, cross_entropy_loss, accuracy_fn, contrastive_loss,
                          weight, single_corr_bs, pass_through):
    """
    Network forwards pass with a module applied at a specified intermediate layer

    pass_through: if True, the module is applied to both identity and corruption features
                  if False, the module is applied only to the corruption features
    """
    if module_level < 0 or module_level >= len(network_blocks):
        raise ValueError("Module must be applied at an intermediate layer. Level {}.".format(module_level))
    total_ctv_loss = 0.0
    features = x
    for i, block in enumerate(network_blocks):
        if i == module_level:
            if len(features.shape) == 4:
                id_features = features[single_corr_bs:, :, :, :]
            elif len(features.shape) == 2:
                id_features = features[single_corr_bs:, :]
            else:
                raise ValueError("Features must be 2d or 4d. Level {}.".format(i))

            if pass_through:
                features = module(features)
            else:
                if len(features.shape) == 4:
                    features = module(features[:single_corr_bs, :, :, :])
                else:
                    features = module(features[:single_corr_bs, :])
                features = torch.cat((features, id_features), dim=0)

            if len(features.shape) == 4:
                corr_features = features[:single_corr_bs, :, :, :]
            elif len(features.shape) == 2:
                corr_features = features[:single_corr_bs, :]
            else:
                raise ValueError("Features must be 2d or 4d. Level {}.".format(i))

            corr_features = corr_features.reshape(corr_features.shape[0], -1)  # Flatten spatial dimensions if necessary
            id_features = id_features.reshape(id_features.shape[0], -1)  # Flatten spatial dimensions if necessary
            total_ctv_loss += contrastive_loss(torch.cat((corr_features, id_features), dim=0), weight, n_views=2)

        if i != len(network_blocks) - 1:
            features = block(features)
        else:
            output = block(features)

    return cross_entropy_loss(output, y), total_ctv_loss, accuracy_fn(output, y)


def train_identity_network(network_blocks, network_block_ckpt_names, data_root, ckpt_path, logging_path, experiment,
                           total_n_classes, es_burn_in, max_epochs, batch_size, lr, n_workers, pin_mem, dev):
    # Log File set up
    log_name = "{}_Identity.log".format(experiment)
    id_logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
    print("Logging file created for experiment {} to train on Identity".format(experiment))

    # Data Set Up
    corruption_path = os.path.join(data_root, "Identity")
    train_classes = list(range(total_n_classes))
    trn_dl, val_dl, _ = get_static_emnist_dataloaders(corruption_path, train_classes, batch_size, True, n_workers,
                                                      pin_mem)

    # Network & Optimizer Set Up
    all_parameters = []
    for block in network_blocks:
        all_parameters += list(block.parameters())
    optim = torch.optim.Adam(all_parameters, lr)

    # Early Stopping Set Up
    es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(block_ckpt_name)) for block_ckpt_name in
                     network_block_ckpt_names]
    early_stoppings = [EarlyStopping(patience=25, verbose=True, path=es_ckpt_path, trace_func=id_logger.info) for
                       es_ckpt_path in es_ckpt_paths]
    assert len(early_stoppings) == len(network_blocks)
    burn_in_count = 0

    # Loss Function Set Up
    cross_entropy_loss = nn.CrossEntropyLoss()
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
            loss, acc = cross_entropy_forwards_pass(network_blocks, x_trn, y_trn, cross_entropy_loss, accuracy_fn)
            epoch_loss += loss.item()
            epoch_acc += acc
            loss.backward()
            optim.step()
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
                loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val, cross_entropy_loss, accuracy_fn)
                valid_loss += loss.item()
                valid_acc += acc
        id_logger.info("Validation CE loss {:6.4f}".format(valid_loss / len(val_dl)))
        id_logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
        # Early Stopping
        burn_in_count += 1
        if burn_in_count >= es_burn_in:
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
            loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val, cross_entropy_loss, accuracy_fn)
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


def find_contrastive_abstraction_level(corruption_names, trn_dls, val_dls, lr, cross_entropy_loss, accuracy_fn,
                                       contrastive_loss, weights, total_n_classes, single_corr_bs, dev,
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
    network_blocks, _ = create_emnist_network(total_n_classes, "temp", "temp", dev)
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
                network_blocks, _ = create_emnist_network(total_n_classes, "temp", "temp", dev)
                temp_parameters = []
                for block in network_blocks:
                    temp_parameters += list(block.parameters())
                temp_optim = torch.optim.Adam(temp_parameters, lr)

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
                                                                       cross_entropy_loss, accuracy_fn,
                                                                       contrastive_loss, abstraction_levels, weights,
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
                                                                           cross_entropy_loss, accuracy_fn,
                                                                           contrastive_loss,  abstraction_levels, weights,
                                                                           single_corr_bs)
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


def find_module_abstraction_level(network_blocks, trn_dls, val_dls, lr, cross_entropy_loss, accuracy_fn,
                                  contrastive_loss, weights, single_corr_bs, dev, pass_through, num_iterations=50,
                                  num_repeats=5):
    """
    Tries training the network with modules at every level of abstraction. Trains for num_iterations update steps.
    Repeats the experiment num_repeats times. Returns the level of abstraction that gives the best mean performance.

    Whichever level of abstraction lowers the training loss the fastest is chosen as the best level of abstraction.

    Optional ToDo: Throw a warning if means are close to each other (within 1 std).
    """
    val_accs = {}
    for n in range(num_repeats):
        temp_modules, _ = create_emnist_modules("temp", "temp", dev)
        val_accs[n] = []
        for i, module in enumerate(temp_modules):
            print("Abstraction Level {}".format(i))
            temp_optim = torch.optim.Adam(module.parameters(), lr)

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
                                                               cross_entropy_loss, accuracy_fn, contrastive_loss,
                                                               weights[i], single_corr_bs, pass_through)
                loss = ce_loss + ctv_loss
                module_ce_loss += ce_loss.item()
                module_ctv_loss += ctv_loss.item()
                module_acc += acc
                loss.backward()
                temp_optim.step()

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
                                                                   cross_entropy_loss, accuracy_fn, contrastive_loss,
                                                                   weights[i], single_corr_bs, pass_through)
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

    mean_val_accs = []
    for i in range(len(val_accs[0])):
        mean_val_accs.append(np.mean([val_accs[n][i] for n in range(num_repeats)]))
    assert len(mean_val_accs) == len(temp_modules)
    # argmax of mean_val_accs
    best_level = mean_val_accs.index(max(mean_val_accs))
    return best_level


def main(corruptions, data_root, ckpt_path, logging_path, vis_path, experiment, weights, temperature, total_n_classes,
         es_burn_in, max_epochs, batch_size, lr, n_workers, pin_mem, dev, vis_data, check_if_run):
    assert max_epochs > es_burn_in
    # Train all models
    for corruption_names in corruptions:
        ckpt_name = "{}_ConvBlock1_{}.pt".format(experiment, '-'.join(corruption_names))
        # Check if training has already completed for the corruption(s) in question.
        if check_if_run and os.path.exists(os.path.join(ckpt_path, ckpt_name)):
            print("Checkpoint already exists at {} \nSkipping training on corruption(s): {}".format(
                os.path.join(ckpt_path, ckpt_name), corruption_names))
            continue

        # Log File set up
        log_name = "{}_{}.log".format(experiment, '-'.join(corruption_names))
        logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
        print("Logging file created for experiment {} to train on corruption(s): {}".format(experiment,
                                                                                            corruption_names))
        # Data Set Up
        corruption_paths = [os.path.join(data_root, corruption_name) for corruption_name in corruption_names]
        train_classes = list(range(total_n_classes))
        if "CrossEntropy" in experiment:  # randomly mix all corruptions
            trn_dl, val_dl, _ = get_multi_static_emnist_dataloaders(corruption_paths, train_classes, batch_size,
                                                                    True, n_workers, pin_mem)
            trn_dls = [trn_dl]
            val_dls = [val_dl]
        elif "Contrastive" in experiment or "Modules" in experiment:  # each batch contains the same images with different corruptions
            generators = [torch.Generator(device='cpu').manual_seed(2147483647) for _ in range(len(corruption_paths))]
            single_corr_bs = batch_size // len(corruption_names)
            trn_dls, val_dls = [], []
            for corruption_path, generator in zip(corruption_paths, generators):
                trn_dl, val_dl, _ = get_static_emnist_dataloaders(corruption_path, train_classes, single_corr_bs,
                                                                  True, n_workers, pin_mem, fixed_generator=generator)
                trn_dls.append(trn_dl)
                val_dls.append(val_dl)

            if "Modules" in experiment:
                if len(corruption_names) != 2:
                    raise ValueError("Initial module training only uses single corruptions (plus the identity)")

                # Load identity network or create it if it doesn't exist
                network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, "ModulesV3",
                                                                                 ["Identity"], dev)
                assert len(network_blocks) >= 2  # assumed when the network is called
                assert len(weights) == len(network_blocks)  # one module before each layer (including image space)
                if os.path.exists(os.path.join(ckpt_path, network_block_ckpt_names[0])):
                    for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
                        block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
                        block.eval()
                    logger.info("Loaded identity network from {}".format(ckpt_path))
                elif os.path.exists(os.path.join(ckpt_path, "es_ckpt_{}.pt".format(network_block_ckpt_names[0]))):
                    raise RuntimeError("Early stopping checkpoint found. Training may be running in another process.")
                else:
                    logger.info("Identity network not found. Training it now.")
                    _ = train_identity_network(network_blocks, network_block_ckpt_names, data_root,
                                               ckpt_path, logging_path, experiment, total_n_classes,
                                               es_burn_in, max_epochs, batch_size, lr, n_workers, pin_mem, dev)
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
        cross_entropy_loss = nn.CrossEntropyLoss()
        accuracy_fn = lambda x, y: accuracy(x, y)
        if "Contrastive" in experiment or "Modules" in experiment:
            contrastive_loss = ContrastiveLayerLoss(single_corr_bs, temperature, dev)

        # Network & Optimizer Set Up
        if "Modules" not in experiment:  # CrossEntropy and Contrastive
            if "Contrastive" in experiment:
                if "Auto" in experiment:
                    logger.info("Choosing Levels of Abstraction")
                    contrastive_levels = find_contrastive_abstraction_level(corruption_names, trn_dls, val_dls, lr,
                                                                            cross_entropy_loss, accuracy_fn,
                                                                            contrastive_loss, weights, total_n_classes,
                                                                            single_corr_bs, dev)
                elif "ModLevel" in experiment:  # Hardcoded to match the best AutoModules levels of abstraction seen so far
                    contrastive_levels = [1, 1, 1, 2, 3, 4]
                else:  # Penultimate layer (fully connected)
                    contrastive_levels = [5] * (len(corruption_names) - 1)
                logger.info("Using Levels of Abstraction {}".format(contrastive_levels))

            network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment,
                                                                             corruption_names, dev)
            assert len(network_blocks) >= 2  # assumed when the network is called
            if weights is not None:
                assert len(weights) == len(network_blocks)
            all_parameters = []
            for block in network_blocks:
                all_parameters += list(block.parameters())
            optim = torch.optim.Adam(all_parameters, lr)
            # Early Stopping Set Up
            es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(block_ckpt_name)) for block_ckpt_name in
                             network_block_ckpt_names]
            early_stoppings = [EarlyStopping(patience=25, verbose=True, path=es_ckpt_path, trace_func=logger.info) for
                               es_ckpt_path in es_ckpt_paths]
            assert len(early_stoppings) == len(network_blocks)
        elif "Modules" in experiment:
            if "NoPassThrough" in experiment:
                pass_through = False
            else:
                pass_through = True
            if "Auto" in experiment:
                logger.info("Choosing Level of Abstraction")
                module_level = find_module_abstraction_level(network_blocks, trn_dls, val_dls, lr, cross_entropy_loss,
                                                             accuracy_fn, contrastive_loss, weights, single_corr_bs,
                                                             dev, pass_through)
                logger.info("Selected Best Level of Abstraction {}".format(module_level))
            else:
                logger.info("Manually Defined Level of Abstraction")
                if "Contrast" in corruption_names or "GaussianBlur" in corruption_names or \
                        "ImpulseNoise" in corruption_names or "Invert" in corruption_names:
                    module_level = 1  # After first conv layer for local corruptions
                else:
                    module_level = 4  # After last conv layer for long range dependencies
                logger.info("Using Level of Abstraction {}".format(module_level))

            modules, module_ckpt_names = create_emnist_modules(experiment, corruption_names, dev)
            # Last conv layer by default
            module = modules[module_level]
            module_ckpt_name = module_ckpt_names[module_level]
            optim = torch.optim.Adam(module.parameters(), lr)
            # Early Stopping Set Up
            es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(module_ckpt_name))]
            early_stoppings = [EarlyStopping(patience=25, verbose=True, path=es_ckpt_path, trace_func=logger.info) for
                               es_ckpt_path in es_ckpt_paths]
            assert len(early_stoppings) == 1
            if check_if_run and os.path.exists(os.path.join(ckpt_path, module_ckpt_name)):
                print("Checkpoint already exists at {} \nSkipping training on corruption(s): {}".format(
                    os.path.join(ckpt_path, module_ckpt_name), corruption_names))
                continue
        else:
            raise NotImplementedError("Experiment {} not implemented".format(experiment))

        # We want to check for early stopping after approximately equal number of batches:
        val_freq = len(trn_dls[0]) // len(corruption_names)
        burn_in_count = 0
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
                    ce_loss, acc = cross_entropy_forwards_pass(network_blocks, x_trn, y_trn, cross_entropy_loss,
                                                               accuracy_fn)
                    loss = ce_loss
                    epoch_ce_loss += ce_loss.item()
                    epoch_acc += acc
                elif "Contrastive" in experiment:
                    ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_trn, y_trn,
                                                                       cross_entropy_loss, accuracy_fn,
                                                                       contrastive_loss, contrastive_levels, weights,
                                                                       single_corr_bs)
                    loss = ce_loss + ctv_loss
                    epoch_ce_loss += ce_loss.item()
                    epoch_ctv_loss += ctv_loss.item()
                    epoch_acc += acc
                elif "Modules" in experiment:
                    ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, module_level, x_trn, y_trn,
                                                                   cross_entropy_loss, accuracy_fn, contrastive_loss,
                                                                   weights[module_level], single_corr_bs, pass_through)
                    loss = ce_loss + ctv_loss
                    epoch_ce_loss += ce_loss.item()
                    epoch_ctv_loss += ctv_loss.item()
                    epoch_acc += acc
                else:
                    raise NotImplementedError("Experiment {} not implemented".format(experiment))
                loss.backward()
                optim.step()
                # # Time batches
                # print("Batch {} of {} in epoch {} took {:.2f} seconds.".format(i, len(trn_dl), epoch,
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
                                                                           cross_entropy_loss, accuracy_fn)
                                valid_ce_loss += ce_loss.item()
                                valid_total_loss += ce_loss.item()
                                valid_acc += acc
                            elif "Contrastive" in experiment:
                                ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_val, y_val,
                                                                                   cross_entropy_loss, accuracy_fn,
                                                                                   contrastive_loss, contrastive_levels,
                                                                                   weights, single_corr_bs)
                                valid_ce_loss += ce_loss.item()
                                valid_ctv_loss += ctv_loss.item()
                                valid_total_loss += ce_loss.item() + ctv_loss.item()
                                valid_acc += acc
                            elif "Modules" in experiment:
                                ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, module_level,
                                                                               x_val, y_val,
                                                                               cross_entropy_loss, accuracy_fn,
                                                                               contrastive_loss,
                                                                               weights[module_level], single_corr_bs,
                                                                               pass_through)
                                valid_ce_loss += ce_loss.item()
                                valid_ctv_loss += ctv_loss.item()
                                valid_total_loss += ce_loss.item() + ctv_loss.item()
                                valid_acc += acc
                            else:
                                raise NotImplementedError("Experiment {} not implemented".format(experiment))
                    if "CrossEntropy" in experiment:
                        logger.info("Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
                    elif "Contrastive" in experiment or "Modules" in experiment:
                        logger.info("Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
                        logger.info("Validation contrastive loss {:6.4f}".format(valid_ctv_loss / len(val_dls[0])))
                    logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
                    # Early Stopping
                    burn_in_count += 1
                    if burn_in_count >= es_burn_in:
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
                                                               cross_entropy_loss, accuracy_fn)
                    valid_ce_loss += ce_loss.item()
                    valid_acc += acc
                elif "Contrastive" in experiment:
                    ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_val, y_val,
                                                                       cross_entropy_loss, accuracy_fn,
                                                                       contrastive_loss, contrastive_levels, weights,
                                                                       single_corr_bs)
                    valid_ce_loss += ce_loss.item()
                    valid_ctv_loss += ctv_loss.item()
                    valid_acc += acc
                elif "Modules" in experiment:
                    ce_loss, ctv_loss, acc = modules_forwards_pass(network_blocks, module, module_level, x_val, y_val,
                                                                   cross_entropy_loss, accuracy_fn, contrastive_loss,
                                                                   weights[module_level], single_corr_bs, pass_through)
                    valid_ce_loss += ce_loss.item()
                    valid_ctv_loss += ctv_loss.item()
                    valid_acc += acc
                else:
                    raise NotImplementedError("Experiment {} not implemented".format(experiment))
        if "CrossEntropy" in experiment:
            logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
        elif "Contrastive" in experiment or "Modules" in experiment:
            logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
            logger.info("Early Stopped validation contrastive loss {:6.4f}".format(valid_ctv_loss / len(val_dls[0])))
        logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))

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
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST5/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST5/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--logging-path', type=str, default='/om2/user/imason/compositions/logs/EMNIST5/',
                        help="path to directory to save logs")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/EMNIST5/visualisations/',
                        help="path to directory to save data visualisations")
    parser.add_argument('--experiment', type=str, default='CrossEntropy',
                        help="which method to use. CrossEntropy or Contrastive or Modules.")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--es-burn-in', type=int, default=10, help="min number of validation steps before early stopping")
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
    parser.add_argument('--weights', type=str, help="comma delimited string of weights for the contrastive loss")
    parser.add_argument('--temperature', type=float, default=0.15, help="contrastive loss temperature")
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

    # Convert comma delimited strings to lists
    if args.weights is not None:
        weights = [float(x) for x in args.weights.split(',')]
    else:
        weights = None

    # Create unmade directories
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

    if "Modules" in args.experiment: # For the Modules approach, we want all elemental corruptions
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
    CUDA_VISIBLE_DEVICES=4 python train_emnist.py --pin-mem --check-if-run --corruption-ID 0 -experiment CrossEntropy
    CUDA_VISIBLE_DEVICES=4 python train_emnist.py --pin-mem --check-if-run --corruption-ID 34 --vis-data --experiment Contrastive --weights  1,1,1,1,1,1
    CUDA_VISIBLE_DEVICES=4 python train_emnist.py --pin-mem --check-if-run --corruption-ID 0 --experiment Modules --weights 1,1,1,1,1,1
    """

    # Searching learning rates. Change --array=0-47 to --array=0-191
    # assert len(corruptions) == 48  # for EMNIST3
    # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    # exp_idx = args.corruption_ID // 48
    # corruptions = corruptions[(args.corruption_ID % len(corruptions)):(args.corruption_ID % len(corruptions)) + 1]
    # args.lr = lrs[exp_idx]

    main(corruptions, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.experiment, weights,
         args.temperature, args.total_n_classes, args.es_burn_in, args.max_epochs, args.batch_size, args.lr,
         args.n_workers, args.pin_mem, dev, args.vis_data, args.check_if_run)


