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
from lib.networks import create_emnist_network
from lib.early_stopping import EarlyStopping
from lib.utils import *


class ContrastiveLayerLoss:
    def __init__(self, single_corr_bs, temperature=0.15, n_views=2, dev=torch.device('cuda')):
        self.single_corr_bs = single_corr_bs
        self.temperature = temperature
        self.n_views = n_views
        self.dev = dev

    def __call__(self, features, weight):
        if weight == 0:
            return torch.tensor(0.0).to(self.dev)
        if self.n_views <= 1:
            return torch.tensor(0.0).to(self.dev)

        # Flatten spatial dimensions if 4D
        features = features.reshape(features.shape[0], -1)
        assert len(features) % self.single_corr_bs == 0
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(self.single_corr_bs) for _ in range(self.n_views)], dim=0)
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
        As starting point for why this works: https://github.com/sthalles/SimCLR/issues/16 (also issue 33)
        """
        for i in range(self.n_views - 1):
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


def contrastive_forwards_pass(network_blocks, x, y, cross_entropy_loss, accuracy_fn, contrastive_loss, weights):
    """
    Network forwards pass with an option to apply the contrastive loss at any intermediate layer (weighted by weights)
    """
    total_ctv_loss = 0.0
    for i, block in enumerate(network_blocks):
        if i == 0:
            features = block(x)
            total_ctv_loss += contrastive_loss(features, weights[i])
        elif i == len(network_blocks) - 1:
            output = block(features)
        else:
            features = block(features)
            total_ctv_loss += contrastive_loss(features, weights[i])

    return cross_entropy_loss(output, y), total_ctv_loss, accuracy_fn(output, y)


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
        if experiment == "CrossEntropy":  # randomly mix all corruptions
            trn_dl, val_dl, _ = get_multi_static_emnist_dataloaders(corruption_paths, train_classes, batch_size,
                                                                    True, n_workers, pin_mem)
            trn_dls = [trn_dl]
            val_dls = [val_dl]
        elif experiment == "Contrastive":  # each batch contains the same images with different corruptions
            generators = [torch.Generator(device='cpu').manual_seed(2147483647) for _ in range(len(corruption_paths))]
            single_corr_bs = batch_size // len(corruption_names)
            trn_dls, val_dls = [], []
            for corruption_path, generator in zip(corruption_paths, generators):
                trn_dl, val_dl, _ = get_static_emnist_dataloaders(corruption_path, train_classes, single_corr_bs,
                                                                  True, n_workers, pin_mem, fixed_generator=generator)
                trn_dls.append(trn_dl)
                val_dls.append(val_dl)
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

        # Network & Optimizer Set Up
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment, corruption_names,
                                                                         dev)
        assert len(network_blocks) >= 2  # assumed when the network is called
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
        # We want to check for early stopping after approximately equal number of batches:
        val_freq = len(trn_dls[0]) // len(corruption_names)
        burn_in_count = 0
        logger.info("Validation frequency: every {} batches".format(val_freq))

        # Loss Function Set Up
        cross_entropy_loss = nn.CrossEntropyLoss()
        accuracy_fn = lambda x, y: accuracy(x, y)
        if experiment == "Contrastive":
            contrastive_loss = ContrastiveLayerLoss(single_corr_bs, temperature, len(corruption_names), dev)
            assert len(weights) == len(network_blocks) - 1

        # Train Loop
        for epoch in range(max_epochs):
            # Training
            for block in network_blocks:
                block.train()
            epoch_ce_loss = 0.0
            epoch_ctv_loss = 0.0
            epoch_acc = 0.0

            # # Time batches
            # start_time = time.time()
            for i, trn_data_tuples in enumerate(zip(*trn_dls), 1):
                x_trn, y_trn = generate_batch(trn_data_tuples, dev)
                optim.zero_grad()
                if experiment == "CrossEntropy":
                    ce_loss, acc = cross_entropy_forwards_pass(network_blocks, x_trn, y_trn, cross_entropy_loss,
                                                               accuracy_fn)
                    loss = ce_loss
                    epoch_ce_loss += ce_loss.item()
                    epoch_acc += acc
                elif experiment == "Contrastive":
                    ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_trn, y_trn,
                                                                       cross_entropy_loss, accuracy_fn,
                                                                       contrastive_loss, weights)
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
                    for block in network_blocks:
                        block.eval()
                    valid_ce_loss = 0.0
                    valid_ctv_loss = 0.0
                    valid_total_loss = 0.0
                    valid_acc = 0.0
                    with torch.no_grad():
                        for val_data_tuples in zip(*val_dls):
                            x_val, y_val = generate_batch(val_data_tuples, dev)
                            if experiment == "CrossEntropy":
                                ce_loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val,
                                                                           cross_entropy_loss, accuracy_fn)
                                valid_ce_loss += ce_loss.item()
                                valid_total_loss += ce_loss.item()
                                valid_acc += acc
                            elif experiment == "Contrastive":
                                ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_val, y_val,
                                                                                   cross_entropy_loss, accuracy_fn,
                                                                                   contrastive_loss, weights)
                                valid_ce_loss += ce_loss.item()
                                valid_ctv_loss += ctv_loss.item()
                                valid_total_loss += ce_loss.item() + ctv_loss.item()
                                valid_acc += acc
                            else:
                                raise NotImplementedError("Experiment {} not implemented".format(experiment))
                    if experiment == "CrossEntropy":
                        logger.info("Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
                    elif experiment == "Contrastive":
                        logger.info("Validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
                        logger.info("Validation contrastive loss {:6.4f}".format(valid_ctv_loss / len(val_dls[0])))
                    logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))
                    # Early Stopping
                    burn_in_count += 1
                    if burn_in_count >= es_burn_in:
                        for es, block in zip(early_stoppings, network_blocks):
                            es(valid_total_loss / len(val_dls[0]), block)  # ES on loss
                        if early_stoppings[0].early_stop:
                            logger.info("Early stopping")
                            break
                    for block in network_blocks:
                        block.train()

            if early_stoppings[0].early_stop:
                break
            if experiment == "CrossEntropy":
                results = [epoch, epoch_ce_loss / len(trn_dls[0]), epoch_acc / len(trn_dls[0])]
                logger.info("Epoch {}. Avg CE train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))
            elif experiment == "Contrastive":
                results = [epoch, epoch_ce_loss / len(trn_dls[0]), epoch_ctv_loss / len(trn_dls[0]),
                           epoch_acc / len(trn_dls[0])]
                logger.info("Epoch {}. Avg CE train loss {:6.4f}. Avg contrastive train loss {:6.4f}. "
                            "Avg train acc {:6.3f}.".format(*results))

        # Save model
        logger.info("Loading early stopped checkpoints")
        for es, block in zip(early_stoppings, network_blocks):
            es.load_from_checkpoint(block)
        for block in network_blocks:
            block.eval()
        valid_ce_loss = 0.0
        valid_ctv_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for val_data_tuples in zip(*val_dls):
                x_val, y_val = generate_batch(val_data_tuples, dev)
                if experiment == "CrossEntropy":
                    ce_loss, acc = cross_entropy_forwards_pass(network_blocks, x_val, y_val,
                                                               cross_entropy_loss, accuracy_fn)
                    valid_ce_loss += ce_loss.item()
                    valid_acc += acc
                elif experiment == "Contrastive":
                    ce_loss, ctv_loss, acc = contrastive_forwards_pass(network_blocks, x_val, y_val,
                                                                       cross_entropy_loss, accuracy_fn,
                                                                       contrastive_loss, weights)
                    valid_ce_loss += ce_loss.item()
                    valid_ctv_loss += ctv_loss.item()
                    valid_acc += acc
                else:
                    raise NotImplementedError("Experiment {} not implemented".format(experiment))
        if experiment == "CrossEntropy":
            logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
        elif experiment == "Contrastive":
            logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
            logger.info("Early Stopped validation contrastive loss {:6.4f}".format(valid_ctv_loss / len(val_dls[0])))
        logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))

        for es in early_stoppings:
            es.delete_checkpoint()  # Removes from disk
        for block_ckpt_name, block in zip(network_block_ckpt_names, network_blocks):
            torch.save(block.state_dict(), os.path.join(ckpt_path, block_ckpt_name))
            logger.info("Saved best network block to {}".format(os.path.join(ckpt_path, block_ckpt_name)))


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

    if args.experiment not in ['CrossEntropy', 'Contrastive', 'Modules']:
        raise ValueError("Experiment must be one of CrossEntropy, Contrastive or Modules.")

    # Set seeding
    reset_rngs(seed=13579, deterministic=True)

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
    # Training on only the corruptions in the composition. Always include identity, remove permutations
    for corr in all_corruptions:
        if 'Identity' not in corr:
            corr.sort()
            corr += ['Identity']
            if corr not in corruptions:
                corruptions.append(corr)
        elif len(corr) == 1:  # identity only
            if corr not in corruptions:
                corruptions.append(corr)
        else:
            raise ValueError("Only expect the identity to appear as its own corruption")

    assert len(corruptions) == 64  # 64 for EMNIST 5, 128 for EMNIST4

    # Using slurm to parallelise the training
    corruptions = corruptions[args.corruption_ID:args.corruption_ID+1]

    """
    If running on polestar
    CUDA_VISIBLE_DEVICES=4 python train_emnist.py --pin-mem --check-if-run --corruption-ID 0
    CUDA_VISIBLE_DEVICES=4 python train_emnist.py --pin-mem --check-if-run --corruption-ID 34 --vis-data --experiment Contrastive --weights 0,0,0,0,10
    Include weights in above?
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


