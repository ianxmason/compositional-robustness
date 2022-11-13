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
from data.data_loaders import get_multi_static_emnist_dataloaders
from lib.networks import SimpleConvBlock, SimpleFullyConnectedBlock, SimpleClassifier
from lib.early_stopping import EarlyStopping
from lib.utils import *


def main(corruptions, data_root, ckpt_path, logging_path, vis_path, total_n_classes, es_burn_in, max_epochs,
         batch_size, lr, n_workers, pin_mem, dev, vis_data, check_if_run):
    assert max_epochs > es_burn_in
    # Train all models
    for corruption_names in corruptions:
        ckpt_name = "ConvBlock1_{}.pt".format('-'.join(corruption_names))
        # Check if training has already completed for the corruption(s) in question.
        if check_if_run and os.path.exists(os.path.join(ckpt_path, ckpt_name)):
            print("Checkpoint already exists at {} \nSkipping training on corruption(s): {}".format(
                os.path.join(ckpt_path, ckpt_name), corruption_names))
            continue

        # Log File set up
        log_name = "{}.log".format('-'.join(corruption_names))
        logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
        print("Logging file created to train on corruption(s): {}".format(corruption_names))
        # Data Set Up
        corruption_paths = [os.path.join(data_root, corruption_name) for corruption_name in corruption_names]
        train_classes = list(range(total_n_classes))
        # Use static dataloader for the potential to change the classes being trained on
        trn_dl, val_dl, tst_dl = get_multi_static_emnist_dataloaders(corruption_paths, train_classes, batch_size, True,
                                                                     n_workers, pin_mem)

        if vis_data:
            x, y = next(iter(trn_dl))
            fig_name = "{}.png".format('-'.join(corruption_names))
            fig_path = os.path.join(vis_path, fig_name)
            # Denormalise Images
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = denormalize(x).astype(np.uint8)
            # And visualise
            visualise_data(x[:36], y[:36], save_path=fig_path, title=fig_name[:-4], n_rows=6, n_cols=6)

        # Network & Optimizer Set Up
        network_blocks = []
        network_block_ckpt_names = []

        network_blocks.append(
            nn.Sequential(
                SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.1)  # 0.1
            ).to(dev)
        )
        network_block_ckpt_names.append("ConvBlock1_{}.pt".format('-'.join(corruption_names)))

        network_blocks.append(
            nn.Sequential(
                SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.3)  # 0.3
            ).to(dev)
        )
        network_block_ckpt_names.append("ConvBlock2_{}.pt".format('-'.join(corruption_names)))

        network_blocks.append(
            nn.Sequential(
                SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5)  # 0.5
            ).to(dev)
        )
        network_block_ckpt_names.append("ConvBlock3_{}.pt".format('-'.join(corruption_names)))

        # Extra block for more capacity. If used changed next block from 256 * 4 * 4 to 256 * 2 * 2.
        network_blocks.append(
            nn.Sequential(
                SimpleConvBlock(256, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5)  # 0.5
            ).to(dev)
        )
        network_block_ckpt_names.append("ConvBlock4_{}.pt".format('-'.join(corruption_names)))

        network_blocks.append(
            nn.Sequential(
                nn.Flatten(),  # Flattens everything except the batch dimension by default
                SimpleFullyConnectedBlock(256 * 2 * 2, 512, batch_norm=False, dropout=0.5)
            ).to(dev)
        )
        network_block_ckpt_names.append("FullyConnected_{}.pt".format('-'.join(corruption_names)))

        network_blocks.append(
            nn.Sequential(
                SimpleClassifier(512, total_n_classes)
            ).to(dev)
        )
        network_block_ckpt_names.append("Classifier_{}.pt".format('-'.join(corruption_names)))

        assert len(network_block_ckpt_names) == len(network_blocks)
        assert len(network_blocks) >= 2  # assumed when the network is called
        all_parameters = []
        for block in network_blocks:
            all_parameters += list(block.parameters())
        optim = torch.optim.Adam(all_parameters, lr)
        criterion = nn.CrossEntropyLoss()
        es_ckpt_paths = [os.path.join(ckpt_path, "es_ckpt_{}.pt".format(block_ckpt_name)) for block_ckpt_name in
                         network_block_ckpt_names]
        early_stoppings = [EarlyStopping(patience=25, verbose=True, path=es_ckpt_path, trace_func=logger.info) for
                           es_ckpt_path in es_ckpt_paths]
        assert len(early_stoppings) == len(network_blocks)


        # We want to check for early stopping after approximately equal number of batches
        # With more corruptions, trn_dl is larger, so the number of batches per epoch is higher
        val_freq = len(trn_dl) // len(corruption_names)
        val_count = 0
        logger.info("Validation frequency: every {} batches".format(val_freq))
        # Train Loop
        for epoch in range(max_epochs):
            # Training
            for block in network_blocks:
                block.train()
            epoch_loss = 0.0
            epoch_acc = 0.0

            # # Time batches
            # start_time = time.time()

            for i, data_tuple in enumerate(trn_dl, 1):
                x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
                optim.zero_grad()
                for j, block in enumerate(network_blocks):
                    if j == 0:
                        features = block(x_trn)
                    elif j == len(network_blocks) - 1:
                        output = block(features)
                    else:
                        features = block(features)
                loss = criterion(output, y_trn)
                acc = accuracy(output, y_trn)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                epoch_acc += acc

                # # Time batches
                # print("Batch {} of {} in epoch {} took {:.2f} seconds.".format(i, len(trn_dl), epoch,
                #                                                                time.time() - start_time))
                # start_time = time.time()

                if i % val_freq == 0:
                    # Validation
                    for block in network_blocks:
                        block.eval()
                    valid_loss = 0.0
                    valid_acc = 0.0
                    with torch.no_grad():
                        for data_tuple in val_dl:
                            x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                            for j, block in enumerate(network_blocks):
                                if j == 0:
                                    features = block(x_val)
                                elif j == len(network_blocks) - 1:
                                    output = block(features)
                                else:
                                    features = block(features)
                            loss = criterion(output, y_val)
                            acc = accuracy(output, y_val)
                            valid_loss += loss.item()
                            valid_acc += acc
                    logger.info("Validation loss {:6.4f}".format(valid_loss / len(val_dl)))
                    logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
                    # Early Stopping
                    val_count += 1
                    if val_count >= es_burn_in:
                        for es, block in zip(early_stoppings, network_blocks):
                            es(valid_loss / len(val_dl), block)  # ES on loss
                            # es(100. - (valid_acc / len(val_dl)), block)  # ES on acc
                        if early_stoppings[0].early_stop:
                            logger.info("Early stopping")
                            break
                    for block in network_blocks:
                        block.train()

            if early_stoppings[0].early_stop:
                break
            results = [epoch, epoch_loss / len(trn_dl), epoch_acc / len(trn_dl)]
            logger.info("Epoch {}. Avg train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

        # Save model
        logger.info("Loading early stopped checkpoints")
        for es, block in zip(early_stoppings, network_blocks):
            es.load_from_checkpoint(block)
        for block in network_blocks:
            block.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuple in val_dl:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                for j, block in enumerate(network_blocks):
                    if j == 0:
                        features = block(x_val)
                    elif j == len(network_blocks) - 1:
                        output = block(features)
                    else:
                        features = block(features)
                loss = criterion(output, y_val)
                acc = accuracy(output, y_val)
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dl)))
        logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
        for es in early_stoppings:
            es.delete_checkpoint()  # Removes from disk
        for block_ckpt_name, block in zip(network_block_ckpt_names, network_blocks):
            torch.save(block.state_dict(), os.path.join(ckpt_path, block_ckpt_name))
            logger.info("Saved best network block to {}".format(os.path.join(ckpt_path, block_ckpt_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to train networks on different corruptions.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST4/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST4/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--logging-path', type=str, default='/om2/user/imason/compositions/logs/EMNIST4/',
                        help="path to directory to save logs")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/EMNIST4/visualisations/',
                        help="path to directory to save data visualisations")
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

    assert len(corruptions) == 128  # for EMNIST4

    # Using slurm to parallelise the training
    corruptions = corruptions[args.corruption_ID:args.corruption_ID+1]

    """
    If running on polestar
    CUDA_VISIBLE_DEVICES=4 python train_emnist.py --pin-mem --check-if-run --corruption-ID 0
    """

    # Searching learning rates. Change --array=0-47 to --array=0-191
    # assert len(corruptions) == 48  # for EMNIST3
    # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    # exp_idx = args.corruption_ID // 48
    # corruptions = corruptions[(args.corruption_ID % len(corruptions)):(args.corruption_ID % len(corruptions)) + 1]
    # args.lr = lrs[exp_idx]

    main(corruptions, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.total_n_classes,
         args.es_burn_in, args.max_epochs, args.batch_size, args.lr, args.n_workers, args.pin_mem, dev, args.vis_data,
         args.check_if_run)


