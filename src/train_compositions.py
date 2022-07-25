"""
Train a model on individual data transformations, pairs of transformations and all transformations. .
"""
import argparse
import os
import pickle
import torch
import torch.nn as nn
from data.data_transforms import denormalize
from data.data_loaders import get_multi_static_emnist_dataloaders
from lib.networks import DTN
from lib.early_stopping import EarlyStopping
from lib.utils import *

# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility
    # Todo: How am setting hparams? - Try a few shifts for quite some epochs? Use early stopping across the board once lr, etc. is set okay? Do loops over lrs?
        # Have done no optimisation of hparams at all. The current parameters are just guessed from experience.
    # Todo: WandB style tracking - loss curves etc.
    # Todo: split over multiple gpus for faster training over corruptions
    # Todo: neatening - write the early stopping output to the logger, so we can see the last saved es_ckpt


def main(corruptions, data_root, ckpt_path, logging_path, vis_path, total_n_classes, min_epochs, max_epochs,
         batch_size, lr, n_workers, pin_mem, dev, vis_data, check_if_run):
    assert max_epochs > min_epochs
    # Train all models
    for corruption_names in corruptions:
        ckpt_name = "{}.pt".format('-'.join(corruption_names))
        # Check if training has already completed for the corruption(s) in question.
        if check_if_run and os.path.exists(os.path.join(ckpt_path, ckpt_name)):
            print("Checkpoint already exists at {}. \n Skipping training on corruption(s): {}".format(
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
        network = DTN(total_n_classes).to(dev)
        optim = torch.optim.Adam(network.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        es_ckpt_path = os.path.join(ckpt_path, "es_ckpt_{}.pt".format('-'.join(corruption_names)))
        early_stopping = EarlyStopping(patience=25, verbose=True, path=es_ckpt_path)

        # Train Loop
        val_freq = len(trn_dl) // len(corruption_names)
        logger.info("Validation frequency: every {} batches".format(val_freq))
        for epoch in range(max_epochs):
            # Training
            network.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, data_tuple in enumerate(trn_dl, 1):
                x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
                optim.zero_grad()
                output = network(x_trn)
                loss = criterion(output, y_trn)
                acc = accuracy(output, y_trn)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                epoch_acc += acc

                if i % val_freq == 0:
                    # Validation
                    network.eval()
                    valid_loss = 0.0
                    valid_acc = 0.0
                    with torch.no_grad():
                        for data_tuple in val_dl:
                            x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                            output = network(x_val)
                            loss = criterion(output, y_val)
                            acc = accuracy(output, y_val)
                            valid_loss += loss.item()
                            valid_acc += acc
                    logger.info("Validation loss {:6.4f}".format(valid_loss / len(val_dl)))
                    logger.info("Validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
                    # Early Stopping
                    if epoch > min_epochs:
                        early_stopping(valid_loss / len(val_dl), network)  # ES on loss
                        # early_stopping(100. - (valid_acc / len(val_dl)), network)  # ES on acc
                        if early_stopping.early_stop:
                            logger.info("Early stopping")
                            break
                    network.train()

            if early_stopping.early_stop:
                break
            results = [epoch, epoch_loss / len(trn_dl), epoch_acc / len(trn_dl)]
            logger.info("Epoch {}. Avg train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

        # Save model
        logger.info("Loading early stopped checkpoint")
        early_stopping.load_from_checkpoint(network)
        network.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data_tuple in val_dl:
                x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                output = network(x_val)
                loss = criterion(output, y_val)
                acc = accuracy(output, y_val)
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dl)))
        logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dl)))
        early_stopping.delete_checkpoint()  # Removes from disk

        torch.save(network.state_dict(), os.path.join(ckpt_path, ckpt_name))
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
    parser.add_argument('--n-workers', type=int, default=4, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--vis-data', action='store_true', help="set to save a png of one batch of data")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
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

    # Corruptions to train on - for now we don't do triples. If trains fast we can add them
    # corruptions = [['identity'],
    #                ['impulse_noise'],
    #                ['inverse'],
    #                ['gaussian_blur'],
    #                ['stripe'],
    #                ['identity', 'impulse_noise'],
    #                ['identity', 'inverse'],
    #                ['identity', 'gaussian_blur'],
    #                ['identity', 'stripe'],
    #                ['impulse_noise', 'inverse'],
    #                ['impulse_noise', 'gaussian_blur'],
    #                ['impulse_noise', 'stripe'],
    #                ['inverse', 'gaussian_blur'],
    #                ['inverse', 'stripe'],
    #                ['gaussian_blur', 'stripe'],
    #                ['identity', 'impulse_noise', 'inverse'],
    #                ['identity', 'impulse_noise', 'gaussian_blur'],
    #                ['identity', 'impulse_noise', 'stripe'],
    #                ['identity', 'inverse', 'gaussian_blur'],
    #                ['identity', 'inverse', 'stripe'],
    #                ['identity', 'gaussian_blur', 'stripe'],
    #                ['identity', 'impulse_noise', 'inverse', 'gaussian_blur', 'stripe']]

    # Add triple corruptions - can then add to heatmap. Can actually add to the above list and check-if-run will skip trained
    # corruptions = [['identity', 'impulse_noise', 'inverse', 'gaussian_blur'],
    #                ['identity', 'impulse_noise', 'inverse', 'stripe'],
    #                ['identity', 'impulse_noise', 'gaussian_blur', 'stripe'],
    #                ['identity', 'inverse', 'gaussian_blur', 'stripe']]

    # Hardcode canny_edges-inverse to see if interesting
    # corruptions = [['identity', 'rotate_fixed'],
    #                ['identity', 'scale'],
    #                ['identity', 'rotate_fixed', 'scale']]

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)

    corruptions = []
    # Always include identity, remove permutations
    for corr in all_corruptions:
        if 'identity' not in corr:
            corr.sort()
            corr += ['identity']
            if corr not in corruptions:
                corruptions.append(corr)
        elif len(corr) == 1:  # identity only
            if corr not in corruptions:
                corruptions.append(corr)
        else:
            raise ValueError("Only expect the identity to appear as its own corruption")

    # Using slurm to parallelise the training
    corruptions = corruptions[args.corruption_ID:args.corruption_ID+1]

    main(corruptions, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.total_n_classes,
         args.min_epochs, args.max_epochs, args.batch_size, args.lr, args.n_workers, args.pin_mem, dev, args.vis_data,
         args.check_if_run)


