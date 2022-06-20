"""
Train a model on individual data transformations, pairs of transformations and all transformations. .
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_transforms import denormalize
from data.data_loaders import get_static_emnist_dataloaders
from lib.networks import DTN
from lib.early_stopping import EarlyStopping
from lib.utils import *

# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility
    # Todo: How am setting hparams? - Try a few shifts for quite some epochs? Use early stopping across the board once lr, etc. is set okay? Do loops over lrs?
        # Have done no optimisation of hparams at all. The current parameters are just guessed from experience.
    # Todo: WandB style tracking - loss curves etc.
    # Todo: split over multiple gpus for faster training over corruptions
    # Todo: Epochs. If multiple corruptions, the number of training steps is larger. Either set iterations instead of epochs or use early stopping well.
    # Todo: neatening - write the early stopping output to the logger, so we can see the last saved es_ckpt


# Todo: When we increase the weight of the invariance loss (say to 10) this encourages
# more neurons to die/fire minimally. Perhaps some type of threshold is needed to encourage them to live?
def invariance_loss(features, single_corr_bs, num_corrs, dev):
    mse = nn.MSELoss()
    if num_corrs == 2:
        f_1 = features[:single_corr_bs]
        f_2 = features[single_corr_bs:]
        return mse(f_1, f_2)
    elif num_corrs == 3:
        # f_1 = features[:single_corr_bs]
        # f_2 = features[single_corr_bs:2*single_corr_bs]
        # f_3 = features[2*single_corr_bs:]
        # return mse(f_1, f_2) + mse(f_1, f_3) + mse(f_2, f_3)

        # # Todo: optional. Rough contrastive loss with MSE.
        # # We just use the triple loss and push away from neighboring sample to avoid collapse to zero.
        # # Extreme hack - but to check quickly we only apply the neighbouring sample to indices after 1
        # similarity = mse(f_1, f_2) + mse(f_1, f_3) + mse(f_2, f_3)
        # difference = mse(f_1[:, 1:], f_2[:, :-1]) + mse(f_3[:, 1:], f_1[:, :-1]) + mse(f_2[:, 1:], f_3[:, :-1])
        # return similarity - difference  # probably this will just blow up the difference vectors. But we can try. -> Indeed they just blow up


        # Todo: replace MSE with NT-Xent loss from simCLR?
        # Todo: try the NT-Xent loss with only positive samples.
        # From https://github.com/sthalles/SimCLR/blob/master/simclr.py
        # Todo: how to make it work with 3 corruptions?
            # Basically consider each of the 3 pairs as positive pairs and everything else as negative pairs
            # Then can sum or take mean

        labels = torch.cat([torch.arange(single_corr_bs) for i in range(num_corrs)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(dev)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # labels and similarity_matrix are of shape (num_corrs * single_corr_bs, num_corrs * single_corr_bs)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(dev)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # labels and similarity_matrix are of shape (num_corrs * single_corr_bs, num_corrs * single_corr_bs - 1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # shape (num_corrs * single_corr_bs, num_corrs - 1) -> (num_corrs * single_corr_bs, 2)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # shape (num_corrs * single_corr_bs, num_corrs * single_corr_bs - (num_corrs - 1) - 1)
        # -> (num_corrs * single_corr_bs, num_corrs * single_corr_bs - 3)

        temperature = 0.15  #  0.07  # Todo: if works make this a hyperparameter
        # logits = torch.cat([positives, negatives], dim=1)
        # logits = logits / temperature
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dev)
        # shape (num_corrs * single_corr_bs, num_corrs * single_corr_bs - 1)

        # Explanation: https://github.com/sthalles/SimCLR/issues/16  https://github.com/sthalles/SimCLR/issues/33
        zero_logits = torch.cat([positives[:, 0:1], negatives], dim=1)
        zero_logits = zero_logits / temperature
        zero_labels = torch.zeros(zero_logits.shape[0], dtype=torch.long).to(dev)

        one_logits = torch.cat([positives[:, 1:2], negatives], dim=1)
        one_logits = one_logits / temperature
        one_labels = torch.ones(one_logits.shape[0], dtype=torch.long).to(dev)

        # Todo: there is weirdness here. All positive pairs are captured but not 'considered' in the same way
        # i.e. we don't take every possible pair of f1/f2/f3 and get all positive and negatives
        # rather we have some positives and some negatives and make things close/far
        # Overall I think it should be okay but can think a bit more about it
        # Also can try larger batch size/different temperature may help.

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(zero_logits, zero_labels)
        loss += criterion(one_logits, one_labels)

        return loss
        # temp 0.07, 50 epochs. Comp acc: 33% (35% with fewer epochs)
        # temp 0.15, 25 epochs. Comp acc 45.0%
        # temp 0.25, 25 epochs. Comp acc: 37%
        # temp 0.25, 25 epochs. Batch size 512 (up from 128). Comp acc 34%
        # temp 0.15, 25 epochs. Batch size 512 (up from 128). Comp acc 45.3%
        # temp 0.15, 150 epochs. Batch size 128. Comp acc 44.8% (early stopped on epoch 40 - so saved epoch 15)
        # temp 0.15, 50 epochs. Batch size 128. Sequential 25 epochs contrastive followed by 25 cross ent.
        #                                       Early stopping off. Comp acc: 14.7%
        # temp 0.1, 25 epochs. Batch size 128. Comp acc 40.1%
        # temp 0.01, 25 epochs. Batch size 128. Comp acc 22.4%
        # Add template loss as well as contrastive loss temp 0.15. Equal weighting. Comp acc: 38.3%
        # temp 0.15, 25 epochs. Batch size 128. Weight = 10. Comp acc 50%.
        # temp 0.15, 25 epochs. Batch size 128. Weight = 100. Comp acc 32%.
        # temp 0.15, 35 epochs. Batch size 128. 25 epochs contrastive followed by 10 cross ent, with cross ent only
        #                                       training the last layer. No early stopping: Comp acc: 43% (and only achieves ~60% on elementals).

        # Training
        # temp 0.15, 25 epochs. Batch size 128. Contrastive loss after conv layers as well as after first FC. Comp acc

        # Todo:
        # Contrastive loss in output of conv layers
            # By itself
            # Jointly w/ contrastive loss in output of FC layer
            # By itself but sequential with CE loss on the FC layers (so we can get better acc on the elementals).
        # Can we do something like contrastive loss in the template loss where positive pairs are themselves
        # pairs of units??

        # Note these are all done with wrong scaling on early stopping (shouldn't matter though)
        # Todo: fix division in early stopping to remove sum over dls.

        # mean loss (or use reduce sum rather than reduce mean)?
        # increase batch size? cosine similarity? weighting?
        # sequential? train on contrastive loss 25 epochs then on MSE 25 epochs (should only train last layer with MSE?)
    else:
        raise NotImplementedError("num_corrs must be 2 or 3")
        # Todo: implement for more corruptions. Maybe another method? This doesn't scale to arbitrary numbers of corrs.
        # Todo: current method has different scaling for different num corrs (because we sum).

# Todo: try out this loss, perhaps can examine it more carefully to check it does what we want and how
# it affects losses etc. At first just try it blindly.
# Todo: Why would first layer templates give composition?
# Todo: What about combining with other templates or invariance in the last layer?
# Todo: this function runs with the max or mean firing. Don't fully understand the differentiability of max
    # may be worth spending time to figure it out.
def first_layer_template_loss(features, single_corr_bs, num_corrs):
    mse = nn.MSELoss()
    if num_corrs == 2:
        assert len(features) == single_corr_bs * 2
        f_1 = features[:single_corr_bs]
        f_2 = features[single_corr_bs:]
        # Encourage each pair to be a template
        return mse(f_1[:, ::2], f_2[:, 1::2]) + mse(f_1[:, 1::2], f_2[:, ::2])
    elif num_corrs == 3:
        assert len(features) == single_corr_bs * 3
        f_1 = features[:single_corr_bs]
        f_2 = features[single_corr_bs:2*single_corr_bs]
        f_3 = features[2*single_corr_bs:]
        # Encourage each pair to be a template. Have some templates for each corruption.
        # For now this is hardcoded - we know there are 64 units = 32 pairs. So 10, 11, 11 templates.
        corr1_loss = mse(f_1[:, :20:2], f_2[:, 1:20:2]) + mse(f_1[:, 1:20:2], f_2[:, :20:2])
        corr2_loss = mse(f_1[:, 20:42:2], f_3[:, 21:42:2]) + mse(f_1[:, 21:42:2], f_3[:, 20:42:2])
        corr3_loss = mse(f_2[:, 42::2], f_3[:, 43::2]) + mse(f_2[:, 43::2], f_3[:, 42::2])
        return corr1_loss + corr2_loss + corr3_loss
    else:
        raise NotImplementedError("num_corrs must be 2 or 3")


class MeanConvHook:  # We calculate invariance using the max but max is not differentiable. Could also try a smooth max?
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # output - batchsize, num_units/channels, width, height (height and width may be other way round - not checked)
        max_spatial_firing, _ = torch.max(output.reshape(output.shape[0], output.shape[1], -1), dim=2)
        # mean_spatial_firing = torch.mean(output.reshape(output.shape[0], output.shape[1], -1), dim=2)
        self.output = max_spatial_firing  # batch_size, num_units

    def close(self):
        self.hook.remove()


def main(corruptions, data_root, ckpt_path, logging_path, vis_path, total_n_classes, max_epochs, batch_size, lr,
         n_workers, pin_mem, dev, vis_data, check_if_run):
    # Train all models
    for corruption_names in corruptions:
        ckpt_name = "{}_invariance-loss.pt".format('-'.join(corruption_names))
        # Check if training has already completed for the corruption(s) in question.
        if check_if_run and os.path.exists(os.path.join(ckpt_path, ckpt_name)):
            print("Checkpoint already exists at {}. \n Skipping training on corruption(s): {}".format(
                os.path.join(ckpt_path, ckpt_name), corruption_names))
            continue

        # Log File set up
        log_name = "{}_invariance-loss.log".format('-'.join(corruption_names))
        logger = custom_logger(os.path.join(logging_path, log_name), stdout=False)  # stdout True to see also in console
        print("Logging file created to train on corruption(s): {}".format(corruption_names))
        # Data Set Up
        corruption_paths = [os.path.join(data_root, corruption_name) for corruption_name in corruption_names]
        train_classes = list(range(total_n_classes))
        generators = [torch.Generator(device='cpu').manual_seed(2147483647) for _ in range(len(corruption_paths))]
        single_corr_bs = batch_size // len(corruption_names)
        # Use static dataloader for the potential to change the classes being trained on
        trn_dls, val_dls, tst_dls = [], [], []
        for corruption_path, generator in zip(corruption_paths, generators):
            trn_dl, val_dl, tst_dl = get_static_emnist_dataloaders(corruption_path, train_classes, single_corr_bs,
                                                                   True, n_workers, pin_mem, fixed_generator=generator)
            trn_dls.append(trn_dl)
            val_dls.append(val_dl)
            tst_dls.append(tst_dl)

        # Todo: if useful add weighting as an argument
        weight = 1.0

        if vis_data:
            for i, trn_dl in enumerate(trn_dls):
                if i == 0:
                    x, y = next(iter(trn_dl))
                else:
                    x_temp, y_temp = next(iter(trn_dl))
                    x = torch.cat((x, x_temp), dim=0)
                    y = torch.cat((y, y_temp), dim=0)
            fig_name = "{}_invariance-loss.png".format('-'.join(corruption_names))
            fig_path = os.path.join(vis_path, fig_name)
            # Denormalise Images
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = denormalize(x).astype(np.uint8)
            # And visualise
            visualise_data(x[:100], y[:100], save_path=fig_path, title=fig_name[:-4], n_rows=10, n_cols=10)

        # Network & Optimizer Set Up
        network = DTN(total_n_classes).to(dev)
        # For template loss need hooks
        for module in network.modules():
            if isinstance(module, nn.Conv2d):
                print("Hooked module: {}".format(module))
                hooked_conv = MeanConvHook(module)
                break
        optim = torch.optim.Adam(network.parameters(), lr)
        # classifier_optim = torch.optim.Adam(network.classifier.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=25, verbose=True, path=os.path.join(ckpt_path, "es_ckpt.pt"))

        # Training
        for epoch in range(max_epochs):
            # Training
            network.train()
            epoch_ce_loss = 0.0
            epoch_inv_loss = 0.0
            epoch_acc = 0.0
            # epoch_visualised = False
            for data_tuples in zip(*trn_dls):
                for i, data_tuple in enumerate(data_tuples):
                    if i == 0:
                        x_trn, y_trn = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    else:
                        x_temp, y_temp = data_tuple[0].to(dev), data_tuple[1].to(dev)
                        x_trn = torch.cat((x_trn, x_temp), dim=0)
                        y_trn = torch.cat((y_trn, y_temp), dim=0)

                # Todo: when happy can remove this. It is a check to see if the data is being loaded in the correct order.
                # if not epoch_visualised and vis_data:
                #     fig_name = "{}_invariance-loss_epoch-{}.png".format('-'.join(corruption_names), epoch)
                #     fig_path = os.path.join(vis_path, fig_name)
                #     # Denormalise Images
                #     x = x_trn.detach().cpu().numpy()
                #     y = y_trn.detach().cpu().numpy()
                #     x = denormalize(x).astype(np.uint8)
                #     # And visualise
                #     visualise_data(x[:100], y[:100], save_path=fig_path, title=fig_name[:-4], n_rows=10, n_cols=10)
                #     epoch_visualised = True

                optim.zero_grad()
                # classifier_optim.zero_grad()
                features = network.conv_params(x_trn)
                features = features.view(features.size(0), -1)
                inv_loss = invariance_loss(features, single_corr_bs, len(trn_dls), dev)
                features = network.fc_params(features)
                inv_loss += invariance_loss(features, single_corr_bs, len(trn_dls), dev)
                output = network.classifier(features)
                # inv_loss += first_layer_template_loss(hooked_conv.output, single_corr_bs, len(trn_dls))
                ce_loss = criterion(output, y_trn)

                loss = ce_loss + weight * inv_loss
                acc = accuracy(output, y_trn)
                loss.backward()
                optim.step()

                # if epoch < 25:
                #     loss = inv_loss
                #     acc = accuracy(output, y_trn)
                #     loss.backward()
                #     optim.step()
                # else:
                #     loss = ce_loss
                #     acc = accuracy(output, y_trn)
                #     loss.backward()
                #     classifier_optim.step()

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
            network.eval()
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
                    features = network.conv_params(x_val)
                    features = features.view(features.size(0), -1)
                    # inv_loss = invariance_loss(features, single_corr_bs, len(val_dls), dev)
                    features = network.fc_params(features)
                    inv_loss = invariance_loss(features, single_corr_bs, len(val_dls), dev)
                    output = network.classifier(features)
                    # inv_loss = first_layer_template_loss(hooked_conv.output, single_corr_bs, len(val_dls))
                    ce_loss = criterion(output, y_val)
                    loss = ce_loss + weight * inv_loss
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
            early_stopping(valid_loss / sum([len(dl) for dl in val_dls]), network)  # ES on loss
            # early_stopping(100. - (valid_acc / sum([len(dl) for dl in val_dls])), network)  # ES on acc
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        # Save model
        logger.info("Loading early stopped checkpoint")
        early_stopping.load_from_checkpoint(network)
        network.eval()
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
                features = network.conv_params(x_val)
                features = features.view(features.size(0), -1)
                # inv_loss = invariance_loss(features, single_corr_bs, len(val_dls), dev)
                features = network.fc_params(features)
                inv_loss = invariance_loss(features, single_corr_bs, len(val_dls), dev)
                output = network.classifier(features)
                # inv_loss = first_layer_template_loss(hooked_conv.output, single_corr_bs, len(val_dls))
                ce_loss = criterion(output, y_val)
                loss = ce_loss + weight * inv_loss
                acc = accuracy(output, y_val)
                valid_ce_loss += ce_loss.item()
                valid_inv_loss += inv_loss.item()
                valid_loss += loss.item()
                valid_acc += acc
        logger.info("Early Stopped validation CE loss {:6.4f}".format(valid_ce_loss / len(val_dls[0])))
        logger.info("Early Stopped validation inv loss {:6.4f}".format(valid_inv_loss / len(val_dls[0])))
        logger.info("Early Stopped validation loss {:6.4f}".format(valid_loss / len(val_dls[0])))
        logger.info("Early Stopped validation accuracy {:6.3f}".format(valid_acc / len(val_dls[0])))

        hooked_conv.close()
        torch.save(network.state_dict(), os.path.join(ckpt_path, ckpt_name))
        logger.info("Saved best model to {}".format(os.path.join(ckpt_path, ckpt_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to train networks on different corruptions.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--logging-path', type=str, default='/om2/user/imason/compositions/logs/EMNIST/',
                        help="path to directory to save logs")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/EMNIST/visualisations/',
                        help="path to directory to save data visualisations")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--max-epochs', type=int, default=50, help="max number of training epochs")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--n-workers', type=int, default=4, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--vis-data', action='store_true', help="set to save a png of one batch of data")
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

    # Hardcode to see if interesting
    corruptions = [['identity', 'rotate_fixed', 'scale'],
                   ['identity', 'rotate_fixed'],
                   ['identity', 'scale']]
    corruptions = [['identity', 'rotate_fixed', 'scale']]

    main(corruptions, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.total_n_classes,
         args.max_epochs, args.batch_size, args.lr, args.n_workers, args.pin_mem, dev, args.vis_data, args.check_if_run)


