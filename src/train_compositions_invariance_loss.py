"""
Train a model on individual data transformations, pairs of transformations and all transformations. .
"""
import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_transforms import denormalize
from data.data_loaders import get_static_emnist_dataloaders
from lib.networks import DTN, DTN_half
from lib.early_stopping import EarlyStopping
from lib.utils import *
from lib.equivariant_hooks import *

# Todo: For if/when run more detailed/designed experiments
    # Todo - seeding/reproducibility
    # Todo: How am setting hparams? - Try a few shifts for quite some epochs? Use early stopping across the board once lr, etc. is set okay? Do loops over lrs?
        # Have done no optimisation of hparams at all. The current parameters are just guessed from experience.
    # Todo: WandB style tracking - loss curves etc.
    # Todo: split over multiple gpus for faster training over corruptions
    # Todo: Epochs. If multiple corruptions, the number of training steps is larger. Either set iterations instead of epochs or use early stopping well.
    # Todo: neatening - write the early stopping output to the logger, so we can see the last saved es_ckpt


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

        contrastive_criterion = torch.nn.CrossEntropyLoss()
        loss = contrastive_criterion(zero_logits, zero_labels)
        loss += contrastive_criterion(one_logits, one_labels)

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
        # temp 0.15, 25 epochs. Batch size 128. Contrastive loss after conv layers as well as after first FC. Comp acc 34.9%
        # temp 0.15, 25 epochs. Batch size 128. Contrastive loss after conv layers only. Comp acc 25.6%

        # Contrastive loss in the first layer to encourage templates. 15.9%
        # Contrastive loss in the first layer to encourage templates (method 2). 16.3%
        # Contrastive loss in the first layer to encourage templates (method 2) + contrastive loss in fc layer. 35.8%
        # Contrastive loss in the first layer to encourage templates on half the units(method 2)
          #  + contrastive loss in fc layer. 33%

        # Training
        # Todo:
        # Implement equivariant first layer for rotation and scale

        # Note these are all done with wrong scaling on early stopping (shouldn't matter though)
        # Todo: fix division in early stopping to remove sum over dls.

        # mean loss (or use reduce sum rather than reduce mean)?
        # increase batch size? cosine similarity? weighting?
        # sequential? train on contrastive loss 25 epochs then on MSE 25 epochs (should only train last layer with MSE?)
    else:
        raise NotImplementedError("num_corrs must be 2 or 3")
        # Todo: implement for more corruptions. Maybe another method? This doesn't scale to arbitrary numbers of corrs.
        # Todo: current method has different scaling for different num corrs (because we sum).


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
        zero_logits = torch.cat([positives[:, 0:1], negatives], dim=1)
        zero_logits = zero_logits / temperature
        zero_labels = torch.zeros(zero_logits.shape[0], dtype=torch.long).to(dev)

        one_logits = torch.cat([positives[:, 1:2], negatives], dim=1)
        one_logits = one_logits / temperature
        one_labels = torch.ones(one_logits.shape[0], dtype=torch.long).to(dev)

        # Todo: make sure am really happy with the contrastive loss for >2 corruptions/positive pairs

        # Todo: there is weirdness here. All positive pairs are captured but not 'considered' in the same way
        # i.e. we don't take every possible pair of f1/f2/f3 and get all positive and negatives
        # rather we have some positives and some negatives and make things close/far
        # Overall I think it should be okay but can think a bit more about it
        # Also can try larger batch size/different temperature may help.
        loss = contrastive_criterion(zero_logits, zero_labels)
        loss += contrastive_criterion(one_logits, one_labels)
    else:
        raise NotImplementedError("Only currently handles 2 or 3 corruptions")

    return loss


# Todo: docs say hooks should really be used for visualising and debugging. Maybe we should split the forward
#   pass into layers so we don't need the hook?
class FlatConvHook:  # We calculate invariance using the max but max is not differentiable. Could also try a smooth max?
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # output - batchsize, num_units/channels, width, height (height and width may be other way round - not checked)

        # max_spatial_firing, _ = torch.max(output.reshape(output.shape[0], output.shape[1], -1), dim=2)
        # self.output = max_spatial_firing  # batch_size, num_units

        # mean_spatial_firing = torch.mean(output.reshape(output.shape[0], output.shape[1], -1), dim=2)
        # self.output = mean_spatial_firing  # batch_size, num_units

        flat_firing = output.reshape(output.shape[0], -1)
        self.output = flat_firing  # batch_size, num_units*feature_map_size

    def close(self):
        self.hook.remove()


def main(corruptions, data_root, ckpt_path, logging_path, vis_path, total_n_classes, max_epochs, batch_size, lr,
         weights, compare_corrs, experiment_name, n_workers, pin_mem, dev, vis_data, check_if_run):
    # Train all models
    for corruption_names in corruptions:
        ckpt_name = "{}_{}.pt".format('-'.join(corruption_names), experiment_name)
        # Check if training has already completed for the corruption(s) in question.
        if check_if_run and os.path.exists(os.path.join(ckpt_path, ckpt_name)):
            print("Checkpoint already exists at {}. \n Skipping training on corruption(s): {}".format(
                os.path.join(ckpt_path, ckpt_name), corruption_names))
            continue

        # Log File set up
        log_name = "{}_{}.log".format('-'.join(corruption_names), experiment_name)
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

        # Todo: if continue add the composition validation accuracy in a non-hardcoded way
        # composition = "rotate_fixed-scale"
        # _, comp_dl, _ = get_static_emnist_dataloaders(os.path.join(data_root, composition), train_classes,
        #                                              batch_size, False, n_workers, pin_mem)

        if vis_data:
            for i, trn_dl in enumerate(trn_dls):
                if i == 0:
                    x, y = next(iter(trn_dl))
                else:
                    x_temp, y_temp = next(iter(trn_dl))
                    x = torch.cat((x, x_temp), dim=0)
                    y = torch.cat((y, y_temp), dim=0)
            fig_name = "{}_{}.png".format('-'.join(corruption_names), experiment_name)
            fig_path = os.path.join(vis_path, fig_name)
            # Denormalise Images
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = denormalize(x).astype(np.uint8)
            # And visualise
            visualise_data(x[:100], y[:100], save_path=fig_path, title=fig_name[:-4], n_rows=10, n_cols=10)

        # Network & Optimizer Set Up
        if "equivariant" in experiment_name:
            network = DTN_half(total_n_classes).to(dev)
        else:
            network = DTN(total_n_classes).to(dev)
        # For template loss need hooks
        conv_count = 0
        for module in network.modules():
            if isinstance(module, nn.Conv2d):
                conv_count += 1
                if conv_count == 1 and "equivariant" in experiment_name:
                    rot_hook = RotationHook(module)
                    # scale_hook = ScaleHook(module)
                    print("Hooked module: {}".format(module))
            if isinstance(module, nn.ReLU):
                if conv_count == 1:
                    hooked_conv = FlatConvHook(module)
                    print("Hooked module: {}".format(module))
                elif conv_count == 2:
                    hooked_conv2 = FlatConvHook(module)
                    print("Hooked module: {}".format(module))
                elif conv_count == 3:
                    hooked_conv3 = FlatConvHook(module)
                    print("Hooked module: {}".format(module))
                    break
        optim = torch.optim.Adam(network.parameters(), lr)
        # classifier_optim = torch.optim.Adam(network.classifier.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        es_ckpt_path = os.path.join(ckpt_path, "es_ckpt_{}_{}.pt".format('-'.join(corruption_names), experiment_name))
        early_stopping = EarlyStopping(patience=25, verbose=True, path=es_ckpt_path)

        # Training
        l1w, l2w, l3w, l4w = weights
        comp1, comp2, comp3, comp4 = compare_corrs
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
                #     fig_name = "{}_{}_epoch-{}.png".format('-'.join(corruption_names), experiment_name, epoch)
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
                inv_loss = 0
                features = network.conv_params(x_trn)
                features = features.view(features.size(0), -1)
                features = network.fc_params(features)
                inv_loss += l4w * contrastive_layer_loss(features, single_corr_bs, dev, compare=comp4)
                inv_loss += l1w * contrastive_layer_loss(hooked_conv.output, single_corr_bs, dev, compare=comp1)
                inv_loss += l2w * contrastive_layer_loss(hooked_conv2.output, single_corr_bs, dev, compare=comp2)
                inv_loss += l3w * contrastive_layer_loss(hooked_conv3.output, single_corr_bs, dev, compare=comp3)
                output = network.classifier(features)
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
                    inv_loss = 0
                    features = network.conv_params(x_val)
                    features = features.view(features.size(0), -1)
                    features = network.fc_params(features)
                    inv_loss += l4w * contrastive_layer_loss(features, single_corr_bs, dev, compare=comp4)
                    inv_loss += l1w * contrastive_layer_loss(hooked_conv.output, single_corr_bs, dev, compare=comp1)
                    inv_loss += l2w * contrastive_layer_loss(hooked_conv2.output, single_corr_bs, dev, compare=comp2)
                    inv_loss += l3w * contrastive_layer_loss(hooked_conv3.output, single_corr_bs, dev, compare=comp3)
                    output = network.classifier(features)
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

            # comp_loss = 0.0
            # comp_acc = 0.0
            # with torch.no_grad():
            #     for data_tuple in comp_dl:
            #         x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
            #         features = network.conv_params(x_val)
            #         features = features.view(features.size(0), -1)
            #         features = network.fc_params(features)
            #         output = network.classifier(features)
            #         loss = criterion(output, y_val)
            #         acc = accuracy(output, y_val)
            #         comp_loss += loss.item()
            #         comp_acc += acc
            # logger.info("Composition CE loss {:6.4f}".format(comp_loss / len(comp_dl)))
            # logger.info("Composition accuracy {:6.3f}".format(comp_acc / len(comp_dl)))

            # Early Stopping
            early_stopping(valid_loss / len(val_dls[0]), network)  # ES on loss
            # early_stopping(100. - (valid_acc / len(val_dls[0]), network)  # ES on acc
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
                inv_loss = 0
                features = network.conv_params(x_val)
                features = features.view(features.size(0), -1)
                features = network.fc_params(features)
                inv_loss += l4w * contrastive_layer_loss(features, single_corr_bs, dev, compare=comp4)
                inv_loss += l1w * contrastive_layer_loss(hooked_conv.output, single_corr_bs, dev, compare=comp1)
                inv_loss += l2w * contrastive_layer_loss(hooked_conv2.output, single_corr_bs, dev, compare=comp2)
                inv_loss += l3w * contrastive_layer_loss(hooked_conv3.output, single_corr_bs, dev, compare=comp3)
                output = network.classifier(features)
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
        if "equivariant" in experiment_name:
            rot_hook.close()
            # scale_hook.close()
        hooked_conv.close()
        hooked_conv2.close()
        hooked_conv3.close()
        torch.save(network.state_dict(), os.path.join(ckpt_path, ckpt_name))
        logger.info("Saved best model to {}".format(os.path.join(ckpt_path, ckpt_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to train networks on different corruptions.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST2/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST2/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--logging-path', type=str, default='/om2/user/imason/compositions/logs/EMNIST2/',
                        help="path to directory to save logs")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/EMNIST2/visualisations/',
                        help="path to directory to save data visualisations")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--max-epochs', type=int, default=50, help="max number of training epochs")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--n-workers', type=int, default=1, help="number of workers (PyTorch)")
    # 4 workers seems to cause hanging. Likely due to zipping multiple dataloaders each with 4 workers
    # https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
    # setting num_workers > num_cores may be complicated.
    # Try solution: use 2 workers, ask for more cores from openmind
    # Each corruption adds 3*num_workers. So for 3 corruptions ask for 3*3*num_workers=18 cores
    # With 2 workers and 18 cores still get some hanging
    # With 1 worker and 18 cores seems to hang less (maybe never - not checked training over all shifts)
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--vis-data', action='store_true', help="set to save a png of one batch of data")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    parser.add_argument('--experiment-name', type=str, default='invariance-loss',
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

    # # Hardcode to see if interesting
    # corruptions = [['identity', 'rotate_fixed', 'scale']]

    # Get all pairs of corruptions (plus identity)
    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)

    corruptions = []
    # Always include identity, remove permutations
    for corr in all_corruptions:
        if 'identity' not in corr:
            corr += ['identity']
            corr.sort()
            if corr not in corruptions and len(corr) == 3:
                corruptions.append(corr)
        else:
            corr.sort()
            if corr not in corruptions and len(corr) == 3:
                corruptions.append(corr)

    # Using slurm to parallelise the training
    corruptions = corruptions[args.corruption_ID:args.corruption_ID + 1]

    main(corruptions, args.data_root, args.ckpt_path, args.logging_path, args.vis_path, args.total_n_classes,
         args.max_epochs, args.batch_size, args.lr, weights, compare_corrs, args.experiment_name, args.n_workers,
         args.pin_mem, dev, args.vis_data, args.check_if_run)
