"""
For hyperparameter searches we want to test different methods in distribution (i.e. on the training domains, i.e. the
elemental corruptions). We also want to use the validation set.

Additionally, for two step methods we want to validate step one using the mean squared error of the autoencoders on the
elemental corruptions they are trained on and the accuracy of the backbone network for the modular approach on the
Identity data.
"""
import argparse
import os
import torchvision
import torch.nn as nn
import data.data_transforms as dt
from test import loss_and_accuracy
from data.data_loaders import get_transformed_static_dataloaders, get_static_dataloaders
from data.emnist import EMNIST_MEAN, EMNIST_STD
from data.cifar import CIFAR10_MEAN, CIFAR10_STD
from data.facescrub import FACESCRUB_MEAN, FACESCRUB_STD
from lib.networks import create_emnist_network, create_emnist_modules, create_emnist_autoencoder, \
                         create_cifar_network, create_cifar_modules, create_cifar_autoencoder, \
                         create_facescrub_network, create_facescrub_modules, create_facescrub_autoencoder
from lib.utils import *


"""
1. Modules on identity - validation accuracy
2. AEs on training corruptions - validation MSE (average over all corruptions). Visualise one batch.
3. Contrastive on training corruptions - validation accuracy (average over all corruptions)
4. Cross entropy on training corruptions - validation accuracy (average over all corruptions)
5. Modules on training corruptions - validation accuracy (average over all corruptions)
6. AE identity classifier - validation accuracy (average over all corruptions)
7. AE joint classifier - validation accuracy (average over all corruptions)

print hparams, print results
"""


def val_module_backbone(experiment, dataset, data_root, ckpt_path, total_n_classes, batch_size, n_workers, pin_mem,
                        dev):
    if dataset == "EMNIST":
        network_blocks, network_block_ckpt_names = create_emnist_network(total_n_classes, experiment,
                                                                         ["Identity"], dev)
    elif dataset == "CIFAR":
        network_blocks, network_block_ckpt_names = create_cifar_network(total_n_classes, experiment,
                                                                        ["Identity"], dev)
    elif dataset == "FACESCRUB":
        network_blocks, network_block_ckpt_names = create_facescrub_network(total_n_classes, experiment,
                                                                            ["Identity"], dev)
    for block, block_ckpt_name in zip(network_blocks, network_block_ckpt_names):
        block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

    # Test the model on the Identity data
    trained_classes = list(range(total_n_classes))
    identity_path = os.path.join(data_root, "Identity")
    transforms = []
    _, val_dl, _ = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                      batch_size, False, n_workers, pin_mem)

    val_loss, val_acc = loss_and_accuracy(network_blocks, val_dl, dev)
    print("{}, Identity backbone. val loss: {:.4f}, val acc: {:.4f}".format(experiment, val_loss, val_acc))
    sys.stdout.flush()


def val_autoencoders_mse(experiment, dataset, data_root, ckpt_path, vis_path, total_n_classes, batch_size, n_workers,
                         pin_mem, dev):
    elemental_corruptions = ["Contrast", "GaussianBlur", "ImpulseNoise", "Invert", "Rotate90", "Swirl"]
    all_val_losses = []
    for corr in elemental_corruptions:
        test_corruptions = [corr, "Identity"]
        if dataset == "EMNIST":
            blocks, block_ckpt_names = create_emnist_autoencoder(experiment, test_corruptions, dev)
            denorm_mean, denorm_std = EMNIST_MEAN, EMNIST_STD
        elif dataset == "CIFAR":
            blocks, block_ckpt_names = create_cifar_autoencoder(experiment, test_corruptions, dev)
            denorm_mean, denorm_std = CIFAR10_MEAN, CIFAR10_STD
        elif dataset == "FACESCRUB":
            blocks, block_ckpt_names = create_facescrub_autoencoder(experiment, test_corruptions, dev)
            denorm_mean, denorm_std = FACESCRUB_MEAN, FACESCRUB_STD
        else:
            raise ValueError("Unknown dataset: {}".format(dataset))
        for block, block_ckpt_name in zip(blocks, block_ckpt_names):
            block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))

        trained_classes = list(range(total_n_classes))
        identity_path = os.path.join(data_root, "Identity")
        single_corr_bs = batch_size // len(test_corruptions)
        val_dls = []
        # During training we needed a fixed generator to get the same shuffle. During validation we just don't shuffle.
        for tc in test_corruptions:
            if dataset == "EMNIST":  # Black and white images.
                transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
                transforms += [getattr(dt, tc)()]
                transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
                transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
            else:  # Color images.
                transforms = [getattr(dt, tc)()]
                transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='RGB'))]
            _, val_dl, _ = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                              single_corr_bs, False, n_workers, pin_mem)
            val_dls.append(val_dl)

        criterion = nn.MSELoss()
        val_loss = 0.0
        collected_imgs = False
        for block in blocks:
            block.eval()
        with torch.no_grad():
            for val_data_tuples in zip(*val_dls):
                corr_tuple = val_data_tuples[0]
                id_tuple = val_data_tuples[1]
                x_val, y_val = corr_tuple[0].to(dev), corr_tuple[1].to(dev)
                x_val_id, y_val_id = id_tuple[0].to(dev), id_tuple[1].to(dev)
                features = x_val

                # Visualise images before autoencoding
                if not collected_imgs:
                    pre_ae_imgs = features.detach().cpu().numpy()
                    pre_ae_lbls = y_val.detach().cpu().numpy()
                    pre_ae_imgs_id = x_val_id.detach().cpu().numpy()
                    pre_ae_lbls_id = y_val_id.detach().cpu().numpy()

                # # Apply autoencoder
                for block in blocks:
                    features = block(features)

                # Visualise images after autoencoding
                if not collected_imgs:
                    post_ae_imgs = features.detach().cpu().numpy()
                    post_ae_lbls = y_val.detach().cpu().numpy()
                    collected_imgs = True

                loss = criterion(features, x_val_id)
                val_loss += loss.item()

        corr_loss = val_loss / len(val_dls[0])
        all_val_losses.append(corr_loss)
        print("{} autoencoder. Val MSE loss: {:.4f}".format(corr, corr_loss))

        # Visualise the autoencoder input and output
        fig_name = "val_before_ae_{}.png".format(corr)
        fig_path = os.path.join(vis_path, fig_name)
        pre_ae_imgs = dt.denormalize_255(pre_ae_imgs, np.array(denorm_mean).astype(np.float32),
                                         np.array(denorm_std).astype(np.float32)).astype(np.uint8)
        visualise_data(pre_ae_imgs[:25], pre_ae_lbls[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)

        fig_name = "val_Identity_imgs.png"
        fig_path = os.path.join(vis_path, fig_name)
        pre_ae_imgs_id = dt.denormalize_255(pre_ae_imgs_id, np.array(denorm_mean).astype(np.float32),
                                            np.array(denorm_std).astype(np.float32)).astype(np.uint8)
        visualise_data(pre_ae_imgs_id[:25], pre_ae_lbls_id[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5,
                       n_cols=5)

        fig_name = "val_after_ae_{}.png".format(corr)
        fig_path = os.path.join(vis_path, fig_name)
        post_ae_imgs = dt.denormalize_255(post_ae_imgs, np.array(denorm_mean).astype(np.float32),
                                          np.array(denorm_std).astype(np.float32)).astype(np.uint8)
        visualise_data(post_ae_imgs[:25], post_ae_lbls[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5,
                       n_cols=5)

    avg_val_loss = np.mean(all_val_losses)
    print("{}. Avg val MSE loss: {:.4f}".format(experiment, avg_val_loss))
    sys.stdout.flush()



def val_monolithic():
    pass


def val_modules():
    pass


def val_autoencoders():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/',
                        help="path to directory to save checkpoints")
    parser.add_argument('--vis-path', type=str, default='/om2/user/imason/compositions/figs/',
                        help="path to directory to save autoencoder visualisations")
    parser.add_argument('--experiment', type=str, default='CrossEntropy',
                        help="which method to use. CrossEntropy or Contrastive or Modules.")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=256, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight', type=float, default=1.0, help="weight for the contrastive loss")
    parser.add_argument('--n-workers', type=int, default=2, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    args = parser.parse_args()

    # Set seeding
    seed = 48121620
    reset_rngs(seed=seed, deterministic=True)

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')


    # Set up and create unmade directories
    variance_dir_name = f"lr-{args.lr}_weight-{args.weight}"
    print(f"Testing with hyperparameters: {variance_dir_name}")
    sys.stdout.flush()
    args.data_root = os.path.join(args.data_root, args.dataset)
    args.ckpt_path = os.path.join(args.ckpt_path, args.dataset, variance_dir_name)
    args.vis_path = os.path.join(args.vis_path, args.dataset, "autoencoder_visualisations", variance_dir_name)
    if "ImgSpace" in args.experiment:
        mkdir_p(args.vis_path)

    """
    CUDA_VISIBLE_DEVICES=4 python validate.py --dataset EMNIST --experiment Modules --total-n-classes 47 --lr 1.0 --weight 1.0 --pin-mem 
    CUDA_VISIBLE_DEVICES=4 python validate.py --dataset EMNIST --experiment ImgSpace --total-n-classes 47 --lr 1.0 --weight 1.0 --pin-mem
    """

    if "Modules" in args.experiment:
        val_module_backbone("Modules", args.dataset, args.data_root, args.ckpt_path, args.total_n_classes,
                            args.batch_size, args.n_workers, args.pin_mem, dev)
    elif "ImgSpace" in args.experiment:
        val_autoencoders_mse("ImgSpace", args.dataset, args.data_root, args.ckpt_path, args.vis_path,
                             args.total_n_classes, args.batch_size, args.n_workers, args.pin_mem, dev)
