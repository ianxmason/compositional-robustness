"""
For hyperparameter searches for step one of methods that train in two steps.
That is, the mean squared error of the autoencoders on the elemental corruptions they are trained on and
the accuracy of the backbone network for the modular approach on the Identity data.
"""

# Todo: write and test this whole test_elemental_mse function. Visualise batch. Matches train MSE?
# Todo: write a function to test the modular backbone network on identity data. Visualise batch. Matches train accuracy? etc.
def test_elemental_mse(experiment, validate, dataset, data_root, ckpt_path, save_path, vis_path, total_n_classes,
                       batch_size, n_workers, pin_mem, dev, check_if_run, total_processes, process):
    """
    Get the MSE for autoencoders on the corruptions they were trained on. For setting the best hyperparameters.
    """
    files = os.listdir(ckpt_path)
    for f in files:
        if "es_" == f[:3]:
            raise ValueError("Early stopping ckpt found {}. Training hasn't finished yet".format(f))
    files = [f for f in files if f.split('_')[0] == experiment]

    ae_ckpts = [f for f in files if len(f.split('-')) == 2]
    ae_corrs = list(set([f.split('_')[-1][:-3] for f in ae_ckpts]))
    ae_corrs.sort()
    assert len(ae_corrs) == 6  # hardcoded for EMNIST5. Encoder and Decoder.

    # Load the ckpt
    if check_if_run and os.path.exists(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(
                                                    experiment + "IdentityClassifier", process, total_processes))):
        raise RuntimeError("Pickle file already exists at {}. \n Skipping testing.".format(
            os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment, process, total_processes))))
    else:
        # Load the autoencoders
        ae_blocks = []
        ae_block_ckpt_names = []
        for corr in ae_corrs:
            if dataset == "EMNIST":
                blocks, block_ckpt_names = create_emnist_autoencoder(experiment, corr.split('-'), dev)
            elif dataset == "CIFAR":
                blocks, block_ckpt_names = create_cifar_autoencoder(experiment, corr.split('-'), dev)
            elif dataset == "FACESCRUB":
                blocks, block_ckpt_names = create_facescrub_autoencoder(experiment, corr.split('-'), dev)
            ae_blocks.append(blocks)
            ae_block_ckpt_names.append(block_ckpt_names)
        for blocks, block_ckpt_names in zip(ae_blocks, ae_block_ckpt_names):
            for block, block_ckpt_name in zip(blocks, block_ckpt_names):
                block.load_state_dict(torch.load(os.path.join(ckpt_path, block_ckpt_name)))
                print("Loaded {}".format(block_ckpt_name))
                print("From {}".format(os.path.join(ckpt_path, block_ckpt_name)))
                sys.stdout.flush()


    # Test the model on the elemental corruptions.
    corruption_mse_losses = {}

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        corruptions = pickle.load(f)
    corruptions.sort()
    assert len(corruptions) == 167  # hardcoded for EMNIST. 149 EMNIST4. 167 EMNIST5.
    assert total_processes <= len(corruptions)
    assert process < total_processes

    corruptions_per_process = len(corruptions) // total_processes
    if process == total_processes - 1:
        corruptions = corruptions[corruptions_per_process * process:]
    else:
        corruptions = corruptions[corruptions_per_process * process:corruptions_per_process * (process + 1)]

    for test_corruption in corruptions:
        print("Testing on {}".format(test_corruption))
        sys.stdout.flush()
        trained_classes = list(range(total_n_classes))

        identity_path = os.path.join(data_root, "Identity")
        if dataset == "EMNIST":
            transforms = [torchvision.transforms.Lambda(lambda im: im.convert('L'))]
            transforms += [getattr(dt, c)() for c in test_corruption]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='L'))]
            transforms += [torchvision.transforms.Lambda(lambda im: im.convert('RGB'))]
        else:
            transforms = [getattr(dt, c)() for c in test_corruption]
            transforms += [torchvision.transforms.Lambda(lambda im: Image.fromarray(np.uint8(im), mode='RGB'))]
        if validate:
            _, tst_dl, _ = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                              batch_size, False, n_workers, pin_mem)
        else:
            _, _, tst_dl = get_transformed_static_dataloaders(dataset, identity_path, transforms, trained_classes,
                                                              batch_size, False, n_workers, pin_mem)

        test_ae_blocks = []
        for c in test_corruption:
            if c == "Identity":
                continue
            else:
                for blocks, block_ckpt_names in zip(ae_blocks, ae_block_ckpt_names):
                    if c in block_ckpt_names[0]:
                        print("Selected autoencoder {}".format(block_ckpt_names))
                        test_ae_blocks.append(blocks)

        tst_loss, tst_acc, pre_ae_imgs, pre_ae_lbls, post_ae_imgs, post_ae_lbls = \
            autoencoders_loss_and_accuracy(test_ae_blocks, id_clsf_blocks, tst_dl, dev)
        id_clsf_corruption_accs['-'.join(test_corruption)] = tst_acc
        id_clsf_corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{} Identity Classifier, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment,
                                                                                       '-'.join(test_corruption),
                                                                                       tst_loss, tst_acc))
        sys.stdout.flush()

        tst_loss, tst_acc, _, _, _, _ = autoencoders_loss_and_accuracy(test_ae_blocks, all_clsf_blocks, tst_dl,
                                                                                dev)
        all_clsf_corruption_accs['-'.join(test_corruption)] = tst_acc
        all_clsf_corruption_losses['-'.join(test_corruption)] = tst_loss
        print("{} Joint Classifier, {}. test loss: {:.4f}, test acc: {:.4f}".format(experiment,
                                                                                    '-'.join(test_corruption),
                                                                                    tst_loss, tst_acc))
        sys.stdout.flush()

        # Visualise the autoencoder input and output
        fig_name = "before_ae_{}.png".format('-'.join(test_corruption))
        fig_path = os.path.join(vis_path, fig_name)
        pre_ae_imgs = dt.denormalize_255(pre_ae_imgs, np.array(denorm_mean).astype(np.float32),
                                         np.array(denorm_std).astype(np.float32)).astype(np.uint8)
        visualise_data(pre_ae_imgs[:25], pre_ae_lbls[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5, n_cols=5)

        fig_name = "after_ae_{}.png".format('-'.join(test_corruption))
        fig_path = os.path.join(vis_path, fig_name)
        post_ae_imgs = dt.denormalize_255(post_ae_imgs, np.array(denorm_mean).astype(np.float32),
                                          np.array(denorm_std).astype(np.float32)).astype(np.uint8)
        visualise_data(post_ae_imgs[:25], post_ae_lbls[:25], save_path=fig_path, title=fig_name[:-4], n_rows=5,
                       n_cols=5)

    # Save the results
    with open(os.path.join(save_path, "{}_all_accs_process_{}_of_{}.pkl".format(experiment + "IdentityClassifier",
                                                                                process, total_processes)), "wb") as f:
        pickle.dump(id_clsf_corruption_accs, f)
    with open(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment + "IdentityClassifier",
                                                                                  process, total_processes)), "wb") as f:
        pickle.dump(id_clsf_corruption_losses, f)
    with open(os.path.join(save_path, "{}_all_accs_process_{}_of_{}.pkl".format(experiment + "JointClassifier",
                                                                                process, total_processes)), "wb") as f:
        pickle.dump(all_clsf_corruption_accs, f)
    with open(os.path.join(save_path, "{}_all_losses_process_{}_of_{}.pkl".format(experiment + "JointClassifier",
                                                                                  process, total_processes)), "wb") as f:
        pickle.dump(all_clsf_corruption_losses, f)
