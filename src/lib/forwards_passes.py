import torch


def cross_entropy_forwards_pass(network_blocks, x, y, cross_entropy_loss_fn, accuracy_fn):
    for i, block in enumerate(network_blocks):
        if i == 0:
            features = block(x)
        elif i == len(network_blocks) - 1:
            output = block(features)
        else:
            features = block(features)

    return cross_entropy_loss_fn(output, y), accuracy_fn(output, y)


def autoencoder_forwards_pass(network_blocks, x, mse_loss_fn, single_corr_bs):
    if len(x.shape) != 4:
        raise ValueError("Image must be 4d. Got {}".format(x.shape))
    id_imgs = x[single_corr_bs:, :, :, :]
    corr_imgs = x[:single_corr_bs, :, :, :]

    for i, block in enumerate(network_blocks):
        if i == 0:
            features = block(corr_imgs)
        elif i == len(network_blocks) - 1:
            output = block(features)
        else:
            features = block(features)

    return mse_loss_fn(output, id_imgs)


def contrastive_forwards_pass(network_blocks, x, y, cross_entropy_loss_fn, accuracy_fn, contrastive_loss_fn,
                              abstraction_levels, weight, single_corr_bs):
    """
    Network forwards pass with an option to apply the contrastive loss at any intermediate layer
    The contrastive loss is weighted by weight
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
            total_ctv_loss += contrastive_loss_fn(loss_features, weight, n_views=n_views)

        if i != len(network_blocks) - 1:
            features = block(features)
        else:
            output = block(features)

    return cross_entropy_loss_fn(output, y), total_ctv_loss, accuracy_fn(output, y)


def modules_forwards_pass(network_blocks, module, module_level, x, y, cross_entropy_loss_fn, accuracy_fn,
                          contrastive_loss_fn, weight, single_corr_bs, pass_through=True):
    """
    Network forwards pass with a module applied at a specified intermediate layer

    pass_through: if True, the module is applied to both identity and corruption features
                  if False, the module is applied only to the corruption features
    """
    if module_level < 0 or module_level >= len(network_blocks):
        raise ValueError("Module must be applied at an intermediate layer (or in image space)."
                         "Received level {}.".format(module_level))
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
            total_ctv_loss += contrastive_loss_fn(torch.cat((corr_features, id_features), dim=0), weight, n_views=2)

        if i != len(network_blocks) - 1:
            features = block(features)
        else:
            output = block(features)

    return cross_entropy_loss_fn(output, y), total_ctv_loss, accuracy_fn(output, y)
