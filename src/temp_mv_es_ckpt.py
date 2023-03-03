"""
This is for speeding up testing with a hack

Basically takes an early stopping checkpoint (that has trained for long enough, but not fully completed), loads
it and saves just the state dict for testing.
"""
import torch
import os
from lib.early_stopping import EarlyStopping
from data.data_loaders import get_multi_static_dataloaders, get_static_dataloaders, get_transformed_static_dataloaders
from data.emnist import EMNIST_MEAN, EMNIST_STD
from data.cifar import CIFAR10_MEAN, CIFAR10_STD
from data.facescrub import FACESCRUB_MEAN, FACESCRUB_STD
from lib.networks import create_facescrub_modules

# Path to checkpoint
ckpt_path = "/om2/user/imason/compositions/ckpts/FACESCRUB"
es_ckpt_path = os.path.join(ckpt_path, "es_ckpt_AutoModules_InceptionModule5_Swirl-Identity.pt")

# Create the network to load es ckpt
dev = torch.device('cuda')
modules, module_ckpt_names = create_facescrub_modules("AutoModules", ["Swirl", "Identity"], dev)
module = modules[5]
module_ckpt_name = module_ckpt_names[5]


# Save model
print("Loading early stopped checkpoints")
ckpt = torch.load(es_ckpt_path, map_location=dev)
last_epoch = ckpt['epoch']
last_loss = ckpt['loss']
module.load_state_dict(ckpt['model_state_dict'])
module.eval()

print("Saving state dict")
torch.save(module.state_dict(), os.path.join(ckpt_path, module_ckpt_name))
print("Saved best module to {}".format(os.path.join(ckpt_path, module_ckpt_name)))

"""Manually delete the ES ckpt on the server once you have saved the model state dict"""""
# for es in early_stoppings:
#     es.delete_checkpoint()  # Removes from disk