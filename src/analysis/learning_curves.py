"""
Plots the accuracy of each module during the first few training iterations to pick which is the best level
of abstraction.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

elem_corrs = ['Contrast', 'GaussianBlur', 'ImpulseNoise', 'Invert', 'Rotate90', 'Swirl']

# Load the data
dataset = "FACESCRUB"
logging_path = "/om2/user/imason/compositions/logs/{}/".format(dataset)
save_path = "/om2/user/imason/compositions/analysis/{}/".format(dataset)

for corr in elem_corrs:
    with open(os.path.join(logging_path, "module_train_accs_{}.pkl".format(corr)), "rb") as f:
        train_accs = pickle.load(f)

    mod_levels = []
    for k in train_accs.keys():
        mod_levels.append(k.split('_')[0].split('-')[1])
    mod_levels = list(set(mod_levels))
    mod_levels.sort()

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.set_title("Training accuracy for {} at different levels of abstraction".format(corr))
    axs.set_ylim(0, 100)
    axs.set_xlabel("Training Iteration")
    axs.set_ylabel("Accuracy (%)")

    if len(mod_levels) < 10:
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(mod_levels)))
    else:
        colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(mod_levels)))
    for i, level in enumerate(mod_levels):
        level_values = []
        for k, v in train_accs.items():
            if "Level-{}_".format(level) in k:
                level_values.append(v)
        level_values = np.array(level_values)
        level_mean = np.mean(level_values, axis=0)
        level_std = np.std(level_values, axis=0)
        axs.plot(range(1, len(level_mean) + 1), level_mean, label="Level {}".format(level), color=colors[i])
        axs.fill_between(range(1, len(level_mean) + 1), level_mean - level_std, level_mean + level_std, alpha=0.2,
                         color=colors[i])

    axs.set_xlim(1, len(level_mean))
    axs.legend(loc='upper left')
    plt.savefig(os.path.join(save_path, "module-{}-learning-curves.pdf".format(corr)), bbox_inches="tight")
    print("Saved learning curves to {}".format(os.path.join(save_path, "module-{}-learning-curves.pdf".format(corr))))