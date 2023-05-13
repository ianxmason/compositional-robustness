# Plot histogram, heatmap, box plots comparing CrossEntropy, Contrastive, Modules etc.
import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
sys.path.append("../")
from lib.utils import *
import data.data_transforms as dt
from analysis.plotting import *


def main(elemental_corruptions, experiments, legend_names, dataset, results_path, save_path):

    if dataset == "EMNIST":
        chance = 100./47.
        ceiling = 89.
    elif dataset == "CIFAR":
        chance = 100./10.
        ceiling = 92.
    elif dataset == "FACESCRUB":
        chance = 100./388.
        ceiling = 96.
    else:
        raise ValueError("Dataset not implemented")

    elemental_corruptions = [getattr(dt, c)() for c in elemental_corruptions]
    all_results_files = os.listdir(os.path.join(results_path))
    for i, experiment in enumerate(experiments):
        all_losses = {}
        all_accs = {}
        results_files = [f for f in all_results_files if f.split("_")[0] == experiment]

        for results_file in sorted(results_files, key=lambda x: (x.count('-'), x.lower())):
            if "process" in results_file:
                with open(os.path.join(results_path, results_file), "rb") as f:
                    results = pickle.load(f)
                    if "_losses" in results_file:
                        name = results_file.split("_")[1]
                        for k, v in results.items():
                            for corruption in elemental_corruptions:
                                k = k.replace(corruption.name, corruption.abbreviation)
                            if k in all_losses:
                                raise RuntimeError("Duplicate key {} in {}".format(k, results_file))
                            else:
                                all_losses[k] = v
                    elif "_accs" in results_file:
                        name = results_file.split("_")[1]
                        for k, v in results.items():
                            for corruption in elemental_corruptions:
                                k = k.replace(corruption.name, corruption.abbreviation)
                            if k in all_accs:
                                raise RuntimeError("Duplicate key {} in {}".format(k, results_file))
                            else:
                                all_accs[k] = v
                    else:
                        raise RuntimeError("Invalid file {} in {}".format(results_file, results_path))
            if "process" not in results_file:
                continue

        assert len(all_accs) == 167  # hardcoded for EMNIST. EMNIST4 149. EMNIST5 167.

        if i == 0:
            losses_df = pd.DataFrame(data=all_losses, index=[legend_names[i]])
            accs_df = pd.DataFrame(data=all_accs, index=[legend_names[i]])
        else:
            losses_df = losses_df.append(pd.DataFrame(data=all_losses, index=[legend_names[i]]))
            accs_df = accs_df.append(pd.DataFrame(data=all_accs, index=[legend_names[i]]))

    # print(accs_df)
    # print(losses_df)
    # print(len(accs_df))

    # Column reordeing shows an easier to read heatmap
    ordered_columns = list(accs_df.columns)
    ordered_columns.sort(key=lambda x: (x.count('-'), x.lower()))
    reordered_cols = []
    for comp_1 in ordered_columns:
        if comp_1 in reordered_cols:
            continue
        reordered_cols.append(comp_1)
        for comp_2 in ordered_columns:
            if comp_2 in reordered_cols:
                continue
            if sorted(comp_1) == sorted(comp_2):
                reordered_cols.append(comp_2)
    accs_df = accs_df[reordered_cols]
    losses_df = losses_df[reordered_cols]

    print(accs_df)
    print(losses_df)
    print(len(accs_df))

    # Heatmap
    sns.set_theme()
    sns.set_context("notebook")
    fig, axs = plt.subplots(1, 2, figsize=(60, 40))

    title = "Accuracy Heatmap"
    sns.heatmap(accs_df.transpose(), annot=True, ax=axs[0])
    axs[0].set_title(title)
    axs[0].set_xlabel("Training Approach")
    axs[0].set_ylabel("Test Corruption(s)")

    title = "Loss Heatmap"
    sns.heatmap(losses_df.transpose(), annot=True, ax=axs[1])
    axs[1].set_title(title)
    axs[1].set_xlabel("Training Approach")
    axs[1].set_ylabel("Test Corruption(s)")

    plt.savefig(os.path.join(save_path, "comparison-heatmap.pdf"), bbox_inches="tight")
    print("Saved heatmap to {}".format(os.path.join(save_path, "comparison-heatmap.pdf")))


    # Boxplot
    col_names = ["Num Elementals", "Composition Accuracy", "Experiment"]
    row_names = []
    df_dict = {k: [] for k in col_names}

    for index in accs_df.index:
        # comp_cols = [sorted(x.split('-')) for x in accs_df.columns]
        for comp in accs_df.columns:
            df_dict["Num Elementals"].append(len(comp.split('-')))
            df_dict["Composition Accuracy"].append(accs_df.loc[index][comp])
            df_dict["Experiment"].append(index)
            row_names.append(comp)

    plot_df = pd.DataFrame(data=df_dict, index=row_names)

    # Get counts of how many points are in each box
    comp_counts = {}
    for i in range(1, 7):
        comp_counts[i] = []
        for exp in legend_names:
            comp_counts[i].append(
                len(plot_df.loc[(plot_df['Num Elementals'] == i) & (plot_df['Experiment'] == exp)]))

    sns.set_theme()
    sns.set_context("poster")
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    sns.boxplot(x="Num Elementals", y="Composition Accuracy", hue="Experiment", data=plot_df, ax=axs)
    axs.axhline(y=chance, color='k', linestyle='--')
    axs.text(0.1, chance + 0.1, "Chance", verticalalignment='bottom', horizontalalignment='center', size='small', color='k',
             weight='semibold')
    axs.axhline(y=ceiling, color='k', linestyle='--')
    axs.text(4.9, ceiling - 0.7, "Ceiling", verticalalignment='top', horizontalalignment='center', size='small', color='k',
             weight='semibold')
    # axs.set_title("Composition accuracy with seen elementals")
    axs.set_ylim(0, 100)
    axs.set_xlabel("Corruptions in Composition")
    axs.set_ylabel("Accuracy (%)")
    axs.legend(loc='center right')
    # plt.xticks(rotation=90)
    # Add a count for how many points are in each box
    # for i, counts in comp_counts.items():
    #     # 3 experiments
    #     axs.text(i - 1 - 0.3, 95, counts[0], c='b', horizontalalignment='center')
    #     axs.text(i - 1 + 0.0, 95, counts[1], c='orange', horizontalalignment='center')
    #     axs.text(i - 1 + 0.3, 95, counts[2], c='g', horizontalalignment='center')

    plt.savefig(os.path.join(save_path, "comparison-boxplot.pdf"), bbox_inches="tight")
    print("Saved boxplot to {}".format(os.path.join(save_path, "comparison-boxplot.pdf")))


    # Violinplot
    # sns.set_theme()
    # sns.set_context("poster")
    # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    # sns.violinplot(x="Num Elementals", y="Composition Accuracy", hue="Experiment", data=plot_df, ax=axs)
    # axs.axhline(y=chance, color='k', linestyle='--')
    # axs.axhline(y=ceiling, color='k', linestyle='--')
    # axs.set_title("Composition accuracy with seen elementals")
    # axs.set_ylim(0, 100)
    # axs.legend(loc='center right')
    # plt.xticks(rotation=90)
    # # Add a count for how many points are in each box
    # # for i, counts in comp_counts.items():
    # #     # 3 experiments
    # #     axs.text(i - 1 - 0.3, 95, counts[0], c='b', horizontalalignment='center')
    # #     axs.text(i - 1 + 0.0, 95, counts[1], c='orange', horizontalalignment='center')
    # #     axs.text(i - 1 + 0.3, 95, counts[2], c='g', horizontalalignment='center')
    #
    # plt.savefig(os.path.join(save_path, "comparison-violins.pdf"), bbox_inches="tight")
    # print("Saved violinplot to {}".format(os.path.join(save_path, "comparison-violins.pdf")))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--results-path', type=str, default='/om2/user/imason/compositions/results/',
                        help="path to directory containing results of testing")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/analysis/',
                        help="path to directory to save analysis plots and pickle files")
    parser.add_argument('--seed', type=int, default=38164641, help="random seed")
    args = parser.parse_args()

    # Set seeding # Final: 13579111 24681012 36912151. Hparams: 48121620
    reset_rngs(seed=args.seed, deterministic=True)

    variance_dir_name = f"seed-{args.seed}"  # f"lr-0.01_weight-1.0"
    args.data_root = os.path.join(args.data_root, args.dataset)
    args.results_path = os.path.join(args.results_path, args.dataset, variance_dir_name)
    args.save_path = os.path.join(args.save_path, args.dataset, variance_dir_name)
    mkdir_p(args.save_path)

    # Most common experiments
    experiments = ["CrossEntropy",
                   "Contrastive",
                   "AutoModules"]  # "Modules",
                   # "ImgSpaceIdentityClassifier", "ImgSpaceJointClassifier"]

    legend_names = ["ERM", "Contrastive", "Modular"] #  "AE-Modular-ID", "AE-Modular-Joint"]

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)

    elemental_corruptions = []
    # Training on only the corruptions in the composition. Always include identity, remove permutations
    for corr in all_corruptions:
        if len(corr) == 1:
            if corr not in elemental_corruptions:
                elemental_corruptions.append(corr[0])

    # Run from inside analysis directory as: python comparison_plots.py
    main(elemental_corruptions, experiments, legend_names, args.dataset, args.results_path, args.save_path)
