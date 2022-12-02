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


def main(elemental_corruptions, experiments, results_path, save_path):

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
            losses_df = pd.DataFrame(data=all_losses, index=[experiment])
            accs_df = pd.DataFrame(data=all_accs, index=[experiment])
        else:
            losses_df = losses_df.append(pd.DataFrame(data=all_losses, index=[experiment]))
            accs_df = accs_df.append(pd.DataFrame(data=all_accs, index=[experiment]))

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
    fig, axs = plt.subplots(1, 2, figsize=(60, 40))

    title = "Accuracy Heatmap"
    sns.heatmap(accs_df.transpose(), annot=True, ax=axs[0])
    axs[0].set_title(title)
    axs[0].set_xlabel("Training Corruption(s) - Single Shifts")
    axs[0].set_ylabel("Test Corruption(s)- Compositions")

    title = "Loss Heatmap"
    sns.heatmap(losses_df.transpose(), annot=True, ax=axs[1])
    axs[1].set_title(title)
    axs[1].set_xlabel("Training Corruption(s) - Single Shifts")
    axs[1].set_ylabel("Test Corruption(s) - Compositions")

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
        for exp in experiments:
            comp_counts[i].append(
                len(plot_df.loc[(plot_df['Num Elementals'] == i) & (plot_df['Experiment'] == exp)]))

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    sns.boxplot(x="Num Elementals", y="Composition Accuracy", hue="Experiment", data=plot_df, ax=axs)
    axs.set_title("Composition accuracy with seen elementals")
    axs.set_ylim(0, 100)
    axs.legend(loc='center right')
    plt.xticks(rotation=90)
    # Add a count for how many points are in each box
    for i, counts in comp_counts.items():
        # 2 experiments
        # axs.text(i - 1 - 0.2, 95, counts[0], c='b', horizontalalignment='center')
        # axs.text(i - 1 + 0.2, 95, counts[1], c='orange', horizontalalignment='center')

        # 3 experiments
        axs.text(i - 1 - 0.3, 95, counts[0], c='b', horizontalalignment='center')
        axs.text(i - 1 + 0.0, 95, counts[1], c='orange', horizontalalignment='center')
        axs.text(i - 1 + 0.3, 95, counts[2], c='g', horizontalalignment='center')

    plt.savefig(os.path.join(save_path, "comparison-boxplot.pdf".format(experiment)), bbox_inches="tight")
    print("Saved boxplot to {}".format(os.path.join(save_path, "comparison-boxplot.pdf".format(experiment))))


    # Violinplot
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    sns.violinplot(x="Num Elementals", y="Composition Accuracy", hue="Experiment", data=plot_df, ax=axs)
    axs.set_title("Composition accuracy with seen elementals")
    axs.set_ylim(0, 100)
    axs.legend(loc='center right')
    plt.xticks(rotation=90)
    # Add a count for how many points are in each box
    for i, counts in comp_counts.items():
        # 2 experiments
        # axs.text(i - 1 - 0.2, 95, counts[0], c='b', horizontalalignment='center')
        # axs.text(i - 1 + 0.2, 95, counts[1], c='orange', horizontalalignment='center')

        # 3 experiments
        axs.text(i - 1 - 0.3, 95, counts[0], c='b', horizontalalignment='center')
        axs.text(i - 1 + 0.0, 95, counts[1], c='orange', horizontalalignment='center')
        axs.text(i - 1 + 0.3, 95, counts[2], c='g', horizontalalignment='center')

    plt.savefig(os.path.join(save_path, "comparison-violins.pdf".format(experiment)), bbox_inches="tight")
    print("Saved violinplot to {}".format(os.path.join(save_path, "comparison-violins.pdf".format(experiment))))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST5/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--results-path', type=str, default='/om2/user/imason/compositions/results/EMNIST5/',
                        help="path to directory containing results of testing")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/analysis/EMNIST5/',
                        help="path to directory to save analysis plots and pickle files")
    args = parser.parse_args()

    # experiments = ["CrossEntropy", "Contrastive", "Modules"]
    # experiments = ["CrossEntropyV2",
    #                "ContrastiveV2", "AutoContrastiveV2", "ModLevelContrastiveV2",
    #                "ModulesV2", "AutoModulesV2"]

    # experiments = ["ModLevelContrastiveV3",
    #                "ContrastiveL3W01", "ContrastiveL3W1", "ContrastiveL3W10",
    #                "ContrastiveL4W01", "ContrastiveL4W1", "ContrastiveL4W10",
    #                "ContrastiveL5W01", "ContrastiveL5W1", "ContrastiveL5W10"]

    experiments = ["ModulesV3", "ModulesV3NoPassThrough", "ModulesV3NoInvariance",
                   "AutoModulesV3", "AutoModulesV3NoPassThrough", "AutoModulesV3NoInvariance"]

    # Set seeding
    reset_rngs(seed=1357911, deterministic=True)

    # Create unmade directories
    mkdir_p(args.save_path)

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)

    elemental_corruptions = []
    # Training on only the corruptions in the composition. Always include identity, remove permutations
    for corr in all_corruptions:
        if len(corr) == 1:
            if corr not in elemental_corruptions:
                elemental_corruptions.append(corr[0])

    """
    Run from inside analysis directory as: python comparison_plots.py
    If use shell script put cd analysis in shell script
    """

    main(elemental_corruptions, experiments, args.results_path, args.save_path)
