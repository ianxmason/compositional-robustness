# Plot histogram, heatmap, box plots of trained on all vs trained on one
# Make analysis directory
# Perhaps include analysis plotting functions for histograms, heatmaps etc that are generic
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


def main(elemental_corruptions, results_path, save_path):

    elemental_corruptions = [getattr(dt, c)() for c in elemental_corruptions]
    all_losses = {}
    all_accs = {}
    specific_losses = {}
    specific_accs = {}
    for results_file in sorted(os.listdir(os.path.join(results_path)), key=lambda x: (x.count('-'), x.lower())):
        if "process" in results_file:
            with open(os.path.join(results_path, results_file), "rb") as f:
                results = pickle.load(f)
                if "_losses" in results_file:
                    name = results_file.split("_")[0]
                    for k, v in results.items():
                        for corruption in elemental_corruptions:
                            k = k.replace(corruption.name, corruption.abbreviation)
                        if k in all_losses:
                            raise RuntimeError("Duplicate key {} in {}".format(k, results_file))
                        else:
                            all_losses[k] = v
                elif "_accs" in results_file:
                    name = results_file.split("_")[0]
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
            with open(os.path.join(results_path, results_file), "rb") as f:
                results = pickle.load(f)
                if "_losses" in results_file:
                    name = results_file.split("_")[0]
                    for k, v in results.items():
                        for corruption in elemental_corruptions:
                            k = k.replace(corruption.name, corruption.abbreviation)
                        if k in specific_losses:
                            raise RuntimeError("Duplicate key {} in {}".format(k, results_file))
                        else:
                            specific_losses[k] = v
                elif "_accs" in results_file:
                    name = results_file.split("_")[0]
                    for k, v in results.items():
                        for corruption in elemental_corruptions:
                            k = k.replace(corruption.name, corruption.abbreviation)
                        if k in specific_accs:
                            raise RuntimeError("Duplicate key {} in {}".format(k, results_file))
                        else:
                            specific_accs[k] = v
                else:
                    raise RuntimeError("Invalid file {} in {}".format(results_file, results_path))

    assert len(specific_accs) == 149  # hardcoded for EMNIST4
    assert len(all_accs) == len(specific_accs) == len(all_losses) == len(specific_losses)

    all_losses_df = pd.DataFrame(data=all_losses, index=["all"])
    all_accs_df = pd.DataFrame(data=all_accs, index=["all"])
    specific_losses_df = pd.DataFrame(data=specific_losses, index=["specific"])
    specific_accs_df = pd.DataFrame(data=specific_accs, index=["specific"])
    losses_df = pd.concat([specific_losses_df, all_losses_df])
    accs_df = pd.concat([specific_accs_df, all_accs_df])

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

    plt.savefig(os.path.join(save_path, "one-vs-all-heatmap.pdf"), bbox_inches="tight")
    print("Saved heatmap to {}".format(os.path.join(save_path, "one-vs-all-heatmap.pdf")))


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
    for i in range(1, 8):
        comp_counts[i] = []
        for exp in ["specific", "all"]:
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
        axs.text(i - 1 - 0.2, 95, counts[0], c='b', horizontalalignment='center')
        axs.text(i - 1 + 0.2, 95, counts[1], c='orange', horizontalalignment='center')

        # 3 experiments
        # axs.text(i - 1 - 0.3, 95, counts[0], c='b', horizontalalignment='center')
        # axs.text(i - 1 + 0.0, 95, counts[1], c='orange', horizontalalignment='center')
        # axs.text(i - 1 + 0.3, 95, counts[2], c='g', horizontalalignment='center')

    plt.savefig(os.path.join(save_path, "one-vs-all-boxplot.pdf"), bbox_inches="tight")
    print("Saved boxplot to {}".format(os.path.join(save_path, "one-vs-all-boxplot.pdf")))


    # Difference-histogram
    hist_corrs, hist_comps = [], []
    for comp in accs_df.columns:
        specific_acc = accs_df.loc["specific"][comp]
        all_acc = accs_df.loc["all"][comp]

        if len(comp.split('-')) == 1:
            hist_corrs.append(all_acc - specific_acc)
        else:
            hist_comps.append(all_acc - specific_acc)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    hist_color = 'orange'
    axs[0].hist(hist_corrs, bins=20, range=(-20, 20), color=hist_color)
    axs[0].axvline(0, c='r', ls='--')
    axs[0].set_ylim(0, 20)
    axs[0].set_title("all - specific. Elementaries.")
    axs[0].set_xlabel("Change in accuracy")
    axs[0].set_ylabel("Count")
    axs[1].hist(hist_comps, bins=20, range=(-70, 70), color=hist_color)
    axs[1].axvline(0, c='r', ls='--')
    axs[1].set_ylim(0, 20)
    axs[1].set_title("all - specific. Compositions.")
    axs[1].set_xlabel("Change in accuracy")
    axs[1].set_ylabel("Count")

    plt.savefig(os.path.join(save_path, "one-vs-all-histogram.pdf"), bbox_inches="tight")
    print("Saved histogram to {}".format(os.path.join(save_path, "one-vs-all-histogram.pdf")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST4/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--results-path', type=str, default='/om2/user/imason/compositions/results/EMNIST4/',
                        help="path to directory containing results of testing")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/analysis/EMNIST4/',
                        help="path to directory to save analysis plots and pickle files")
    args = parser.parse_args()

    # Set seeding
    reset_rngs(seed=13579, deterministic=True)

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
    Run from inside analysis directory as: python one_vs_all.py
    If use shell script put cd analysis in shell script
    """

    main(elemental_corruptions, args.results_path, args.save_path)
