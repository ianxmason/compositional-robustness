# Plot histogram, heatmap, box plots comparing CrossEntropy, Contrastive, Modules etc.
import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
sys.path.append("../")
from lib.utils import *
import data.data_transforms as dt
from analysis.plotting import *

PLOT_COLORS = sns.color_palette()

# Todo: Refactor the three annotate functions into one
def annotate_elem_comp(data, **kws):
    plot_elem_scores = data["Elemental Invariance Score"]
    plot_comp_scores = data["Composition Invariance Score"]
    plot_experiment = data["Experiment"].iloc[0]
    slope, intercept, r_value, p_value, std_err = linregress(plot_elem_scores, plot_comp_scores)
    ax = plt.gca()
    for l_idx, l_name in enumerate(legend_names):
        if l_idx == 0:
            ax.text(0.5, -0.1, "Elemental Invariance Score", c='k', fontsize='medium', horizontalalignment='center',
                    transform=ax.transAxes)
        if l_name == plot_experiment:
            ax.text(0.5, -0.15 - l_idx * 0.05, f"\u2022 {plot_experiment}. r = {r_value:.2f}. p = {p_value:.2f}.",
                    c=PLOT_COLORS[l_idx], fontsize='medium', horizontalalignment='center', transform=ax.transAxes)
            # x_range = np.arange(0, 1.05, 0.05)
            # ax.plot(x_range, slope * x_range + intercept, c=PLOT_COLORS[l_idx], linestyle="--")


def annotate_elem_acc(data, **kws):
    plot_elem_scores = data["Elemental Invariance Score"]
    plot_accs = data["Accuracy (%)"]
    plot_experiment = data["Experiment"].iloc[0]
    slope, intercept, r_value, p_value, std_err = linregress(plot_elem_scores, plot_accs)
    ax = plt.gca()
    for l_idx, l_name in enumerate(legend_names):
        if l_idx == 0:
            ax.text(0.5, -0.1, "Elemental Invariance Score", c='k', fontsize='medium', horizontalalignment='center',
                    transform=ax.transAxes)
        if l_name == plot_experiment:
            ax.text(0.5, -0.15 - l_idx * 0.05, f"\u2022 {plot_experiment}. r = {r_value:.2f}. p = {p_value:.2f}.",
                    c=PLOT_COLORS[l_idx], fontsize='medium', horizontalalignment='center', transform=ax.transAxes)
            # x_range = np.arange(0, 1.05, 0.05)
            # ax.plot(x_range, slope * x_range + intercept, c=PLOT_COLORS[l_idx], linestyle="--")


def annotate_comp_acc(data, **kws):
    plot_comp_scores = data["Composition Invariance Score"]
    plot_accs = data["Accuracy (%)"]
    plot_experiment = data["Experiment"].iloc[0]
    slope, intercept, r_value, p_value, std_err = linregress(plot_comp_scores, plot_accs)
    ax = plt.gca()
    for l_idx, l_name in enumerate(legend_names):
        if l_idx == 0:
            ax.text(0.5, -0.1, "Composition Invariance Score", c='k', fontsize='medium', horizontalalignment='center',
                    transform=ax.transAxes)
        if l_name == plot_experiment:
            ax.text(0.5, -0.15 - l_idx * 0.05, f"\u2022 {plot_experiment}. r = {r_value:.2f}. p = {p_value:.2f}.",
                    c=PLOT_COLORS[l_idx], fontsize='medium', horizontalalignment='center', transform=ax.transAxes)
            # x_range = np.arange(0, 1.05, 0.05)
            # ax.plot(x_range, slope * x_range + intercept, c=PLOT_COLORS[l_idx], linestyle="--")



def main(all_corruptions, experiments, legend_names, dataset, total_n_classes, results_path, activations_path,
         save_path, pairs_only):

    if dataset == "EMNIST" or dataset == "CIFAR":
        num_neurons = 512
    elif dataset == "FACESCRUB":
        num_neurons = 2048
    else:
        raise ValueError("Dataset not implemented")

    elemental_corruptions = []
    composition_corruptions = []
    # Training on only the corruptions in the composition. Always include identity, remove permutations
    for corr in all_corruptions:
        if len(corr) == 1:
            if corr not in elemental_corruptions:
                elemental_corruptions.append(corr[0])
        else:
            if corr not in composition_corruptions:
                composition_corruptions.append(corr)
    assert len(elemental_corruptions) == 7  # Includes Identity
    assert len(composition_corruptions) == 160

    elemental_corruptions = [getattr(dt, c)() for c in elemental_corruptions]
    all_results_files = os.listdir(os.path.join(results_path))
    for i, experiment in enumerate(experiments):
        all_accs = {}
        results_files = [f for f in all_results_files if f.split("_")[0] == experiment]

        for results_file in sorted(results_files, key=lambda x: (x.count('-'), x.lower())):
            if "process" in results_file:
                with open(os.path.join(results_path, results_file), "rb") as f:
                    results = pickle.load(f)
                    if "_losses" in results_file:
                        continue
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

        assert len(all_accs) == 167  # hardcoded for EMNIST.

        if i == 0:
            accs_df = pd.DataFrame(data=all_accs, index=[experiment])
        else:
            accs_df = accs_df.append(pd.DataFrame(data=all_accs, index=[experiment]))

    print(accs_df)
    print(len(accs_df))

    min_elemental_median = 100
    min_composition_median = 100
    all_activations_files = os.listdir(os.path.join(activations_path))
    scores_dict = {"Experiment": [], "Elemental Invariance Score": [], "Composition Invariance Score": [],
                   "Accuracy (%)": [], "Corruptions in Composition": []}
    for i, experiment in enumerate(experiments):
        df_cols = ["Neuron Idx", "Corruption"]
        df_cols += ["Class {}".format(i) for i in range(total_n_classes)]
        activations_df = pd.DataFrame(columns=df_cols)
        activations_files = [f for f in all_activations_files if f.split("_")[0] == experiment]

        # First process maxes to get the max activations for each neuron over all corruptions
        # If using this normalisation this asks that neurons fire in exactly the same way over corruptions
        # If instead we normalise each corruption with its own max activation this asks that neurons fire in the same
        # relative way over corruptions, (not necessarily with the same magnitude)
        max_activations = np.zeros(num_neurons)
        for activations_file in sorted(activations_files, key=lambda x: (x.count('-'), x.lower())):
            if "process" in activations_file:
                with open(os.path.join(activations_path, activations_file), "rb") as f:
                    activations = pickle.load(f)
                    for k, v in activations.items():
                        _, _, corr_max_activations, _ = v
                        # Shapes: _, _, (num_units), _
                        assert corr_max_activations.shape == (num_neurons,)
                        max_activations = np.maximum(max_activations, corr_max_activations)

        for activations_file in sorted(activations_files, key=lambda x: (x.count('-'), x.lower())):
            if "process" in activations_file:
                with open(os.path.join(activations_path, activations_file), "rb") as f:
                    activations = pickle.load(f)
                    for k, v in activations.items():
                        for corruption in elemental_corruptions:
                            k = k.replace(corruption.name, corruption.abbreviation)

                        class_avg_firings, class_std_firings, _, raw_activations = v
                        # Shapes: (num_classes, num_units), (num_classes, num_units), _, (num_dpoints, num_units)
                        assert class_avg_firings.shape == (total_n_classes, num_neurons)

                        activations_dict = {k: [] for k in df_cols}
                        for neuron_idx, class_firings in enumerate(class_avg_firings.T):
                            activations_dict["Neuron Idx"].append(neuron_idx)
                            activations_dict["Corruption"].append(k)
                            for class_idx, firings in enumerate(class_firings):
                                if max_activations[neuron_idx] == 0:
                                    activations_dict["Class {}".format(class_idx)].append(0)
                                else:
                                    activations_dict["Class {}".format(class_idx)].append(firings /
                                                                                          max_activations[neuron_idx])
                        activations_df = pd.concat([activations_df, pd.DataFrame(data=activations_dict)])
            if "process" not in activations_file:
                continue
        assert len(activations_df) == 167 * num_neurons
        print(activations_df)
        print(len(activations_df))

        median_elemental_invariance_scores = {}
        median_composition_invariance_scores = {}
        for composition_corruption in composition_corruptions:
            composition_corruption = [getattr(dt, c)().abbreviation for c in composition_corruption]
            print(composition_corruption)
            if pairs_only and len(composition_corruption) != 2:
                continue

            elemental_df = activations_df[activations_df["Corruption"].isin(composition_corruption +
                                                                            [getattr(dt, "Identity")().abbreviation])]
            composition_df = activations_df[activations_df["Corruption"] == "-".join(composition_corruption)]
            composition_df = pd.concat([elemental_df, composition_df])
            assert len(composition_df) == num_neurons * (len(composition_corruption) + 2)

            elemental_invariance_scores = []
            composition_invariance_scores = []
            for neuron_idx in range(num_neurons):
                if max_activations[neuron_idx] <= 1e-6:
                    print("Skipped dead neuron with idx {}".format(neuron_idx))
                    continue
                elemental_neuron_df = elemental_df[elemental_df["Neuron Idx"] == neuron_idx]
                composition_neuron_df = composition_df[composition_df["Neuron Idx"] == neuron_idx]
                elemental_neuron_df = elemental_neuron_df.drop(columns=["Neuron Idx", "Corruption"])
                composition_neuron_df = composition_neuron_df.drop(columns=["Neuron Idx", "Corruption"])

                # Normalize the activations grid following Madan et al. B.1 https://arxiv.org/pdf/2007.08032.pdf
                #  "The activations grid is then normalized to be between 0 and 1. To do so, we subtract the minimum of
                #  the activations grid and then divide it by the maximum."
                elem_max = elemental_neuron_df.max().max()
                elem_min = elemental_neuron_df.min().min()
                elemental_neuron_df = (elemental_neuron_df - elem_min) / (elem_max - elem_min)
                comp_max = composition_neuron_df.max().max()
                comp_min = composition_neuron_df.min().min()
                composition_neuron_df = (composition_neuron_df - comp_min) / (comp_max - comp_min)

                # Preferred category is the category which maximally activates over all relevant corruptions
                # We want to use the same prefrred category for elemental and composition.
                # I.e. if an "F"  detection neuron for elementals, we want to know how it fires for "F" on compositions.
                elemental_pref_cat = elemental_neuron_df.sum(axis=0).idxmax()
                # composition_pref_cat = composition_neuron_df.sum(axis=0).idxmax()

                # Difference between max and min firing rate for preferred category is worst case invariance score
                elemental_invariance_score = 1 - (elemental_neuron_df[elemental_pref_cat].max()
                                                  - elemental_neuron_df[elemental_pref_cat].min())

                # Firing rate has decreased for the composition, so invariance change is when it increases to max
                # composition_drop_invariance_score = (elemental_neuron_df[elemental_pref_cat].max()
                #                                      - composition_neuron_df.iloc[-1][elemental_pref_cat])
                # # Firing rate has increased for the composition, so invariance change is when it drops to the min
                # composition_incr_invariance_score = (composition_neuron_df.iloc[-1][elemental_pref_cat]
                #                                      - elemental_neuron_df[elemental_pref_cat].min())
                #
                # composition_invariance_score = 1 - max(abs(composition_incr_invariance_score),
                #                                        abs(composition_drop_invariance_score))

                # max() is worst case invariance. min() is best case invariance - i.e. want to be close to one elemental
                composition_invariance_score = 1 - (elemental_neuron_df[elemental_pref_cat] -
                                                    composition_neuron_df.iloc[-1][elemental_pref_cat]).abs().min()

                # print(composition_neuron_df)
                # print(max_activations[neuron_idx])
                # print(elemental_invariance_score)
                # print(composition_invariance_score)
                # print(elem_max, elem_min, comp_max, comp_min)
                # print("++++++++++++++++++++++++++++++++")
                # print(elemental_invariance_score, composition_invariance_score)


                # Without preferred category
                # elemental_invariance_score = 1 - ((elemental_neuron_df.max() - elemental_neuron_df.min()).mean())
                #
                # composition_invariance_diff = (elemental_neuron_df.max() - composition_neuron_df.iloc[-1]).abs()
                # composition_invariance_diff = pd.concat([composition_invariance_diff,
                #                                          (composition_neuron_df.iloc[-1] - elemental_neuron_df.min()).abs()],
                #                                         axis=1)
                # composition_invariance_score = 1 - composition_invariance_diff.max(axis=1).mean()
                # print(composition_neuron_df)
                # print(composition_invariance_diff)
                # print(composition_invariance_diff.max(axis=1))

                assert elemental_invariance_score >= 0
                assert composition_invariance_score >= 0
                elemental_invariance_scores.append(elemental_invariance_score)
                composition_invariance_scores.append(composition_invariance_score)

            print(len(elemental_invariance_scores))
            print(len(composition_invariance_scores))

            median_elemental_invariance_scores["-".join(composition_corruption)] = np.median(elemental_invariance_scores)
            median_composition_invariance_scores["-".join(composition_corruption)] = np.median(composition_invariance_scores)

        print(median_elemental_invariance_scores)
        print(median_composition_invariance_scores)
        min_elemental_median = min(min_elemental_median, min(median_elemental_invariance_scores.values()))
        min_composition_median = min(min_composition_median, min(median_composition_invariance_scores.values()))
        for k in median_elemental_invariance_scores.keys():
            scores_dict["Experiment"].append(legend_names[i])
            scores_dict["Elemental Invariance Score"].append(median_elemental_invariance_scores[k])
            scores_dict["Composition Invariance Score"].append(median_composition_invariance_scores[k])
            scores_dict["Accuracy (%)"].append(accs_df.loc[experiment, k])
            scores_dict["Corruptions in Composition"].append(len(k.split("-")))

    # Axis limits
    min_elemental_median = np.floor(min_elemental_median * 10) / 10  # Round down to 1 decimal place
    min_composition_median = np.floor(min_composition_median * 10) / 10
    min_joint_median = min(min_elemental_median, min_composition_median)
    if min_elemental_median < 0.5:
        lim_elemental_median = 0
    else:
        lim_elemental_median = 0.5
    if min_composition_median < 0.5:
        lim_composition_median = 0
    else:
        lim_composition_median = 0.5
    if min_joint_median < 0.5:
        lim_joint_median = 0
    else:
        lim_joint_median = 0.5

    scores_df = pd.DataFrame(scores_dict)
    print(scores_df)
    sns.set_theme()
    sns.set_context("poster")
    elem_comp_plot = sns.lmplot(data=scores_df, x="Elemental Invariance Score", y="Composition Invariance Score",
                                hue="Experiment", col="Corruptions in Composition", col_wrap=2, height=12,
                                legend=False, facet_kws=dict(sharex=False, sharey=False, xlim=(lim_joint_median, 1),
                                                             ylim=(lim_joint_median, 1)))
    elem_comp_plot.set_xlabels("")  # x labelling is handled by annotate_elem_comp
    elem_comp_plot.map_dataframe(annotate_elem_comp)
    elem_comp_fig = elem_comp_plot.fig
    elem_comp_fig.savefig(os.path.join(save_path, f"invariance-plots-elem-comp-{dataset}.pdf"), bbox_inches="tight")
    print("Saved invariance plot to {}".format(os.path.join(save_path, f"invariance-plots-elem-comp-{dataset}.pdf")))

    elem_acc_plot = sns.lmplot(data=scores_df, x="Elemental Invariance Score", y="Accuracy (%)",
                               hue="Experiment", col="Corruptions in Composition", col_wrap=2, height=12,
                               legend=False, facet_kws=dict(sharex=False, sharey=False, xlim=(lim_elemental_median, 1),
                                                            ylim=(0, 100)))
    elem_acc_plot.set_xlabels("")  # x labelling is handled by annotate_elem_acc
    elem_acc_plot.map_dataframe(annotate_elem_acc)
    elem_acc_fig = elem_acc_plot.fig
    elem_acc_fig.savefig(os.path.join(save_path, f"invariance-plots-elem-acc-{dataset}.pdf"), bbox_inches="tight")
    print("Saved invariance plot to {}".format(os.path.join(save_path, f"invariance-plots-elem-acc-{dataset}.pdf")))

    comp_acc_plot = sns.lmplot(data=scores_df, x="Composition Invariance Score", y="Accuracy (%)",
                               hue="Experiment", col="Corruptions in Composition", col_wrap=2, height=12,
                               legend=False, facet_kws=dict(sharex=False, sharey=False,
                                                            xlim=(lim_composition_median, 1),
                                                            ylim=(0, 100)))
    comp_acc_plot.set_xlabels("")  # x labelling is handled by annotate_comp_acc
    comp_acc_plot.map_dataframe(annotate_comp_acc)
    comp_acc_fig = comp_acc_plot.fig
    comp_acc_fig.savefig(os.path.join(save_path, f"invariance-plots-comp-acc-{dataset}.pdf"), bbox_inches="tight")
    print("Saved invariance plot to {}".format(os.path.join(save_path, f"invariance-plots-comp-acc-{dataset}.pdf")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--results-path', type=str, default='/om2/user/imason/compositions/results/',
                        help="path to directory containing results of testing")
    parser.add_argument('--activations-path', type=str, default='/om2/user/imason/compositions/activations/',
                        help="path to directory to save neural activations")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/analysis/',
                        help="path to directory to save analysis plots and pickle files")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--seed', type=int, default=38164641, help="random seed")
    parser.add_argument('--pairs-only', action='store_true', help="Plots only the compositions with 2 corruptions")
    args = parser.parse_args()

    # Set seeding # Final: 13579111 24681012 36912151. Hparams: 48121620
    reset_rngs(seed=args.seed, deterministic=True)

    variance_dir_name = f"seed-{args.seed}"  # f"lr-0.01_weight-1.0"
    args.data_root = os.path.join(args.data_root, args.dataset)
    args.results_path = os.path.join(args.results_path, args.dataset, variance_dir_name)
    args.activations_path = os.path.join(args.activations_path, args.dataset, variance_dir_name)
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

    """
    Run from inside analysis directory as: python invariance_plots.py --dataset EMNIST --total-n-classes 47 --pairs-only
    """

    main(all_corruptions, experiments, legend_names, args.dataset, args.total_n_classes, args.results_path,
         args.activations_path, args.save_path, args.pairs_only)
