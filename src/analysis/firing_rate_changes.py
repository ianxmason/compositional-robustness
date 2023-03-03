import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, rankdata

"""
Takes pkl files of activations for different corruptions and plots the change in firing rate at each layer
for each corruption.
"""

# Todo:
# Load the pkl, check the shapes etc.
# Create plot for each corruption over all layers - FR change. Maybe a bar is okay, maybe violin, not sure?

def main(corruptions, dataset, results_path, save_path):
    fname = os.path.join(results_path, "avg_firing_rates_Modules_Identity.pkl")
    with open(fname, "rb") as f:
        loaded_firing_rates = pickle.load(f)
    id_avg_firing_rates = loaded_firing_rates[0]  # num_layers, classes, neurons
    id_std_firing_rates = loaded_firing_rates[1]  # num_layers, classes, neurons
    id_max_firing_rates = loaded_firing_rates[2]  # num_layers, neurons

    for corruption in corruptions:
        print(corruption)
        fname = os.path.join(results_path, "avg_firing_rates_Modules_{}.pkl".format(corruption))
        with open(fname, "rb") as f:
            loaded_firing_rates = pickle.load(f)
        corr_avg_firing_rates = loaded_firing_rates[0]
        corr_std_firing_rates = loaded_firing_rates[1]
        corr_max_firing_rates = loaded_firing_rates[2]


        # One fig per corruption
        fig, axs = plt.subplots(2, 3, figsize=(6, 6))
        fig.suptitle("Firing Rate Changes - {}".format(corruption))

        all_novelty_scores = []
        all_spurious_scores = []
        all_geometric_means = []
        for i in range(len(id_avg_firing_rates)):
            # We have per-class firing rates for each layer
            id_layer_avg_frs = id_avg_firing_rates[i]
            corr_layer_avg_frs = corr_avg_firing_rates[i]
            id_layer_std_frs = id_std_firing_rates[i]
            corr_layer_std_frs = corr_std_firing_rates[i]
            id_layer_max_frs = id_max_firing_rates[i]
            corr_layer_max_frs = corr_max_firing_rates[i]  # neurons

            # Easiest thing is to average over neurons and classes and plot bars
            # But this quantity is probably meaningless
            # id_avg_frs = np.mean(id_layer_frs)
            # corr_avg_frs = np.mean(corr_layer_frs)
            # # axs.bar([2*i, 2*i+1], [id_avg_frs, corr_avg_frs], label="Level {}".format(i))

            # Perhaps better is change per neuron per class, then average?
            # fr_change = np.abs(id_layer_frs - corr_layer_frs)
            # mean_fr_change = np.mean(fr_change)
            # axs.bar([i], [mean_fr_change], label="Level {}".format(i))
            # Todo: try box plot, with and without absolute value

            # % Change in firing rate.
            # fr_change = np.abs((id_layer_avg_frs - corr_layer_avg_frs) / id_layer_avg_frs)  # absolute percentage change
            # mean_fr_change = np.mean(fr_change)
            # axs[0, 0].bar([i], [mean_fr_change], label="Level {}".format(i))
            # axs[0, 0].set_ylabel("Absolute % change in firing rate")

            # Change in avg firing rate as a multiple of std
            # fr_change = np.abs((id_layer_avg_frs - corr_layer_avg_frs) / id_layer_std_frs)
            # mean_fr_change = np.mean(fr_change)
            # axs[0, 1].bar([i], [mean_fr_change], label="Level {}".format(i))
            # axs[0, 1].set_ylabel("Change in firing rate as multiple of std")

            # Next we try collecting the max firing rate for each neuron over all classes. And plot the ratio
            # fr_ratio = corr_layer_max_frs / id_layer_max_frs
            # mean_fr_ratio = np.median(fr_ratio)
            # axs[1, 0].bar([i], [mean_fr_ratio], label="Level {}".format(i))
            # axs[1, 0].set_ylabel("Ratio of max firing rates")

            # Max category per neuron (Anirban's novelty?)
            id_max_avg_frs = np.max(id_layer_avg_frs, axis=0)  # max over classes
            corr_max_avg_frs = np.max(corr_layer_avg_frs, axis=0)
            #####
            ## Approx averagin - better to collect average. May be better to divide botj by id_layer_max_frs
            # id_max_avg_frs  = id_max_avg_frs / id_layer_max_frs
            # corr_max_avg_frs = corr_max_avg_frs / id_layer_max_frs  # corr_layer_max_frs
            ######
            # Todo: compare abs vs. throwing away negative cases
            # Todo: think about scale - on average what is novelty score and what is spurious score? c
                # basically the geometric mean is fairly dominated by the spurious score, novelty isn't doing that much
            # Todo: a problem is that currently the activations are normalised by the max firing the corruption
                # they should be normalised by the max firing of the identity.
                # another issue is that this means the novelty score is not always necessarily between 0 and 1

            # novelty_scores = np.abs(id_max_avg_frs - corr_max_avg_frs)  # Could normalize by id_max_avg_frs. Or rather than abs, take only the positive cases?

            # novelty_scores = 1 - (corr_max_avg_frs / id_max_avg_frs)  # 1 has meaning. It is when corruption activity goes to 0.

            novelty_scores = np.abs(((id_max_avg_frs - corr_max_avg_frs) / id_max_avg_frs)) ** 0.5  # Same as above but goes from 0 to 1 non-linearly
            novelty_scores = np.clip(novelty_scores, 0, 1)

            mean_novelty = np.mean(novelty_scores)
            axs[0, 0].bar([i], [mean_novelty], label="Level {}".format(i))
            axs[0, 0].set_ylabel("Something similar to Novelty Score")
            axs[0, 0].set_ylim([0, 1])
            all_novelty_scores.append(mean_novelty)

            # Spearman correlation between firing rates (Anirban's spurious score?)
            layer_corrs = []
            for neuron_idx in range(id_layer_avg_frs.shape[1]):
                neuron_frs = id_layer_avg_frs[:, neuron_idx]
                neuron_frs_corr = corr_layer_avg_frs[:, neuron_idx]
                # Spearman is rank correlation.
                corr, p = spearmanr(neuron_frs, neuron_frs_corr)
                # We may actually care about linear correlation?
                # corr, p = pearsonr(neuron_frs, neuron_frs_corr)
                layer_corrs.append(corr)

            spurious_scores = [1 - np.abs(corr) for corr in layer_corrs]

            # spurious_scores = [(1 - corr) / 2 for corr in layer_corrs]  # Corr is between -1 and 1. So this is between 0 and 1

            mean_spurious = np.mean(spurious_scores)
            axs[0, 1].bar([i], [mean_spurious], label="Level {}".format(i))
            axs[0, 1].set_ylabel("Something similar to Spurious Score")
            axs[0, 1].set_ylim([0, 1])
            all_spurious_scores.append(mean_spurious)

            # Geometric mean of novelty and spurious
            # print(np.max(novelty_scores))
            # print(np.min(novelty_scores))
            # print(np.max(spurious_scores))
            # print(np.min(spurious_scores))
            # print("-----")
            geometric_mean = np.sqrt((1-mean_novelty) * (1-mean_spurious))
            axs[0,2].bar([i], [geometric_mean], label="Level {}".format(i))
            axs[0,2].set_ylabel("Geometric mean of 1 - novelty and 1 - spurious")
            axs[0,2].set_ylim([0, 1])
            all_geometric_means.append(geometric_mean)

            if geometric_mean < 0.6201547955843831:
                print("Level {} has geometric mean of {}".format(i, geometric_mean))
                print("-----")

        # Try ranking novelty and spurious scores and multiplying the rank to get a new score
        # novelty_rank = rankdata(all_novelty_scores)
        # spurious_rank = rankdata(all_spurious_scores)
        # print(novelty_rank)
        # print(spurious_rank)
        # print(np.max(novelty_rank))
        # print(np.max(spurious_rank))
        # print("=====")
        # novelty_rank = novelty_rank / np.max(novelty_rank)
        # spurious_rank = spurious_rank / np.max(spurious_rank)
        # rank_product = novelty_rank * spurious_rank
        # axs[0, 1].bar(range(len(rank_product)), rank_product)
        # axs[0, 1].set_ylabel("Rank product of novelty and spurious")
        # axs[0, 1].set_ylim([0, 1])

        # Try renormalize novelty and spurious scores to be between 0 and 1
        # print(np.max(all_novelty_scores))  # 1.4030147,1.5543685,1.1788225,0.65049833,0.16229688,0.26230538
        # print(np.min(all_novelty_scores))  # 0.014689291,0.013895003,0.01684662,0.016389195,0.0152671235,0.014328909
        # print(np.max(all_spurious_scores))  # 0.7326699311768443,0.7239713329025892,0.8395478681262918,0.7904388784669589,0.6860070251020572,0.6303336294039232
        # print(np.min(all_spurious_scores))  # 0.2833708150536486,0.2128843048128395,0.6073656321951966,0.46689772382593514,0.1382879067482768,0.036943493958516566
        # # print("-----")
        # all_novelty_scores = np.array(all_novelty_scores)  # / np.max([1.4030147,1.5543685,1.1788225,0.65049833,0.16229688,0.26230538])  # np.max(all_novelty_scores)
        # # Don't normalilse spurious, 1 already has meaning - it is when the correlation is 0
        # all_spurious_scores = np.array(all_spurious_scores) # / np.max([0.7326699311768443,0.7239713329025892,0.8395478681262918,0.7904388784669589,0.6860070251020572,0.6303336294039232])  # np.max(all_spurious_scores)
        # axs[1, 0].bar(range(len(all_novelty_scores)), all_novelty_scores)
        # axs[1, 0].set_ylabel("Renormalized novelty scores")
        # axs[1, 0].set_ylim([0, 1])
        # axs[1, 1].bar(range(len(all_spurious_scores)), all_spurious_scores)
        # axs[1, 1].set_ylabel("Renormalized spurious scores")
        # axs[1, 1].set_ylim([0, 1])
        # arithmetic_mean = ((1 - all_novelty_scores) + (1 - all_spurious_scores)) / 2
        # axs[1,2].bar(range(len(arithmetic_mean)), arithmetic_mean)
        # axs[1,2].set_ylabel("Arithmetic mean of 1 - novelty and 1 - spurious")
        # axs[1,2].set_ylim([0, 1])

        # print(all_geometric_means)
        """
        EMNIST - mean of the below (ignoring the nan) = 0.5528080672220326. 
        [0.3353870696001096, 0.37068140645623093, 0.3873869279374966, 0.4100045897922033, nan, 0.5834092136285474],
        [0.268270185572241, 0.43999605003393294, 0.5209942628984611, 0.570304844262882, 0.5352255424802825, 0.6676016908638244],
        [0.48044291969071007, 0.6915675165128047, 0.7035729506676625, 0.7212999815163935, 0.6293931744691903, 0.6837458750863457],
        [0.5479854342232231, 0.39193354440620143, 0.34454575623445294, 0.3483280201503628, 0.33831292999036244, 0.5150028848079768],
        [0.6279222849414324, 0.5554119833420216, 0.5989595984711098, 0.5581815626883471, 0.49938465456982795, 0.4528953773255412],
        [0.7493351291661956, 0.7757254648588963, 0.7778385861843197, 0.796154942334446, 0.7201742094767243, 0.7509057881303774]
        
        CIFAR - mean of the below = 0.7499073134634917. 
        [0.5883443404224722, 0.5973665436906237, 0.6180585718692841, 0.7092913487228746, 0.7322420212450893, 0.7592939406899171, 0.7731092742296138, 0.7328284016474201, 0.7553170041381755, 0.6443722649157922],
        [0.6312029170234162, 0.6638089544238891, 0.6632754516028148, 0.7839471973562089, 0.8022575583560707, 0.8458540652465584, 0.849840211298821, 0.8242505246281486, 0.8570731278888235, 0.7459038181751501],
        [0.6550529393249177, 0.6863445453845226, 0.7012749647591207, 0.6810505775819244, 0.6907592024924789, 0.6588856160377438, 0.650582293106199, 0.6226165417512066, 0.6458489305153312, 0.6069440699106359],
        [0.556279454383554, 0.5538864285399224, 0.6150201065604478, 0.6920285789194527, 0.6869630656147052, 0.7371745955248191, 0.7239548723365963, 0.6842694347039141, 0.7494894402366855, 0.6433318044885684],
        [0.8178958897655594, 0.7811704563517564, 0.786256022735888, 0.8286360524134895, 0.8281026707127506, 0.8048097850709907, 0.7964707994718974, 0.7660102875395126, 0.8189924993851372, 0.7520785988013664],
        [0.9141662809825316, 0.9293329354935482, 0.9286476250374613, 0.9416012318076645, 0.9451871749286052, 0.9287847201531532, 0.9238985122239132, 0.8933948746875672, 0.9146442530622696, 0.8749631374405329]
        
        FACESCRUB - mean of the below = 0.6201547955843831. 
        [0.4735223378444431, 0.5744362419832016, 0.5995882186964399, 0.6428930696229798, 0.6621919396818574, 0.699212003338763, 0.6917530951027945, 0.6815096269324864, 0.6568537782653255, 0.6469398061884754, 0.6404087797853024, 0.6346623587882181, 0.6386080088425622, 0.5596367354375495, 0.5014257372936631, 0.4581188116323929, 0.430673038798217],
        [0.6440115413552033, 0.5663837046494253, 0.5907250884034243, 0.6929115212695061, 0.7016461349347741, 0.7396622477454515, 0.7296733204698308, 0.7242133582491203, 0.6937377501877053, 0.6809193980911, 0.6605983019823676, 0.6495765359126145, 0.65046312486003, 0.5752066411027753, 0.519423430444539, 0.4669778194186927, 0.4326576631093286],
        [0.42707864169237525, 0.41326815044899656, 0.4013176163964202, 0.4465758007833093, 0.4528040527762616, 0.5252802293088606, 0.5298093220965981, 0.5272365707461493, 0.504022387456057, 0.4983693312758747, 0.49089496865502125, 0.4851199639678365, 0.4936290893960497, 0.4268305279132787, 0.3868095191100238, 0.35421027652802645, 0.3243482478168793],
        [0.351279879856467, 0.4929128328857428, 0.5699621699407865, 0.5258718037616772, 0.5729379884157951, 0.5996056667708404, 0.6365929735201172, 0.6694745781465941, 0.6228488998674152, 0.6303129820252484, 0.6375783154098129, 0.6412917454272687, 0.6618989756532981, 0.5516786013106748, 0.4790518029632555, 0.44781285437307694, 0.4136116565100102],
        [0.8607961411444628, 0.7984798298250636, 0.768052243060396, 0.765485927738506, 0.737957052395747, 0.7521170298925355, 0.7616899015864338, 0.7627757735579958, 0.723767578597038, 0.7199874524809546, 0.7181864811436174, 0.7199626861755372, 0.733138634688977, 0.6225523828491033, 0.5433027305740118, 0.4960480684621726, 0.4634893331939993],
        [0.8668226118939365, 0.8545936618468889, 0.8627152366984728, 0.885483509299795, 0.8812689882127465, 0.8670987016536329, 0.8545639507204434, 0.8487750966153247, 0.8083590438583614, 0.799970817259636, 0.786017889030867, 0.7793840116095446, 0.7821443934155949, 0.6805947947145602, 0.601793413449049, 0.5412063768622887, 0.5036578114727253]
        """


            # Alternate - try and come up with 2 scores?
            # 1. relative firings/patter of firings stays the same, but firing rate drops - we still have extracted the right structure
            # 2. pattern of firings change, but firing rate stays similar - we still have useful features

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "firing_rate_changes_{}.pdf".format(corruption)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--results-path', type=str, default='/om2/user/imason/compositions/analysis/',
                        help="path to directory containing avg firing rates")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/analysis/',
                        help="path to directory to save analysis plots and pickle files")
    args = parser.parse_args()

    args.data_root = os.path.join(args.data_root, args.dataset)
    args.results_path = os.path.join(args.results_path, args.dataset)
    args.save_path = os.path.join(args.save_path, args.dataset)

    with open(os.path.join(args.data_root, "corruption_names.pkl"), "rb") as f:
        all_corruptions = pickle.load(f)

    elemental_corruptions = []
    for corr in all_corruptions:
        if len(corr) == 1:
            if corr not in elemental_corruptions:
                elemental_corruptions.append(corr[0])

    main(elemental_corruptions, args.dataset, args.results_path, args.save_path)