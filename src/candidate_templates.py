"""
Initially we found candidate templates by looking for selectivity to the data
Now we find candidate templates by assuming they exist and thus searching for permuted firings for the data and
the corrupted data.

Q: Should we only look for sets of size 2 (say template for identity and for the corruption), then each set of size 2
   is a set of templates?
Q: Is the identity special - should we consider (id, corr1), (id, corr2), (corr1, corr2) to search over
Q: When looking for permuted elements how close do we need the firing rates to be?

Example (fr for template neurons being permuted by corr):
    SAMPLE 1 in batch   SAMPLE 2 in batch
    FR_id  FR_corr1     FR_id  FR_corr1
    6      2            7      1
    4      6            3      7
    2      4            1      3

Pseudo code outline:
    1. Take sample 1 FR_id
    2. For element 1 look in FR_corr1 for all cases of the element, using something like isclose() probably with a relatively large tolerance
    3. Check if the same similarity exists for sample 2
    4. Repeat for all elements (can add more samples to batch to check carefully) to find potential template neurons
    5. (Optional) Determine the cycles of the template neurons e.g. are the permutations all pairs, groups of 3 etc.
"""
import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from collect_activations import FiringConvHook, FiringLinHook
from data.data_loaders import get_static_emnist_dataloaders
from data.data_transforms import denormalize
from lib.networks import DTN
from lib.utils import *


# Kind of like max patches
# Take unshuffled test data, get the activations with the hook, then get the candidate templates
def main(data_root, ckpt_path, save_path, total_n_classes, batch_size, n_workers, pin_mem, dev, check_if_run):
    ckpt = "identity-gaussian_blur-stripe.pt"  # "identity-rotate_fixed-scale.pt"  # "identity-canny_edges-inverse.pt"
    network = DTN(total_n_classes).to(dev)
    network.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt)))
    network.eval()
    # Try first conv layer for now
    conv_count = 0
    for module in network.modules():
        if isinstance(module, nn.ReLU):
            conv_count += 1
            if conv_count == 1:
                hook = FiringConvHook(module)
                break
    print("Testing {}".format(ckpt))

    trained_classes = list(range(total_n_classes))
    corruption_1 = "gaussian_blur"  # "rotate_fixed"  # "canny_edges"
    corruption_2 = "stripe"  # "scale"  # "inverse"
    composition = "gaussian_blur-stripe"  # "rotate_fixed-scale"  # "canny_edges-inverse"
    corruption_path_1 = os.path.join(data_root, corruption_1)
    corruption_path_2 = os.path.join(data_root, corruption_2)

    ex_per_class = 400  # hardcoded and known
    # class_to_use = 10
    # assert batch_size > total_n_classes
    # samples_per_class = batch_size // total_n_classes
    # assert samples_per_class < ex_per_class
    # batch_size = total_n_classes * samples_per_class

    # Shuffle=False gives identical samples across corruptions - I have checked this through visualisation
    # Shuffling in the same way is tricky, if want new classes easiest will be to skip a few batches
    _, val_dl_1, _ = get_static_emnist_dataloaders(corruption_path_1, trained_classes, ex_per_class, False,
                                                   n_workers, pin_mem)
    _, val_dl_2, _ = get_static_emnist_dataloaders(corruption_path_2, trained_classes, ex_per_class, False,
                                                   n_workers, pin_mem)

    mins_pairs = []
    mins_pairs_indices = []
    for class_to_use in range(total_n_classes):
        with torch.no_grad():
            class_count = -1
            for data_tuple in val_dl_1:
                class_count += 1
                if class_count == class_to_use:
                    x_tst = data_tuple[0][0:batch_size].to(dev)
                    break
            _ = network(x_tst)

            # # # Vis to check images are same across batches
            # fig_name = "Corr1.png"
            # vis_path = "/om2/user/imason/compositions/"
            # fig_path = os.path.join(vis_path, fig_name)
            # # Denormalise Images
            # x = x_tst.detach().cpu().numpy()
            # y = y_tst.detach().cpu().numpy()
            # x = denormalize(x).astype(np.uint8)
            # # And visualise
            # visualise_data(x[:16], y[:16], save_path=fig_path, title=fig_name[:-4], n_rows=4, n_cols=4)

        activations_1 = hook.output.detach().cpu().numpy()

        with torch.no_grad():
            class_count = -1
            for data_tuple in val_dl_2:
                class_count += 1
                if class_count == class_to_use:
                    x_tst = data_tuple[0][0:batch_size].to(dev)
                    break
            _ = network(x_tst)

            # Vis to check images are same across batches
            # fig_name = "Corr2.png"
            # vis_path = "/om2/user/imason/compositions/"
            # fig_path = os.path.join(vis_path, fig_name)
            # # Denormalise Images
            # x = x_tst.detach().cpu().numpy()
            # y = y_tst.detach().cpu().numpy()
            # x = denormalize(x).astype(np.uint8)
            # # And visualise
            # visualise_data(x[:16], y[:16], save_path=fig_path, title=fig_name[:-4], n_rows=4, n_cols=4)

        activations_2 = hook.output.detach().cpu().numpy()

        max_templates = activations_1.shape[1] // 2  # Never want more than half the units, so can turn off the other half

        distances = np.zeros([activations_1.shape[1], activations_1.shape[1]])  # num_units, num_units
        for sample_1, sample_2 in zip(activations_1, activations_2):  # Over the batch
            for i, unit in enumerate(sample_1):  # Todo: I imagine this can be vectorized
                distances[i] += np.abs(sample_2 - unit)

        # Find closest units to each other (in both directions, distances matrix is not symmetric)
        # in distances row i is all units in corr2 compared to unit i in corr1
        # min in each row is the unit that is closest
        mins_1 = np.min(distances, axis=1) / len(distances)  # take average distance for comparison across layers
        min_indices_1 = np.argmin(distances, axis=1)
        # distances.T is the other direction - so row i is all units in corr1 compared to unit i in corr2
        mins_2 = np.min(distances, axis=0) / len(distances)
        min_indices_2 = np.argmin(distances, axis=0)

        # {2, 3, 4, 6, 7, 8, 11, 14, 15, 17, 18, 20, 22, 23, 26, 27, 28, 29, 30, 34, 36, 37, 40, 41, 42, 44, 45, 47, 48, 49, 50, 51, 53, 57, 58, 59, 61, 63}

        # Find the pairs that are closest to each other
        for i, j in enumerate(min_indices_1):
            if min_indices_2[j] == i:  # Todo: can this also be vectorized?
                mins_pairs.append(mins_1[i] + mins_2[j])
                mins_pairs_indices.append((i, j))

    print("Post-processing found candidate pairs")
    # Make pairs of neurons unique. If pairs are duplicated could do any of:
    # best case distance, worst case distance, average distance, sort by number of times the pair appears over classes.
    sorted_mins = defaultdict(float)
    pair_counts = defaultdict(float)

    # Calculate average distance
    # for i, pair in enumerate(mins_pairs_indices):
    #     sorted_mins[pair] += mins_pairs[i]
    #     pair_counts[pair] += 1.
    # for k, v in sorted_mins.items():
    #     sorted_mins[k] = v / pair_counts[k]
    # Calculate best case - i.e. if we have template that is good for only one class (?), then we still want to keep it
    for i, pair in enumerate(mins_pairs_indices):
        if sorted_mins[pair] >= mins_pairs[i] or sorted_mins[pair] == 0.:
            sorted_mins[pair] = mins_pairs[i]
    # Worst case - i.e. compare using the furthest distance apart
    # for i, pair in enumerate(mins_pairs_indices):
    #     if sorted_mins[pair] <= mins_pairs[i]:
    #         sorted_mins[pair] = mins_pairs[i]

    # exclude invariant neurons. I am not sure invariant neurons should be removed - they could be a necessary part of an invariant represenation (e.g. a triple of 2 templates and 1 invariant)
    # for k in list(sorted_mins):  # https://stackoverflow.com/questions/11941817/how-to-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
    #     if k[0] == k[1]:
    #         del sorted_mins[k]
    #         print("Deleted invariant neuron pair:", k)

    print(sorted_mins)

    # Sort by min distance
    sorted_mins = sorted(sorted_mins.items(), key=lambda x: x[1])
    # Threshold
    # # by hyperparameter:
    # is_close_threshold = 0.5
    # not_dead_threshold = 0.01
    # print("Thresholded between {} and {}".format(not_dead_threshold, is_close_threshold))
    # sorted_mins = [x for x in sorted_mins if not_dead_threshold < x[1] < is_close_threshold]
    # by number of units desired
    max_templates = 20
    all_pairs = []
    turn_off_units = []
    for pair in sorted_mins:
        all_pairs.append(pair[0])
        turn_off_units.append(pair[0][0])
        turn_off_units.append(pair[0][1])
        if len(set(turn_off_units)) >= max_templates - 1:
            break
        # -1 because we don't want to risk adding another 2 units from a pair to go above max_templates
        # nor do we want to add a single unit from a pair without adding its partner
    print("All pairs over all classes {}".format(all_pairs))
    print("Units to turn off {}".format(list(set(turn_off_units))))


    # Worst case (using test set)
    # [2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 18, 23, 27, 29, 30, 34, 36, 37, 41, 44, 45, 47, 49, 50, 51, 53, 57, 59, 61, 63]  55.9771
    # Best case (using test set)
    # [2, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 18, 20, 23, 27, 29, 30, 34, 36, 41, 42, 44, 45, 47, 49, 50, 53, 57, 59, 61, 63]  55.2815
        # More strangeness. If turn off the remaining neurons, leaving only these on the accuracy goes to 20.4302
        # neurons off are: [0, 1, 3, 9, 10, 12, 16, 19, 21, 22, 24, 25, 26, 28, 31, 32, 33, 35, 37, 38, 39, 40, 43, 46, 48, 51, 52, 54, 55, 56, 58, 60, 62]
    # Average (using test set)
    # [2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 18, 20, 23, 27, 29, 30, 34, 36, 41, 42, 44, 45, 47, 49, 50, 53, 57, 59, 61, 63]  # 56.5711

    # Best case with invariant removed (val set)
    # [2, 3, 4, 6, 7, 8, 11, 14, 17, 18, 20, 23, 26, 27, 28, 29, 30, 34, 36, 37, 41, 42, 44, 47, 49, 50, 53, 57, 59, 61, 63]  # 59.9850
    # Best case with invariant removed (test set)
    # [2, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 18, 20, 23, 27, 29, 30, 34, 36, 41, 42, 44, 45, 47, 49, 50, 53, 57, 59, 61, 63]  # 55.2815
    # Difference between test and val set shows this isn't a perfect method and templates don't perfectly exist.

    # Best case with invariant removed (val set) using canny vs canny-inverse composition
    # [0, 1, 2, 4, 6, 7, 8, 11, 14, 17, 18, 19, 23, 25, 26, 27, 29, 30, 34, 37, 41, 42, 45, 46, 47, 49, 50, 53, 57, 59, 61, 63]


    # INVESTIGATION
    # Best case (using test set)
    # [2, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 18, 20, 23, 27, 29, 30, 34, 36, 41, 42, 44, 45, 47, 49, 50, 53, 57, 59, 61, 63]  55.2815
        # If turn off the remaining neurons, leaving only these on the accuracy goes to 20.4302
        # neurons off are: [0, 1, 3, 9, 10, 12, 16, 19, 21, 22, 24, 25, 26, 28, 31, 32, 33, 35, 37, 38, 39, 40, 43, 46, 48, 51, 52, 54, 55, 56, 58, 60, 62]
    # Take the set with 20% acc above and compare with this list found using old method of thresholding over all classes
    # Take intersection to see if there are some "key units"
    # [2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 18, 20, 23, 26, 27, 29, 30, 34, 36, 37, 41, 42, 44, 45, 47, 49, 50, 51, 53, 57, 58, 59, 61, 63]
    # Intersection is [3, 26, 37, 51, 58]
    # Moving these neurons from one list to the other makes acc go 55 -> 53. But more importantly acc of other half goes 20% -> 70%
    # Suggests some interaction between [3, 26, 37, 51, 58] and other neurons in the list that makes the acc go to 20%???
    # Seems much more complicated than specific neurons - if we turn off say groups of 5 neurons, either the "key units"
    # or some other 5, the accuracy immediately jumps to 70%+ (sometimes 85%). This makes it very hard to say why this grouping
    # specifically is 20%.

    # So what can I say so far
    # If I turn off 10 units selected with this method, is it consistently better than random over say 50 runs
        # Mean and worst case?
    # If I choose 10 units with the "best case" method on the val set they are: [41, 47, 18, 50, 53, 57, 27, 29, 63]

    # # Todo: method 1 - store distances matrix over whole dataset and take sorted_mins from this
    # # Todo: method 2 - calculate distances for each class, take sorted_mins from this and combine
    # # Another thing we could try - combine (id, corr1) templates with (id, corr2) templates
    # # Another idea - find neurons most invariant to (id, corr1) or (id, corr2) alone in layer 2 and turn them off
    # # or even to (corr1, corr2). Need to think more about this - intuition is that if we have corr 1 templates in
    # # layer 1 then the (2D or higher) output should be invariant to corr 1; is there something that pools this info
    # # to form a 1D invariant output to corr1 in layer 2?

    # Without thresholding
    # id, canny = [4, 6, 7, 8, 14, 15, 17, 24, 26, 29, 30, 34, 36, 37, 41, 42, 44, 48, 53, 57, 59, 63]  # 84.0807, still not lower than selective units
    # id, inv = [0, 4, 6, 7, 8, 9, 10, 13, 18, 21, 22, 23, 24, 29, 30, 34, 35, 37, 41, 42, 44, 45, 47, 48, 50, 53, 55, 57, 59, 61, 63]
    # canny, inv = [2, 4, 6, 8, 14, 15, 17, 18, 23, 24, 26, 29, 30, 36, 41, 42, 43, 47, 50, 51, 53, 59]

    # set = [0, 2, 4, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 21, 22, 23, 24, 26, 29, 30, 34, 35, 36, 37, 41, 42, 43, 44, 45, 47, 48, 50, 51, 53, 55, 57, 59, 61, 63]

    # If use this set and turn them off acc is: 67.3427
    # If leave on all other units (which is more units left on so not fair): acc is 85.5790

    # Let's try thin the set of template units
    # One way - just take the canny, inverse list
    # [2, 4, 6, 8, 14, 15, 17, 18, 23, 24, 26, 29, 30, 36, 41, 42, 43, 47, 50, 51, 53, 59]  # Acc: 70.1413
    # Seems very strong - turning off random units of same length (excluding these) is ~85% acc - may need to do more runs though (also, perhaps there is a lot of dead units?)
    # Plus if only leave these units on get acc 59.8245
    # Random runs of same number of units: 47.8863, 35.1884, 45.5533, 63.5595, 72.4262
    # So hard to say much, need to do more thorough random experiments to see what is happening here

    # Another way, by thresholding (too close to zero suggests off, to far away suggests on)
    # Thresholded from above at 0.5
    # Tholded id, canny: [34, 4, 36, 6, 7, 8, 41, 42, 44, 14, 15, 17, 53, 57, 59, 29, 30, 63]
    # Tholded id, inv: [0, 4, 6, 7, 8, 9, 10, 13, 18, 21, 22, 23, 24, 29, 30, 34, 37, 41, 42, 44, 45, 47, 48, 50, 53, 57, 59, 61, 63]
    # Tholded canny, inv: [2, 36, 4, 6, 8, 41, 42, 14, 47, 17, 18, 50, 53, 23, 26, 59, 29, 30]
    # If turn off the tholded canny, inv. Acc: 73.4375
    # Here's the crazy part I absolutely do not understand
    # If leave only these units on, get acc: 18.8249
    # This means the 4 units difference between the tholded and not tholded give a 40% increase in accuracy. Why???
    # Are all these results noise? Is it like we need some strongly firing non-templates?
    # Is it that these are actually important templates? Is this a direction for missing pieces in the idea/theory?
    # The 4 units tholded away are [15, 24, 43, 51]

    # Thresholded from below 0.3
    # Tholded canny, inv: [2, 36, 4, 6, 8, 42, 43, 14, 15, 17, 51, 53, 23, 24, 26, 59, 29, 30]  Acc: 81.3142
    # Suggests being "True templates" is very important, that the lower threshold is set too high
    # The 4 units tholded away are [18, 41, 47, 50]. If just turn off these 4, acc: 84.9797 - hard to say much from this value.

    # IMPORTANT - when tholding away 4 units, need to also think about their pairs (the template unit to which they are associated)
    # Otherwise removing 4 units could actually be removing 8 units (4 pairs).
    # The pairs (corr1, corr2) are: [(2, 36), (4, 8), (6, 26), (14, 59), (15, 51), (17, 23), (18, 41), (24, 43), (29, 50), (30, 2), (41, 47), (42, 4), (53, 29)]

    # Thresholded from below at 0.3 and above at 0.5
    # Tholded canny, inv: [2, 36, 4, 6, 8, 42, 14, 17, 53, 23, 26, 59, 29, 30]  Acc: 82.8714







    # Todo: issues
    # Check carefully code is doing what I want
    # Check thresholding indexing is correct
    # Only checks for pairs
    # Ease of use - add a loop over all corruptions to be checked?
    # How to find templates in later layers? Do need to use the composed stuff as data? (What about if use composed stuff in first layer?)

    # Todo: when change to all classes the acc is worse than the templates for one class. Why?
    # The min values are much larger! So threshold? But still are missing some because the accuracy is worse than with just one class?
    # Do need class dependent templates?
    # Trying class dependent templates
    # Class 1: [2, 4, 6, 8, 14, 15, 17, 18, 23, 24, 26, 29, 30, 36, 41, 42, 43, 47, 50, 51, 53, 59]
    # Class 2: [2, 4, 5, 6, 8, 10, 12, 14, 18, 23, 25, 27, 29, 30, 34, 36, 41, 47, 48, 53, 58, 59, 63]
    # Class 3: [1, 2, 4, 5, 6, 8, 12, 14, 17, 18, 23, 25, 26, 29, 30, 36, 39, 41, 43, 47, 49, 53, 59, 62]
    # Class 4: [4, 6, 7, 8, 11, 17, 18, 23, 29, 30, 37, 41, 43, 47, 48, 53, 58, 60, 63]
    # Class 5: [2, 4, 37, 8, 41, 42, 14, 47, 48, 17, 18, 20, 23, 26, 27, 29, 30]
    # Problem - combining just these 5 is 40 out of 64 units
    # Another option - test accuracy on class 0 only in test_without_templates?
    # But, the theory tells us class dependence shouldn't matter - so should we take the intersection?
    # The intersection is [4, 8, 41, 47, 18, 23, 29, 30]. Acc: 85 - not great
    # Another option is thresholding then the union.
    # if we threshold between 0.01 and 0.5
    # Class 0: [2, 36, 4, 6, 8, 41, 42, 14, 47, 17, 18, 50, 53, 23, 26, 59, 29, 30]
    # Class 1: [34, 4, 36, 6, 27, 41, 8, 14, 18, 53, 23, 59, 29, 63]
    # Class 2: [4, 6, 8, 41, 47, 17, 18, 49, 53, 23, 59, 29, 30]
    # Class 3: [4, 37, 6, 7, 8, 41, 11, 47, 17, 18, 53, 23, 58, 29, 30, 63]
    # Class 4: [4, 8, 41, 42, 14, 47, 17, 18, 20, 23, 27, 29]
    # Union of these: [2, 4, 6, 7, 8, 11, 14, 17, 18, 20, 23, 26, 27, 29, 30, 34, 36, 37, 41, 42, 47, 49, 50, 53, 58, 59, 63]
    # With these: Acc = 57.9623
    # Sampling randomly from set without these: 85.3596

    # Add more classes
    # Class 5: [4, 6, 8, 41, 42, 14, 47, 17, 18, 53, 23, 59, 29, 63]
    # Class 6: [4, 7, 8, 41, 42, 11, 15, 47, 18, 29]
    # Class 7: [7, 8, 41, 47, 17, 18, 49, 53, 23, 29, 63]
    # Class 8: [2, 4, 8, 41, 42, 11, 47, 17, 18, 53, 23, 27, 29, 30, 63]
    # Class 9: [34, 4, 6, 8, 41, 14, 47, 49, 18, 20, 53, 59, 29, 30, 63]
    # Union: [2, 4, 6, 7, 8, 11, 14, 15, 17, 18, 20, 23, 26, 27, 29, 30, 34, 36, 37, 41, 42, 47, 49, 50, 53, 58, 59, 63]
    # With these turned off: Acc : 57.4165
    # Sampling randomly from set without these: 84.8138

    # Loop over classes (eventually maybe just go over the whole dataset - perhaps separate classes or perhaps not?)
    # [2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 17, 18, 20, 23, 26, 27, 29, 30, 34, 36, 37, 41, 42, 44, 45, 47, 49, 50, 51, 53, 57, 58, 59, 61, 63]

    # length 36. Acc with these turned off: 53.0073
    # The set without these turned off: 79.9765  (Note: this is equal to having only these units on)
    # To equalize sets remove last 4 of the list, then acc is 56.3463 with these turned off and 73.6569 with the other half






    # Layer 2. No Thresholding. Ignored identity as a candidate to search over.
    # canny, canny-inv: [4, 6, 7, 12, 16, 20, 24, 25, 26, 27, 34, 35, 41, 49, 55, 61, 65, 69, 70, 74, 75, 77, 79, 80, 85, 88, 91, 96, 100, 101, 104, 111, 112, 114, 121, 126]
    # inv, canny-inv: [5, 10, 11, 17, 20, 22, 33, 34, 35, 42, 46, 55, 61, 72, 73, 75, 77, 80, 89, 101, 105, 111, 114, 121, 126]
    # combination of above 2: [4, 5, 6, 7, 10, 11, 12, 16, 17, 20, 22, 24, 25, 26, 27, 33, 34, 35, 41, 42, 46, 49, 55, 61, 65, 69, 70, 72, 73, 74, 75, 77, 79, 80, 85, 88, 89, 91, 96, 100, 101, 104, 105, 111, 112, 114, 121, 126]
    # canny, inv: [5, 7, 8, 12, 16, 17, 20, 27, 28, 30, 32, 35, 41, 44, 50, 51, 53, 56, 58, 60, 62, 63, 64, 65, 70, 72, 74, 75, 77, 79, 84, 89, 91, 92, 95, 96, 97, 103, 105, 111, 114, 123, 126]
    # combination of all 3: [4, 5, 6, 7, 8, 10, 11, 12, 16, 17, 20, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 41, 42, 44, 46, 49, 50, 51, 53, 55, 56, 58, 60, 61, 62, 63, 64, 65, 69, 70, 72, 73, 74, 75, 77, 79, 80, 84, 85, 88, 89, 91, 92, 95, 96, 97, 100, 101, 103, 104, 105, 111, 112, 114, 121, 123, 126]
    # If use combination of all 3 with the canny, inv units in layer 1 acc goes to: 39.5013. This is more than half the units off (68).
    # If turn off all other units acc is: 58.0212 (but number of units is not the same)
    # This is with the canny, inv units off in layer 1. If we use rand units in layer 1 (excluding canny, inv) acc is: 69.7881 (one run)
    # If remove last 4 units from the overall list acc is: 42.9473.
    # As opposed to 55.3778 for the other half of the units.
    # Todo: points - (1) this gap is not as big as I hoped. (2) Can we thin the list of units to turn off somehow?


    # Layer 2. No Thresholding. Ignored identity as a candidate to search over.
    # canny, canny-inv: [4, 6, 7, 12, 16, 20, 24, 25, 26, 27, 34, 35, 41, 49, 55, 61, 65, 69, 70, 74, 75, 77, 79, 80, 85, 88, 91, 96, 100, 101, 104, 111, 112, 114, 121, 126]
    # inv, canny-inv: [5, 10, 11, 17, 20, 22, 33, 34, 35, 42, 46, 55, 61, 72, 73, 75, 77, 80, 89, 101, 105, 111, 114, 121, 126]
    # combination of above 2: [4, 5, 6, 7, 10, 11, 12, 16, 17, 20, 22, 24, 25, 26, 27, 33, 34, 35, 41, 42, 46, 49, 55, 61, 65, 69, 70, 72, 73, 74, 75, 77, 79, 80, 85, 88, 89, 91, 96, 100, 101, 104, 105, 111, 112, 114, 121, 126]
    # canny, inv: [5, 7, 8, 12, 16, 17, 20, 27, 28, 30, 32, 35, 41, 44, 50, 51, 53, 56, 58, 60, 62, 63, 64, 65, 70, 72, 74, 75, 77, 79, 84, 89, 91, 92, 95, 96, 97, 103, 105, 111, 114, 123, 126]
    # combination of all 3: [4, 5, 6, 7, 8, 10, 11, 12, 16, 17, 20, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 41, 42, 44, 46, 49, 50, 51, 53, 55, 56, 58, 60, 61, 62, 63, 64, 65, 69, 70, 72, 73, 74, 75, 77, 79, 80, 84, 85, 88, 89, 91, 92, 95, 96, 97, 100, 101, 103, 104, 105, 111, 112, 114, 121, 123, 126]
    # If use combination of all 3 with the canny, inv units in layer 1 acc goes to: 39.5013. This is more than half the units off (68).
    # If turn off all other units acc is: 58.0212 (but number of units is not the same)
        # This is with the canny, inv units off in layer 1. If we use rand units in layer 1 (excluding canny, inv) acc is: 69.7881 (one run)
    # If remove last 4 units from the overall list acc is: 42.9473.
    # As opposed to 55.3778 for the other half of the units.
    # Todo: points - (1) this gap is not as big as I hoped. (2) Can we thin the list of units to turn off somehow?

    # Using iteration over all classes
    # canny, inv:  [0, 4, 5, 8, 10, 11, 16, 20, 21, 25, 27, 30, 31, 32, 33, 34, 35, 47, 50,53, 55, 56, 61, 62, 63, 65,
    #               66, 70, 73, 74, 75, 76, 77, 79, 80, 83, 84, 89, 91, 95, 97, 100, 101, 112, 113, 114, 118, 123, 126]
    # Need to use smaller threshold: 0.01 and 0.2
    # Note: with canny, inv there are a lot of invariannt neurons (not surprising based on template ideas)
        # Do we want to turn these ones off?
    # Acc with these turned off (and layer 1 turned off): 40.9942
    # With random units turned off (and layer 1 templates off): 41.8985, 49.7753, 44.4189

    # canny, canny-inv: [0, 5, 8, 11, 20, 21, 24, 25, 27, 33, 35, 41, 50, 53, 54, 55, 61, 65, 66,
    #                    74, 75, 77, 79, 80, 88, 89, 95, 97, 100, 101, 110, 112, 113, 118, 123, 126]
    # Threshold: 0.01 and 0.2
    # Note again we have a lot of invariant neurons - do we want these on or off?
    # Acc with these turned off (and layer 1 turned off): 53.7083 - basically same as with only layer 1 turned off
    # With random units turned off (and layer 1 templates off): 48.2930

    # inv, canny-inv: [0, 8, 10, 11, 20, 21, 24, 27, 31, 33, 34, 35, 50, 54, 55, 61, 62, 65, 66,
    #                 75, 77, 79, 80, 83, 88, 89, 95, 97, 99, 100, 101, 104, 110, 111, 112, 113, 117, 118, 123, 125, 126]
    # Threshold: 0.01 and 0.2
    # Again a lot of invariant neurons
    # Acc with these turned off (and layer 1 turned off): 51.4020
    # With random units turned off (and layer 1 templates off): 42.2250  - again suggests this selection is not quite right yet

    # All 3 combined: [0, 4, 5, 8, 10, 11, 16, 20, 21, 24, 25, 27, 30, 31, 32, 33, 34, 35, 41, 47, 50, 53, 54, 55, 56, 61,
    # 62, 63, 65, 66, 70, 73, 74, 75, 76, 77, 79, 80, 83, 84, 88, 89, 91, 95, 97, 99, 100, 101, 104, 110, 111, 112, 113,
    # 114, 117, 118, 123, 125, 126]
    # Acc with these turned off (and layer 1 turned off): 33.9041
    # With random units turned off (and layer 1 templates off):  38.8378, 32.6413
        # Maybe a small difference but nothing to write home about. See the ToDos below
        # Actively do not turn off invariant representations?!?


    # Todo: To formalize this:
        # Instead of threshold set a max number of neurons N and take the N neurons with min pairwise distance (in both dirs)
            # Note somewhere the possibility to relax the requirement for bidirectionality
        # Do corr1, corr2 and use join to make the composition. What about the identity (need to decide this but maybe post Fujitsu)???
        # Do the template calculation using the validation set
        # Compare the corr1 vs corr2 neurons with, for example corr1 vs composition neurons
        # In layer 2 look for neurons than are invariant to corr1, comp and pairwise invariant to corr2, comp - do these exist?
        # In the testing of turning off neurons get also the accuracy to the individual corruptions
        # Explore more all classes vs individual classes. Consider using all data.
        # When testing accuracy auto compare to say 20 random runs mean/std.
            # Compare both selecting random units excluding the templates and selecting 2 sets of random units (to see the natural variance)

    # For Fujitsu
        # Do this formalizing and create a simple table - for id/canny/inverse or for id/rot/scale
        # Do the invariance analysis for id/rot/scale
        # Make slides with heatmaps, inv results and tease these results - do not explain yet the theory, still working on it

    # Todo: think - do want invariant neurons on or off - what does this mean with the templates framework?
    # Todo: intuition - invariant neurons on - this is exactly what we want (invariance).
    # Todo: therefore redo to not allow invariant neurons (perhaps with a flag - maybe we want depending on data)
    # Loose theory/intuition:
        # It is useufl to build templates at various levels of abstraction for single corruptions because of the "other stuff".
        # This allows some first layer neurons to do something other than be templates. But, by coincidence, this
        # also gives you generalisation to composition.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/EMNIST/',
                        help="path to directory containing directories of different corruptions")
    parser.add_argument('--ckpt-path', type=str, default='/om2/user/imason/compositions/ckpts/EMNIST/',
                        help="path to directory to load checkpoints from")
    parser.add_argument('--save-path', type=str, default='/om2/user/imason/compositions/results/EMNIST/',
                        help="path to directory to save to")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--n-workers', type=int, default=4, help="number of workers (PyTorch)")
    parser.add_argument('--pin-mem', action='store_true', help="set to turn pin memory on (PyTorch)")
    parser.add_argument('--cpu', action='store_true', help="set to train with the cpu (PyTorch) - untested")
    parser.add_argument('--check-if-run', action='store_true', help="If set, skips corruptions for which training has"
                                                                    " already been run. Useful for slurm interruption")
    args = parser.parse_args()

    # Set device
    if args.cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # Create unmade directories
    mkdir_p(args.save_path)

    main(args.data_root, args.ckpt_path, args.save_path, args.total_n_classes, args.batch_size, args.n_workers,
         args.pin_mem, dev, args.check_if_run)

