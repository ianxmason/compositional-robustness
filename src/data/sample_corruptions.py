import os
import pickle
import random
from itertools import combinations, permutations

random.seed(2778)
root_dir = "/om2/user/imason/compositions/datasets/"
output_dir = os.path.join(root_dir, "EMNIST4/")

# Which corruptions to use - EMNIST4
base_corruption_names = ['Contrast', 'GaussianBlur', 'ImpulseNoise', 'Invert', 'Mirror', 'Rotate180', 'Rotate90']

# All elemental corruptions. (8 base elementals)
corruption_names = [['Identity']]
corruption_names += [[corruption] for corruption in base_corruption_names]

# All pairs. (7P2 = 42)
corruption_names += [list(corruptions) for corruptions in permutations(base_corruption_names, 2)]

# For triples and up sample every combination and take one random order
# (7C3 = 7C4 = 35), (7C5 = 21), (7C6 = 7), (7C7 = 1)
for n in range(3, len(base_corruption_names) + 1):
    all_combinations = [list(corruptions) for corruptions in combinations(base_corruption_names, n)]
    for combination in all_combinations:
        random.shuffle(combination)  # in place
    corruption_names += all_combinations

# Save as pickle (for easy loading). If need to view the entire list can print the pickle easily.
with open(os.path.join(output_dir, "corruption_names.pkl"), "wb") as f:
    pickle.dump(corruption_names, f)

print(corruption_names)
print("Saved {} corruptions to {}".format(len(corruption_names), os.path.join(output_dir, "corruption_names.pkl")))

