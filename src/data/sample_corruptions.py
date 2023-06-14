import os
import pickle
import random
import math
from itertools import combinations, permutations

random.seed(2778)
root_dir = "<your_data_root_here>"
output_dir = os.path.join(root_dir, "EMNIST/")

# Which corruptions to use
base_corruption_names = ['Contrast', 'GaussianBlur', 'ImpulseNoise', 'Invert', 'Rotate90', 'Swirl']

# All elemental corruptions. (7 base elementals)
corruption_names = [['Identity']]
corruption_names += [[corruption] for corruption in base_corruption_names]

# All pairs. (6P2 = 30)
corruption_names += [list(corruptions) for corruptions in permutations(base_corruption_names, 2)]

# For triples and above:
# (6C3 = 20) -> sample 2 orders per combination
# (6C4 = 15) -> sample 2 orders per combination
# (6C5 = 6) -> sample 5 orders per combination
# (6C6 = 1) -> sample 30 orders per combination
for n in range(3, len(base_corruption_names) + 1):
    all_combinations = [list(corruptions) for corruptions in combinations(base_corruption_names, n)]
    sample_count = math.ceil(30 / len(all_combinations))  # 30 comes from 6C2=30
    for combination in all_combinations:
        samples = random.sample(list(permutations(combination)), sample_count)
        for sample in samples:
            corruption_names += [list(sample)]

# Save as pickle (for easy loading). If need to view the entire list can print the pickle easily.
with open(os.path.join(output_dir, "corruption_names.pkl"), "wb") as f:
    pickle.dump(corruption_names, f)

print(corruption_names)
print("Saved {} corruptions to {}".format(len(corruption_names), os.path.join(output_dir, "corruption_names.pkl")))

