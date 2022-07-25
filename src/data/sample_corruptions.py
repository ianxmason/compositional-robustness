import os
import pickle
import random
from itertools import combinations, permutations

random.seed(2778)
root_dir = "/om2/user/imason/compositions/datasets/"
output_dir = os.path.join(root_dir, "EMNIST3/")

# Which corruptions to use
# EMNIST2
# base_corruption_names = ['impulse_noise', 'inverse', 'gaussian_blur', 'rotate_fixed', 'scale', 'thinning']
# EMNIST3
base_corruption_names = ['impulse_noise', 'inverse', 'gaussian_blur', 'rotate_fixed', 'scale', 'shear_fixed']

# All elemental corruptions
corruption_names = [['identity']]
corruption_names += [[corruption] for corruption in base_corruption_names]

# All pairs
corruption_names += [list(corruptions) for corruptions in permutations(base_corruption_names, 2)]

# All triples - expensive but useful to see if some triples are solvable. From 6 base corrs this is 120 permutations.
corruption_names += [list(corruptions) for corruptions in permutations(base_corruption_names, 3)]
# Alternate - sample 4 triples
# # hardcode the non-geometric and geometric corruptions then sample 2 more
# triple_sample = [(base_corruption_names[0], base_corruption_names[1], base_corruption_names[2]),
#                  (base_corruption_names[3], base_corruption_names[4], base_corruption_names[5])]
# possible_triples = list(combinations(base_corruption_names, 3))
# possible_triples.remove(triple_sample[0])
# possible_triples.remove(triple_sample[1])
# triple_sample += random.sample(possible_triples, 2)
# # Sample 2 orders of corruptions for each triple
# for triple in triple_sample:
#     corruption_names += [list(corruptions) for corruptions in random.sample(list(permutations(triple)), 2)]

# Sample 3 quadruples
possible_quadruples = list(combinations(base_corruption_names, 4))
quadruple_sample = random.sample(possible_quadruples, 3)
# Sample 2 orders of corruptions for each quadruple
for quadruple in quadruple_sample:
    corruption_names += [list(corruptions) for corruptions in random.sample(list(permutations(quadruple)), 2)]

# Sample 2 quintuples
possible_quintuples = list(combinations(base_corruption_names, 5))
quintuple_sample = random.sample(possible_quintuples, 2)
# Sample 2 orders of corruptions for each quintuples
for quintuple in quintuple_sample:
    corruption_names += [list(corruptions) for corruptions in random.sample(list(permutations(quintuple)), 2)]

# Sample 2 orders of the sextuple
corruption_names += [list(corruptions) for corruptions in random.sample(list(permutations(base_corruption_names)), 2)]

# Save as pickle (for easy loading). If need to view the entire list can print the pickle easily.
with open(os.path.join(output_dir, "corruption_names.pkl"), "wb") as f:
    pickle.dump(corruption_names, f)

print(corruption_names)
print("Saved {} corruptions to {}".format(len(corruption_names), os.path.join(output_dir, "corruption_names.pkl")))

