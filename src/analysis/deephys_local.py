import argparse
import numpy as np
import os
import deephys as dp


def main(dp_path, dataset, total_n_classes):
    classes = tuple([str(i) for i in range(47)])  # 47 for EMNIST
    num_module_neurons = {"Contrast": 64, "GaussianBlur": 64, "ImpulseNoise": 3, "Invert": 128, "Rotate90": 256,
                          "Swirl": 128}

    for test_corruption, num_neurons in num_module_neurons.items():
        dp_model = dp.model(
            name=os.path.join(dp_path, f"{dataset}-{test_corruption}-network"),
            layers={  # include any additional layers
                "pre_module_layer": num_neurons,
                "post_module_layer": num_neurons,
                "output": total_n_classes,
            },
            classification_layer="output"
        )
        dp_model.save()

        data = np.load(os.path.join(dp_path, f"{dataset}-{test_corruption}.npz"), allow_pickle=True)
        all_images = data["all_images"]
        all_cats = data["all_cats"]
        all_pre_mods = data["all_pre_mods"]
        all_post_mods = data["all_post_mods"]
        all_outputs = data["all_outputs"]

        test = dp.import_test_data(
                    name=os.path.join(dp_path, f"{dataset}-{test_corruption}"),
                    pixel_data=all_images,  # Images resized to 32x32 pixels
                    ground_truths=all_cats.tolist(),  # Labels
                    classes=classes,  # List with all category names
                    state=[all_pre_mods, all_post_mods, all_outputs],  # List with neural activity
                    model=dp_model
                )
        test.save()

        data = np.load(os.path.join(dp_path, f"{dataset}-Identity-With-{test_corruption}-Module.npz"), allow_pickle=True)
        all_images = data["all_images"]
        all_cats = data["all_cats"]
        all_pre_mods = data["all_pre_mods"]
        all_post_mods = data["all_post_mods"]
        all_outputs = data["all_outputs"]

        test = dp.import_test_data(
            name=os.path.join(dp_path, f"{dataset}-Identity-With-{test_corruption}-Module"),
            pixel_data=all_images,  # Images resized to 32x32 pixels
            ground_truths=all_cats.tolist(),  # Labels
            classes=classes,  # List with all category names
            state=[all_pre_mods, all_post_mods, all_outputs],  # List with neural activity
            model=dp_model
        )
        test.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args to test networks on all corruptions in a given directory.')
    parser.add_argument('--dataset', type=str, default='EMNIST', help="which dataset to use")
    parser.add_argument('--deephys-path', type=str,
                        help="path to directory to save checkpoints")
    parser.add_argument('--seed', type=int, default=38164641, help="random seed")
    parser.add_argument('--total-n-classes', type=int, default=47, help="output size of the classifier")
    args = parser.parse_args()

    variance_dir_name = f"seed-{args.seed}"
    args.deephys_path = os.path.join(args.deephys_path, args.dataset, variance_dir_name)

    main(args.deephys_path, args.dataset, args.total_n_classes)
