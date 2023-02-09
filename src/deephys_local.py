import numpy as np
import os
import deephys as dp

dp_path = "/Users/ian/compositions/outputs/deephys/"

dp_model = dp.model(
                name=os.path.join(dp_path, "emnist_simple_net"),
                layers={  # include any additional layers
                    "penultimate_layer": 512,
                    "output": 47,
                },
                classification_layer="output"
            )
dp_model.save()
classes = tuple([str(i) for i in range(47)])  # 47 for EMNIST

for test_corruption in ["Identity", "Contrast"]:
    data = np.load(os.path.join(dp_path, f"EMNIST-{test_corruption}.npz"), allow_pickle=True)
    all_images = data["all_images"]
    all_cats = data["all_cats"]
    all_activs = data["all_activs"]
    all_outputs = data["all_outputs"]

    test = dp.import_test_data(
                name=os.path.join(dp_path, f"EMNIST-{test_corruption}"),  # Ugly, uses test_corruption from last iteration of loop
                pixel_data=all_images,  # Images resized to 32x32 pixels
                ground_truths=all_cats.tolist(),  # Labels
                classes=classes,  # List with all category names
                state=[all_activs, all_outputs],  # List with neural activity
                model=dp_model
            )
    test.save()
