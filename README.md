# Compositions/Invariance is Not Enough
Which neural mechanisms underpin combinatorial generalisation?

## Dependencies
The dependencies for this project can be found in ```docker/Dockerfile```. The Docker
image is also currently available on [DockerHub](https://hub.docker.com/repository/docker/ianxmason/pytorch/general).

The code should also be runnable in a conda environment with a fairly standard PyTorch set up using python3. 
This is untested, but the key packages should be: ```pytorch```, ```torchvision```, ```numpy```, ```matplotlib```, 
```pillow```, ```pandas```, ```seaborn```, ```scikit-image```, ```opencv-python```, ```scipy```.

#### A Note on Slurm & Singularity
The code in this repo is written to be run on a cluster using [Slurm](https://slurm.schedmd.com/documentation.html) and
[Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html). The Docker image is converted into a Singularity image.

The code is almost all executed via ```sbatch``` and the shell scripts in ```src/scripts```.
- If using Singularity and Slurm you should only need to change the bind paths (```-B /om,/om2/user/$USER```) 
and path to the Singularity image (```/om2/user/imason/singularity/imason-pytorch.simg```) in these scripts.
- If working on a system without these tools it should be possible to run the code by taking the commands in these scripts
  (beginning ```singularity exec```) and running only the part of the command that starts with ```python```.
  - Commands that contain ```$SLURM_ARRAY_TASK_ID``` will need to be run with ```$SLURM_ARRAY_TASK_ID``` set to each 
    of the values in the range shown in ```#SBATCH --array```.

## Data Generation

Everything is run from the ```src``` directory.

#### EMNIST
First download the base dataset to the directory where you want to generate the training data. 
In an interactive python session run:
```python
from data.emnist import _get_emnist_datasets
_get_emnist_datasets("<path to data root>")
```
This should create a directory ```EMNIST/raw``` in ```<path to data root>``` containing the base dataset.

To generate the corruptions for training.
Run the first python command in ```generate_data.sh```.
You will need to add the ```--data-root``` flag to match your ```<path to data root>```.
```
sbatch scripts/generate_data.sh
```

#### CIFAR10
CIFAR10 follows a similar process as EMNIST. First download the base data
```python
from data.cifar import _get_cifar_datasets
_get_cifar_datasets("<path to data root>")
```
You should see the directory ```cifar-10-batches-py``` and file ```cifar-10-python.tar.gz``` in the data root directory.

Then run the second python command in ```generate_data.sh```. Again changing ```--data-root``` as needed.
```
sbatch scripts/generate_data.sh
```

#### FACESCRUB
FaceScrub cannot be directly downloaded, you can request access to the dataset [here](http://vintage.winklerbros.net/facescrub.html).
We used an internal version of the dataset which is a subset of the original where identities with less than 100 images
are discarded following [this paper](https://www.pnas.org/doi/epdf/10.1073/pnas.1800901115) (section _Simulation 2_). 
We end up with 388 identities (classes). These are in a flat directory structure in ```<path to data root>/FaceScrub/``` with the images named as ```<identity number>_<image number>.jpg```. 
Where ```<identity number>``` is between 100 and 487 (inclusive).

Given this directory structure you can run the third python command in ```generate_data.sh```. Again changing ```--data-root``` as needed.
```
sbatch scripts/generate_data.sh
```

#### Generating Corruption Names for Training
Running the following command will generate a file called ```corruption_names.pkl``` in the ```data_root/EMNIST``` directory.
You will need to change the absolute path to your data root in line 8 before running.
```python data/sample_corruptions.py ```
This file is a random sample to get all the combinations and permuations of corruptions that will be used for testing our models.

CIFAR and FACESCRUB also need a corruption names file. The sampling script can be run again, but easiest is to copy the EMNIST file
In your data root run:
```
cp EMNIST/corruption_names.pkl CIFAR/
cp EMNIST/corruption_names.pkl FACESCRUB/
```

#### Checking the Data Generation
After generation you should have in your data root directories called 

- ```EMNIST/```
- ```CIFAR/```
- ```FACESCRUB/```

Inside each of these directories you can run the following command to count the number of files
```du -a | cut -d/ -f2 | sort | uniq -c | sort -nr```

This should output something like this for each of the datasets (the numbers are important)
![file count](assets/file_count.png)

You can also inspect the data visually
You can look directly in the folders in the data root or you can use the script in the tests directory
```check_data.py``` (manually changing the root_dir and save_path) to generate a much smaller version of the datasets 
which can be copied and inspected locally


## Training
Everything is run from the ```src``` directory.

Figure showing monolithic, modular, identity training. 
Broken into step_one, step_two. Which scripts to run etc.

Description of how to do the training.

Something like
For cross-entropy/contrastive: run the correct line in train_monolithic.sh

For modules: first run train_identity.sh to pretrain the network on clean data.
Then run train_modules.sh to train the modules

For ImgSpace: first run the correct line in train_modules.sh to train the Autoencoders
Then run the correct line in train_monolithic.sh to train the classifiers on top.

These should create ckpts and logs for each method. Then run test.sh setting --dataset, --total-n-classes and --experiment to the correct values

Then run comparison_plots.py and learning_curves.py on polestar to get the result plots.
Setting the arguments in the python scripts correctly



## Testing & Analysis
Description of how to the testing and the plotting.

Deephys? (Or delete deephys files?)

## Hyperparameter Searching
Difference for hyperparameter searching 
E.g. change file name. step_one_test.py

Link to sheet or csv of hyperparameter searching results.

## Citation


### To run (OLD)

One time things
- Generate corruptions.pkl (should be length 167, mostly for testing, also used in training)
- Generate data (just elementals??)

Training (a figure to explain)
- Monolithic training (contrastive, cross entropy)
- Modular training (modulels, autoencoders)
- Identity training (pre-modules, post-autoencoders)

Testing
- Explain test.sh

Analysis
- Plotting etc.


