# compositions
Which neural mechanisms underpin combinatorial generalisation?

### Openmind/Slurm
Run files in the scripts folder from the src directory (e.g. sbatch scripts/train_compositions.sh)

### To run

Run each line in train.sh on openmind
Should create ckpts and logs for each method

Then run test.sh setting --dataset, --total-n-classes and --experiment to the correct values

Then run comparison_plots.py and learning_curves.py on polestar to get the result plots
Setting the arguments in the python scripts correctly



### Analysis Notebooks

[Initial EMNIST Heatmaps.](https://colab.research.google.com/drive/10Kz_uupMghu33v31QhOux7WslLZZVNWa?usp=sharing)
Using a small number of corruptions, creates heatmaps of loss and accuracy for each corruption and some compositions.

[EMNIST Selectivity and Invariance](https://colab.research.google.com/drive/1i9ZehCj3ps7OKnQBAvWzMVezTxJTeENo?usp=sharing)
Using selected corruptions from the heatmaps plot selectivity and invariance for comparison.