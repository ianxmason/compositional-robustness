### To just run right now
For cross-entropy/contrastive: run the correct line in train_monolithic.sh

For modules: first run train_identity.sh to pretrain the network on clean data.
Then run train_modules.sh to train the modules

For ImgSpace: first run the correct line in train_modules.sh to train the Autoencoders
Then run the correct line in train_monolithic.sh to train the classifiers on top.

These should create ckpts and logs for each method. Then run test.sh setting --dataset, --total-n-classes and --experiment to the correct values

Then run comparison_plots.py and learning_curves.py on polestar to get the result plots.
Setting the arguments in the python scripts correctly

---
# compositions
Which neural mechanisms underpin combinatorial generalisation?

### Dependencies
Currently using which container
May be good to make container specific to project

Also can say if not using slurm, in the scripts remove everything before ```python train.py```, you may want to add CUDA_VISIBLE_DEVICES or however you specify gpu.
E.g. change ```singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "EMNIST" --experiment "CrossEntropy" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run```
To ```CUDA_VISIBLE_DEVICES=0 python train.py --dataset "EMNIST" --experiment "CrossEntropy" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run```

### Openmind/Slurm
Run files in the scripts folder from the src directory (e.g. sbatch scripts/train_compositions.sh)

### To run

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
