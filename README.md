# Conditional WGAN with full GP
Training scripts for 1D densities 

## Directories
``Data`` contains data generation scripts

``Scripts`` contains training and testing scripts

Remember to run all scripts inside the virtual environment which contains the Torch modules

## Generating the data
Run the following command from inside the ``Data`` directory

```
python3 generate_data.py
```

This will create 3 datasets, one each for tanh, bimodal and swissroll


## Training the network
Run the training scripts from inside the ``Scripts`` folder.

To train with tanh dataset 

```
python3 trainer.py \
        --dataset=tanh \
        --n_epoch=600000 \
        --n_train=2000 \
        --batch_size=2000 \
        --save_freq=5000 \
        --n_critic=20
```

To train with bimodal or swissroll, change the argument value of ``--dataset`` to ``bimodal`` or ``swissroll``. A full description of what each argument means, use the command

```
python3 trainer.py -h
```

The trained networks and associated results are save in a the ``trained_models`` directory.


## Testing networks
To test the trained networks by loading from a saved checkpoint, execute the following command inside the ``Scripts`` directory

```
python3 test.py \
        --dataset=tanh \
        --n_epoch=600000 \
        --n_train=2000 \
        --batch_size=2000 \
        --save_freq=5000 \
        --n_critic=20 \
        --n_test=100000 \
        --ckpt=600000
```
The results are saved inside the appropriate subfolder (default is ``Test_results``) inside the appropriate cGAN subfolder. The checkpoint id can be any checkpioint saved inside the ``checkpoints`` subfolder.
