# TANL: Structured Prediction as Translation between Augmented Natural Languages

Code for the paper "[Structured Prediction as Translation between Augmented Natural Languages](http://arxiv.org/abs/2101.05779)" (ICLR 2021) and [fine-tuned multi-task model](#fine-tuned-multi-task-model).

If you use this code, please cite the paper using the bibtex reference below.
```
@inproceedings{tanl,
    title={Structured Prediction as Translation between Augmented Natural Languages},
    author={Giovanni Paolini and Ben Athiwaratkun and Jason Krone and Jie Ma and Alessandro Achille and Rishita Anubhai and Cicero Nogueira dos Santos and Bing Xiang and Stefano Soatto},
    booktitle={9th International Conference on Learning Representations, {ICLR} 2021},
    year={2021},
}
```


## Requirements

- Python 3.6+
- PyTorch (tested with version 1.7.1)
- Transformers (tested with version 4.0.0)
- NetworkX (tested with version 2.5, only used in coreference resolution)

You can install all required Python packages with `pip install -r requirements.txt`


## Datasets

By default, datasets are expected to be in `data/DATASET_NAME`.
Dataset-specific code is in [datasets.py](datasets.py).

The CoNLL04 and ADE datasets (joint entity and relation extraction) in the correct format can be downloaded using https://github.com/markus-eberts/spert/blob/master/scripts/fetch_datasets.sh.
For other datasets, we provide sample processing code which does not necessarily match the format of publicly available versions (we do not plan to adapt the code to load datasets in other formats).



## Running the code

Use the following command:
`python run.py JOB`

The `JOB` argument refers to a section of the config file, which by default is `config.ini`.
A [sample config file](config.ini) is provided, with settings that allow for a faster training and less memory usage than the settings used to obtain the final results in the paper.

For example, to replicate the paper's results on CoNLL04, have the following section in the config file:
```
[conll04_final]
datasets = conll04
model_name_or_path = t5-base
num_train_epochs = 200
max_seq_length = 256
max_seq_length_eval = 512
train_split = train,dev
per_device_train_batch_size = 8
per_device_eval_batch_size = 16
do_train = True
do_eval = False
do_predict = True
episodes = 1-10
num_beams = 8
```
Then run `python run.py conll04_final`.
Note that the final results will differ slightly from the ones reported in the paper, due to small code changes and randomness.

Config arguments can be overwritten by command line arguments.
For example: `python run.py conll04_final --num_train_epochs 50`.


### Additional details

If `do_train = True`, the model is trained on the given train split (e.g., `'train'`) of the given datasets.
The final weights and intermediate checkpoints are written in a directory such as `experiments/conll04_final-t5-base-ep200-len256-b8-train`, with one subdirectory per episode.
Results in JSON format are also going to be saved there.

In every episode, the model is trained on a different (random) permutation of the training set.
The random seed is given by the episode number, so that every episode always produces the same exact model.

Once a model is trained, it is possible to evaluate it without training again.
For this, set `do_train = False` or (more easily) provide the `-e` command-line argument: `python run.py conll04_final -e`.

If `do_eval = True`, the model is evaluated on the `'dev'` split.
If `do_predict = True`, the model is evaluated on the `'test'` split.


### Arguments

The following are the most important command-line arguments for the `run.py` script.
Run `python run.py -h` for the full list.

- `-c CONFIG_FILE`: specify config file to use (default is `config.ini`)
- `-e`: only run evaluation (overwrites the setting `do_train` in the config file)
- `-a`: evaluate also intermediate checkpoints, in addition to the final model
- `-v` : print results for each evaluation run
- `-g GPU`: specify which GPU to use for evaluation

The following are the most important arguments for the config file. 
See the [sample config file](config.ini) to understand the format.

- `datasets` (str): comma-separated list of datasets for training
- `eval_datasets` (str): comma-separated list of datasets for evaluation (default is the same as for training)
- `model_name_or_path` (str): path to pretrained model or model identifier from [huggingface.co/models](https://huggingface.co/models) (e.g. `t5-base`)
- `do_train` (bool): whether to run training (default is False)
- `do_eval` (bool): whether to run evaluation on the `dev` set (default is False)
- `do_predict` (bool): whether to run evaluation on the `test` set (default is False)
- `train_split` (str): comma-separated list of data splits for training (default is `train`)
- `num_train_epochs` (int): number of train epochs
- `learning_rate` (float): initial learning rate (default is 5e-4)
- `train_subset` (float > 0 and <=1): portion of training data to effectively use during training (default is 1, i.e., use all training data)
- `per_device_train_batch_size` (int): batch size per GPU during training (default is 8)
- `per_device_eval_batch_size` (int): batch size during evaluation (default is 8; only one GPU is used for evaluation)
- `max_seq_length` (int): maximum input sequence length after tokenization; longer sequences are truncated
- `max_output_seq_length` (int): maximum output sequence length (default is `max_seq_length`)
- `max_seq_length_eval` (int): maximum input sequence length for evaluation (default is `max_seq_length`)
- `max_output_seq_length_eval` (int): maximum output sequence length for evaluation (default is `max_output_seq_length` or `max_seq_length_eval` or `max_seq_length`)
- `episodes` (str): episodes to run (default is `0`; an interval can be specified, such as `1-4`; the episode number is used as the random seed)
- `num_beams` (int): number of beams for beam search during generation (default is 1)
- `multitask` (bool): if True, the name of the dataset is prepended to each input sentence (default is False)

See [arguments.py](arguments.py) and [transformers.TrainingArguments](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py) for additional config arguments.


## Fine-tuned multi-task model

The weights of our multi-task model (released under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)) can be downloaded here: https://tanl.s3.amazonaws.com/tanl-multitask.zip

Extract the zip file in the `experiments/` directory. This will create a subdirectory called `multitask-t5-base-ep50-len512-b8-train,dev-overlap96`. For example, to test the multi-task model on the CoNLL04 dataset, run `python run.py multitask -e --eval_datasets conll04`.

Note that: the `multitask` job is defined in [config.ini](config.ini); the `-e` flag is used to skip training and run evaluation only; the name of the subdirectory containing the weights is compatible with the definition of the `multitask` job.

The multi-task model was fine-tuned as described in the paper. The results differ slightly from what is reported in the paper due to small code changes.


## Licenses

The code of this repository is released under the [Apache 2.0 license](LICENSE).
The weights of the [fine-tuned multi-task model](#fine-tuned-multi-task-model) are released under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
