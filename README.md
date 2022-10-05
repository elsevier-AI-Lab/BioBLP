# discovery-LP
Initial repository hosting code &amp; experiments of the Discovery Lab around link prediction (and more..).


## Setup

Common tasks are managed with a **Makefile**. 

The python dependencies for this project are managed with [Poetry](https://python-poetry.org). For development, it is recommended to use Poetry, but the repository contains a `requirements.txt` to support alternative installations.


### Setup and install python project dependencies
The following only needs to be run once, for initial setup.


If on Entellect JupyterHub environment:
```shell
   make setup_jh_env
```

Otherwise:
```shell
    make setup
    source $HOME/.poetry/env
    make install
    make create_ipython_kernel
```

### Benchmark tasks
* Pre-generate the input dataset with flags indicating if they are known or novel links. 
* Run `bioblp.benchmarking.preprocess.py` to prepare BM dataset for ML by shuffling, splits, etc.
* `bioblp.benchmarking.featurize.py` can be used to featurize a list of pair wise entities into vectors composed from individual vector entities.

Custom usage:
```bash
python -m bioblp.benchmarking.preprocess --dataset=resnet --data_path=./data/raw --preprocessed_path=./data/preprocessed --encode_relation_endpoint_type --encode_relation_effect --filter_relation_refcount=3
```


### 20220520 - Archived stuff below

Note, you can open a terminal window in the jupyterhub environment, that enables you to carry on working as in any linux-based/ec2 instance over the command line.

### Setting up github repo/ssh keys
* Setup/Add an ssh key to your github account, and to your ec2 workspace. (Example: https://docs.github.com/en/enterprise-server@3.0/github/authenticating-to-github/connecting-to-github-with-ssh)
* After configuring your ssh keys, simply `git clone git@github.com:elsevier-AI-Lab/discovery-LP.git` to clone the repository
* Checkout desired remote branch as local `git checkout -b <branch_name> remotes/origin/<branch_name>`


### Create conda environment with dependencies

* The Makefile in project root contains targets that can be run to setup the virtual environment. 
* Create the virtual env, and install dependencies
```bash
make create_env
```

* Create jupyter kernel

```bash
make set_kernel
```

* Next, one can open a jupyter notebook from within the project root, and set the appropriate kernel (Example: Python(`bio-blp`))

## Training experiments 

### Fetch the data for training

### Preprocessing

The `bioblp.preprocessing` module contains the code to perform additional preprocessing to the triples.

- Split relations on entity type endpoints (`DirectRegulation_SmallMol_Protein`, `Binding_Protein_Protein`)
- Split relations on effect (`DirectRegulation_Protein_Protein_positive`, `DirectRegulation_Protein_Protein_negative`)
- Filter on refcount threshold (`triple['refcount] >= thresh`)

Default usage:
```bash
python -m bioblp.preprocess --dataset=DSP-mini --data_path=./data/raw --preprocessed_path=./data/preprocessed
```

Custom usage:
```bash
python -m bioblp.preprocess --dataset=DSP-mini --data_path=./data/raw --preprocessed_path=./data/preprocessed --encode_relation_endpoint_type --encode_relation_effect --filter_relation_refcount=3
```

### Preparing the data for training

[Optional] Conditional Filtering of triples based on availability of entity properties

This refers to the filtering of triples based on which source and target entity nodes are known to have retrievable properties (such as Protein: amino acid sequences, SmallMolecule: molecular structure file, etc)

Assuming that the initial queried set of triples is available at `data/preprocessed`, you can use `bioblp.utils.filter_triples` to obtain a set of filtered triples 
- The input raw triples are accompanied by supplementary information such as pubyear, entity types of source and target nodes, etc). These can be produced, for example, with `notebooks/filter_triples.ipynb` or `notebooks/get-triples-subgraph-variants.ipynb`
- Illustration of required fields (however they may be called) in input triples dataset is presented below:  
```
src             edg         tgt         directed    pubyear     refcount    src_type    tgt_type
72057594XX      Biomarker   725474XX    True        2010        1           Protein     Disease      
```
- The conditional filtering requires a list of allowed entity nodes belonging to each entity type of interest. For ready lists, unpack the tarball `data/preprocessed/entity_w_property_lists.tar.gz`

Let's say we have a raw (preprocessed) set of triples of required shape at `data/preprocessed/dsp_mini_triples.tsv`, and the lists with allowed entity nodes organised by entity type are present at default location of `data/preprocessed/entity_w_property_lists`. Then, the following command can be run to perform the conditional filtering to create a subgraph of triples where each node has a retrievable property

```sh
python -m bioblp.utils.filter_triples --dataset=dsp_mini_triples-preprocessed --data_path=data/preprocessed --output_path=data/preprocessed --entity_w_property_lists_path=data/preprocessed/entity_w_property_lists
```
#### Data Splits for training 
Assuming that the initial set of raw triples is located at `data/raw`, you can use `bioblp.utils.triples` to split it into training, validation, and test sets. The sets have the following properties:

- Splits are based on `pubyear`, so that the training set contains the oldest edges, up to 80% of the edges.
- All entities and relations in the validation and test sets are contained in the training set.
- The original proportion of relations is preserved in the training set.

Let's say we have a raw set of triples at `data/raw/triples_devgraph.tsv`. The following command will store the splits at `data/processed`:

```sh
python -m bioblp.utils.triples data/raw/triples_devgraph.tsv
```

### Setting up experiment trackers

We use [Weights & Biases](https://wandb.ai/) to track experiments. Included in the requirements is the associated `wandb` library for Python.

When running for the first time, you have to specify your identity so that experiments can be logged in our Team's project at W&B. To do this, run

```sh
wandb login
```

and follow the instructions.



### Running the experiments

Run the following commands to reproduce the experiments.

#### 1. Link prediction baselines trained on the DSP subgraph, without entity properties.

The `data` path contains the `raw` and `processed` folders with triples for training. For example:

```sh
data
├── processed
│   ├── triples_devgraph-test.tsv
│   ├── triples_devgraph-train.tsv
│   └── triples_devgraph-valid.tsv
└── raw
│   └── triples_devgraph.tsv
│
```



The "dataset" is then `triples_devgraph`. 

**Single run:** Use the following command to train a ComplEx model with default hyperparameters:

```sh
python -m bioblp.train --data_path=./data --dataset=triples_devgraph --model=complex
```

Replace `complex` by `transe` to train a TransE model.



If you want to do some tests without logging to wandb, add the flag `--offline`.



**Hyperparameter search:** To find the best set of hyperparameters, pass the `search` command as well as the number of trials to run. For example:

```sh
python -m bioblp.train --data_path=./data --dataset=triples_devgraph --model=complex --command=search --num_trials=10
```



