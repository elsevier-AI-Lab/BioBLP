# Benchmark

## Experiment preparation
Command to prepare experimental data, given config file. This script will load the raw benchmark dataset, perform negative sampling, generate features and splits:

```bash
python bioblp/benchmarking/experiment.py \
    --conf=conf/dpi-benchmark-cv-20230413.toml \
    --override_data_root=./ \
    --bm_file=data/benchmarks/transductive/dpi_fda.tsv \
    --n_proc=1
```

You can execute the steps in `experiment.py` individually with the below.

1. Negative sampling.
```bash
python bioblp/benchmarking/preprocess.py \
    --bm_data_path=data/benchmarks/experiments/DPI/1681398697/features/raw.pt \
    --kg_triples_dir=data/benchmarks/experiments/encoders/rotate/training_triples/ \
    --num_negs_per_pos=10 \
    --outdir=data/benchmarks/experiments/DPI/1681398697/sampled/ \
    --override_run_id=1681398697
```

2. Generate features.

```bash
python bioblp/benchmarking/featurise.py \
    --conf=conf/dpi-benchmark-cv-20230413.toml \
    --bm_file=data/benchmarks/experiments/DPI/1681398697/sampled/dpi_fda_p2n-1-10.tsv \
    --override_data_root=./ \
    --override_run_id=1681398697

```

3. Preparing data splits for cross validation.

```bash
python bioblp/benchmarking/split.py \
    --conf=conf/dpi-benchmark-cv-20230413.toml \
    --data=data/benchmarks/experiments/DPI/1681398697/features/raw.pt \
    --outdir=data/benchmarks/experiments/DPI/1681398697/splits/ \
    --n_folds=5 \
    --override_data_root=./ \
    --override_run_id=1681398697
```

## Model training

Sample command for `train.py`. This script performs the training procedure for one model configuration, on one particular data split.
```bash
python bioblp/benchmarking/train.py \
    --model_clf=RF \
    --model_feature=complex \
    --feature_dir=data/benchmarks/experiments/dpi_fda/1681301749/features/ \
    --splits_path=data/benchmarks/experiments/dpi_fda/1681301749/splits/train-test-split.pt \
    --split_idx=0 \
    --n_iter=3 \
    --refit_params=AUCPR,AUCROC \
    --outdir=data/benchmarks/experiments/dpi_fda/1681301749/models/ \
    --model_label=complex__RF \
    --timestamp=1681301749 \
    --wandb_tag=dev
```

The `train_runner` script contains the procedure to run a full experiment, given a configuration file. This will perform the complete CV routine for all model configurations contained in the config file. Also supports multiprocessing through the `--n_proc` flag. For example, 
```bash
python bioblp/benchmarking/train_runner.py \
    --conf conf/dpi-benchmark-cv-20230413.toml \
    --override_data_root=./ \
    --override_run_id=1681398697 \
    --tag=dpi-20230413 \
    --n_proc=5
```

In its current implementations here, the multiprocessing capability conflicts with PyTorch on GPU. For MLP models using GPU, we recommend setting `--n_proc=1`:
```bash
python bioblp/benchmarking/train_runner.py \
    --conf conf/dpi-benchmark-cv-20230413-mlp.toml \
    --override_data_root=./ \
    --override_run_id=1681398697 \
    --tag=dpi-20230413 \
    --n_proc=1
```

## WandB logging

By default logging to WandB is turned off. Change the assignments to `LOG_WANDB = True` in `train.py` for logging.