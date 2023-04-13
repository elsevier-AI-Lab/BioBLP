# Benchmark

Command to prepare experimental data, given config file:
```bash
python bioblp/benchmarking/experiment.py \
    --conf conf/dpi-benchmark-cv-20230413.toml \
    --override_data_root=./ \
    --bm_file=data/benchmarks/transductive/dpi_fda.tsv \
    --n_proc=1
```


Preparing data splits
```bash
python bioblp/benchmarking/split.py \
    --data=data/benchmarks/experiments/DPI/1681398697/features/raw.pt \
    --outdir=data/benchmarks/experiments/DPI/1681398697/splits/ \
    --n_folds=5
```

Sample command for `train.py`

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

Sample to run using train_runner, for Logistic Regression and Random Forest
```bash
python bioblp/benchmarking/train_runner.py \
    --conf conf/dpi-benchmark-cv-20230413.toml \
    --override_data_root=./ \
    --override_run_id=1681398697 \
    --tag=dpi-20230413 \
    --n_proc=5
```

For MLP models using GPU
```bash
python bioblp/benchmarking/train_runner.py \
    --conf conf/dpi-benchmark-cv-20230413-mlp.toml \
    --override_data_root=./ \
    --override_run_id=1681398697 \
    --tag=dpi-20230413 \
    --n_proc=1
```