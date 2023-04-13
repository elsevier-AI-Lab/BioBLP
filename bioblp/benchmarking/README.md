# Benchmark




Sample command for `train.py`

```
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