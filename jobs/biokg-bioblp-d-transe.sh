#!/bin/bash
#SBATCH --job-name=bioblp-d-transe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1

PROJ_FOLDER=bioblp
OUT_FOLDER=models

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate bioblp

git checkout fix_bioblp_init

python -m bioblp.train \
        --train_triples=data/biokgb/graph/biokg.links-train.csv \
        --valid_triples=data/biokgb/graph/biokg.links-valid.csv \
        --test_triples=data/biokgb/graph/biokg.links-test.csv \
        --text_data=data/biokgb/properties/biokg_meshid_to_descr_name.tsv \
        --model=transe \
        --dimension=512 \
        --loss_fn=marginranking \
        --loss_margin=8.155451890616455 \
        --optimizer=adam \
        --learning_rate=2e-5 \
        --warmup_fraction=0.05 \
        --num_epochs=100 \
        --batch_size=1024 \
        --eval_batch_size=64 \
        --num_negatives=512 \
        --in_batch_negatives=True \
        --log_wandb=True \
        --notes="TransE BioBLP-D, margin from sage-shadow-1047"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
