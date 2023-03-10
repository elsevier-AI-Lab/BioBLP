#!/bin/bash
#SBATCH --job-name=complex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
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
        --model=complex \
        --dimension=256 \
        --loss_fn=bcewithlogits \
        --learning_rate=0.3595182058943781 \
        --regularizer=3.7579365087382533e-05 \
        --num_epochs=100 \
        --batch_size=256 \
        --eval_batch_size=64 \
        --num_negatives=512 \
        --log_wandb=True \
        --notes="ComplEx best hparams, rep"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
