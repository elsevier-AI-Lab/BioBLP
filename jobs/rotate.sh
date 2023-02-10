#!/bin/bash
#SBATCH --job-name=bioblp-complex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1

PROJ_FOLDER=bioblp
OUT_FOLDER=models

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate bioblp

git checkout disease-encoder

python -m bioblp.train \
        --train_triples=data/biokgb/graph/biokg.links-train.csv \
        --valid_triples=data/biokgb/graph/biokg.links-valid.csv \
        --test_triples=data/biokgb/graph/biokg.links-test.csv \
        --text_data=data/biokgb/properties/biokg_meshid_to_descr_name.tsv \
        --model=rotate \
        --dimension=256 \
        --loss_fn=crossentropy \
        --optimizer=adam \
        --learning_rate=2e-5 \
        --warmup_fraction=0.05 \
        --num_epochs=10 \
        --batch_size=1024 \
        --search_eval_batch_size=True \
        --eval_every=1 \
        --num_negatives=512 \
        --in_batch_negatives=True \
        --log_wandb=True \
        --notes="BioBLP-D 10 epoch test"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
