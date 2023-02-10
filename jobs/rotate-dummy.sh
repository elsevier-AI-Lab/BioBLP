#!/bin/bash
#SBATCH --job-name=bioblp-rotate-dummy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1

PROJ_FOLDER=bioblp
OUT_FOLDER=models

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate bioblp

git checkout disease-encoder-dummy

python -m bioblp.train \
        --train_triples=data/biokgb/graph/biokg.links-train.csv \
        --valid_triples=data/biokgb/graph/biokg.links-valid.csv \
        --test_triples=data/biokgb/graph/biokg.links-test.csv \
        --text_data=data/biokgb/properties/dummy_biokg_meshid_to_descr_name.tsv \
        --model=rotate \
        --dimension=256 \
        --loss_fn=crossentropy \
        --optimizer=adagrad \
        --regularizer=0.0002757262741946316 \
        --learning_rate=0.07300713133641318 \
        --num_epochs=100 \
        --batch_size=1024 \
        --eval_batch_size=64 \
        --num_negatives=512 \
        --in_batch_negatives=False \
        --log_wandb=True \
        --notes="BioBLP-D RotatE, no descriptions, fixed eval batch size"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
