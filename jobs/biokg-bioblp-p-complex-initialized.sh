#!/bin/bash
#SBATCH --job-name=bioblp-p
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

git checkout freeze-embeddings

python -m bioblp.train \
        --train_triples=data/biokgb/graph/biokg.links-train.csv \
        --valid_triples=data/biokgb/graph/biokg.links-valid.csv \
        --test_triples=data/biokgb/graph/biokg.links-test.csv \
        --protein_data=data/biokgb/properties/protein_prottrans_embeddings_24_12.pt \
        --model=complex \
        --dimension=256 \
        --loss_fn=bcewithlogits \
        --regularizer=7.54616261352196e-05 \
        --freeze_pretrained_embeddings=True \
        --learning_rate=0.344274380857535 \
        --num_epochs=100 \
        --batch_size=512 \
        --eval_batch_size=64 \
        --num_negatives=512 \
        --from_checkpoint=models/1e9b4f4o \
        --log_wandb=True \
        --notes="ComplEx BioBLP-P initialized with 1e9b4f4o"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
