#!/bin/bash
#SBATCH --job-name=bioblp-complex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
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
--search_eval_batch_size=True \
--batch_size=64 \
--num_negatives=512 \
--learning_rate=2e-5 \
--warmup_fraction=0.05 \
--dimension=256 \
--num_epochs=1 \
--log_wandb=True \
--notes="BioBLP-D 1 epoch test with in-batch negatives"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
