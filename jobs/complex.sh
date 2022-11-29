#!/bin/bash
#SBATCH --job-name=bioblp-complex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

PROJ_FOLDER=bioblp
OUT_FOLDER=models

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate bioblp

python -m bioblp.train \
--train_triples=data/biokgb/graph/biokg.links-train.csv \
--valid_triples=data/biokgb/graph/biokg.links-valid.csv \
--test_triples=data/biokgb/graph/biokg.links-test.csv \
--protein_data=data/biokgb/properties/protein_prottrans_embeddings.pt \
--search_eval_batch_size=True \
--search_train_batch_size=True \
--dimension=1024 \
--num_epochs=1 \
--batch_size=128 \
--log_wandb=True \
--notes="BioBLP-prot search batch sizes"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
