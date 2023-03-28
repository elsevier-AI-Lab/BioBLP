#!/bin/bash
#SBATCH --job-name=biokg-bioblp-m-rotate-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=72:00:00
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
wandb agent --count 1 discoverylab/bioblp/oouxbq6p

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
