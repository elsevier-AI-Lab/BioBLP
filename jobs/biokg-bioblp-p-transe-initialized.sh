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
        --model=transe \
        --dimension=512 \
        --loss_fn=marginranking \
        --loss_margin=7.234906889602847 \
        --regularizer=0.0006031667561379036 \
        --freeze_pretrained_embeddings=True \
        --learning_rate=0.03569964236328523 \
        --num_epochs=100 \
        --batch_size=256 \
        --eval_batch_size=64 \
        --num_negatives=512 \
        --from_checkpoint=models/394htt2x \
        --log_wandb=True \
        --notes="TransE BioBLP-P initialized with 394htt2x"

# Keep files generated during job
RESULTS_FOLDER=$HOME/$PROJ_FOLDER-$OUT_FOLDER
mkdir -p $RESULTS_FOLDER
cp -r $TMPDIR/$PROJ_FOLDER/$OUT_FOLDER/* $RESULTS_FOLDER
