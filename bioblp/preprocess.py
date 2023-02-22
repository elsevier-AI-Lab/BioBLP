import torch
import pandas as pd
import numpy as np
import bio_embeddings
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder, prottrans_t5_embedder, esm_embedder


# Here we can change the Protein Embedder to w/e we want from the above.
# TODO: An experiment with t5 embedding
prot_trans_embedder = ProtTransBertBFDEmbedder()


def get_protein_repr(amino_repr):
    """ Here we need to go from a collection of amino-acid embeddings to a full protein embedding

    # Example:
    #
    #   M : (1,1024)
    #   A : (1,1024)
    #   S : (1,1024)
    #
    #  Output: An aggregated representation for proteins
    #
    #  Type: Dict(protein_id: (embedding))
    #
       e.g Dict(: (LENG8_MOUSE, 1024)) """

    emb_matrix = torch.Tensor(amino_repr)

    # We average over columns
    protein_emb = torch.mean(emb_matrix, dim=0)

    return protein_emb


def get_protein_embedding(path, embedder="prottrans"):
    """
        Wrapper over different protein embedders
    Parameters
    ----------
    embedder: The model to embed proteins
    path: The data path

    Returns
    -------
    """
    print('Im in')

    # Load sequences
    sequence_data = pd.read_csv(path, sep='\t')

    # Sample : Uncomment for testing
    # sequence_data = sequence_data.sample(2)

    # Select correct columns
    sequence_data = sequence_data[['From', 'Sequence']]

    # Embed sequences
    sequence_data['embedding'] = sequence_data['Sequence'].apply(lambda x: prot_trans_embedder.embed(x))

    # Aggregate sequences
    sequence_data['squashed'] = sequence_data['embedding'].apply(lambda x: get_protein_repr(x))


    # Save sequences
    sequence_data.to_csv('../data/processed/uniprot_seq_embeddings.tsv')


get_protein_embedding('../data/uniprot_sequences.tsv')
