from torch import nn

from bioblp.models.encoders import PropertyEncoder
from bioblp.models.encoders import PretrainedLookupTableEncoder


class KGEEncoder(PropertyEncoder):
    def __init__(self):
        super().__init__()
        pass


class EntityPairEncoder(nn.Module):
    def __init__(self, left_encoder: PretrainedLookupTableEncoder, right_encoder: PretrainedLookupTableEncoder, dim: int, entity_to_id: dict):
        # Here we need to run preprocess to get the id mappings etx
        # borrow logic from bioblp

        self.left_encoder: PretrainedLookupTableEncoder = left_encoder
        self.right_encoder: PretrainedLookupTableEncoder = right_encoder

        # batch
        #
        dim_encoders = self.left_encoder.embeddings.shape[-1] + \
            self.right_encoder.embeddings.shape[-1]

        self.linear = nn.Linear(dim_encoders, dim)

    def forward(self, x):
        left = x[:, 0]
        right = x[:, 1]

        left_enc = self.left_encoder(left)
        right_enc = self.right_encoder(right)

        concat_enc = torch.stack([left_enc, right_enc], dim=0)
        return self.linear(concat_enc)


class ClassifierLayer(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.3, n_labels: int = 2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(in_dim, n_labels)

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        return self.clf(x)


class StructuralEClassifier(nn.Module):
    def __init__(self, hidden_enc_dim: int = 128, dropout: float = 0.3, n_labels: int = 2):
        super().__init__()
        self.smiles_encoder = PretrainedLookupTableEncoder(
            filepath="path_to_smiles_encodings")
        self.protein_seq_encoder = PretrainedLookupTableEncoder(
            filepath="path_to_prot_seq_encodings")

        self.pair_encoder = EntityPairEncoder(left_encoder=self.smiles_encoder,
                                              right_encoder=self.protein_seq_encoder,
                                              dim=hidden_enc_dim)

        self.clf = ClassifierLayer(hidden_enc_dim, dropout, n_labels)

    def forward(self, x):
        x = self.pair_encoder(x)
        return self.clf(x)


class NoiseEClassifier(nn.Module):
    def __init__(self, hidden_enc_dim: int = 128, dropout=0.3, n_labels: int = 2):
        super().__init__()
        self.noise_encoder = PretrainedLookupTableEncoder(
            "path_to_noise_encodings")

        self.pair_encoder = EntityPairEncoder(left_encoder=self.noise_encoder,
                                              right_encoder=self.noise_encoder,
                                              dim=hidden_enc_dim)
        self.clf = ClassifierLayer(hidden_enc_dim, dropout, n_labels)

    def forward(self, x):
        x = self.pair_encoder(x)
        return self.clf(x)


class KGEClassifier(nn.Module):
    def __init__(self, hidden_enc_dim: int = 128, dropout=0.3, n_labels: int = 2):
        super().__init__()
        self.kge_encoder = KGEEncoder(filepath="path_to_kge_model")

        self.pair_encoder = EntityPairEncoder(left_encoder=self.kge_encoder,
                                              right_encoder=self.kge_encoder,
                                              dim=hidden_enc_dim)

        self.clf = ClassifierLayer(hidden_enc_dim, dropout, n_labels)

    def forward(self, x):
        x = self.pair_encoder(x)
        return self.clf(x)
