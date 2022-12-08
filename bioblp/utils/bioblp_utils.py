from typing import Mapping

import bioblp.models.encoders as encoders


def build_encoders(dim: int,
                   entity_to_id: Mapping[str, int],
                   protein_data: str = None,
                   protein_learning_rate: float = None,
                   molecule_data: str = None,
                   molecule_learning_rate: float = None,
                   text_data: str = None,
                   text_learning_rate: float = None,
                   ) -> encoders.PropertyEncoderRepresentation:
    if not any((protein_data, molecule_data, text_data)):
        raise ValueError("No entity data provided to build encoders.")

    encoders_list = []

    if protein_data:
        protein_encoder = encoders.PretrainedLookupTableEncoder(
            file_path=protein_data,
            dim=dim,
            learning_rate=protein_learning_rate
        )
        encoders_list.append(protein_encoder)

    if molecule_data:
        molecule_encoder = encoders.MoleculeEmbeddingEncoder(
            file_path=molecule_data,
            dim=dim,
            learning_rate=molecule_learning_rate
        )
        encoders_list.append(molecule_encoder)

    if text_data:
        text_encoder = encoders.TransformerTextEncoder(
            file_path=text_data,
            dim=dim,
            learning_rate=text_learning_rate
        )
        encoders_list.append(text_encoder)

    entity_encoders = encoders.PropertyEncoderRepresentation(
        dim=dim,
        entity_to_id=entity_to_id,
        encoders=encoders_list
    )

    return entity_encoders
