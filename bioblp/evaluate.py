import os.path as osp

import numpy as np
from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.rank_based_evaluator import _iter_ranks
from pykeen.triples import TriplesFactory
from tap import Tap
import torch


class Arguments(Tap):
    model_path: str


class SavedRanksEvaluator(RankBasedEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_ranks = None

    def finalize(self) -> RankBasedMetricResults:
        if self.num_entities is None:
            raise ValueError

        result = RankBasedMetricResults.from_ranks(
            metrics=self.metrics,
            rank_and_candidates=_iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates),
        )

        self.saved_ranks = self.ranks.copy()
        self.ranks.clear()
        self.num_candidates.clear()

        return result


def get_triple_ranks(args: Arguments):
    model_file = osp.join(args.model_path, 'trained_model.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_file).to(device)
    train = TriplesFactory.from_path_binary(osp.join(args.model_path,
                                                     'training_triples'))

    graph_path = osp.join('data', 'biokgb', 'graph')
    valid_triples = 'biokg.links-valid.csv'
    test_triples = 'biokg.links-test.csv'

    valid, test = [TriplesFactory.from_path(osp.join(graph_path, f),
                                            entity_to_id=train.entity_to_id,
                                            relation_to_id=train.relation_to_id)
                   for f in (valid_triples, test_triples)]

    evaluator = SavedRanksEvaluator(filtered=True)
    evaluator.evaluate(model,
                       test.mapped_triples,
                       additional_filter_triples=[train.mapped_triples,
                                                  valid.mapped_triples])

    head_ranks = evaluator.saved_ranks[('head', 'realistic')]
    tail_ranks = evaluator.saved_ranks[('tail', 'realistic')]
    ranks = np.concatenate(head_ranks + tail_ranks)
    # Save ranks to a csv file, specifying the integer format
    np.savetxt(osp.join(args.model_path, 'ranks.csv'), ranks, fmt='%d')


if __name__ == '__main__':
    get_triple_ranks(Arguments().parse_args())
