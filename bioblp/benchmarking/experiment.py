from argparse import ArgumentParser
from time import time
from pathlib import Path
from bioblp.benchmarking.preprocess import main as sampling_main
from bioblp.benchmarking.config import BenchmarkPreprocessConfig

from bioblp.benchmarking.featurise import main as featurise_main
from bioblp.benchmarking.split import main as split_main


def run_experiment(args):

    experiment_id = str(int(time()))

    override_data_root = Path(
        args.override_data_root) if args.override_data_root is not None else None

    #
    # Negative sampling
    #
    preprocess_config = BenchmarkPreprocessConfig.from_toml(
        args.conf, run_id=experiment_id)

    if override_data_root:
        preprocess_config.data_root = override_data_root

    sampled_bm_filepath = sampling_main(bm_data_path=args.bm_file,
                                        kg_triples_dir=preprocess_config.kg_triples_dir,
                                        num_negs_per_pos=preprocess_config.num_negs_per_pos,
                                        outdir=preprocess_config.resolve_outdir(),
                                        override_run_id=experiment_id)
    #
    # Prepare features
    #
    featurise_main(bm_file=sampled_bm_filepath,
                   conf=args.conf,
                   override_data_root=override_data_root,
                   override_run_id=experiment_id)
    #
    # Prepare splits
    #
    split_main(data=sampled_bm_filepath,
               conf=args.conf,
               override_data_root=override_data_root,
               override_run_id=experiment_id)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run full benchmark experiment procedure")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment configuration")
    parser.add_argument("--bm_file", type=str, help="Path to benchmark data")
    parser.add_argument("--override_data_root", type=str, default=None,
                        help="Path to root of data tree")
    parser.add_argument(
        "--n_proc", type=int, default=-1, help="Number of cores to use in process."
    )
    parser.add_argument("--tag", type=str,
                        help="Optional tag to add to wandb runs")
    parser.add_argument("--dev_run", action='store_true',
                        help="Quick dev run")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    run_experiment(args)
