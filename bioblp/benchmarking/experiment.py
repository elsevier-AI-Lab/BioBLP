from argparse import ArgumentParser
from time import time
from pathlib import Path
from bioblp.benchmarking.preprocess import main as sampling_main
from bioblp.benchmarking.preprocess import parse_preprocess_config, PreprocessConfig
from bioblp.benchmarking.featurise import main as featurise_main
from bioblp.benchmarking.train import run as train_main


def run_experiment(args):

    experiment_id = str(int(time()))

    # Negative sampling
    preprocess_args = parse_preprocess_config(args.conf)
    data_root = Path(preprocess_args["data_root"])
    preprocess_args["kg_triples_dir"] = data_root.joinpath(
        preprocess_args["kg_triples_dir"])
    preprocess_args["outdir"] = data_root.joinpath(preprocess_args["outdir"])
    preprocess_cfg = PreprocessConfig(**preprocess_args)

    sampling_main(bm_data_path=args.bm_file,
                  kg_triples_dir=preprocess_cfg.kg_triples_dir,
                  num_negs_per_pos=preprocess_cfg.num_negs_per_pos,
                  outdir=preprocess_cfg.outdir,
                  override_run_id=experiment_id)

    # Prepare features

    featurise_main(bm_file=args.bm_file,
                   conf=args.conf,
                   override_data_root=args.override_data_root,
                   override_run_id=experiment_id)

    # Run training

    train_main(conf=args.conf,
               n_proc=args.n_proc,
               tag=args.tag,
               override_data_root=args.override_data_root,
               override_run_id=experiment_id)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run full benchmark experiment procedure")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment configuration")
    parser.add_argument("--bm_file", type=str, help="Path to benchmark data")
    parser.add_argument("--outdir", type=str, help="Path to write output")
    parser.add_argument("--override_data_root", type=str,
                        help="Path to root of data tree")
    parser.add_argument(
        "--n_proc", type=int, default=1, help="Number of cores to use in process."
    )
    parser.add_argument("--tag", type=str,
                        help="Optional tag to add to wandb runs")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    run_experiment(args)
