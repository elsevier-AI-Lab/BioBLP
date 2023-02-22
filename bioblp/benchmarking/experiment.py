from argparse import ArgumentParser


def run_experiment(args):

    # Negative sampling

    # Prepare features

    # Run training

    pass


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run full benchmark experiment procedure")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment configuration")
    parser.add_argument("--outdir", type=str, help="Path to write output")


if __name__ == "__main__":

    args = parser.parse_args()
    run_experiment(args)
