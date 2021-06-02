from argparse import ArgumentParser


def args():
    # DARK_TODO: Get better at writing ArgumentParser, for now ghetto version is fine to save time.

    parser = ArgumentParser(prog="project", description="CS231N 2021 Project (dark@stanford.edu) (darkhan@baimyrza.com)")
    parser.add_argument(
        "--model",
        choices=[ "baseline", "model1", "model2" ],
        required=True,
        help="a model",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train a model",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="evaluate a model",
    )

    parser.add_argument(
        "--train_timesteps_per_update",
        type=int,
        help="timesteps per training update",
    )
    parser.add_argument(
        "--train_total_timesteps",
        type=int,
        help="total timesteps to train a model for",
    )

    parser.add_argument(
        "--eval_train_iteration",
        help="training iteration number or 'current', look for directory name in iterations directory, if 'current' then the current training iteration is automatically selected",
    )

    args = parser.parse_args()

    if not args.train and not args.eval:
        raise AssertionError("at least one of --train or --eval argument is required")
    if args.train:
        if not args.train_timesteps_per_update or args.train_timesteps_per_update < 1:
            raise AssertionError("if training, --train_timesteps_per_update argument is required and has to be more than 1")
        if not args.train_total_timesteps or args.train_total_timesteps < 1:
            raise AssertionError("if training, --train_total_timesteps argument is required and has to be more than 1")

    if args.eval:
        if not args.eval_train_iteration:
            raise AssertionError("if evaluating, --eval_train_iteration argument is required")
            if args.eval_train_iteration == "current" and not args.train:
                raise AssertionError("--eval_train_iteration argument can be set to 'current' only when also --train argument is present")

    return args
