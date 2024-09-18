# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: 27 August 2023

import argparse
import textwrap


def formatter(prog): return argparse.HelpFormatter(prog, max_help_position=50)


def cla():
    parser = argparse.ArgumentParser(
        description="list of arguments", formatter_class=formatter
    )

    # Data parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="tanh",
        choices=["tanh", "bimodal", "swissroll"],
        help=textwrap.dedent(
            """Either bimodal, tanh or swissroll for the 1D examples."""
        ),
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=1500,
        help=textwrap.dedent(
            """Number of training samples to use. Cannot be more than that available."""
        ),
    )
    parser.add_argument(
        "--n_valid",
        type=int,
        default=500,
        help=textwrap.dedent(
            """Number of validation samples to use. Cannot be more than that available."""
        ),
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=500,
        help=textwrap.dedent(
            """Number of test samples to use. Cannot be more than that available."""
        ),
    )

    # Dataset Arguments
    parser.add_argument(
        "--X_dataset",
        type=str,
        default="../Data 4/Type1/X_data-4000.npy",
        help=textwrap.dedent("""X dataset"""),
    )
    parser.add_argument(
        "--Y_clean_dataset",
        type=str,
        default="../Data 4/Type1/Y_clean-4000.npy",
        help=textwrap.dedent("""Clean Y dataset"""),
    )
    parser.add_argument(
        "--Y_noisy_dataset",
        type=str,
        default="../Data 4/Type1/Y_noisy-4000.npy",
        help=textwrap.dedent("""Noisy Y dataset"""),
    )

    # Network parameters
    parser.add_argument(
        "--act_func",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "elu"],
        help=textwrap.dedent("""Type of critic to use"""),
    )
    parser.add_argument(
        "--gp_coef",
        type=float,
        default=0.1,
        help=textwrap.dedent("""Gradient penalty parameter"""),
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=32,
        help=textwrap.dedent(
            """Number of critic updates per generator update"""),
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=1000,
        help=textwrap.dedent("""Maximum number of epochs"""),
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=1,
        help=textwrap.dedent(
            """Dimension of the latent variable."""
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=textwrap.dedent("""Batch size while training"""),
    )
    parser.add_argument(
        "--reg_param",
        type=float,
        default=1e-7,
        help=textwrap.dedent("""Regularization parameter"""),
    )

    parser.add_argument(
        "--seed_no",
        type=int,
        default=1008,
        help=textwrap.dedent("""Set the random seed"""),
    )

    # Output parameters
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1000,
        help=textwrap.dedent(
            """Number of epochs after which a snapshot is saved"""
        ),
    )
    parser.add_argument(
        "--plot_freq",
        type=int,
        default=100,
        help=textwrap.dedent(
            """Number of epochs after which plots are created"""
        )
    )
    parser.add_argument(
        "--sdir_suffix",
        type=str,
        default="",
        help=textwrap.dedent(
            """Suffix to directory where trained network/results are saved"""
        ),
    )
    parser.add_argument(
        "--z_n_MC",
        type=int,
        default=800,
        help=textwrap.dedent(
            """Number of (z) samples used to generate emperical statistics."""
        ),
    )
    parser.add_argument(
        "--y_pert_sigma",
        type=float,
        default=0.0,
        help=textwrap.dedent(
            """Std dev for the normal perturbation of y when using full gp gan in eval mode."""
        ),
    )

    # Testing parameters
    parser.add_argument(
        "--GANdir",
        type=str,
        default=None,
        help=textwrap.dedent(
            """Load checkpoint from user specified GAN directory. Else path will be infered from hyperparameters."""
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="Test_results",
        help=textwrap.dedent("""Directory where test results are saved"""),
    )
    parser.add_argument(
        "--ckpt_id",
        type=int,
        default=-1,
        help=textwrap.dedent("""The checkpoint index to load when testing"""),
    )

    return parser.parse_args()
