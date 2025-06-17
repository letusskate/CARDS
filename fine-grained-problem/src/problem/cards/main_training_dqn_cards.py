import sys
import os
import argparse

sys.path.append(os.path.join(sys.path[0], "..", "..", ".."))

from src.problem.cards.learning.trainer_dqn import TrainerDQN

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Problem instance parameters
    parser.add_argument("--n_L", type=int, default=10)
    parser.add_argument("--n_N", type=int, default=8)
    parser.add_argument("--n_O", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    # Hyper parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--n_step", type=int, default=-1)
    parser.add_argument(
        "--max_softmax_beta", type=int, default=2, help="max_softmax_beta"
    )
    parser.add_argument("--hidden_layer", type=int, default=2)
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="dimension of latent layers"
    )
    parser.add_argument(
        "--n_action", type=int, default=11, help="number of possible actions for z"
    )

    # Argument for Trainer
    parser.add_argument("--n_episode", type=int, default=1000000)
    parser.add_argument("--save_dir", type=str, default="./result-default")
    parser.add_argument("--plot_training", type=int, default=1)
    parser.add_argument("--mode", default="cpu", help="cpu/gpu")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] TRAINING ON RANDOM INSTANCES: Cards (DQN)")
    print("[INFO] n_L: %d" % args.n_L)
    print("[INFO] n_N: %d" % args.n_N)
    print("[INFO] n_O: %d" % args.n_O)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")
    print("[INFO] TRAINING PARAMETERS")
    print("[INFO] algorithm: DQN")
    print("[INFO] batch_size: %d" % args.batch_size)
    print("[INFO] learning_rate: %f" % args.learning_rate)
    print("[INFO] hidden_layer: %d" % args.hidden_layer)
    print("[INFO] latent_dim: %d" % args.latent_dim)
    print("[INFO] softmax_beta: %d" % args.max_softmax_beta)
    print("[INFO] n_step: %d" % args.n_step)
    print("***********************************************************")
    sys.stdout.flush()

    trainer = TrainerDQN(args)
    trainer.run_training()
