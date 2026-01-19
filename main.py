# main.py

import argparse
from data_processing import process_file
from train import main as training_main

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Lyrics Generation Training Script")

    # Data & File Paths
    parser.add_argument(
        "--train_file_path", 
        type=str, 
        required=True, 
        help="Path to the training CSV/text file."
    )

    # Hyperparameters
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=5,
        help="Number of input words before predicting the next word."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=300,
        help="Dimension of word embeddings."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size of the LSTM."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for text generation."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: Read the training data
    df_train = process_file(args.train_file_path)

    # Step 2: Run training and generation
    training_main(
        df_train, 
        sequence_length=args.sequence_length, 
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim, 
        hidden_size=args.hidden_size,
        epochs=args.epochs, 
        lr=args.lr, 
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
