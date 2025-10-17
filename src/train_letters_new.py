"""Train the new letters k-NN model from EMNIST letters split.

Usage:
  python src/train_letters_new.py --max-samples 20000 --neighbors 5
Saves model to models_new/letters_knn.pkl
"""

import argparse
from letters_new.model_knn import train_knn, save_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=20000)
    ap.add_argument('--neighbors', type=int, default=5)
    args = ap.parse_args()
    model, mapping = train_knn(max_samples=args.max_samples, n_neighbors=args.neighbors)
    path = save_model(model, mapping)
    print(f"Saved new letters model to {path}")


if __name__ == '__main__':
    main()

