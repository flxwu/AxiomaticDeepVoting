"""
Reproduction of Table 3: 10-fold cross-validation of MLP / CNN / WEC on
100,000 IC-sampled Plurality profiles (up to 55 voters, 5 alternatives).

Per fold: 90,000 train / 10,000 test, 8 epochs, batch size 200 (3,600 gradient
steps), AdamW + cosine-annealing warm restarts, BCE-with-logits loss. We record
train and test loss + accuracy on each fold for each architecture.
"""

import os
import sys
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from gensim.models import Word2Vec

import utils
import generate_data
import models
from models import MLP, CNN, WEC
import train_and_eval


MAX_NUM_VOTERS = 55
MAX_NUM_ALTERNATIVES = 5
DATASET_SIZE = 100_000
NUM_FOLDS = 10
FOLD_SIZE = DATASET_SIZE // NUM_FOLDS
BATCH_SIZE = 200
EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SCHEDULER_T0 = 100
RULE_NAME = "Plurality"
ELECTION_SAMPLING = {"probmodel": "IC"}
WE_CORPUS_SIZE = 100_000
WE_DIM = 200
WE_WINDOW = 7
WE_ALGORITHM = 1  # skip-gram
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataloader(architecture, X, y, embeddings, shuffle):
    if architecture == "MLP":
        return generate_data.tensorize_profile_data_MLP(
            X, y, MAX_NUM_VOTERS, MAX_NUM_ALTERNATIVES, BATCH_SIZE, shuffle=shuffle
        )
    if architecture == "CNN":
        return generate_data.tensorize_profile_data_CNN(
            X, y, MAX_NUM_VOTERS, MAX_NUM_ALTERNATIVES, BATCH_SIZE, shuffle=shuffle
        )
    if architecture == "WEC":
        sentences = [
            [models.ranking_to_string(r) for r in profile.rankings]
            for profile in X
        ]
        loader, _ = generate_data.tensorize_profile_data_WEC(
            embeddings,
            sentences,
            y,
            MAX_NUM_VOTERS,
            MAX_NUM_ALTERNATIVES,
            BATCH_SIZE,
            num_of_unks=False,
            shuffle=shuffle,
        )
        return loader
    raise ValueError(architecture)


def build_model(architecture, embeddings):
    if architecture == "MLP":
        return MLP(MAX_NUM_VOTERS, MAX_NUM_ALTERNATIVES)
    if architecture == "CNN":
        return CNN(
            MAX_NUM_VOTERS,
            MAX_NUM_ALTERNATIVES,
            kernel1=[5, 1],
            kernel2=[1, 5],
            channels=64,
        )
    if architecture == "WEC":
        return WEC(embeddings, MAX_NUM_VOTERS, MAX_NUM_ALTERNATIVES)
    raise ValueError(architecture)


def train_one_fold(architecture, X_train, y_train, X_test, y_test, embeddings):
    model = build_model(architecture, embeddings)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0
    )

    train_loader = make_dataloader(architecture, X_train, y_train, embeddings, shuffle=True)
    test_loader = make_dataloader(architecture, X_test, y_test, embeddings, shuffle=False)
    train_eval_loader = make_dataloader(architecture, X_train, y_train, embeddings, shuffle=False)

    for _ in range(EPOCHS):
        train_and_eval.train(train_loader, model, loss_fn, optimizer)
        scheduler.step()

    train_loss = train_and_eval.loss(model, loss_fn, train_eval_loader)
    train_acc = train_and_eval.accuracy(model, train_eval_loader)
    test_loss = train_and_eval.loss(model, loss_fn, test_loader)
    test_acc = train_and_eval.accuracy(model, test_loader)
    return {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }


def main():
    set_seed(SEED)

    out_dir = os.path.join(
        os.path.dirname(__file__),
        f"results_table3_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving results to {out_dir}")

    print(f"Generating {DATASET_SIZE} IC-{RULE_NAME} profiles...")
    t0 = time.time()
    X, y, _ = generate_data.generate_profile_data(
        MAX_NUM_VOTERS,
        MAX_NUM_ALTERNATIVES,
        DATASET_SIZE,
        ELECTION_SAMPLING,
        [utils.dict_rules_all[RULE_NAME]],
        merge="accumulative",
        progress_report=10_000,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("Pretraining Word2Vec embeddings for WEC...")
    t0 = time.time()
    sentences = [
        [models.ranking_to_string(r) for r in profile.rankings] for profile in X
    ]
    sentences_with_special = sentences + [["UNK"], ["PAD"]]
    embeddings = Word2Vec(
        sentences_with_special,
        vector_size=WE_DIM,
        window=WE_WINDOW,
        min_count=1,
        workers=8,
        sg=WE_ALGORITHM,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    indices = np.arange(DATASET_SIZE)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    folds = [indices[k * FOLD_SIZE : (k + 1) * FOLD_SIZE] for k in range(NUM_FOLDS)]

    list_of_architectures = ["MLP", "CNN", "WEC"]
    results = {
        "location": out_dir,
        "list_of_architectures": list_of_architectures,
        "num_folds": NUM_FOLDS,
        "rule_name": RULE_NAME,
        "max_num_voters": MAX_NUM_VOTERS,
        "max_num_alternatives": MAX_NUM_ALTERNATIVES,
        "dataset_size": DATASET_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler_T0": SCHEDULER_T0,
        "election_sampling": ELECTION_SAMPLING,
        "we_corpus_size": WE_CORPUS_SIZE,
        "we_dim": WE_DIM,
        "we_window": WE_WINDOW,
        "we_algorithm": WE_ALGORITHM,
        "random_seed": SEED,
    }

    def dump():
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    dump()

    for arch in list_of_architectures:
        print(f"\n=== {arch} ===")
        for k in range(NUM_FOLDS):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(NUM_FOLDS) if j != k])
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            t0 = time.time()
            metrics = train_one_fold(
                arch, X_train, y_train, X_test, y_test, embeddings
            )
            runtime = time.time() - t0
            results[f"{arch}_fold_{k}"] = {
                "result": metrics,
                "runtime_sec": runtime,
            }
            print(
                f"  fold {k}: "
                f"train_loss={metrics['train_loss']:.4f}, "
                f"train_acc={metrics['train_accuracy']:.4f}, "
                f"test_loss={metrics['test_loss']:.4f}, "
                f"test_acc={metrics['test_accuracy']:.4f} "
                f"({runtime:.0f}s)"
            )
            dump()

    print("\n=== Table 3 ===")
    import plot_and_visual
    plot_and_visual.exp1_table_crossval(out_dir)


if __name__ == "__main__":
    main()
