import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import CNNReg, KNNPartialFit
from src.resnet import ResNet
from test import TestDataset


def fit(branch, images, labels, epochs, batch_size, lr, device):
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(branch.model.parameters(), lr=lr)
    branch.train()
    for _ in range(epochs):
        for x, y in loader:
            prediction, _ = branch.model(x.to(device))
            loss = nn.functional.mse_loss(prediction.squeeze(-1), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    branch.eval()
    with torch.no_grad():
        _, features = branch.model(images.to(device))
    branch.h.knn.fit(features.cpu().numpy(), labels.numpy())


def select_pseudo_labels(branches, images, iterations, device):
    banks = [b.h.knn._fit_X.copy() for b in branches]
    targets = [b.h.knn._y.copy() for b in branches]
    regressors = [clone(b.h.knn).fit(x, y) for b, x, y in zip(branches, banks, targets)]
    additions = [[], []]
    remaining = [set(range(len(images))), set(range(len(images)))]
    with torch.no_grad():
        outputs = [b.model(images.to(device)) for b in branches]
    features = [o[1].cpu().numpy() for o in outputs]
    labels = [o[0].squeeze(-1).cpu().numpy() for o in outputs]
    for _ in range(iterations):
        choices = []
        for source in range(2):
            best_score, best_id = 0, None
            for index in sorted(remaining[source]):
                x = features[source][index:index + 1]
                y = labels[source][index]
                nearest = regressors[source].kneighbors(x, return_distance=False)[0]
                before = regressors[source].predict(banks[source][nearest])
                temporary = clone(regressors[source]).fit(
                    np.vstack((banks[source], x)), np.append(targets[source], y))
                after = temporary.predict(banks[source][nearest])
                score = np.sum((before - targets[source][nearest]) ** 2)
                score -= np.sum((after - targets[source][nearest]) ** 2)
                if score > best_score:
                    best_score, best_id = score, index
            if best_id is not None:
                choices.append((source, best_id))
        if not choices:
            break
        # Pass accepted labels to the other branch
        for source, index in choices:
            target = 1 - source
            remaining[source].remove(index)
            additions[target].append((index, labels[source][index]))
            banks[target] = np.vstack((banks[target], features[target][index:index + 1]))
            targets[target] = np.append(targets[target], labels[source][index])
            regressors[target].fit(banks[target], targets[target])
    return additions


def main():
    parser = argparse.ArgumentParser(description="Train the two CoDFE branches.")
    parser.add_argument("--data-dir", default="data/release")
    parser.add_argument("--train-csv", default="test.csv")
    parser.add_argument("--unlabeled-csv", default="test.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pseudo-iterations", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", default="outputs/codfe.pth")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)
    dataset = TestDataset(args.data_dir, args.train_csv)
    images, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    branches = [CNNReg(ResNet(), KNNPartialFit(p=p)).to(device) for p in (2, 3)]
    for branch in branches:
        fit(branch, images, labels, args.epochs, args.batch_size, args.lr, device)
    if args.pseudo_iterations:
        pool = TestDataset(args.data_dir, args.unlabeled_csv)
        unlabeled, _ = next(iter(DataLoader(pool, batch_size=len(pool))))
        additions = select_pseudo_labels(branches, unlabeled, args.pseudo_iterations, device)
        for branch, accepted in zip(branches, additions):
            if accepted:
                indices, values = zip(*accepted)
                combined_images = torch.cat((images, unlabeled[list(indices)]))
                combined_labels = torch.cat((labels, torch.tensor(values, dtype=labels.dtype)))
                fit(branch, combined_images, combined_labels, args.epochs,
                    args.batch_size, args.lr, device)
    # Save the same format that test.py reads
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"CoModel1": branches[0].cpu(), "CoModel2": branches[1].cpu()}, output)


if __name__ == "__main__":
    main()
