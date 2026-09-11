import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import model
import src.resnet

# Names stored in the original checkpoints
sys.modules["D_model_CoResNet"] = model
sys.modules["src.D_model_ResNet"] = src.resnet


class TestDataset(Dataset):
    def __init__(self, directory, csv_name="test.csv"):
        self.directory = Path(directory)
        self.samples = pd.read_csv(self.directory / csv_name)
        with np.load(self.directory / "normalization.npz") as stats:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((3, 3)),
                transforms.Normalize(stats["means"].astype(np.float32),
                                     stats["stds"].astype(np.float32)),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        with rasterio.open(self.directory / "patches" / f"{int(row.RegionIndex)}.tif") as patch:
            image = patch.read().astype(np.float32).transpose(1, 2, 0)
        return self.transform(image), torch.tensor(float(row.TGB), dtype=torch.float32)


def evaluate(checkpoint_path, loader, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    first = checkpoint["CoModel1"].to(device).eval()
    second = checkpoint["CoModel2"].to(device).eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            _, _, left = first.predict(images)
            _, _, right = second.predict(images)
            # Average the branches, then convert plot kg to t/ha
            predictions.append((left.item() + right.item()) * 0.5 * 0.025)
            labels.append(target.item() * 0.025)
    return r2_score(labels, predictions), mean_squared_error(labels, predictions) ** 0.5


def main():
    parser = argparse.ArgumentParser(description="Evaluate the released CoDFE checkpoints.")
    parser.add_argument("--checkpoint", default="weights/codfe_pseudo_4_4.pth")
    parser.add_argument("--test-dir", default="data/release")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    # Keep each prediction paired with its CSV row
    loader = DataLoader(TestDataset(args.test_dir), batch_size=1, shuffle=False)
    r2, rmse = evaluate(args.checkpoint, loader, torch.device(args.device))
    print(f"R2: {r2:.5f}")
    print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
