import numpy as np
import torch
from torch import nn
from sklearn.neighbors import KNeighborsRegressor


class CNNReg(nn.Module):
    def __init__(self, model, h):
        super().__init__()
        self.model = model
        self.h = h

    def predict(self, images):
        invalid = ~torch.isfinite(images)
        clean = images.masked_fill(invalid, 0)
        with torch.no_grad():
            prediction, features = self.model(clean)
            features = features.cpu().numpy()
            # KNN stays fixed during testing
            neighbors = self.h.predict(features)
            neighbors[invalid.any(dim=(1, 2, 3)).cpu().numpy()] = np.nan
        return prediction, features, neighbors


class KNNPartialFit:
    def __init__(self, n_neighbors=3, p=2):
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors, p=p)

    def predict(self, features):
        return self.knn.predict(features)
