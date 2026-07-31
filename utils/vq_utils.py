import os
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans


def softmax_to_topk_soft_code(logits, k):
    """
    Sparse Coefficient
    """
    # Apply softmax to get probabilities
    y_soft = logits.softmax(dim=1)  # [batch_size, K]

    values, indices = torch.topk(y_soft, k, dim=1)
    mask = torch.zeros_like(y_soft, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    zero_tensor = torch.full_like(y_soft, 0)
    y_soft_topk = torch.where(mask, y_soft, zero_tensor)
    y_soft_topk = y_soft_topk / (y_soft_topk.sum(dim=1).unsqueeze(1) + 1e-10)
    soft_code_topk = y_soft_topk

    return soft_code_topk

def get_weights_and_indices(logits, k):
    # Apply softmax to get probabilities
    y_soft = logits.softmax(dim=1)  # [batch_size, K]
    values, indices = torch.topk(y_soft, k, dim=1)
    mask = torch.zeros_like(y_soft, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    zero_tensor = torch.full_like(y_soft, 0)
    y_soft_topk = torch.where(mask, y_soft, zero_tensor)
    y_soft_topk = y_soft_topk / (y_soft_topk.sum(dim=1).unsqueeze(1) + 1e-10)
    soft_code_topk = y_soft_topk
    non_zero_mask = soft_code_topk != 0
    weights = soft_code_topk[non_zero_mask].view(soft_code_topk.shape[0], k)
    indices = torch.arange(y_soft_topk.shape[1], device=y_soft_topk.device).expand_as(soft_code_topk)[non_zero_mask].view(soft_code_topk.shape[0], k)

    return weights.float(), indices.float()


class ResidualVectorQuantizationWithClustering(nn.Module):
    def __init__(self, num_levels, num_clusters, feature_dim, device):
        super(ResidualVectorQuantizationWithClustering, self).__init__()
        self.num_levels = num_levels
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.device = device
        # Store the quantizers for each level
        self.quantizers = []

    def fit_quantizers(self, features):
        """
        Perform clustering on residuals to initialize quantizers for each level.
        """
        residuals = features.cpu().detach().numpy()  # Start with original features on CPU for clustering

        for level in range(self.num_levels):
            # Perform K-means clustering on the residuals
            print("Level", level)
            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters)
            kmeans.fit(residuals)
            # Save the cluster centers as the quantizer for this level
            self.quantizers.append(torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32))
            # Compute quantized values and update residuals
            quantized = self._quantize_with_centers(residuals, kmeans.cluster_centers_).cpu().numpy()
            residuals = residuals - quantized

    def _quantize_with_centers(self, data, centers):
        """
        Given data and quantization centers, return the quantized data.
        """
        data_tensor = torch.tensor(data, device=self.device)
        centers_tensor = torch.tensor(centers, device=self.device)
        # Calculate distances and find nearest centers
        distances = torch.cdist(data_tensor, centers_tensor, p=2)
        indices = distances.argmin(dim=1)
        quantized_data = centers_tensor[indices]

        return quantized_data

    def forward(self, features):
        residuals = features
        quantized_outputs = []
        quantization_indices = []

        for level, centers in enumerate(self.quantizers):
            # Calculate distances to each cluster center and get the closest one
            print(level)
            print(torch.norm(centers, dim=1))
            print(torch.norm(residuals, dim=1).mean())
            distances = torch.cdist(residuals, centers, p=2)
            indices = distances.argmin(dim=1)
            # Retrieve quantized values based on closest centers
            quantized = centers[indices]
            # Store the quantized output and indices for each level
            quantized_outputs.append(quantized)
            quantization_indices.append(indices)
            # Update residuals for the next level
            residuals = residuals - quantized        
        quantized_result = sum(quantized_outputs)

        return quantized_result, quantization_indices

def load_2d_language_feature(data_dir, device):
    """
    Load language feature from 2D images
    """
    data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
    for i in range(len(data_names)):
        features = np.load(data_names[i])
        if i == 0:
            data = features
        else:
            data = np.concatenate([data, features], axis=0)
    data = torch.from_numpy(data).to(device)
    
    return data