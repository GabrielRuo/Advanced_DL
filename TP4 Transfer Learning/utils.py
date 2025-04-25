# import numpy as np

import torch

from torchvision import datasets, transforms, models
from torch.functional import F
import torch.nn as nn
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics import Metric

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm



def precompute_features(
    model: models.ResNet, 
    dataset: torch.utils.data.Dataset, 
    device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is 
    the rest of the model, it is not necessary to recompute $g(x)$ at 
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and 
    create a new dataset 
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation
    
    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    #pass the original dataset in batches to not overload the compute
    batch_size = 10
    num_classes = len(dataset.classes)
    loader = DataLoader(dataset, batch_size, shuffle = False)# we don't want to shuffle to get all the data once
    extracted_model = nn.Sequential(*list(model.children())[:-1])
    extracted_model.to(device)
    extracted_model.eval()

    all_features = []
    all_labels = []

    for inputs, labels in tqdm(loader,"precomputing dataset"):
      with torch.no_grad():

        inputs, labels = inputs.to(device), labels.to(device)

        features = extracted_model(inputs)
        features = features.view(features.shape[0],-1) #not squeeze as if batch_size = 1 would collapse batch size

        all_features.append(features.cpu())
        all_labels.append(labels)

    #create dataset
    all_features = torch.concat(all_features,dim = 0) #concatenate along the batch dimension and creates tensor
    all_labels = torch.concat(all_labels,dim = 0) #concatenate along the batch dimension and creates tensor
    dataset = TensorDataset(all_features,all_labels)

    return dataset


class LastLayer(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        in_features = 512; num_classes = 2
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x

class FinalModel(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        # <YOUR CODE>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <YOUR CODE>
        raise NotImplementedError("Implement the forward pass of the LastLayer module")