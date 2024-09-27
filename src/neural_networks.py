"""Train and infer on a feature transformer."""

import math

import pandas
import sklearn
from tqdm import tqdm
import torch
import scipy
import numpy
import delu
from torch import Tensor
import torch.nn.functional as F


def infer_on_feature_transformer(model, batch) -> Tensor:
    if isinstance(batch, numpy.ndarray):
        return model(torch.as_tensor(batch, device="cpu", dtype=torch.float), None).squeeze(-1)
    else:
        return model(batch, None).squeeze(-1)

@torch.no_grad()
def evaluate(model, validation) -> float:
    model.eval()

    eval_batch_size = validation["data"].size()[0]
    y_pred = (
        torch.cat(
            [
                infer_on_feature_transformer(model, batch["data"])
                for batch in delu.iter_batches(validation, eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = validation["labels"].cpu().numpy()

    y_pred = numpy.round(scipy.special.expit(y_pred))
    score = sklearn.metrics.accuracy_score(y_true, y_pred)

    return score

def train_feature_transformer(model, train_data: pandas.DataFrame, validation_data: pandas.DataFrame,
                              number_epochs: int):
    batch_size = 256  # how many instances to look at to compute the gradient?
    epoch_size = math.ceil(train_data.shape[0] / batch_size)  # how many batches do we need to optimize over the whole dataset?
    optimizer = model.make_default_optimizer()  # how do we optimize the model?
    loss_function = F.binary_cross_entropy_with_logits  # what does the model optimize?

    torch_train_data = torch.as_tensor(train_data.values, device="cpu", dtype=torch.float)
    torch_validation_data = torch.as_tensor(validation_data.values, device="cpu", dtype=torch.float)
    torch_train_data, torch_train_labels = torch_train_data[:, :-1], torch_train_data[:, -1]
    torch_validation_data, torch_validation_labels = torch_validation_data[:, :-1], torch_validation_data[:, -1]
    train = {"data": torch_train_data, "labels": torch_train_labels}
    validation = {"data": torch_validation_data, "labels": torch_validation_labels}

    # forward-backward loop
    for epoch in range(number_epochs):
        for batch in tqdm(delu.iter_batches(train, batch_size, shuffle=True),  # iterates over batches of data
                          desc=f"Epoch {epoch}",
                          total=epoch_size):
            model.train()
            optimizer.zero_grad()
            loss = loss_function(infer_on_feature_transformer(model, batch["data"]), batch["labels"])
            loss.backward()
            optimizer.step()

    return model
