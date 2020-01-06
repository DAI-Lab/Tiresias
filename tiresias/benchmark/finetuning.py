import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from time import time
from sklearn.metrics import accuracy_score
from tiresias.core.federated_learning import get_gradients, put_gradients, merge_gradients

class ImageClassifier(object):

    def __init__(self, epsilon, delta, nb_classes, lot_size=100, base_model="squeezenet", feature_extraction=False):
        self.lr = 0.01
        self.epochs = 1
        self.lot_size = lot_size
        self.epsilon = epsilon
        self.delta = delta
        if base_model == "vgg":
            self.model = models.vgg16(pretrained=True)
            if feature_extraction:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier[6] = nn.Linear(4096, nb_classes)
        elif base_model == "squeezenet":
            self.model = models.squeezenet1_1(pretrained=True)
            if feature_extraction:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier[1] = nn.Conv2d(512, nb_classes, kernel_size=(1,1), stride=(1,1))

    def fit(self, dataset, distributed=False):
        if distributed:
            self.fit_distributed(dataset)
        else:
            self.fit_centralized(dataset)

    def fit_distributed(self, dataset):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        epsilon = self.epsilon / self.epochs
        delta = self.delta / self.epochs
        for epoch in range(self.epochs):
            gradients = []
            iterator = tqdm(dataset)
            running_loss, running_acc = [], []
            for x, y in iterator:
                optimizer.zero_grad()
                x = torch.FloatTensor(x).unsqueeze(0)
                y = torch.tensor(y).unsqueeze(0)
                y_pred = self.model(x)
                loss = nn.functional.cross_entropy(y_pred, y)
                loss.backward()

                gradients.append(get_gradients(self.model, epsilon, delta))
                running_loss.append(loss.item())
                running_acc.append(torch.argmax(y_pred[0])==y[0])

                if len(gradients) >= self.lot_size:
                    iterator.set_description("Loss %.3f | Acc %.3f" % (np.mean(running_loss), np.mean(running_acc)))
                    put_gradients(self.model, merge_gradients(gradients))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    gradients = []

    def fit_centralized(self, dataset):
        self.model.train()
        delta = self.delta / self.epochs
        epsilon = self.epsilon / self.epochs
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            losses = []
            for x, y in tqdm(dataset):
                x = torch.FloatTensor(x).unsqueeze(0)
                y = torch.tensor(y).unsqueeze(0)
                y_pred = self.model(x)
                losses.append(nn.functional.cross_entropy(y_pred, y))
                if len(losses) >= self.lot_size:
                    (sum(losses) / len(losses)).backward()
                    put_gradients(self.model, get_gradients(self.model, epsilon, delta))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    losses = []
                    optimizer.zero_grad()

    def score(self, dataset):
        self.model.eval()
        Y_true, Y_pred = [], []
        for x, y in dataset:
            x = torch.FloatTensor(x).unsqueeze(0)
            y_pred = self.model(x)[0]
            Y_true.append(y)
            Y_pred.append(y_pred.detach().numpy())
        return accuracy_score(Y_true, np.argmax(Y_pred, axis=1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--feature_extraction', type=int, default=1)
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test = datasets.STL10("/tmp", split="test", download=True, transform=transform)
    train = datasets.STL10("/tmp", split="train", download=True, transform=transform)

    start = time()
    model = ImageClassifier(
        epsilon=args.epsilon, 
        delta=1e-5, 
        nb_classes=len(train.classes), 
        feature_extraction=args.feature_extraction
    )
    results = []
    for epoch in range(args.epochs):
        model.fit(train, args.distributed)
        results.append({
            "Epoch": epoch,
            "Accuracy": model.score(test),
            "Elapsed Time": time() - start,
        })
        print(results[-1])
    if not args.csv:
        args.csv = "stl10-n%s-e%s-d%s-f%s.csv" % (args.epochs, args.epsilon, args.distributed, args.feature_extraction)
    pd.DataFrame(results).to_csv(args.csv, index=False)
