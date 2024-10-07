#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################


import argparse
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from torch import nn
import random

class SimpleVisionTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleVisionTransformer, self).__init__()
        self.patch_embed = nn.Conv2d(3, 64, kernel_size=4, stride=4)
        self.positional_encoding = nn.Parameter(torch.randn(1, 64, 8, 8))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1),
            num_layers=2
        )
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.positional_encoding
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).flatten(1)
        x = self.fc(x)
        return x



#%% main
if __name__ == '__main__':

    print("Starting training.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", dest="epochs", type=int)
    parser.add_argument("-lr", dest="lr", type=float)
    parser.add_argument("-batch", dest="batch_size", type=int)
    args = parser.parse_args()

    if args.epochs is not None:
        num_epochs = args.epochs
    else:
        num_epochs = 5

    if args.lr is not None:
        lr = args.lr
    else:
        lr = 1e-4

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 32

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device available", device)

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset (replace with your own dataset)
    dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    # Randomly sample 50% of the dataset simply to shorten training time for this example
    num_samples = len(dataset)
    sampled_indices = random.sample(range(num_samples), num_samples // 2)
    dataset = Subset(dataset, sampled_indices)

    # Split dataset into train and test sets
    # This is an example so we are going to remove most data
    # so that the training time is not too slow
    train_size = int(0.6 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained ViT model and image processor
    model = SimpleVisionTransformer().to(device)

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


    print("Training complete")

    # Evaluation
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    # Calculate average test set loss
    avg_test_loss = total_loss / total_samples
    print(f"Test set loss: {avg_test_loss:.4f}")

    with open("custom_test_metric.txt", 'w') as f:
        f.write(str(avg_test_loss))