#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################


import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from torch import nn
import os

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


def train(local_rank, global_rank, world_size, args):

    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SimpleVisionTransformer().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in tqdm(train_loader, disable=global_rank != 0):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")

    if global_rank == 0:
        print("Training complete")

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

        avg_test_loss = total_loss / total_samples
        print(f"Test set loss: {avg_test_loss:.4f}")

        with open("custom_test_metric.txt", 'w') as f:
            f.write(str(avg_test_loss))

    dist.destroy_process_group()

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", dest="epochs", type=int, default=300)
    parser.add_argument("-lr", dest="lr", type=float, default=1e-4)
    parser.add_argument("-batch", dest="batch_size", type=int, default=32)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"world_size: {world_size}, global_rank: {global_rank}, local_rank: {local_rank}")

    train(local_rank, global_rank, world_size, args)
