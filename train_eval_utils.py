import numpy as np
import torch


# @TODO: add exception handling
def load_dataset(dataset_path):
    data = torch.load(dataset_path)
    return data


def generate_optical_flow_dataloader(dataset_path, batch_size=1, shuffle=True):
    flow_dataset = load_dataset(dataset_path)
    flow_dataloader = torch.utils.data.DataLoader(flow_dataset, batch_size=batch_size,
                                                  num_workers=4, shuffle=shuffle,
                                                  pin_memory=True)
    return flow_dataloader


def generate_optical_flow_dataloader_split(dataset_path, validation_split, train_batch_size=8):
    flow_dataset = load_dataset(dataset_path)
    dataset_size = len(flow_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_data = torch.utils.data.Subset(flow_dataset, train_indices)
    val_data = torch.utils.data.Subset(flow_dataset, val_indices)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size,
                                               num_workers=4, shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                             num_workers=4, shuffle=False,
                                             pin_memory=True)
    return train_loader, val_loader
