# data_loader.py

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ISICDataset


def get_data_loaders(data_root, batch_size=32, task="melanoma"):
    # Paths
    train_dir = f"{data_root}/ISIC-2017_Training_Data"
    valid_dir = f"{data_root}/ISIC-2017_Validation_Data"
    test_dir = f"{data_root}/ISIC-2017_Test_v2_Data"

    train_csv = f"{data_root}/ISIC-2017_Training_Part3_GroundTruth.csv"
    valid_csv = f"{data_root}/ISIC-2017_Validation_Part3_GroundTruth.csv"
    test_csv = f"{data_root}/ISIC-2017_Test_v2_Part3_GroundTruth.csv"

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),  # for EfficientNet-B3
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = ISICDataset(train_dir, train_csv, transform=train_transform, task=task)
    valid_dataset = ISICDataset(valid_dir, valid_csv, transform=test_transform, task=task)
    test_dataset = ISICDataset(test_dir, test_csv, transform=test_transform, task=task)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader
