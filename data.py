import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from upsample import upsample_wavelet  # Adjust the import path as necessary


class ImageNet1K(Dataset):
    def __init__(self, root, split='train', transform=None, wavelet_transform=None):
        """
        Initializes the ImageNet1K dataset.

        Args:
            root (str): Root directory of the ImageNet1K dataset.
            split (str): Dataset split to use ('train' or 'val').
            transform (callable, optional): Transformations to apply to the images.
            wavelet_transform (callable, optional): Transformations to apply for wavelet representation.
        """
        self.root = os.path.join(root, split)
        self.transform = transform
        self.wavelet_transform = wavelet_transform

        # Initialize ImageFolder dataset
        self.dataset = datasets.ImageFolder(self.root, transform=self.transform)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves the wavelet representation and the original image tensor for a given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            wavelet_coeffs (torch.Tensor): Upsampled wavelet coefficients of shape (c, 512, 512).
            image (torch.Tensor): Original image tensor of shape (c, 256, 256).
        """
        # Retrieve image and label (labels can be ignored if not needed)
        image, _ = self.dataset[index]

        return image  # You can also return mask if required

if __name__ == "__main__":
    # Example usage of the ImageNet1K Dataset and DataLoader

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Reshape to 256x256
        transforms.ToTensor(),          # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    # Initialize the dataset
    dataset = ImageNet1K(root='/path/to/ImageNet1K', split='train', transform=transform)

    # Define the DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Iterate through the DataLoader
    for batch_idx, images in enumerate(dataloader):
        wavelet_coeffs, mask = upsample_wavelet(images, scale_factor=2)
        print(f"Batch {batch_idx + 1}")
        print(f"Wavelet Coefficients Shape: {wavelet_coeffs.shape}")  # Expected: (32, 3, 512, 512)
        print(f"Images Shape: {images.shape}")                        # Expected: (32, 3, 256, 256)")
        # Add your training or processing logic here

        # For demonstration, we'll break after the first batch
        break
