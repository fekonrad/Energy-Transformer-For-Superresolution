import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Import the Haar transform that is already implemented in `train.py`
from train import haar_wavelet_transform


def mean_highfreq_haar_norm(batch_size: int = 512, device = None) -> float:
    """Compute the mean L2-norm of the high-frequency Haar coefficients (LH, HL, HH)
    over the entire CIFAR-10 **training** split.

    Args:
        batch_size (int, optional): Batch size used when iterating through the dataset. Defaults to ``512``.
        device (str | None, optional): Device on which the computations are performed. If ``None`` the
            routine automatically selects *cuda* (if available), *mps* (Apple Silicon) or finally *cpu*.

    Returns:
        float: Mean of ‖LH‖₂ + ‖HL‖₂ + ‖HH‖₂ taken over all images in the dataset.
    """

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # CIFAR-10 pixel values in [0, 1]
    dataset = CIFAR10(root="./", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    total_norm = 0.0
    norm_value = 0.0
    n_images = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)

            # Forward Haar transform
            coeffs = haar_wavelet_transform(imgs)
            _, _, h, w = coeffs.shape

            # Extract the three high-frequency quadrants
            ll = coeffs[:, :, : h // 2, : w // 2]  # Low-Low (top-left)
            lh = coeffs[:, :, : h // 2, w // 2 :]  # Low-High (top-right)
            hl = coeffs[:, :, h // 2 :, : w // 2]  # High-Low (bottom-left)
            hh = coeffs[:, :, h // 2 :, w // 2 :]  # High-High (bottom-right)

            # Compute L2 norm of each component then sum them per image
            batch_norm = (
                torch.norm(lh.reshape(lh.size(0), -1), dim=1) +
                torch.norm(hl.reshape(hl.size(0), -1), dim=1) +
                torch.norm(hh.reshape(hh.size(0), -1), dim=1)
            )

            total_norm += batch_norm.sum().item()
            n_images += imgs.size(0)

            norm_value += batch_norm.sum().item() + torch.norm(ll.reshape(ll.size(0), -1), dim=1).sum().item()

    print(f"Mean Norm of Target: {norm_value / n_images}")
    # Mean over the whole dataset
    return total_norm / n_images


if __name__ == "__main__":
    mean_value = mean_highfreq_haar_norm()
    print(f"Mean L2-norm of LH+HL+HH over CIFAR-10: {mean_value:.6f}") 