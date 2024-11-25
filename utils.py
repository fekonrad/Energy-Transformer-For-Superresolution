import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gen_structured_mask_id(num_patch: int, batch_size: int):
    """
    Generates structured mask indices for 2x2 block masking pattern where the top-left pixel
    of each 2x2 block remains unmasked and the other three pixels are masked.

    Args:
        num_patch (int): Total number of patches in the image. For a 32x32 image with 
            patch_size=4, this would be 64 (8x8 grid of patches).
        batch_size (int): Number of images in the batch.

    Returns:
        tuple: Contains two tensors:
            - batch_id (torch.Tensor): Shape [batch_size, num_patch * 3] tensor indicating 
              which batch each mask belongs to.
            - mask_id (torch.Tensor): Shape [batch_size, num_patch * 3] tensor indicating 
              which patch each mask belongs to.

    Example:
        >>> batch_size = 2
        >>> num_patch = 4  # 2x2 grid of patches
        >>> batch_id, mask_id = gen_structured_mask_id(num_patch, batch_size)
        >>> print(batch_id.shape)  # torch.Size([2, 12])
        >>> print(mask_id.shape)   # torch.Size([2, 12])
        >>> # Each patch has 3 masked positions (bottom-left, top-right, bottom-right)
        >>> # Total masks per image = 4 patches × 3 masks = 12 masked positions
    """
    patches_per_side = int(np.sqrt(num_patch))
    masks_per_patch = 3  # 3 masked pixels per 2x2 block
    total_masks = num_patch * masks_per_patch
    
    # Create batch indices
    batch_id = torch.arange(batch_size)[:, None].repeat(1, total_masks)
    
    # Create mask indices for one batch
    mask_indices = []
    for patch_idx in range(num_patch):
        # For each patch, add indices for the 3 masked pixels
        mask_indices.extend([patch_idx] * masks_per_patch)
    
    # Convert to tensor and repeat for each batch
    mask_id = torch.tensor(mask_indices)[None, :].repeat(batch_size, 1)
    
    return batch_id, mask_id



def wavelet_reparameterization(x: torch.Tensor, wavelet_type: str = 'haar') -> torch.Tensor:
    """
    Performs Haar wavelet transformation on each 2x2 block of the input images and
    stores the coefficients in a new tensor `wavelet_x`.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        wavelet_type (str, optional): Type of wavelet to use. Currently supports 'haar'.
                                      Defaults to 'haar'.

    Returns:
        torch.Tensor: Tensor of the same shape as `x`, containing wavelet coefficients.
                      Each 2x2 block in `wavelet_x` contains:
                        - Top-Left: Approximation coefficient
                        - Top-Right: Horizontal detail coefficient
                        - Bottom-Right: Vertical detail coefficient
                        - Bottom-Left: Diagonal detail coefficient

    Example:
        >>> batch_size, channels, height, width = 2, 3, 4, 4
        >>> x = torch.arange(batch_size * channels * height * width, dtype=torch.float32).reshape(batch_size, channels, height, width)
        >>> wavelet_x = wavelet_reparameterization(x)
        >>> print(wavelet_x.shape)  # torch.Size([2, 3, 4, 4])
    """
    if wavelet_type != 'haar':
        raise NotImplementedError(f"Wavelet type '{wavelet_type}' is not supported.")

    batch_size, channels, height, width = x.shape

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Height and Width of the input tensor must be even numbers.")

    # Fold the spatial dimensions into non-overlapping 2x2 blocks
    # Reshape x to (batch_size, channels, height//2, 2, width//2, 2)
    x_reshaped = x.view(batch_size, channels, height // 2, 2, width // 2, 2)

    # Permute to bring the 2x2 blocks together: (batch_size, channels, height//2, width//2, 2, 2)
    x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Now, reshape to (batch_size, channels, height//2, width//2, 4)
    # where the last dimension represents the 2x2 block
    x_blocks = x_permuted.view(batch_size, channels, height // 2, width // 2, 4)

    # Define Haar wavelet transform matrix
    haar_matrix = torch.tensor([[1, 1, 1, 1],
                                [1, -1, 1, -1],
                                [1, 1, -1, -1],
                                [1, -1, -1, 1]], dtype=x.dtype, device=x.device) / 2.0

    # Perform matrix multiplication to get wavelet coefficients
    # x_blocks shape: (batch_size, channels, height//2, width//2, 4)
    # haar_matrix shape: (4, 4)
    # Output shape: same as x_blocks
    wavelet_blocks = torch.matmul(x_blocks, haar_matrix.T)

    # Assign the coefficients to the corresponding positions in wavelet_x
    # Initialize wavelet_x with zeros
    wavelet_x = torch.zeros_like(x)

    # Map the coefficients back to spatial positions
    # wavelet_blocks has shape (batch_size, channels, height//2, width//2, 4)
    # We need to map it back to (batch_size, channels, height, width)
    # with the corresponding coefficients in 2x2 blocks

    # Permute back to (batch_size, channels, height//2, width//2, 2, 2)
    wavelet_blocks_perm = wavelet_blocks.view(batch_size, channels, height // 2, width // 2, 2, 2)

    # Permute to (batch_size, channels, height//2, 2, width//2, 2)
    wavelet_blocks_perm = wavelet_blocks_perm.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Finally, reshape to (batch_size, channels, height, width)
    wavelet_x = wavelet_blocks_perm.view(batch_size, channels, height, width)

    return wavelet_x


def wavelet_inverse_reparameterization(wavelet_x: torch.Tensor, wavelet_type: str = 'haar') -> torch.Tensor:
    """
    Reconstructs the original image from its Haar wavelet coefficients.

    Args:
        wavelet_x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
                                   containing Haar wavelet coefficients.
        wavelet_type (str, optional): Type of wavelet to use. Currently supports 'haar'.
                                      Defaults to 'haar'.

    Returns:
        torch.Tensor: Reconstructed original image tensor of shape (batch_size, channels, height, width).
    """
    if wavelet_type != 'haar':
        raise NotImplementedError(f"Wavelet type '{wavelet_type}' is not supported.")

    batch_size, channels, height, width = wavelet_x.shape

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Height and Width of the input tensor must be even numbers.")

    # Define Inverse Haar wavelet transform matrix (transpose of the forward matrix)
    inverse_haar_matrix = torch.tensor([[1,  1,  1,  1],
                                        [1, -1,  1, -1],
                                        [1,  1, -1, -1],
                                        [1, -1, -1,  1]], dtype=wavelet_x.dtype, device=wavelet_x.device) / 2.0

    # Reshape wavelet_x to (batch_size, channels, height//2, width//2, 4)
    wavelet_blocks = wavelet_x.view(batch_size, channels, height // 2, 2, width // 2, 2)
    wavelet_blocks = wavelet_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    wavelet_blocks = wavelet_blocks.view(batch_size, channels, height // 2, width // 2, 4)

    # Perform inverse Haar wavelet transformation
    # wavelet_blocks: (batch_size, channels, height//2, width//2, 4)
    # inverse_haar_matrix: (4, 4)
    x_blocks = torch.matmul(wavelet_blocks, inverse_haar_matrix)

    # Reshape back to (batch_size, channels, height//2, 2, width//2, 2)
    x_blocks = x_blocks.view(batch_size, channels, height // 2, width // 2, 2, 2)

    # Permute to (batch_size, channels, height//2, 2, width//2, 2)
    x_blocks = x_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Finally, reshape to (batch_size, channels, height, width)
    x_reconstructed = x_blocks.view(batch_size, channels, height, width)

    return x_reconstructed


def test_wavelet_reparameterization():
    avg_mse = 0
    for _ in range(1000):
        x = torch.randn(1, 3, 256, 256)
        wavelet_x = wavelet_reparameterization(x)
        x_reconstructed = wavelet_inverse_reparameterization(wavelet_x)
        mse = F.mse_loss(x, x_reconstructed)
        avg_mse += mse
    avg_mse /= 1000
    print(f"Average MSE: {avg_mse}")


if __name__ == "__main__":
    # test_wavelet_reparameterization()

    import matplotlib.pyplot as plt
    import torchvision
    x = torchvision.io.read_image("Energy-Transformer\data\cat.jpg")
    x = F.upsample(x, scale_factor=2, mode='nearest')
    wavelet_x = wavelet_reparameterization(x)
    plt.imshow(wavelet_x[0].permute(1,2,0))
    plt.show()


    x = torch.tensor([[[[0.0, 1.0], 
                        [1.0, 0.0]]]], dtype=torch.float32)
    print(x.shape)
    wavelet_x = wavelet_reparameterization(x)
    print(wavelet_x)

    x_reconstructed = wavelet_inverse_reparameterization(wavelet_x)
    print(x_reconstructed)
