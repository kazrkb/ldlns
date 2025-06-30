import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Add for 3D plotting

# Path to your .npy file
NPY_PATH = "NODULE_MASK_VOLUME.npy"

# Load the 3D mask volume
mask_volume = np.load(NPY_PATH)
print(f"Loaded mask volume with shape: {mask_volume.shape}")

# Function to view a specific slice

def view_slice(slice_idx):
    plt.imshow(mask_volume[slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    plt.show()

def view_3d_voxels():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    filled = mask_volume > 0  # Only show mask voxels
    ax.voxels(filled, facecolors='red', edgecolor='k', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Nodule Mask Volume (Voxels)")
    plt.show()

# Example usage: view the middle slice
if __name__ == "__main__":
    num_slices = mask_volume.shape[0]
    print(f"Number of slices: {num_slices}")
    # View the middle slice by default
    view_slice(num_slices // 2)
    # To view the 3D voxel plot, uncomment below:
    # view_3d_voxels()
    # Or, uncomment below to scroll through all slices
    # for i in range(num_slices):
    #     view_slice(i)
