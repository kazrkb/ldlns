import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Use v2 to avoid deprecation warning
from tqdm import tqdm  # Progress bar

# Folders
ct_folder = "CONVERTED_PNG"
mask_folder = "NODULE_MASK_IMAGES"
output_folder = "OVERLAY_VISUALS"
os.makedirs(output_folder, exist_ok=True)

# Get sorted list of PNGs (assumes same names in both folders)
ct_files = sorted([f for f in os.listdir(ct_folder) if f.lower().endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith('.png')])

for ct_file, mask_file in tqdm(list(zip(ct_files, mask_files)), desc="Saving overlays"):
    ct_path = os.path.join(ct_folder, ct_file)
    mask_path = os.path.join(mask_folder, mask_file)
    ct_img = imageio.imread(ct_path)
    mask_img = imageio.imread(mask_path)
    if mask_img.ndim > 2:
        mask_img = mask_img[..., 0]
    # Convert CT image to RGB if RGBA
    if ct_img.ndim == 3 and ct_img.shape[2] == 4:
        ct_img = ct_img[..., :3]
    if ct_img.ndim == 2:
        ct_img = np.stack([ct_img]*3, axis=-1)
    # Make mask RGB for overlay (magenta: R=255, G=0, B=255)
    mask_rgb = np.zeros((*mask_img.shape, 3), dtype=np.uint8)
    mask_rgb[..., 0] = mask_img  # Red channel
    mask_rgb[..., 1] = 0         # Green channel
    mask_rgb[..., 2] = mask_img  # Blue channel
    # Overlay: blend CT and mask (magenta, alpha=0.4)
    overlay = ct_img.astype(float)
    mask_alpha = (mask_img > 0)[:, :, None]
    overlay = overlay * (1 - 0.4 * mask_alpha) + mask_rgb * (0.4 * mask_alpha)
    overlay = overlay.astype(np.uint8)
    # Plot side by side
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(ct_img, cmap='gray')
    axs[0].set_title('CT Slice')
    axs[1].imshow(mask_img, cmap='gray')
    axs[1].set_title('Nodule Mask')
    axs[2].imshow(overlay)
    axs[2].set_title('Overlay (Magenta)')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f'overlay_{ct_file}')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
print(f"Saved overlays to {output_folder}")
