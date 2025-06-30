#!/usr/bin/env python3
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure
import argparse
import shutil
from pathlib import Path
import re


# DICOM Processing Functions
def load_scan(path):
    """Load DICOM files and sort them by slice position"""
    slices = [
        pydicom.dcmread(os.path.join(path, s), force=True)
        for s in os.listdir(path)
        if s.endswith(".dcm")
    ]

    # Sort slices by position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)

    # Calculate slice thickness
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    """Convert pixel values to Hounsfield units"""
    # Stack all slices into a 3D array
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    # Set outside-of-scan pixels to 0
    image[image == -2000] = 0

    # Apply slope and intercept to convert to HU
    for slice_number, slice_data in enumerate(slices):
        if slice_data.RescaleSlope != 1:
            image[slice_number] = slice_data.RescaleSlope * image[slice_number].astype(
                np.float64
            )
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(slice_data.RescaleIntercept)

    return image


def normalize_image(image):
    """Normalize image to 0-1 range"""
    min_val, max_val = np.min(image), np.max(image)
    return (image - min_val) / (max_val - min_val) if max_val != min_val else image


def largest_label_volume(im, bg=-1):
    """Find the largest labeled volume"""
    vals, counts = np.unique(im, return_counts=True)

    # Filter background values
    background_mask = vals != bg
    counts = counts[background_mask]
    vals = vals[background_mask]

    return vals[np.argmax(counts)] if len(counts) > 0 else None


def segment_lung_mask(image, fill_lung_structures=True):
    """Segment the lung from the given 3D image"""
    # Create binary image
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Fill the air around the person
    background_label = labels[0, 0, 0]
    binary_image[background_label == labels] = 2

    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    # Make binary and invert
    binary_image = 1 - (binary_image - 1)

    # Remove air pockets
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image


def process_and_save(patient_pixels, output_dir, processing_func, **kwargs):
    """Generic function to process and save slices"""
    os.makedirs(output_dir, exist_ok=True)

    for i, slice_pixels in enumerate(tqdm(patient_pixels)):
        # Process the slice according to the provided function
        processed_image = processing_func(slice_pixels, **kwargs)

        # Save as PNG
        output_path = os.path.join(output_dir, f"slice_{i:04d}.png")
        plt.imsave(output_path, processed_image, cmap=plt.cm.gray)

    return len(patient_pixels)


def basic_processing(slice_pixels, **kwargs):
    """Basic normalization processing"""
    return normalize_image(slice_pixels)


def raw_processing(slice_pixels, window_center=20, window_width=400, **kwargs):
    """Raw processing with window settings"""
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    windowed_image = np.clip(slice_pixels, min_value, max_value)

    return ((windowed_image - min_value) / (max_value - min_value) * 255).astype(
        np.uint8
    ) / 255.0


def segmented_processing(slice_pixels, lung_mask=None, slice_idx=0, **kwargs):
    """Segmented processing with lung mask"""
    slice_mask = lung_mask[slice_idx]
    masked_slice = slice_pixels * slice_mask
    normalized_slice = normalize_image(masked_slice)

    # Create black background for use with gray_r colormap
    final_image = np.zeros_like(normalized_slice)
    final_image[slice_mask == 1] = 1.0 - normalized_slice[slice_mask == 1]

    return final_image


def convert_dicom(
    input_dir, output_dir, method="segmented", window_center=20, window_width=400
):
    """Main conversion function with method selection"""
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist!")
        return

    try:
        print(f"Loading DICOM files from {input_dir}...")
        patient_scans = load_scan(input_dir)

        print("Converting to Hounsfield units...")
        patient_pixels = get_pixels_hu(patient_scans)

        # Method-specific processing
        if method == "segmented":
            print("Segmenting lungs...")
            lung_mask = segment_lung_mask(patient_pixels, fill_lung_structures=True)
            print(f"Saving PNG files to {output_dir}...")
            count = 0
            for i, slice_pixels in enumerate(tqdm(patient_pixels)):
                processed = segmented_processing(
                    slice_pixels, lung_mask=lung_mask, slice_idx=i
                )
                output_path = os.path.join(output_dir, f"slice_{i+1:03d}.png")
                os.makedirs(output_dir, exist_ok=True)
                plt.imsave(output_path, processed, cmap=plt.cm.gray_r)
                count += 1
        elif method == "raw":
            print(f"Saving PNG files to {output_dir} with window settings...")
            count = 0
            for i, slice_pixels in enumerate(tqdm(patient_pixels)):
                processed = raw_processing(
                    slice_pixels, window_center=window_center, window_width=window_width
                )
                output_path = os.path.join(output_dir, f"slice_{i+1:03d}.png")
                os.makedirs(output_dir, exist_ok=True)
                plt.imsave(output_path, processed, cmap=plt.cm.gray)
                count += 1
        else:  # basic
            print(f"Saving PNG files to {output_dir} with basic normalization...")
            count = 0
            for i, slice_pixels in enumerate(tqdm(patient_pixels)):
                processed = basic_processing(slice_pixels)
                output_path = os.path.join(output_dir, f"slice_{i+1:03d}.png")
                os.makedirs(output_dir, exist_ok=True)
                plt.imsave(output_path, processed, cmap=plt.cm.gray)
                count += 1

        print(f"Successfully converted {count} DICOM slices to PNG format.")

    except Exception as e:
        print(f"An error occurred: {e}")


# LIDC Dataset Processing Functions
def count_dicom_files(folder_path):
    """Count the number of DICOM files in a folder"""
    return len([f for f in os.listdir(folder_path) if f.endswith(".dcm")])


def get_valid_folders(base_path, min_dicom_files=30):
    """Find, for each patient, the inner folder containing the most DICOM files (if above threshold)"""
    valid_folders = []

    # Walk through the LIDC-IDRI directory structure
    for patient_folder in os.listdir(base_path):
        if not patient_folder.startswith("LIDC-IDRI-"):
            continue

        patient_path = os.path.join(base_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        # Find all inner folders and count DICOM files
        max_dicom_count = 0
        max_inner_folder = None
        for subfolder in os.listdir(patient_path):
            subfolder_path = os.path.join(patient_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for inner_folder in os.listdir(subfolder_path):
                inner_folder_path = os.path.join(subfolder_path, inner_folder)
                if not os.path.isdir(inner_folder_path):
                    continue
                dicom_count = count_dicom_files(inner_folder_path)
                if dicom_count > max_dicom_count:
                    max_dicom_count = dicom_count
                    max_inner_folder = inner_folder_path
        if max_inner_folder and max_dicom_count >= min_dicom_files:
            valid_folders.append((max_inner_folder, max_dicom_count))
    return valid_folders


def create_output_folder(output_base, patient_num):
    """Create output folder with proper naming"""
    folder_name = f"P-{patient_num:04d}"
    output_path = os.path.join(output_base, folder_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def process_dataset(input_base_path, output_base_path, method="segmented"):
    """Process the entire LIDC-IDRI dataset"""
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)

    # Get all valid folders
    print("Scanning for valid DICOM folders...")
    valid_folders = get_valid_folders(input_base_path)

    if not valid_folders:
        print("No valid folders found!")
        return

    print(f"Found {len(valid_folders)} valid folders to process")

    # Process each valid folder
    for idx, (input_folder, dicom_count) in enumerate(valid_folders, 1):
        # Create output folder
        output_folder = create_output_folder(output_base_path, idx)

        print(f"\nProcessing patient {idx:04d}")
        print(f"Input folder: {input_folder}")
        print(f"Number of DICOM files: {dicom_count}")
        print(f"Output folder: {output_folder}")

        try:
            # Convert DICOM to PNG
            convert_dicom(input_folder, output_folder, method=method)
            print(f"Successfully processed patient {idx:04d}")

        except Exception as e:
            print(f"Error processing patient {idx:04d}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Process LIDC-IDRI dataset and convert DICOM to PNG"
    )
    parser.add_argument("--input", required=True, help="Base path to LIDC-IDRI dataset")
    parser.add_argument(
        "--output", required=True, help="Base path for output PNG files"
    )
    parser.add_argument(
        "--method",
        default="segmented",
        choices=["basic", "raw", "segmented"],
        help="Conversion method: basic, raw, or segmented",
    )
    parser.add_argument(
        "--window-center", type=int, default=20, help="Window center for raw conversion"
    )
    parser.add_argument(
        "--window-width", type=int, default=400, help="Window width for raw conversion"
    )

    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist!")
        exit(1)

    # Process the dataset
    process_dataset(args.input, args.output, args.method)


if __name__ == "__main__":
    main()

# C:/Users/rakib/anaconda3/Scripts/conda.exe run -p C:\Users\rakib\anaconda3 --no-capture-output python lidc_processor.py --input LIDC-IDRI --output Segmentation