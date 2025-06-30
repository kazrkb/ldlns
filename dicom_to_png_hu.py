import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import nibabel as nib

def parse_xml_nodule(xml_path):
    """Parse LIDC XML and return a list of nodules with their z position and edge coordinates."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'lidc': 'http://www.nih.gov'}
    nodules = []
    for nodule in root.findall('.//lidc:unblindedReadNodule', ns):
        for roi in nodule.findall('lidc:roi', ns):
            z = float(roi.find('lidc:imageZposition', ns).text.strip())
            edge_points = [
                (int(edge.find('lidc:xCoord', ns).text.strip()), int(edge.find('lidc:yCoord', ns).text.strip()))
                for edge in roi.findall('lidc:edgeMap', ns)
            ]
            if edge_points:
                nodules.append({'z': z, 'edge': edge_points})
    return nodules

def get_pixels_hu(dicom):
    image = dicom.pixel_array.astype(np.int16)
    image[image == -2000] = 0
    intercept = getattr(dicom, 'RescaleIntercept', 0)
    slope = getattr(dicom, 'RescaleSlope', 1)
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return image

def window_image(image, window_center=-600, window_width=1500):
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    windowed_image = np.clip(image, min_value, max_value)
    norm_image = ((windowed_image - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(norm_image)

def dicom_to_png(input_folder, output_folder, window_center=-600, window_width=1500):
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            file_path = os.path.join(output_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder, exist_ok=True)
    dicom_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dcm')]
    for dicom_file in dicom_files:
        dcm_path = os.path.join(input_folder, dicom_file)
        ds = pydicom.dcmread(dcm_path, force=True)
        hu_image = get_pixels_hu(ds)
        norm_image = window_image(hu_image, window_center, window_width)
        png_name = os.path.splitext(dicom_file)[0] + '.png'
        png_path = os.path.join(output_folder, png_name)
        plt.imsave(png_path, norm_image, cmap='gray')
    print(f"Saved {len(dicom_files)} PNG files to {output_folder}")

def dicom_to_png_with_nodule_marking(input_folder, output_folder, xml_path, window_center=-600, window_width=1500, z_tolerance=0.1):
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            file_path = os.path.join(output_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder, exist_ok=True)
    dicom_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dcm')]
    dcm_info = []
    for dicom_file in dicom_files:
        dcm_path = os.path.join(input_folder, dicom_file)
        ds = pydicom.dcmread(dcm_path, force=True)
        z = float(ds.ImagePositionPatient[2])
        dcm_info.append((z, dicom_file, ds))
    nodules = parse_xml_nodule(xml_path)
    for z, dicom_file, ds in dcm_info:
        hu_image = get_pixels_hu(ds)
        norm_image = window_image(hu_image, window_center, window_width)
        overlay = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
        nodule_marks = 0
        for nodule in nodules:
            if abs(nodule['z'] - z) < z_tolerance:
                pts = np.array(nodule['edge'], np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=(255,0,0), thickness=2)
                nodule_marks += 1
        png_name = os.path.splitext(dicom_file)[0] + '.png'
        png_path = os.path.join(output_folder, png_name)
        plt.imsave(png_path, overlay)
        print(f"{png_name}: {nodule_marks} nodules (red) marked.")
    print(f"Saved {len(dicom_files)} PNG files with nodule markings to {output_folder}")

def dicom_to_nodule_mask_images(input_folder, xml_path, output_folder="NODULE_MASK_IMAGES", z_tolerance=0.1):
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            file_path = os.path.join(output_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder, exist_ok=True)
    dicom_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dcm')]
    dcm_info = []
    for dicom_file in dicom_files:
        dcm_path = os.path.join(input_folder, dicom_file)
        ds = pydicom.dcmread(dcm_path, force=True)
        z = float(ds.ImagePositionPatient[2])
        dcm_info.append((z, dicom_file, ds))
    nodules = parse_xml_nodule(xml_path)
    for z, dicom_file, ds in dcm_info:
        img_shape = ds.pixel_array.shape
        mask = np.zeros(img_shape, dtype=np.uint8)
        nodule_marks = 0
        for nodule in nodules:
            if abs(nodule['z'] - z) < z_tolerance:
                pts = np.array(nodule['edge'], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
                nodule_marks += 1
        png_name = os.path.splitext(dicom_file)[0] + '.png'
        png_path = os.path.join(output_folder, png_name)
        plt.imsave(png_path, mask, cmap='gray')
        print(f"{png_name}: {nodule_marks} nodule regions (filled white) in mask.")
    print(f"Saved {len(dicom_files)} mask images to {output_folder}")

def create_nodule_mask_volume(mask_folder, output_npy_path):
    """Create a 3D numpy array (volume) from the mask images and save as .npy. Shape: (height, width, num_slices) with correct axial orientation."""
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith('.png')])
    if not mask_files:
        print(f"No mask PNG files found in {mask_folder}")
        return
    # Read the first image to get shape
    first_mask = imageio.imread(os.path.join(mask_folder, mask_files[0]))
    if first_mask.ndim > 2:
        first_mask = first_mask[..., 0]  # Take first channel if not grayscale
    height, width = first_mask.shape
    num_slices = len(mask_files)
    mask_volume = np.zeros((height, width, num_slices), dtype=np.uint8)
    for i, fname in enumerate(mask_files):
        mask = imageio.imread(os.path.join(mask_folder, fname))
        if mask.ndim > 2:
            mask = mask[..., 0]  # Convert to grayscale if needed
        # Flip left-right, then rotate 90 deg clockwise
        mask_corrected = np.rot90(np.fliplr(mask), k=-1)
        mask_volume[:, :, i] = mask_corrected
    np.save(output_npy_path, mask_volume)
    print(f"Saved 3D mask volume to {output_npy_path} with shape {mask_volume.shape}")

def save_mask_volume_as_nifti(npy_path, nii_path):
    """Save the 3D mask volume as a NIfTI file (.nii.gz) for medical imaging viewers."""
    mask_volume = np.load(npy_path)
    nii_img = nib.Nifti1Image(mask_volume.astype(np.uint8), affine=np.eye(4))
    nib.save(nii_img, nii_path)
    print(f"Saved mask volume as NIfTI file: {nii_path} (shape: {mask_volume.shape})")

if __name__ == "__main__":
    input_folder = "INPUT"
    output_nodule_folder = "OUTPUT_NODULES"
    converted_png_folder = "CONVERTED_PNG"
    mask_output_folder = "NODULE_MASK_IMAGES"
    mask_volume_npy = "NODULE_MASK_VOLUME.npy"
    mask_volume_nii = "NODULE_MASK_VOLUME.nii.gz"
    xml_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.xml')]
    if not xml_files:
        print("No XML file found in the INPUT folder.")
    else:
        xml_path = os.path.join(input_folder, xml_files[0])
        print(f"Using XML file: {xml_path}")
        dicom_to_png(input_folder, converted_png_folder)
        dicom_to_png_with_nodule_marking(input_folder, output_nodule_folder, xml_path)
        dicom_to_nodule_mask_images(input_folder, xml_path, mask_output_folder)
        create_nodule_mask_volume(mask_output_folder, mask_volume_npy)
        save_mask_volume_as_nifti(mask_volume_npy, mask_volume_nii)