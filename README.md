# LIDC-IDRI DICOM to PNG Converter

This tool processes the LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative) dataset, converting DICOM files to PNG images with various processing methods including lung segmentation.

## Features

- **Automatic Dataset Scanning**: Automatically finds and processes valid DICOM folders in the LIDC-IDRI dataset
- **Multiple Processing Methods**: 
  - Basic normalization
  - Raw windowing
  - **Lung segmentation** (recommended for lung analysis)
- **Lung Segmentation**: Isolates lung tissue from CT scans with white backgrounds for non-lung areas
- **Batch Processing**: Processes entire dataset (1000+ patients) automatically
- **Progress Tracking**: Real-time progress bars for each patient

## Requirements

```bash
pip install numpy pydicom matplotlib tqdm scikit-image argparse
```

## Usage

### Basic Command Structure

```bash
python lidc_processor.py --input <input_path> --output <output_path> [options]
```

### Examples

#### 1. Segmented Processing (Recommended)
```bash
cd e:\LDLN && python lidc_processor.py --input "LIDC-IDRI" --output "Segmentation"
```
Or simply:
```bash
python lidc_processor.py --input "LIDC-IDRI" --output "Segmentation"
```

#### 2. Raw Processing with Custom Window Settings
```bash
python lidc_processor.py --input "LIDC-IDRI" --output "Raw_Output" --method raw --window-center 20 --window-width 400
```

#### 3. Basic Normalization
```bash
python lidc_processor.py --input "LIDC-IDRI" --output "Basic_Output" --method basic
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | ✅ | - | Base path to LIDC-IDRI dataset |
| `--output` | ✅ | - | Base path for output PNG files |
| `--method` | ❌ | `segmented` | Processing method: `basic`, `raw`, or `segmented` |
| `--window-center` | ❌ | `20` | Window center for raw conversion (HU) |
| `--window-width` | ❌ | `400` | Window width for raw conversion (HU) |

## Processing Methods

### 1. Segmented Processing (`--method segmented`)
**Recommended for lung analysis**

- Automatically segments lung tissue from CT scans
- Creates binary lung masks using HU threshold (-320)
- Outputs images with:
  - **White backgrounds** for areas without lung tissue
  - **Grayscale lung regions** with proper contrast
- Perfect for machine learning applications focusing on lung pathology

**Use cases:**
- Lung nodule detection
- Lung disease classification
- Computer-aided diagnosis systems

### 2. Raw Processing (`--method raw`)
- Applies windowing to highlight specific tissue types
- Configurable window center and width
- Standard medical imaging display

**Use cases:**
- General medical imaging display
- Custom tissue contrast enhancement

### 3. Basic Processing (`--method basic`)
- Simple min-max normalization
- Preserves all original image information
- No segmentation or windowing applied

**Use cases:**
- Research requiring full image data
- Custom preprocessing pipelines

## Output Structure

The tool creates organized output folders:

```
Segmentation/
├── P-0001/
│   ├── slice_001.png
│   ├── slice_002.png
│   └── ...
├── P-0002/
│   ├── slice_001.png
│   ├── slice_002.png
│   └── ...
└── ...
```

## Technical Details

### Lung Segmentation Algorithm

1. **HU Thresholding**: Uses -320 HU threshold to separate air/lung from tissue
2. **Connected Component Analysis**: Identifies largest lung regions
3. **Background Removal**: Removes external air around patient
4. **Hole Filling**: Fills internal lung structures for better segmentation
5. **Post-processing**: Removes small air pockets and artifacts

### Image Processing Pipeline

```
DICOM Files → HU Conversion → Lung Segmentation → Normalization → PNG Output
```

### Key Functions

- `segment_lung_mask()`: Core lung segmentation algorithm
- `get_pixels_hu()`: Converts DICOM pixel values to Hounsfield Units
- `segmented_processing()`: Applies mask and creates final images
- `normalize_image()`: Normalizes pixel values to 0-1 range

## Performance

- **Processing Speed**: ~50-80 slices per second
- **Memory Usage**: Processes one patient at a time to minimize memory usage
- **Dataset Coverage**: Handles all 1000+ patients in LIDC-IDRI dataset
- **Error Handling**: Continues processing even if individual patients fail

## Troubleshooting

### Common Issues

1. **Black images instead of white backgrounds**
   - Fixed in current version using `plt.cm.gray_r` colormap
   - Ensures proper color mapping for segmented images

2. **Memory errors**
   - Script processes one patient at a time
   - Close other memory-intensive applications

3. **Missing DICOM files**
   - Tool automatically skips folders with <30 DICOM files
   - Check input path structure matches LIDC-IDRI format

### Expected Console Output

```
Scanning for valid DICOM folders...
Found 1010 valid folders to process

Processing patient 0001
Input folder: LIDC-IDRI\LIDC-IDRI-0001\...
Number of DICOM files: 133
Output folder: Segmentation\P-0001
Loading DICOM files...
Converting to Hounsfield units...
Segmenting lungs...
Saving PNG files...
100%|████████████████████| 133/133 [00:02<00:00, 67.67it/s]
Successfully converted 133 DICOM slices to PNG format.
Successfully processed patient 0001
```

## File Information

- **Script**: `lidc_processor.py`
- **Input Format**: DICOM (.dcm files)
- **Output Format**: PNG (8-bit grayscale)
- **Image Naming**: `slice_001.png`, `slice_002.png`, etc.
- **Folder Naming**: `P-0001`, `P-0002`, etc.

## Research Applications

This tool is particularly useful for:

- **Lung Cancer Detection**: Segmented images highlight lung regions for nodule detection
- **COVID-19 Analysis**: Lung segmentation helps focus on affected areas
- **Pulmonary Disease Classification**: Clean lung masks improve model accuracy
- **Medical Image Processing Research**: Standardized preprocessing for lung CT analysis

## Citation

If you use this tool in your research, please cite the LIDC-IDRI dataset:

```
Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., 
Reeves, A. P., ... & Clarke, L. P. (2011). The lung image database consortium 
(LIDC) and image database resource initiative (IDRI): a completed reference 
database of lung nodules on CT scans. Medical physics, 38(2), 915-931.
```

## License

This tool is provided as-is for research purposes. Please respect the LIDC-IDRI dataset license terms.
# ldlns
