# Segmentation to Bounding Box Annotator

This is a tool for converting segmentation masks to bounding boxes with class label annotations. It provides a user-friendly interface for assigning class labels to objects detected in segmentation masks.

## Features

- Automatically extract bounding boxes from segmentation masks
- Assign class labels to each detected object
- Navigate through images in your dataset
- Save annotations in JSON format
- Export annotations to YOLO format

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/seg-to-box.git
cd seg-to-box
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the following structure:
```
dataset/
  ├── images/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── annotations/
      ├── image1.jpg (or .png)
      ├── image2.jpg (or .png)
      └── ...
```

2. Run the application:

**Windows:**
```
run.bat
```

**Unix/Linux/macOS:**
```
chmod +x run.sh
./run.sh
```

Or directly:
```
streamlit run app.py
```

3. In the application:
   - Enter the path to your dataset
   - Enter the class labels (comma-separated)
   - Click "Load Dataset"
   - Assign class labels to each detected bounding box
   - Navigate through images using the "Previous" and "Next" buttons
   - Click "Save JSON" to save in JSON format
   - Click "Save YOLO" to export in YOLO format

## Annotations Format

### JSON Format
Annotations are saved in JSON format in the `output` directory inside your dataset folder. The format is as follows:

```json
{
  "image1.jpg": {
    "boxes": [[x1, y1, x2, y2], ...],
    "labels": ["class1", "class2", ...]
  },
  "image2.jpg": {
    "boxes": [[x1, y1, x2, y2], ...],
    "labels": ["class1", "class3", ...]
  },
  ...
}
```

### YOLO Format
YOLO format annotations are saved in the `output_yolo` directory with one .txt file per image. Each .txt file contains one line per object in the format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id` is the index of the class in your labels list
- All coordinates are normalized to [0, 1]

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pillow

## License

MIT License 