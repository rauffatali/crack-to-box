# Segmentation to Bounding Box Annotator

This is a tool for converting segmentation masks to bounding boxes with class label annotations. It provides both desktop GUI and web-based interfaces for assigning class labels to objects detected in segmentation masks.

## Features

- **Two Interface Options:**
  - **Tkinter GUI** - Full-featured desktop application (recommended)
  - **Streamlit Web Interface** - Web-based interface (*Note: Not fully implemented - TODO*)
- Automatically extract bounding boxes from segmentation masks
- Interactive box editing (resize, move, delete)
- Assign class labels to each detected object
- Navigate through images in your dataset
- Save annotations in multiple formats: JSON, YOLO, and COCO
- Real-time visualization of annotations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/seg-to-box.git
cd seg-to-box
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

**Interactive Mode (Choose Interface):**
```bash
python main.py
```

**Direct Launch Options:**

**Tkinter Desktop GUI (Recommended):**
```bash
python main.py --tk
```

**Streamlit Web Interface (*Not fully implemented*):**
```bash
python main.py --st
```

**Alternative Methods:**

**Windows:**
```bash
run.bat
```

**Unix/Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

### Dataset Structure

Prepare your dataset in the following structure:
```
dataset/
  ├── images/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── annotations/
      ├── image1.png (segmentation masks)
      ├── image2.png
      └── ...
```

### Application Workflow

1. **Launch the application** using one of the methods above
2. **Setup Phase:**
   - Enter the path to your dataset folder
   - Enter class labels (comma-separated, e.g., "person, car, bicycle")
   - Click "Setup" or "Load Dataset"
3. **Annotation Phase:**
   - Click "Generate Boxes" to automatically create bounding boxes from segmentation masks
   - Use mouse to:
     - Move boxes by dragging
     - Resize boxes using corner handles
     - Right-click to delete boxes
   - Assign class labels by clicking on boxes and selecting from dropdown
   - Navigate between images using "Previous"/"Next" buttons
4. **Export Phase:**
   - Save annotations in your preferred format:
     - **JSON** - Custom format for easy reading
     - **YOLO** - For YOLO model training
     - **COCO** - For COCO-style datasets

## Annotation Formats

### JSON Format
Annotations are saved in JSON format in the `output` directory:

```json
{
  "image1.jpg": {
    "boxes": [[x1, y1, x2, y2], ...],
    "labels": ["class1", "class2", ...]
  },
  "image2.jpg": {
    "boxes": [[x1, y1, x2, y2], ...],
    "labels": ["class1", "class3", ...]
  }
}
```

### YOLO Format
YOLO format annotations are saved in the `output_yolo` directory with one `.txt` file per image:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where all coordinates are normalized to [0, 1].

### COCO Format
COCO format annotations are saved in the `output_coco` directory as a complete COCO dataset structure with `annotations.json`.

## Project Structure

```
seg-to-box/
├── main.py                    # Entry point with interface selection
├── gui/
│   ├── setup_window.py        # Dataset configuration window
│   ├── annotation_window.py   # Main annotation interface
│   └── streamlit_app.py       # Web interface (TODO: fix imports)
├── utils/
│   ├── mask_to_boxes.py       # Mask to bounding box conversion
│   ├── image_loader.py        # Dataset loading and validation
│   ├── io_utils.py           # Export functionality
│   └── box_editor.py         # Interactive box editing
├── assets/                    # Sample images and resources
└── requirements.txt          # Python dependencies
```

## Requirements

- Python 3.7+
- torch==2.0.1
- torchvision==0.15.2
- opencv-python==4.7.0.72
- numpy==1.24.3
- matplotlib==3.7.1
- Pillow==9.5.0
- streamlit==1.22.0 (for web interface, when fully implemented)

## Known Issues & TODOs

- **Streamlit Interface**: The web-based interface is not fully implemented and has import path issues that need to be resolved
- **Future Enhancements**: 
  - Complete Streamlit implementation
  <!-- - Add keyboard shortcuts for faster annotation
  - Implement batch processing features -->
  - Add annotation validation tools

## License

MIT License 