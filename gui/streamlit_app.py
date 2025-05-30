"""
Streamlit web-based interface for the Image Annotation App.

This module provides:
- Web-based annotation interface using Streamlit
- Dataset setup and configuration
- Image display and navigation
- Box generation and labeling
- Export functionality
"""

import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any

try:
    from utils_new.mask_to_boxes import mask_to_boxes, visualize_boxes
    from utils_new.image_loader import load_image_and_mask, load_dataset, find_mask_file, validate_dataset_structure
    from utils_new.io_utils import save_annotations_to_json, export_annotations_to_yolo, export_annotations_to_coco
except ImportError:
    # Fallback for development
    import sys
    sys.path.append('..')
    from utils_new.mask_to_boxes import mask_to_boxes, visualize_boxes
    from utils_new.image_loader import load_image_and_mask, load_dataset, find_mask_file, validate_dataset_structure
    from utils_new.io_utils import save_annotations_to_json, export_annotations_to_yolo, export_annotations_to_coco


# Set page configuration
st.set_page_config(
    page_title="Image Annotation App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'dataset_path' not in st.session_state:
        st.session_state.dataset_path = ""
    if 'class_labels' not in st.session_state:
        st.session_state.class_labels = []
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0
    if 'image_files' not in st.session_state:
        st.session_state.image_files = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False


def setup_page():
    """Display the dataset setup page."""
    st.markdown('<h1 class="main-header">🖼️ Image Annotation App Setup</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to the Image Annotation App!</h3>
        <p>This app helps you convert segmentation masks to bounding box annotations.</p>
        <p><strong>Required Dataset Structure:</strong></p>
        <pre>
📂 dataset_folder/
├── 📂 images/         (Contains .jpg, .png, or other image files)
└── 📂 annotations/    (Contains segmentation mask files)
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset configuration
    st.subheader("📁 Dataset Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        dataset_path = st.text_input(
            "Dataset Path:",
            value=st.session_state.dataset_path,
            placeholder="Enter the path to your dataset folder...",
            help="Path to the folder containing 'images' and 'annotations' subdirectories"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        validate_btn = st.button("✅ Validate", key="validate_dataset")
    
    # Validation results
    if validate_btn and dataset_path:
        with st.spinner("Validating dataset structure..."):
            is_valid, message = validate_dataset_structure(dataset_path)
            
        if is_valid:
            st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
            st.session_state.dataset_path = dataset_path
        else:
            st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
    
    # Class labels configuration
    st.subheader("🏷️ Class Labels Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        class_labels_input = st.text_input(
            "Class Labels (comma-separated):",
            value=",".join(st.session_state.class_labels) if st.session_state.class_labels else "",
            placeholder="person, car, bicycle, dog, cat...",
            help="Enter all possible object classes you want to annotate"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Use Defaults", key="use_defaults"):
            default_labels = ["person", "car", "bicycle", "dog", "cat"]
            st.session_state.class_labels = default_labels
            st.rerun()
    
    # Example
    st.caption("💡 Example: person, car, bicycle, dog, cat")
    
    # Load dataset button
    st.subheader("🚀 Load Dataset")
    
    if st.button("Load Dataset", type="primary", key="load_dataset"):
        # Validate inputs
        if not dataset_path:
            st.error("Please enter a valid dataset path")
            return
        
        # Validate dataset structure
        is_valid, message = validate_dataset_structure(dataset_path)
        if not is_valid:
            st.error(f"Dataset validation failed: {message}")
            return
        
        # Parse class labels
        labels = [label.strip() for label in class_labels_input.split(",") if label.strip()]
        if not labels:
            st.error("Please enter at least one class label")
            return
        
        # Load image files
        with st.spinner("Loading dataset..."):
            image_files = load_dataset(dataset_path)
            
        if not image_files:
            st.error("No images found in the specified directory")
            return
        
        # Success - update session state
        st.session_state.dataset_path = dataset_path
        st.session_state.class_labels = labels
        st.session_state.image_files = image_files
        st.session_state.current_image_idx = 0
        st.session_state.annotations = {}
        st.session_state.setup_complete = True
        
        st.success(f"✅ Dataset loaded successfully! Found {len(image_files)} images.")
        st.rerun()


def annotation_page():
    """Display the main annotation interface."""
    st.markdown('<h1 class="main-header">🖼️ Image Annotation Interface</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Dataset info
        st.subheader("📊 Dataset Info")
        st.write(f"**Path:** {st.session_state.dataset_path}")
        st.write(f"**Images:** {len(st.session_state.image_files)}")
        st.write(f"**Classes:** {len(st.session_state.class_labels)}")
        
        st.write("**Class Labels:**")
        for i, label in enumerate(st.session_state.class_labels):
            st.write(f"  {i+1}. {label}")
        
        st.divider()
        
        # New dataset button
        if st.button("📁 Load New Dataset", key="new_dataset"):
            st.session_state.setup_complete = False
            st.rerun()
    
    # Main content area
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    # Navigation controls
    with col1:
        if st.button("⬅️ Previous", key="prev_image", disabled=st.session_state.current_image_idx == 0):
            st.session_state.current_image_idx -= 1
            st.rerun()
    
    with col2:
        if st.button("Next ➡️", key="next_image", disabled=st.session_state.current_image_idx >= len(st.session_state.image_files) - 1):
            st.session_state.current_image_idx += 1
            st.rerun()
    
    with col3:
        st.write(f"**Image {st.session_state.current_image_idx + 1} of {len(st.session_state.image_files)}**")
    
    # Get current image info
    image_filename = st.session_state.image_files[st.session_state.current_image_idx]
    image_path = os.path.join(st.session_state.dataset_path, "images", image_filename)
    
    # Display current image info
    st.subheader(f"📷 Current Image: {image_filename}")
    
    # Find and load mask
    annotations_dir = os.path.join(st.session_state.dataset_path, "annotations")
    mask_filename = find_mask_file(image_filename, annotations_dir)
    
    if mask_filename is None:
        st.error(f"❌ No corresponding mask file found for {image_filename}")
        return
    
    mask_path = os.path.join(annotations_dir, mask_filename)
    
    # Load image and mask
    with st.spinner("Loading image and mask..."):
        image, mask = load_image_and_mask(image_path, mask_path)
    
    if image is None or mask is None:
        st.error(f"❌ Failed to load image or mask: {image_filename}")
        return
    
    # Generate boxes button and box generation
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🎯 Generate Boxes", key="generate_boxes"):
            with st.spinner("Generating bounding boxes..."):
                boxes = mask_to_boxes(mask)
            
            if boxes:
                # Initialize annotations for this image
                st.session_state.annotations[image_path] = {
                    'boxes': boxes,
                    'labels': ["" for _ in range(len(boxes))]
                }
                st.success(f"✅ Generated {len(boxes)} bounding boxes")
                st.rerun()
            else:
                st.warning("⚠️ No objects found in the mask")
    
    # Get current annotations
    current_annotations = st.session_state.annotations.get(image_path, {'boxes': [], 'labels': []})
    boxes = current_annotations['boxes']
    labels = current_annotations['labels']
    
    # Display image with boxes
    if boxes:
        # Create visualization
        labels_for_viz = [label if label else f"Box {i+1}" for i, label in enumerate(labels)]
        fig = visualize_boxes(image, boxes, labels_for_viz)
        
        # Display the image
        st.pyplot(fig, use_container_width=True)
        
        # Box labeling interface
        st.subheader("🏷️ Assign Labels to Bounding Boxes")
        
        # Create columns for better layout
        n_boxes = len(boxes)
        if n_boxes > 0:
            # Create a grid layout for box labeling
            cols_per_row = 3
            rows = (n_boxes + cols_per_row - 1) // cols_per_row
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    box_idx = row * cols_per_row + col_idx
                    if box_idx < n_boxes:
                        with cols[col_idx]:
                            st.write(f"**Box {box_idx + 1}:**")
                            
                            # Current label
                            current_label = labels[box_idx] if box_idx < len(labels) else ""
                            
                            # Create selectbox for this box
                            selected_label = st.selectbox(
                                "Class:",
                                [""] + st.session_state.class_labels,
                                index=0 if not current_label else st.session_state.class_labels.index(current_label) + 1 if current_label in st.session_state.class_labels else 0,
                                key=f"box_label_{box_idx}",
                                label_visibility="collapsed"
                            )
                            
                            # Update the label in session state
                            if box_idx < len(labels):
                                st.session_state.annotations[image_path]['labels'][box_idx] = selected_label
                            
                            # Show box info
                            x1, y1, x2, y2 = boxes[box_idx]
                            st.caption(f"Size: {x2-x1} × {y2-y1}")
    else:
        # No boxes generated yet, show original image
        st.image(image, use_column_width=True)
        st.info("👆 Click 'Generate Boxes' to create bounding boxes from the segmentation mask")
    
    # Statistics
    if boxes:
        st.subheader("📊 Annotation Statistics")
        total_boxes = len(boxes)
        labeled_boxes = sum(1 for label in labels if label)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Boxes", total_boxes)
        with col2:
            st.metric("Labeled Boxes", labeled_boxes)
        with col3:
            st.metric("Progress", f"{labeled_boxes/total_boxes*100:.1f}%" if total_boxes > 0 else "0%")
    
    # Save annotations section
    st.subheader("💾 Save Annotations")
    
    if not st.session_state.annotations:
        st.info("No annotations to save. Generate some bounding boxes first!")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save JSON", key="save_json", type="secondary"):
                output_dir = os.path.join(st.session_state.dataset_path, "output")
                os.makedirs(output_dir, exist_ok=True)
                
                json_path = os.path.join(output_dir, "annotations.json")
                if save_annotations_to_json(st.session_state.annotations, json_path):
                    st.success(f"✅ JSON saved to {json_path}")
                else:
                    st.error("❌ Failed to save JSON")
        
        with col2:
            if st.button("💾 Save YOLO", key="save_yolo", type="secondary"):
                output_dir = os.path.join(st.session_state.dataset_path, "output_yolo")
                os.makedirs(output_dir, exist_ok=True)
                
                # Get image dimensions
                sample_image = Image.open(image_path)
                image_width, image_height = sample_image.size
                
                if export_annotations_to_yolo(
                    st.session_state.annotations, 
                    output_dir, 
                    image_width, 
                    image_height, 
                    st.session_state.class_labels
                ):
                    st.success(f"✅ YOLO saved to {output_dir}")
                else:
                    st.error("❌ Failed to save YOLO")
        
        with col3:
            if st.button("💾 Save COCO", key="save_coco", type="secondary"):
                output_dir = os.path.join(st.session_state.dataset_path, "output_coco")
                os.makedirs(output_dir, exist_ok=True)
                
                if export_annotations_to_coco(
                    st.session_state.annotations,
                    output_dir,
                    st.session_state.class_labels,
                    st.session_state.image_files,
                    st.session_state.dataset_path
                ):
                    st.success(f"✅ COCO saved to {output_dir}")
                else:
                    st.error("❌ Failed to save COCO")
        
        # Show overall progress
        total_images = len(st.session_state.image_files)
        annotated_images = len(st.session_state.annotations)
        
        st.divider()
        st.write("**Overall Progress:**")
        progress = annotated_images / total_images if total_images > 0 else 0
        st.progress(progress)
        st.write(f"{annotated_images}/{total_images} images annotated ({progress*100:.1f}%)")


def main():
    """Main function for the Streamlit app."""
    # Initialize session state
    initialize_session_state()
    
    # Show appropriate page
    if not st.session_state.setup_complete:
        setup_page()
    else:
        annotation_page()


if __name__ == "__main__":
    main() 