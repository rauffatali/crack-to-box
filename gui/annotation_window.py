"""
Main annotation window with image canvas, box tools, and annotation controls.

This module provides:
- SegmentationAnnotator: Main annotation interface class
- Image display and navigation
- Interactive bounding box editing
- Label assignment and management
- Save/export functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
from PIL import Image, ImageTk, ImageOps
from typing import List, Optional
import numpy as np

from utils.mask_to_boxes import mask_to_boxes
from utils.image_loader import load_image_and_mask, find_mask_file
from utils.io_utils import save_annotations_to_json, export_annotations_to_yolo, export_annotations_to_coco
from utils.box_editor import BoxEditor
from gui.setup_window import show_setup_dialog


class SegmentationAnnotator:
    """
    Main annotation interface for the Image Annotation App.
    
    This class provides:
    - Image display and navigation
    - Automatic box generation from masks
    - Interactive box editing and labeling
    - Save/export in multiple formats
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the annotation interface.
        
        Args:
            root (tk.Tk): Main window root
        """
        self.root = root
        self.setup_window()
        
        # Application state
        self.dataset_path = ""
        self.class_labels = []
        self.current_image_idx = 0
        self.image_files = []
        self.annotations = {}  # Will store image_path: {boxes: [], labels: []}
        
        # Image display state
        self.original_image = None
        self.current_image = None
        self.displayed_image = None
        self.canvas_image = None
        self.scale_factor = 1.0
        self.display_width = 0
        self.display_height = 0
        
        # Mask overlay state
        self.current_mask = None
        self.displayed_mask = None
        self.canvas_mask = None
        self.mask_overlay_visible = False
        
        # Welcome message state
        self.welcome_text_id = None
        
        # Box editor
        self.box_editor: Optional[BoxEditor] = None
        
        # Setup initial UI
        self.setup_initial_ui()
        
    def setup_window(self):
        """Setup the main window properties."""
        # Get screen dimensions for responsive sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size (80% of screen)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # Center window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.title("Image Annotation App - Segmentation to Bounding Box")
        self.root.state('normal')
        
        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
    def setup_initial_ui(self):
        """Setup the initial UI before dataset is loaded."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        self._create_toolbar()
        self._create_left_panel()
        self._create_canvas_area()
        self._create_right_panel()
        self._create_status_bar()
        self._show_welcome_message()
        
    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar_frame = ttk.Frame(self.main_frame)
        toolbar_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        # Dataset management
        ttk.Button(toolbar_frame, text="📁 Open Dataset", command=self.show_setup_dialog).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        # Mask overlay button (initially disabled)
        self.mask_overlay_btn = ttk.Button(toolbar_frame, text="👁️ Show Mask (Ctrl+M)", command=self.toggle_mask_overlay, state=tk.DISABLED)
        self.mask_overlay_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Tools
        self.tools_frame = ttk.Frame(toolbar_frame)
        self.tools_frame.pack(side=tk.LEFT, padx=5)
        
        # Add undo button to the right corner
        self.undo_btn = ttk.Button(toolbar_frame, text="↶ Undo (Ctrl+Z)", command=self.undo_last_action, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
    def _create_left_panel(self):
        """Create the left control panel."""
        left_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="10")
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        left_frame.configure(width=200)
        
        # Image info
        info_frame = ttk.LabelFrame(left_frame, text="Image Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.image_info_text = tk.Text(info_frame, height=6, width=25, font=("Courier", 8))
        self.image_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Box controls
        box_frame = ttk.LabelFrame(left_frame, text="Box Controls", padding="10")
        box_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(box_frame, text="🎯 Generate from Mask", command=self.generate_boxes).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(box_frame, text="🧹 Clear All Boxes", command=self.clear_boxes).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(box_frame, text="📏 Filter Small Boxes", command=self.filter_small_boxes).pack(fill=tk.X)
        
        # View controls
        view_frame = ttk.LabelFrame(left_frame, text="View Controls", padding="10")
        view_frame.pack(fill=tk.X)
        
        ttk.Button(view_frame, text="🔍 Zoom In", command=self.zoom_in).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(view_frame, text="🔍 Zoom Out", command=self.zoom_out).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(view_frame, text="🏠 Fit to Window", command=self.fit_to_window).pack(fill=tk.X)
        
    def _create_canvas_area(self):
        """Create the central canvas area for image display."""
        canvas_frame = ttk.LabelFrame(self.main_frame, text="Image Canvas", padding="5")
        canvas_frame.grid(row=1, column=1, sticky="nsew", padx=5)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, takefocus=True)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Bind to resize events
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        
        # Bind keyboard shortcuts
        self.canvas.bind('<Control-m>', lambda e: self.toggle_mask_overlay())
        self.canvas.bind('<Control-z>', lambda e: self.undo_last_action())
        self.canvas.focus_set()
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Navigation frame at bottom of canvas
        nav_frame = ttk.Frame(canvas_frame)
        nav_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        nav_frame.grid_columnconfigure(1, weight=1)  # Center the navigation controls
        
        self.prev_btn = ttk.Button(nav_frame, text="⬅️ Previous", command=self.previous_image, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=0, padx=5)
        
        self.image_label = ttk.Label(nav_frame, text="No images loaded", font=("Arial", 10, "bold"))
        self.image_label.grid(row=0, column=1, padx=5)
        
        self.next_btn = ttk.Button(nav_frame, text="Next ➡️", command=self.next_image, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=2, padx=5)
        
        # Initialize box editor
        self.box_editor = BoxEditor(self.canvas)
        self.box_editor.on_box_selected = self._on_box_selected
        self.box_editor.on_box_modified = self._on_box_modified
        
    def _create_right_panel(self):
        """Create the right panel for box list and properties."""
        right_frame = ttk.LabelFrame(self.main_frame, text="Annotations", padding="10")
        right_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 0))
        right_frame.configure(width=250)
        
        # Properties panel (moved to top)
        props_frame = ttk.LabelFrame(right_frame, text="Box Properties", padding="10")
        props_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Label assignment
        ttk.Label(props_frame, text="Class Label:").pack(anchor=tk.W)
        
        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(props_frame, textvariable=self.label_var, state="readonly")
        self.label_combo.pack(fill=tk.X, pady=(5, 10))
        self.label_combo.bind("<<ComboboxSelected>>", self._on_label_changed)
        
        # Action buttons
        ttk.Button(props_frame, text="🏷️ Edit Label", command=self.edit_selected_label).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(props_frame, text="🗑️ Delete Box", command=self.delete_selected_box).pack(fill=tk.X)
        
        # Box list (with reduced height)
        list_frame = ttk.LabelFrame(right_frame, text="Bounding Boxes", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for box list
        columns = ("ID", "Label", "Size")
        self.box_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=6)  # Reduced height from 10 to 6
        
        for col in columns:
            self.box_tree.heading(col, text=col)
            self.box_tree.column(col, width=60)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.box_tree.yview)
        self.box_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.box_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.box_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        
        # Save format selection frame
        save_frame = ttk.LabelFrame(right_frame, text="Auto-Save Format", padding="10")
        save_frame.pack(fill=tk.X, pady=(0, 0))
        
        # Add dropdown for save format selection
        self.save_format_var = tk.StringVar(value="none")
        self.save_format_combo = ttk.Combobox(
            save_frame, 
            textvariable=self.save_format_var,
            values=["none", "JSON", "YOLO", "COCO"],
            state="readonly"
        )
        self.save_format_combo.pack(fill=tk.X)
        ttk.Label(save_frame, text="Format will be auto-saved on next image", font=("Arial", 8)).pack(fill=tk.X, pady=(5,0))
        
    def _create_status_bar(self):
        """Create the bottom status bar."""
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Open a dataset to begin annotation")
        
        status_label = ttk.Label(self.status_frame, textvariable=self.status_var, font=("Arial", 9))
        status_label.pack(side=tk.LEFT)
        
        # Progress info
        self.progress_var = tk.StringVar()
        progress_label = ttk.Label(self.status_frame, textvariable=self.progress_var, font=("Arial", 9))
        progress_label.pack(side=tk.RIGHT)
        
    def _show_welcome_message(self):
        """Show welcome message on the canvas."""
        self.canvas.delete("all")
        self.welcome_text_id = None  # Reset the text ID
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas is not ready yet, wait and retry
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self._show_welcome_message)
            return
        
        welcome_text = """🖼️ Image Annotation App

Welcome! To get started:

1. Click 'Open Dataset' to load your images and masks
2. Navigate through images using Previous/Next buttons
3. Use 'Generate from Mask' in controls to create boxes
4. Click on boxes to assign class labels
5. Use Ctrl+Z to undo operations (current image only)
6. Your work will auto-save when moving to next image

Ready to annotate your images!"""
        
        # Create text in the exact center of the canvas and store its ID
        self.welcome_text_id = self.canvas.create_text(
            canvas_width / 2,  # Horizontal center
            canvas_height / 2,  # Vertical center
            text=welcome_text,
            font=("Segoe UI", 12),
            fill="#7f8c8d",
            justify=tk.CENTER,
            anchor=tk.CENTER  # Ensure the text is centered on its position
        )
        
    def show_setup_dialog(self):
        """Show the dataset setup dialog."""
        show_setup_dialog(self.root, self.on_dataset_loaded)
        
    def on_dataset_loaded(self, dataset_path: str, class_labels: List[str], image_files: List[str]):
        """
        Callback when dataset is loaded from setup dialog.
        
        Args:
            dataset_path (str): Path to the dataset
            class_labels (list): List of class labels
            image_files (list): List of image filenames
        """
        self.dataset_path = dataset_path
        self.class_labels = class_labels
        self.image_files = image_files
        self.current_image_idx = 0
        self.annotations = {}
        
        # Clear undo history for new dataset
        if self.box_editor:
            self.box_editor.clear_history()
        
        # Update UI state
        self._update_ui_state()
        
        # Load first image
        self.load_current_image()
        
        self.status_var.set(f"Dataset loaded: {len(image_files)} images")
        
    def _update_ui_state(self):
        """Update UI state based on loaded dataset."""
        has_dataset = bool(self.image_files)
        
        # Enable/disable controls based on current position
        self.prev_btn.config(state=tk.NORMAL if has_dataset and self.current_image_idx > 0 else tk.DISABLED)
        
        # Enable next button if we have dataset and not at completion state
        # (Allow next even on last image for saving purposes)
        self.next_btn.config(state=tk.NORMAL if has_dataset and self.current_image_idx < len(self.image_files) else tk.DISABLED)
        
        # Update combobox values
        if has_dataset:
            self.label_combo['values'] = self.class_labels
            # Enable mask overlay button when dataset is loaded
            self.mask_overlay_btn.config(state=tk.NORMAL)
        else:
            # Disable mask overlay button when no dataset
            self.mask_overlay_btn.config(state=tk.DISABLED)
            self.mask_overlay_visible = False
            self._update_mask_overlay_button_text()
            
        # Update image counter
        if has_dataset:
            if self.current_image_idx < len(self.image_files):
                self.image_label.config(text=f"Image {self.current_image_idx + 1} of {len(self.image_files)}")
            else:
                self.image_label.config(text="All images completed")
        else:
            self.image_label.config(text="No images loaded")
            
        # Update undo button state
        self._update_undo_button_state()
        
    def load_current_image(self):
        """Load and display the current image."""
        if not self.image_files:
            return
        
        # Clear undo history for new image session
        if self.box_editor:
            self.box_editor.clear_history()
            self._update_undo_button_state()
            
        image_filename = self.image_files[self.current_image_idx]
        image_path = os.path.join(self.dataset_path, "images", image_filename)
        
        try:
            # Load image
            self.original_image = Image.open(image_path)
            self.image_format = self.original_image.format

            # Rotate image if necessary
            self.original_image = ImageOps.exif_transpose(self.original_image)
            
            # Load corresponding mask
            self._load_current_mask(image_filename)
            
            # Calculate initial scale to fit window
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.original_image.size
            
            # Calculate scale factors for both dimensions
            scale_x = (canvas_width - 20) / img_width
            scale_y = (canvas_height - 20) / img_height
            
            # Use the smaller scale to ensure entire image is visible
            self.scale_factor = min(scale_x, scale_y)
            
            self.display_image()
            
            # Update image info
            self._update_image_info(image_filename, self.original_image)
            
            # Load existing boxes if available
            self._load_existing_annotations(image_path)
            
            self.status_var.set(f"Loaded: {image_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set(f"Error loading image: {image_filename}")
            
    def display_image(self):
        """Display the current image on canvas."""
        if not self.original_image:
            return
            
        # Calculate display size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, schedule for later
            self.root.after(100, self.display_image)
            return
            
        # Calculate scaled dimensions
        img_width, img_height = self.original_image.size
        
        # Calculate new dimensions based on scale factor
        self.display_width = int(img_width * self.scale_factor)
        self.display_height = int(img_height * self.scale_factor)
        
        # Resize image for display
        self.current_image = self.original_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.displayed_image = ImageTk.PhotoImage(self.current_image)
        
        # Prepare mask overlay for the new scale
        if self.current_mask:
            self._prepare_mask_overlay()
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate padding to ensure image is centered
        pad_x = max(0, (canvas_width - self.display_width) // 2)
        pad_y = max(0, (canvas_height - self.display_height) // 2)
        
        # Create image at calculated position
        self.canvas_image = self.canvas.create_image(
            pad_x,  # Left edge of image
            pad_y,  # Top edge of image
            anchor="nw",  # Anchor to top-left corner
            image=self.displayed_image
        )
        
        # Show mask overlay if it should be visible
        if self.mask_overlay_visible:
            self._show_mask_overlay()
        
        # Set scroll region to encompass the entire image plus padding
        total_width = max(canvas_width, self.display_width + 2 * pad_x)
        total_height = max(canvas_height, self.display_height + 2 * pad_y)
        
        # Add extra padding for scrolling
        total_width += 40
        total_height += 40
        
        self.canvas.configure(scrollregion=(-20, -20, total_width - 20, total_height - 20))
        
        # Center the view
        self.canvas.xview_moveto(0.5 - (canvas_width / (2 * total_width)))
        self.canvas.yview_moveto(0.5 - (canvas_height / (2 * total_height)))
        
        # Set image bounds for box editor to constrain editing within image area
        if self.box_editor and self.canvas_image:
            bbox = self.canvas.bbox(self.canvas_image)
            if bbox:
                # Image bounds are (left, top, right, bottom)
                image_bounds = (bbox[0], bbox[1], bbox[2], bbox[3])
                self.box_editor.set_image_bounds(image_bounds)
        
    def _update_image_info(self, filename: str, image: Image.Image):
        """Update the image information display."""
        info = f"""Filename: {filename}
Size: {image.size[0]} × {image.size[1]}
Mode: {image.mode}
Format: {self.image_format or 'Unknown'}

Scale: {self.scale_factor:.2f}x
Display: {self.display_width} × {self.display_height}"""
        
        self.image_info_text.delete(1.0, tk.END)
        self.image_info_text.insert(1.0, info)
        
    def _load_existing_annotations(self, image_path: str):
        """Load existing annotations for the current image."""
        if image_path in self.annotations:
            data = self.annotations[image_path]
            boxes = data['boxes']
            labels = data['labels']
            
            # Get the image position on canvas
            bbox = self.canvas.bbox(self.canvas_image)
            if not bbox:
                return
                
            # Image position and size on canvas
            image_x, image_y = bbox[0], bbox[1]
            
            # Scale boxes to display size and adjust for image position
            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                # Scale coordinates relative to image size
                scaled_box = [
                    image_x + int(x1 * self.scale_factor),
                    image_y + int(y1 * self.scale_factor),
                    image_x + int(x2 * self.scale_factor),
                    image_y + int(y2 * self.scale_factor)
                ]
                scaled_boxes.append(scaled_box)
            
            self.box_editor.set_boxes(scaled_boxes, labels)
            self._update_box_list()
            
            # Update image bounds for editing constraints
            if self.canvas_image:
                bbox = self.canvas.bbox(self.canvas_image)
                if bbox:
                    image_bounds = (bbox[0], bbox[1], bbox[2], bbox[3])
                    self.box_editor.set_image_bounds(image_bounds)
        else:
            # Clear boxes
            self.box_editor.clear_boxes()
            self._update_box_list()
            
    def generate_boxes(self):
        """Generate bounding boxes from segmentation masks."""
        if not self.image_files:
            return
            
        image_filename = self.image_files[self.current_image_idx]
        image_path = os.path.join(self.dataset_path, "images", image_filename)
        annotations_dir = os.path.join(self.dataset_path, "annotations")
        
        # Find mask file
        mask_filename = find_mask_file(image_filename, annotations_dir)
        if not mask_filename:
            messagebox.showerror("Error", f"No mask file found for {image_filename}")
            return
            
        mask_path = os.path.join(annotations_dir, mask_filename)
        
        try:
            # Load image and mask
            _, mask = load_image_and_mask(image_path, mask_path)
            if mask is None:
                messagebox.showerror("Error", f"Failed to load mask: {mask_filename}")
                return
                
            # Generate boxes
            boxes = mask_to_boxes(mask)
            if not boxes:
                messagebox.showinfo("Info", "No objects found in the mask")
                return
                
            # Get the actual image position on canvas (same method as _load_existing_annotations)
            bbox = self.canvas.bbox(self.canvas_image)
            if not bbox:
                messagebox.showerror("Error", "Could not determine image position on canvas")
                return
                
            # Image position on canvas
            image_x, image_y = bbox[0], bbox[1]
            
            # Scale boxes to display size and add image position offset
            # Use floating-point arithmetic for better precision
            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                # Scale coordinates relative to image size and add image position
                scaled_box = [
                    int(round(image_x + x1 * self.scale_factor)),
                    int(round(image_y + y1 * self.scale_factor)),
                    int(round(image_x + x2 * self.scale_factor)),
                    int(round(image_y + y2 * self.scale_factor))
                ]
                scaled_boxes.append(scaled_box)
            
            # Set boxes in editor
            labels = ["" for _ in scaled_boxes]
            self.box_editor.set_boxes(scaled_boxes, labels)
            
            # Set image bounds for editing constraints
            if self.canvas_image:
                bbox = self.canvas.bbox(self.canvas_image)
                if bbox:
                    image_bounds = (bbox[0], bbox[1], bbox[2], bbox[3])
                    self.box_editor.set_image_bounds(image_bounds)
            
            # Store original boxes in annotations
            self.annotations[image_path] = {
                'boxes': boxes,
                'labels': labels
            }
            
            self._update_box_list()
            self.status_var.set(f"Generated {len(boxes)} bounding boxes")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate boxes: {str(e)}")
            
    def _update_box_list(self):
        """Update the box list display."""
        # Clear existing items
        for item in self.box_tree.get_children():
            self.box_tree.delete(item)
            
        if not self.box_editor:
            return
            
        boxes, labels = self.box_editor.get_boxes()
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            size_text = f"{width}×{height}"
            
            label_text = label if label else "None"
            
            self.box_tree.insert("", "end", values=(f"Box {i+1}", label_text, size_text))
            
        # Update progress info
        total_boxes = len(boxes)
        labeled_boxes = sum(1 for label in labels if label)
        self.progress_var.set(f"Boxes: {labeled_boxes}/{total_boxes} labeled")
        
    def _on_box_selected(self, box_index: Optional[int]):
        """Handle box selection from editor."""
        if box_index is not None:
            # Update tree selection
            children = self.box_tree.get_children()
            if 0 <= box_index < len(children):
                self.box_tree.selection_set(children[box_index])
                self.box_tree.focus(children[box_index])
                
                # Update label combo
                boxes, labels = self.box_editor.get_boxes()
                if 0 <= box_index < len(labels):
                    current_label = labels[box_index]
                    self.label_var.set(current_label if current_label else "")
        else:
            # Clear selection
            self.box_tree.selection_remove(self.box_tree.selection())
            self.label_var.set("")
            
    def _on_box_modified(self, box_index: int):
        """Handle box modification from editor."""
        # Check if we're in completion state (beyond last image)
        if self.current_image_idx >= len(self.image_files):
            return
            
        # Update stored annotations
        if self.image_files:
            image_filename = self.image_files[self.current_image_idx]
            image_path = os.path.join(self.dataset_path, "images", image_filename)
            
            if image_path in self.annotations:
                # Get current boxes from editor and convert to original coordinates
                display_boxes, labels = self.box_editor.get_boxes()
                original_boxes = []
                
                for box in display_boxes:
                    x1, y1, x2, y2 = box
                    # Remove offset and scale back to original size
                    orig_x1 = int((x1 - 10) / self.scale_factor)
                    orig_y1 = int((y1 - 10) / self.scale_factor)
                    orig_x2 = int((x2 - 10) / self.scale_factor)
                    orig_y2 = int((y2 - 10) / self.scale_factor)
                    original_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                
                self.annotations[image_path]['boxes'] = original_boxes
                self.annotations[image_path]['labels'] = labels
                
        self._update_box_list()
        self._update_undo_button_state()
        
    def _on_tree_select(self, event):
        """Handle selection in the box tree."""
        selection = self.box_tree.selection()
        if selection:
            item = selection[0]
            index = self.box_tree.index(item)
            self.box_editor.select_box(index)
            
    def _on_label_changed(self, event):
        """Handle label change in the combobox."""
        selected_index = self.box_editor.selected_idx
        if selected_index is not None:
            new_label = self.label_var.get()
            self.box_editor.update_box_label(selected_index, new_label)
            self._on_box_modified(selected_index)
            
    def edit_selected_label(self):
        """Edit the label of the selected box."""
        selected_index = self.box_editor.selected_idx
        if selected_index is None:
            messagebox.showinfo("Info", "Please select a box first.")
            return
            
        boxes, labels = self.box_editor.get_boxes()
        current_label = labels[selected_index] if selected_index < len(labels) else ""
        
        # Show dialog to edit label
        new_label = simpledialog.askstring(
            "Edit Label",
            f"Enter label for Box {selected_index + 1}:",
            initialvalue=current_label
        )
        
        if new_label is not None:  # User didn't cancel
            self.box_editor.update_box_label(selected_index, new_label)
            self.label_var.set(new_label)
            self._on_box_modified(selected_index)
            
    def delete_selected_box(self):
        """Delete the selected box."""
        selected_index = self.box_editor.selected_idx
        if selected_index is None:
            messagebox.showinfo("Info", "Please select a box first.")
            return
            
        if messagebox.askyesno("Confirm", f"Delete Box {selected_index + 1}?"):
            self.box_editor.remove_box(selected_index)
            self._on_box_modified(0)  # Trigger update
            
    def clear_boxes(self):
        """Clear all bounding boxes."""
        if self.box_editor and self.box_editor.boxes:
            if messagebox.askyesno("Confirm", "Clear all bounding boxes?"):
                self.box_editor.clear_boxes()
                
                # Clear from annotations
                if self.image_files:
                    image_filename = self.image_files[self.current_image_idx]
                    image_path = os.path.join(self.dataset_path, "images", image_filename)
                    if image_path in self.annotations:
                        del self.annotations[image_path]
                        
                self._update_box_list()
                self.status_var.set("All boxes cleared")
                
    def filter_small_boxes(self):
        """Filter out small bounding boxes."""
        if not self.box_editor or not self.box_editor.boxes:
            messagebox.showinfo("Info", "No boxes to filter.")
            return
            
        # Ask for minimum area
        min_area = simpledialog.askinteger(
            "Filter Small Boxes",
            "Enter minimum area (pixels):",
            initialvalue=100,
            minvalue=1
        )
        
        if min_area:
            # Convert to display coordinates
            min_area_display = min_area * (self.scale_factor ** 2)
            self.box_editor.filter_boxes_by_size(int(min_area_display))
            self._on_box_modified(0)  # Trigger update
            self.status_var.set(f"Filtered boxes smaller than {min_area} pixels")
            
    def _handle_zoom_change(self, new_scale_factor):
        """Handle zoom change by updating display and maintaining view center."""
        if not self.original_image:
            return
            
        old_scale = self.scale_factor
        self.scale_factor = new_scale_factor
        
        # Only proceed if scale actually changed
        if old_scale != self.scale_factor:
            # Get current view center relative to the image
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Get the current scroll position
            x_view = self.canvas.xview()
            y_view = self.canvas.yview()
            
            # Calculate the center point in the current view
            center_x = x_view[0] + (x_view[1] - x_view[0]) / 2
            center_y = y_view[0] + (y_view[1] - y_view[0]) / 2
            
            # Update display
            self.display_image()
            
            # Reload annotations with new scale factor
            self._load_existing_annotations(
                os.path.join(self.dataset_path, "images", self.image_files[self.current_image_idx])
            )
            
            # Get the new scroll region size
            bbox = self.canvas.bbox("all")
            if bbox:
                scroll_width = bbox[2] - bbox[0]
                scroll_height = bbox[3] - bbox[1]
                
                # Calculate and apply the new scroll position
                new_x = center_x - (canvas_width / (2 * scroll_width))
                new_y = center_y - (canvas_height / (2 * scroll_height))
                
                self.canvas.xview_moveto(new_x)
                self.canvas.yview_moveto(new_y)
            
    def zoom_in(self):
        """Zoom in on the image."""
        new_scale = min(self.scale_factor * 1.2, 5.0)
        self._handle_zoom_change(new_scale)
            
    def zoom_out(self):
        """Zoom out from the image."""
        new_scale = max(self.scale_factor / 1.2, 0.1)
        self._handle_zoom_change(new_scale)
            
    def fit_to_window(self):
        """Fit the image to the window."""
        if self.original_image:
            # Calculate scale to fit window
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.original_image.size
            
            # Calculate scale factors for both dimensions
            scale_x = (canvas_width - 20) / img_width
            scale_y = (canvas_height - 20) / img_height
            
            # Use the smaller scale to ensure entire image is visible
            self.scale_factor = min(scale_x, scale_y)
            
            # Update display
            self.display_image()
            self._load_existing_annotations(
                os.path.join(self.dataset_path, "images", self.image_files[self.current_image_idx])
            )
            
    def previous_image(self):
        """Navigate to the previous image."""
        if self.current_image_idx > 0:
            # If we're in completion state, go back to last actual image
            if self.current_image_idx >= len(self.image_files):
                self.current_image_idx = len(self.image_files) - 1
            else:
                self.current_image_idx -= 1
            self._update_ui_state()
            self.load_current_image()
            
    def next_image(self):
        """Navigate to the next image."""
        if self.current_image_idx < len(self.image_files):
            # Check if all boxes are labeled
            if self.box_editor and self.box_editor.boxes:
                boxes, labels = self.box_editor.get_boxes()
                unlabeled_boxes = sum(1 for label in labels if not label)
                if unlabeled_boxes > 0:
                    if not messagebox.askyesno(
                        "Warning",
                        f"There are {unlabeled_boxes} unlabeled boxes. Do you want to proceed anyway?"
                    ):
                        return

            # Check if save format is selected
            if self.save_format_var.get() == "none":
                messagebox.showwarning(
                    "Warning",
                    "Please select an output format for annotations before proceeding."
                )
                return
                
            # Save current annotations based on selected format
            save_format = self.save_format_var.get()
            if save_format == "JSON":
                self._auto_save_json()
            elif save_format == "YOLO":
                self._auto_save_yolo()
            elif save_format == "COCO":
                self._auto_save_coco()
            
            # Check if we're at the last image
            if self.current_image_idx == len(self.image_files) - 1:
                # We're at the last image, show completion message
                self.current_image_idx += 1  # Move beyond last image
                self._show_completion_message()
                self._update_ui_state()
                self.status_var.set("All images have been annotated and saved!")
            else:
                # Normal navigation to next image
                self.current_image_idx += 1
                self._update_ui_state()
                self.load_current_image()

    def _auto_save_json(self):
        """Auto-save annotations in JSON format."""
        if not self.annotations:
            return
            
        output_dir = os.path.join(self.dataset_path, "output")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "annotations.json")
        
        if save_annotations_to_json(self.annotations, json_path):
            self.status_var.set("JSON annotations auto-saved")
        else:
            messagebox.showerror("Error", "Failed to auto-save JSON annotations")

    def _auto_save_yolo(self):
        """Auto-save annotations in YOLO format."""
        if not self.annotations or not self.image_files:
            return
            
        # Get sample image dimensions AFTER EXIF transpose (same as used for box generation)
        sample_image_path = os.path.join(self.dataset_path, "images", self.image_files[0])
        sample_image = Image.open(sample_image_path)
        sample_image = ImageOps.exif_transpose(sample_image)  # Apply EXIF transpose
        image_width, image_height = sample_image.size  # Now get the correct dimensions
        
        output_dir = os.path.join(self.dataset_path, "output_yolo")
        
        if export_annotations_to_yolo(
            self.annotations, 
            output_dir, 
            image_width, 
            image_height, 
            self.class_labels
        ):
            self.status_var.set("YOLO annotations auto-saved")
        else:
            messagebox.showerror("Error", "Failed to auto-save YOLO annotations")

    def _auto_save_coco(self):
        """Auto-save annotations in COCO format."""
        if not self.annotations:
            return
            
        output_dir = os.path.join(self.dataset_path, "output_coco")
        
        if export_annotations_to_coco(
            self.annotations,
            output_dir,
            self.class_labels,
            self.image_files,
            self.dataset_path
        ):
            self.status_var.set("COCO annotations auto-saved")
        else:
            messagebox.showerror("Error", "Failed to auto-save COCO annotations")

    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        if not self.image_files and self.welcome_text_id:
            # Only handle welcome text repositioning if no image is loaded
            self._update_welcome_text_position()

    def _update_welcome_text_position(self):
        """Update the welcome text position to the center of the canvas."""
        if self.welcome_text_id:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Move the existing text to the new center
            self.canvas.coords(self.welcome_text_id, canvas_width / 2, canvas_height / 2)

    def undo_last_action(self):
        """Undo the last box editing action."""
        # Prevent undo in completion state
        if self.current_image_idx >= len(self.image_files):
            self.status_var.set("Cannot undo - all images completed")
            return
            
        if self.box_editor:
            if self.box_editor.undo():
                self.status_var.set("Undid last action for current image")
                self._update_undo_button_state()
            else:
                self.status_var.set("Nothing to undo for current image")
    
    def _update_undo_button_state(self):
        """Update the state of the undo button based on history availability."""
        if hasattr(self, 'undo_btn') and self.box_editor:
            # Check if we're in completion state (beyond last image)
            if self.current_image_idx >= len(self.image_files):
                self.undo_btn.config(state=tk.DISABLED)
                return
                
            # Normal state - check if we have history
            has_history = bool(self.box_editor.history_stack)
            self.undo_btn.config(state=tk.NORMAL if has_history else tk.DISABLED)

    def _show_completion_message(self):
        """Show completion message when all images are annotated."""
        self.canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas is not ready yet, wait and retry
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self._show_completion_message)
            return
        
        completion_text = """🎉 All Images Annotated!

Congratulations! You have successfully annotated all images in your dataset.

Summary:
• Total images processed: {}
• Annotations saved in {} format
• Output saved to dataset folder

You can now:
• Use Previous button to review annotations
• Change output format and re-save if needed
• Open a new dataset to continue annotating

Great job on completing your annotation task!""".format(
            len(self.image_files),
            self.save_format_var.get()
        )
        
        # Create completion text in the center of the canvas
        self.canvas.create_text(
            canvas_width / 2,
            canvas_height / 2,
            text=completion_text,
            font=("Segoe UI", 12),
            fill="#27ae60",  # Green color for success
            justify=tk.CENTER,
            anchor=tk.CENTER
        )
        
        # Clear box editor since there's no image
        if self.box_editor:
            self.box_editor.clear_boxes()
            self.box_editor.clear_history()  # Clear undo history in completion state
            self._update_box_list()
            self._update_undo_button_state()  # Update undo button state

    def toggle_mask_overlay(self):
        """Toggle the mask overlay visibility."""
        self.mask_overlay_visible = not self.mask_overlay_visible
        self._update_mask_overlay_button_text()
        
        if self.mask_overlay_visible:
            # Show mask overlay
            self._show_mask_overlay()
        else:
            # Hide mask overlay
            self._hide_mask_overlay()
    
    def _show_mask_overlay(self):
        """Show the mask overlay without redrawing the entire canvas."""
        if not self.current_mask or not self.canvas_image:
            return
            
        # Prepare mask overlay if not already prepared
        if not self.displayed_mask:
            self._prepare_mask_overlay()
            
        if self.displayed_mask:
            # Get the current image position
            bbox = self.canvas.bbox(self.canvas_image)
            if bbox:
                pad_x, pad_y = bbox[0], bbox[1]
                # Create mask overlay at the same position as the image
                self.canvas_mask = self.canvas.create_image(
                    pad_x,
                    pad_y,
                    anchor="nw",
                    image=self.displayed_mask
                )
                
                # Position the mask overlay appropriately
                # First, ensure it's above the image
                self.canvas.tag_raise(self.canvas_mask, self.canvas_image)
                
                # If there are boxes, position mask below them
                # Check if any items with "box" tag exist first
                box_items = self.canvas.find_withtag("box")
                if box_items:
                    # There are boxes, so put mask below them but above image
                    self.canvas.tag_lower(self.canvas_mask, "box")
    
    def _hide_mask_overlay(self):
        """Hide the mask overlay without affecting other canvas elements."""
        if hasattr(self, 'canvas_mask') and self.canvas_mask:
            self.canvas.delete(self.canvas_mask)
            self.canvas_mask = None

    def _update_mask_overlay_button_text(self):
        """Update the text of the mask overlay button based on its state."""
        if self.mask_overlay_visible:
            self.mask_overlay_btn.config(text="👁️ Hide Mask (Ctrl+M)")
        else:
            self.mask_overlay_btn.config(text="👁️ Show Mask (Ctrl+M)")

    def _load_current_mask(self, image_filename: str):
        """Load the corresponding mask for the current image."""
        self.current_mask = None
        self.displayed_mask = None
        
        annotations_dir = os.path.join(self.dataset_path, "annotations")
        mask_filename = find_mask_file(image_filename, annotations_dir)
        if mask_filename:
            try:
                mask_path = os.path.join(annotations_dir, mask_filename)
                # Load mask directly using PIL
                mask_pil = Image.open(mask_path)
                
                # Convert to numpy array for processing
                mask_array = np.array(mask_pil)
                
                # Handle different mask formats
                if len(mask_array.shape) == 3:
                    if mask_array.shape[2] == 4:  # RGBA
                        # Use alpha channel if available
                        if np.any(mask_array[:,:,3] < 255):
                            mask_array = mask_array[:,:,3]  # Use alpha channel as mask
                        else:
                            # Convert RGB part to grayscale
                            mask_array = np.mean(mask_array[:,:,:3], axis=2).astype(np.uint8)
                    elif mask_array.shape[2] == 3:  # RGB
                        mask_array = np.mean(mask_array, axis=2).astype(np.uint8)
                    else:
                        # Other multi-channel format - use first channel
                        mask_array = mask_array[:,:,0]
                
                # Ensure mask is uint8
                if mask_array.dtype != np.uint8:
                    mask_array = mask_array.astype(np.uint8)
                
                # Create binary mask
                unique_values = np.unique(mask_array)
                if len(unique_values) > 2:
                    mask_array = np.where(mask_array > 128, 255, 0).astype(np.uint8)
                elif len(unique_values) == 2:
                    # Already binary, just ensure values are 0 and 255
                    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                
                # Convert back to PIL Image
                self.current_mask = Image.fromarray(mask_array, mode='L')
                # Don't prepare overlay immediately - wait until display dimensions are set
                
            except Exception as e:
                print(f"Warning: Could not load mask {mask_filename}: {e}")
                self.current_mask = None
                self.displayed_mask = None
                
    def _prepare_mask_overlay(self):
        """Prepare the mask overlay for display."""
        if not self.current_mask:
            return
            
        # Check if display dimensions are valid
        if self.display_width <= 0 or self.display_height <= 0:
            print("Display dimensions not ready yet, skipping mask overlay preparation")
            return
            
        try:
            # Resize mask to match current image display size
            mask_resized = self.current_mask.resize((self.display_width, self.display_height), Image.Resampling.NEAREST)
            
            # Create a colored overlay with transparency
            mask_colored = Image.new("RGBA", (self.display_width, self.display_height), (0, 0, 0, 0))
            
            # Convert mask to RGBA with red color and transparency
            mask_array = np.array(mask_resized)
            overlay_array = np.zeros((self.display_height, self.display_width, 4), dtype=np.uint8)
            
            # Set red color where mask is present (alpha = 100 for semi-transparency)
            overlay_array[mask_array > 128] = [255, 0, 0, 100]  # Red with 100/255 transparency
            
            self.displayed_mask = ImageTk.PhotoImage(Image.fromarray(overlay_array, mode='RGBA'))
        except Exception as e:
            print(f"Warning: Could not prepare mask overlay: {e}")
            self.displayed_mask = None


def main():
    """Main function to run the annotation app."""
    root = tk.Tk()
    app = SegmentationAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main() 