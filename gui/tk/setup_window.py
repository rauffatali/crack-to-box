"""
Initial setup window for dataset path and class labels configuration.

This module provides:
- DatasetSetupWindow: Main setup window class
- Dataset validation and configuration
- User-friendly interface for initial setup
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
import os
from typing import Callable, List, Tuple, Optional

from utils.image_loader import validate_dataset_structure, get_dataset_info, load_dataset


class DatasetSetupWindow:
    """
    Initial setup window for dataset configuration.
    
    This window allows users to:
    - Select dataset directory path
    - Enter class labels for annotation
    - Validate dataset structure
    - Configure initial settings
    """
    
    def __init__(self, root: tk.Tk, callback: Callable[[str, List[str], List[str]], None]):
        """
        Initialize the setup window.
        
        Args:
            root (tk.Tk): Parent window
            callback (callable): Callback function to call when setup is complete
                                Signature: callback(dataset_path, class_labels, image_files)
        """
        self.root = root
        self.callback = callback
        self.window: Optional[tk.Toplevel] = None
        
        # State variables
        self.dataset_var = tk.StringVar()
        self.labels_var = tk.StringVar()
        self.status_var = tk.StringVar()
        
        # Default class labels
        self.default_labels = ["alligator", "longitudinal", "transverse", "diagonal"]
        
        self.setup_window()
        self.setup_ui()
        
    def setup_window(self):
        """Setup the dataset setup window properties."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Dataset Setup - Image Annotation App")
        self.window.geometry("700x720")
        self.window.resizable(True, False)
        
        # Center the window
        self.window.transient(self.root)
        self.window.grab_set()
        
        # Center on screen
        self._center_window()
        
        # Force window to front and focus
        self.window.lift()
        self.window.focus_force()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        
        # Handle window close event
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)
        
    def _center_window(self):
        """Center the window on screen."""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() - width) // 2
        y = (self.window.winfo_screenheight() - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
    def setup_ui(self):
        """Setup the user interface components."""
        # Main container with padding
        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        self._create_title_section(main_frame)
        
        # Dataset configuration section
        self._create_dataset_section(main_frame)
        
        # Class labels section
        self._create_labels_section(main_frame)
        
        # Info/Preview section
        self._create_info_section(main_frame)
        
        # Buttons section
        self._create_buttons_section(main_frame)
        
        # Status section
        self._create_status_section(main_frame)
        
    def _create_title_section(self, parent):
        """Create the title section."""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(
            title_frame,
            text="🖼️ Image Annotation App Setup",
            font=("Arial", 18, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Configure your dataset and class labels for annotation",
            font=("Arial", 10),
            foreground="#7f8c8d"
        )
        subtitle_label.pack(pady=(5, 0))
        
    def _create_dataset_section(self, parent):
        """Create the dataset configuration section."""
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Configuration", padding="10")
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dataset structure info
        info_text = """Required Dataset Structure:
                    📂 dataset_folder/
                    ├── 📂 images/         (Contains .jpg, .png, or other image files)
                    └── 📂 masks/    (Contains segmentation mask files)
                    """
        info_label = ttk.Label(
            dataset_frame,
            text=info_text,
            font=("Courier", 8),
            justify=tk.LEFT,
            foreground="#34495e"
        )
        info_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Dataset path input
        ttk.Label(dataset_frame, text="Dataset Path:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        path_frame = ttk.Frame(dataset_frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.dataset_entry = ttk.Entry(path_frame, textvariable=self.dataset_var, font=("Arial", 10))
        self.dataset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = ttk.Button(path_frame, text="Browse", command=self.browse_dataset)
        browse_btn.pack(side=tk.RIGHT)
        
        # Validation button
        validate_btn = ttk.Button(dataset_frame, text="✅ Validate Dataset", command=self.validate_dataset)
        validate_btn.pack(anchor=tk.W)
        
    def _create_labels_section(self, parent):
        """Create the class labels configuration section."""

        
        labels_frame = ttk.LabelFrame(parent, text="Class Labels Configuration", padding="10")
        labels_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(
            labels_frame,
            text="Enter the class labels you want to use for annotation (comma-separated):",
            font=("Arial", 10),
            foreground="#2c3e50"
        )
        instructions.pack(anchor=tk.W, pady=(0, 5))
        
        # Labels input
        labels_input_frame = ttk.Frame(labels_frame)
        labels_input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.labels_entry = ttk.Entry(labels_input_frame, textvariable=self.labels_var, font=("Arial", 10))
        self.labels_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        default_btn = ttk.Button(labels_input_frame, text="Use Defaults", command=self.use_default_labels)
        default_btn.pack(side=tk.RIGHT)
        
        # Example and hint
        example_label = ttk.Label(
            labels_frame,
            text=f"Example: {', '.join(self.default_labels)}",
            font=("Arial", 9),
            foreground="#7f8c8d"
        )
        example_label.pack(anchor=tk.W)
        
        hint_label = ttk.Label(
            labels_frame,
            text="💡 Tip: Enter all possible object classes you want to annotate",
            font=("Arial", 9),
            foreground="#3498db"
        )
        hint_label.pack(anchor=tk.W, pady=(5, 0))
        
    def _create_info_section(self, parent):
        """Create the dataset information/preview section."""
        self.info_frame = ttk.LabelFrame(parent, text="Dataset Information", padding="15")
        self.info_frame.pack(fill=tk.X)
        
        self.info_text = tk.Text(
            self.info_frame,
            height=8,
            font=("Courier", 8),
            state=tk.DISABLED,
            background="#f8f9fa",
            wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.X)
        
        # Initial message
        self._update_info_display("Select a dataset directory to see information here...")
        
    def _create_buttons_section(self, parent):
        """Create the buttons section."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create custom button style
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        
        # Cancel button
        cancel_btn = ttk.Button(
            button_frame,
            text="❌ Cancel",
            command=self.cancel,
            style="TButton",
            padding=(15, 8)
        )
        cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Load Dataset button (primary action)
        self.load_btn = ttk.Button(
            button_frame,
            text="Load Dataset",
            command=self.load_dataset,
            style="Accent.TButton",
            padding=(15, 8)
        )
        self.load_btn.pack(side=tk.RIGHT)
        
    def _create_status_section(self, parent):
        """Create the status message section."""
        status_label = ttk.Label(
            parent,
            textvariable=self.status_var,
            font=("Arial", 9),
            foreground="#7f8c8d"
        )
        status_label.pack(pady=(10, 0))
        
        self.status_var.set("Ready to configure dataset...")
        
    def browse_dataset(self):
        """Open file dialog to browse for dataset folder."""
        folder = filedialog.askdirectory(
            title="Select Dataset Folder",
            mustexist=True
        )
        
        if folder:
            self.dataset_var.set(folder)
            self.status_var.set("Dataset folder selected. Click 'Validate Dataset' to check structure.")
            self.validate_dataset()
            
    def use_default_labels(self):
        """Set default class labels."""
        self.labels_var.set(", ".join(self.default_labels))
        self.status_var.set("Default class labels loaded.")
        
    def validate_dataset(self):
        """Validate the selected dataset structure."""
        dataset_path = self.dataset_var.get().strip()
        
        if not dataset_path:
            self.status_var.set("Please select a dataset directory first.")
            return False
            
        try:
            # Validate structure
            is_valid, message = validate_dataset_structure(dataset_path)
            
            if is_valid:
                # Get detailed info
                info = get_dataset_info(dataset_path)
                self._display_dataset_info(info)
                self.status_var.set("✅ Dataset validation successful!")
                return True
            else:
                self._update_info_display(f"❌ Validation Error:\n{message}")
                self.status_var.set("❌ Dataset validation failed. Check the structure.")
                return False
                
        except Exception as e:
            error_msg = f"Error validating dataset: {str(e)}"
            self._update_info_display(f"❌ Error:\n{error_msg}")
            self.status_var.set("❌ Error during validation.")
            return False
            
    def _display_dataset_info(self, info: dict):
        """Display dataset information in the info section."""
        if info["valid"]:
            info_text = f"""✅ Dataset Status: Valid

📊 Statistics:
• Number of images: {info['num_images']}
• Number of masks: {info['num_masks']}
• Image formats: {', '.join(info['image_extensions']) if info['image_extensions'] else 'None'}
• Mask formats: {', '.join(info['mask_extensions']) if info['mask_extensions'] else 'None'}

📂 Path: {info['path']}

✅ Ready for annotation!"""
        else:
            info_text = f"""❌ Dataset Status: Invalid

Error: {info.get('error', 'Unknown error')}

Please check the dataset structure and try again."""
            
        self._update_info_display(info_text)
        
    def _update_info_display(self, text: str):
        """Update the info display with new text."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)
        
    def load_dataset(self):
        """Validate inputs and load the dataset."""
        dataset_path = self.dataset_var.get().strip()
        labels_input = self.labels_var.get().strip()
        
        # Validate dataset path
        if not dataset_path:
            messagebox.showerror("Error", "Please enter a valid dataset path")
            return
            
        # Validate class labels
        if not labels_input:
            messagebox.showerror("Error", "Please enter at least one class label")
            return
            
        # Parse class labels
        class_labels = [label.strip() for label in labels_input.split(",") if label.strip()]
        if not class_labels:
            messagebox.showerror("Error", "Please enter valid class labels")
            return
            
        # Validate dataset structure one more time
        if not self.validate_dataset():
            messagebox.showerror("Error", "Dataset validation failed. Please check the dataset structure.")
            return
            
        try:
            # Load image files
            from utils.image_loader import get_image_files_ordered_by_mask
            image_files = get_image_files_ordered_by_mask(dataset_path, annotations_subdir="masks")
            # image_files = load_dataset(dataset_path)
            if not image_files:
                messagebox.showerror("Error", "No images found in the specified directory")
                return
                
            # Success - call callback and close window
            self.status_var.set("🚀 Loading annotation interface...")
            self.window.update()
            
            self.callback(dataset_path, class_labels, image_files)
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.status_var.set("❌ Failed to load dataset.")
            
    def cancel(self):
        """Cancel setup and close window."""
        if messagebox.askyesno("Confirm", "Are you sure you want to cancel the setup?"):
            self.window.destroy()
            
    def show(self):
        """Show the setup window."""
        if self.window:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
            
    def hide(self):
        """Hide the setup window."""
        if self.window:
            self.window.withdraw()


def show_setup_dialog(root: tk.Tk, callback: Callable[[str, List[str], List[str]], None]) -> DatasetSetupWindow:
    """
    Show the dataset setup dialog.
    
    Args:
        root (tk.Tk): Parent window
        callback (callable): Callback function for when setup is complete
        
    Returns:
        DatasetSetupWindow: The setup window instance
    """
    return DatasetSetupWindow(root, callback)


# For testing
if __name__ == "__main__":
    def test_callback(dataset_path, class_labels, image_files):
        print(f"Dataset: {dataset_path}")
        print(f"Labels: {class_labels}")
        print(f"Images: {len(image_files)} files")
        
    root = tk.Tk()
    root.withdraw()  # Hide main window for testing
    
    setup_window = show_setup_dialog(root, test_callback)
    
    root.mainloop() 