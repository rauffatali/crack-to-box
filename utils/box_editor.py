"""
Box editing functionality for interactive bounding box manipulation.
Uses a robust state pattern for handling different editing modes.
Includes undo functionality with history tracking.
"""

import tkinter as tk
import copy
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
from enum import Enum, auto


class EditMode(Enum):
    """Different modes for box interaction"""
    IDLE = auto()
    SELECT = auto()
    MOVE = auto()
    RESIZE = auto()


@dataclass
class Box:
    """Represents a bounding box with coordinates and label."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    canvas_objects: List[int] = None
    
    def __post_init__(self):
        self.canvas_objects = []
    
    @property
    def width(self) -> int:
        return abs(self.x2 - self.x1)
    
    @property
    def height(self) -> int:
        return abs(self.y2 - self.y1)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def contains_point(self, x: int, y: int, margin: int = 5) -> bool:
        """Check if point is inside box with margin."""
        return (self.x1 - margin <= x <= self.x2 + margin and 
                self.y1 - margin <= y <= self.y2 + margin)
    
    def get_handle_at(self, x: int, y: int, size: int = 8) -> Optional[str]:
        """Get resize handle at point if any."""
        half = size // 2
        
        # Check corners
        corners = {
            'nw': (self.x1, self.y1),
            'ne': (self.x2, self.y1),
            'sw': (self.x1, self.y2),
            'se': (self.x2, self.y2)
        }
        
        for handle, (hx, hy) in corners.items():
            if abs(x - hx) <= half and abs(y - hy) <= half:
                return handle
        
        return None
    
    def move_by(self, dx: int, dy: int, min_size: int = 10, bounds: tuple = None):
        """Move box by delta while ensuring minimum size and staying within bounds."""
        new_x1 = self.x1 + dx
        new_y1 = self.y1 + dy
        new_x2 = self.x2 + dx
        new_y2 = self.y2 + dy
        
        # Apply boundary constraints if provided
        if bounds:
            bound_x1, bound_y1, bound_x2, bound_y2 = bounds
            
            # Constrain within bounds
            if new_x1 < bound_x1:
                offset = bound_x1 - new_x1
                new_x1 = bound_x1
                new_x2 += offset
            if new_y1 < bound_y1:
                offset = bound_y1 - new_y1
                new_y1 = bound_y1
                new_y2 += offset
            if new_x2 > bound_x2:
                offset = new_x2 - bound_x2
                new_x2 = bound_x2
                new_x1 -= offset
            if new_y2 > bound_y2:
                offset = new_y2 - bound_y2
                new_y2 = bound_y2
                new_y1 -= offset
            
            # Final check to ensure we're still within bounds
            new_x1 = max(bound_x1, new_x1)
            new_y1 = max(bound_y1, new_y1)
            new_x2 = min(bound_x2, new_x2)
            new_y2 = min(bound_y2, new_y2)
        
        self.x1 = new_x1
        self.y1 = new_y1
        self.x2 = new_x2
        self.y2 = new_y2
    
    def resize_by(self, handle: str, dx: int, dy: int, min_size: int = 10, bounds: tuple = None):
        """Resize box using handle while ensuring minimum size and staying within bounds."""
        new_x1, new_y1, new_x2, new_y2 = self.x1, self.y1, self.x2, self.y2
        
        # Apply resize based on handle
        if 'n' in handle:
            new_y1 = self.y1 + dy
        if 's' in handle:
            new_y2 = self.y2 + dy
        if 'w' in handle:
            new_x1 = self.x1 + dx
        if 'e' in handle:
            new_x2 = self.x2 + dx
        
        # Apply boundary constraints if provided
        if bounds:
            bound_x1, bound_y1, bound_x2, bound_y2 = bounds
            new_x1 = max(bound_x1, min(bound_x2, new_x1))
            new_y1 = max(bound_y1, min(bound_y2, new_y1))
            new_x2 = max(bound_x1, min(bound_x2, new_x2))
            new_y2 = max(bound_y1, min(bound_y2, new_y2))
        
        # Ensure minimum size
        if abs(new_x2 - new_x1) >= min_size and abs(new_y2 - new_y1) >= min_size:
            self.x1 = new_x1
            self.y1 = new_y1
            self.x2 = new_x2
            self.y2 = new_y2


class BoxEditor:
    """
    Manages box editing operations using a state pattern approach.
    """
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.boxes: List[Box] = []
        self.selected_idx: Optional[int] = None
        
        # Image boundary constraints (x1, y1, x2, y2)
        self.image_bounds: Optional[Tuple[int, int, int, int]] = None
        
        # History system for undo functionality
        self.history_stack: List[Tuple[List[Box], Optional[int]]] = []
        self.max_history_size = 50  # Limit to prevent memory issues
        self.undo_in_progress = False  # Flag to prevent recursive history saves
        
        # State variables
        self.mode = EditMode.IDLE
        self.active_handle: Optional[str] = None
        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None
        
        # Visual settings
        self.colors = {
            'normal': '#2ecc71',
            'selected': '#3498db',
            'handle': '#e74c3c'
        }
        self.box_width = 2
        self.handle_size = 8
        
        # Callbacks
        self.on_box_selected: Optional[Callable[[Optional[int]], None]] = None
        self.on_box_modified: Optional[Callable[[int], None]] = None
        
        self._bind_events()
    
    def _bind_events(self):
        """Set up canvas event bindings."""
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Motion>', self._on_mouse_motion)
        self.canvas.bind('<Double-Button-1>', self._on_double_click)
        
        # Keyboard bindings for undo
        self.canvas.bind('<Control-z>', self._on_undo)
        self.canvas.bind('<Control-Z>', self._on_undo)
        
        # Make canvas focusable for keyboard events
        self.canvas.focus_set()
    
    def _get_canvas_coords(self, event) -> Tuple[int, int]:
        """Get true canvas coordinates from event."""
        try:
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            return x, y
        except (tk.TclError, ValueError):
            return 0, 0
    
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        try:
            # Ensure canvas has focus for keyboard events
            self.canvas.focus_set()
            
            x, y = self._get_canvas_coords(event)
            
            # Save current state to history before any modifications
            operation_will_modify = False
            
            # Check if this will be a modification operation
            if self.selected_idx is not None:
                box = self.boxes[self.selected_idx]
                handle = box.get_handle_at(x, y, self.handle_size)
                if handle:
                    operation_will_modify = True
                elif box.contains_point(x, y):
                    operation_will_modify = True
            
            # Save history if we're about to modify
            if operation_will_modify:
                self._save_to_history()
            
            # First check if clicked on a handle of selected box
            if self.selected_idx is not None:
                box = self.boxes[self.selected_idx]
                handle = box.get_handle_at(x, y, self.handle_size)
                if handle:
                    self.mode = EditMode.RESIZE
                    self.active_handle = handle
                    self.last_x, self.last_y = x, y
                    return
            
            # Then check if clicked inside any box
            for i, box in enumerate(self.boxes):
                if box.contains_point(x, y):
                    self.select_box(i)
                    self.mode = EditMode.MOVE
                    self.last_x, self.last_y = x, y
                    return
            
            # If no box clicked, clear selection
            self.select_box(None)
            self.mode = EditMode.IDLE
            
        except Exception as e:
            print(f"Error in mouse down: {e}")
            self.mode = EditMode.IDLE
    
    def _on_mouse_move(self, event):
        """Handle mouse movement."""
        try:
            if self.mode == EditMode.IDLE:
                return
                
            x, y = self._get_canvas_coords(event)
            if self.last_x is None:
                self.last_x, self.last_y = x, y
                return
            
            dx = x - self.last_x
            dy = y - self.last_y
            
            if self.selected_idx is not None and 0 <= self.selected_idx < len(self.boxes):
                box = self.boxes[self.selected_idx]
                
                if self.mode == EditMode.MOVE:
                    box.move_by(dx, dy, bounds=self.image_bounds)
                elif self.mode == EditMode.RESIZE and self.active_handle:
                    box.resize_by(self.active_handle, dx, dy, bounds=self.image_bounds)
                
                self._redraw_all()
                
                if self.on_box_modified:
                    self.on_box_modified(self.selected_idx)
            
            self.last_x, self.last_y = x, y
            
        except Exception as e:
            print(f"Error in mouse move: {e}")
            self.mode = EditMode.IDLE
    
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        self.mode = EditMode.IDLE
        self.active_handle = None
        self.last_x = None
        self.last_y = None
    
    def _on_mouse_motion(self, event):
        """Handle mouse motion for cursor updates."""
        if self.mode != EditMode.IDLE:
            return
            
        x, y = self._get_canvas_coords(event)
        
        if self.selected_idx is not None:
            box = self.boxes[self.selected_idx]
            handle = box.get_handle_at(x, y, self.handle_size)
            
            if handle:
                if handle in ['nw', 'se']:
                    self.canvas.config(cursor='size_nw_se')
                elif handle in ['ne', 'sw']:
                    self.canvas.config(cursor='size_ne_sw')
                return
                
            if box.contains_point(x, y):
                self.canvas.config(cursor='fleur')
                return
        
        self.canvas.config(cursor='')
    
    def _on_double_click(self, event):
        """Handle double click for label editing."""
        if self.selected_idx is not None and self.on_box_selected:
            self.on_box_selected(self.selected_idx)
    
    def _on_undo(self, event):
        """Handle Ctrl+Z undo keyboard shortcut."""
        self.undo()
        return "break"  # Prevent event propagation
    
    def _redraw_all(self):
        """Redraw all boxes and handles."""
        # Clear all existing drawings
        for box in self.boxes:
            for obj_id in box.canvas_objects:
                self.canvas.delete(obj_id)
            box.canvas_objects.clear()
        
        # Redraw each box
        for i, box in enumerate(self.boxes):
            color = self.colors['selected'] if i == self.selected_idx else self.colors['normal']
            
            # Draw box
            rect_id = self.canvas.create_rectangle(
                box.x1, box.y1, box.x2, box.y2,
                outline=color,
                width=self.box_width
            )
            box.canvas_objects.append(rect_id)
            
            # Draw label if present
            if box.label:
                text_id = self.canvas.create_text(
                    box.x1, box.y1 - 5,
                    text=box.label,
                    anchor='sw',
                    fill=color
                )
                box.canvas_objects.append(text_id)
            
            # Draw handles for selected box
            if i == self.selected_idx:
                self._draw_handles(box)
    
    def _draw_handles(self, box: Box):
        """Draw resize handles for a box."""
        size = self.handle_size
        half = size // 2
        
        corners = [
            (box.x1, box.y1), (box.x2, box.y1),
            (box.x1, box.y2), (box.x2, box.y2)
        ]
        
        for x, y in corners:
            handle_id = self.canvas.create_rectangle(
                x - half, y - half,
                x + half, y + half,
                fill=self.colors['handle'],
                outline='white'
            )
            box.canvas_objects.append(handle_id)
    
    def set_boxes(self, boxes: List[List[int]], labels: List[str] = None):
        """Set the boxes to be managed."""
        # Save current state to history if we have existing boxes
        if self.boxes:
            self._save_to_history()
            
        self.boxes.clear()
        self.selected_idx = None
        
        if labels is None:
            labels = ["" for _ in boxes]
        
        for box_coords, label in zip(boxes, labels):
            x1, y1, x2, y2 = box_coords
            self.boxes.append(Box(x1, y1, x2, y2, label))
        
        self._redraw_all()
    
    def get_boxes(self) -> Tuple[List[List[int]], List[str]]:
        """Get current boxes and labels."""
        coords = [[box.x1, box.y1, box.x2, box.y2] for box in self.boxes]
        labels = [box.label for box in self.boxes]
        return coords, labels
    
    def select_box(self, index: Optional[int]):
        """Select a box by index."""
        if index != self.selected_idx:
            self.selected_idx = index
            self._redraw_all()
            if self.on_box_selected:
                self.on_box_selected(index)
    
    def update_box_label(self, index: int, label: str):
        """Update box label."""
        if 0 <= index < len(self.boxes):
            self.boxes[index].label = label
            self._redraw_all()
            if self.on_box_modified:
                self.on_box_modified(index)
    
    def clear_boxes(self):
        """Remove all boxes and their canvas objects."""
        # Save state before clearing if we have boxes
        if self.boxes:
            self._save_to_history()
            
        # First delete all canvas objects
        for box in self.boxes:
            for obj_id in box.canvas_objects:
                try:
                    self.canvas.delete(obj_id)
                except tk.TclError:
                    pass  # Object may already be deleted
        
        # Clear all boxes
        self.boxes.clear()
        self.selected_idx = None
        
        # Reset state
        self.mode = EditMode.IDLE
        self.active_handle = None
        self.last_x = None
        self.last_y = None
        
        # Force a canvas update
        self.canvas.update_idletasks()
        
        # Clear any remaining objects with box-related tags
        self.canvas.delete('box')
        self.canvas.delete('handle')
    
    def remove_box(self, index: int):
        """Remove a specific box."""
        if 0 <= index < len(self.boxes):
            # Save state before removing
            self._save_to_history()
            
            box = self.boxes.pop(index)
            for obj_id in box.canvas_objects:
                self.canvas.delete(obj_id)
            if self.selected_idx == index:
                self.selected_idx = None
            elif self.selected_idx > index:
                self.selected_idx -= 1
            self._redraw_all()
    
    def filter_boxes_by_size(self, min_area: int = 0):
        """Remove boxes smaller than min_area."""
        original_len = len(self.boxes)
        
        # Save state before filtering if we have boxes
        if self.boxes:
            self._save_to_history()
            
        self.boxes = [box for box in self.boxes if box.area >= min_area]
        
        if len(self.boxes) != original_len:
            self.selected_idx = None
            self._redraw_all()
    
    def set_image_bounds(self, bounds: Tuple[int, int, int, int]):
        """Set the image boundary constraints for box editing."""
        self.image_bounds = bounds 
    
    def _save_to_history(self):
        """Save current state to history stack."""
        if self.undo_in_progress:
            return
            
        # Create deep copies of boxes to avoid reference issues
        boxes_copy = []
        for box in self.boxes:
            box_copy = Box(box.x1, box.y1, box.x2, box.y2, box.label)
            boxes_copy.append(box_copy)
        
        # Save state (boxes and selected index)
        state = (boxes_copy, self.selected_idx)
        self.history_stack.append(state)
        
        # Limit history size
        if len(self.history_stack) > self.max_history_size:
            self.history_stack.pop(0)
    
    def undo(self):
        """Undo the last operation."""
        if not self.history_stack:
            return False
            
        # Set flag to prevent recursive history saves
        self.undo_in_progress = True
        
        try:
            # Restore previous state
            boxes_copy, selected_idx = self.history_stack.pop()
            
            # Clear current canvas objects
            for box in self.boxes:
                for obj_id in box.canvas_objects:
                    try:
                        self.canvas.delete(obj_id)
                    except tk.TclError:
                        pass
            
            # Restore boxes
            self.boxes = []
            for box_copy in boxes_copy:
                restored_box = Box(box_copy.x1, box_copy.y1, box_copy.x2, box_copy.y2, box_copy.label)
                self.boxes.append(restored_box)
            
            # Restore selection
            self.selected_idx = selected_idx
            
            # Redraw everything
            self._redraw_all()
            
            # Notify about changes
            if self.on_box_modified and self.boxes:
                self.on_box_modified(0)  # Trigger update
            
            return True
            
        finally:
            self.undo_in_progress = False 
    
    def clear_history(self):
        """Clear the undo history stack. Use when starting a new image session."""
        self.history_stack.clear() 