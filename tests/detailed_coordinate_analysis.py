import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_image_and_mask, mask_to_boxes

def analyze_image_mask_alignment():
    """Detailed analysis of image-mask alignment issues"""
    
    # Test with image 100.jpg which clearly shows the misalignment
    dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
    image_path = os.path.join(dataset_path, "images", "100.jpg")
    mask_path = os.path.join(dataset_path, "annotations", "100.png")
    
    print("🔬 Detailed Coordinate Analysis")
    print("=" * 60)
    
    # Load raw files without any processing
    print("1. Loading raw files...")
    raw_image = Image.open(image_path)
    raw_mask = Image.open(mask_path)
    
    print(f"   Raw image size (W×H): {raw_image.size}")
    print(f"   Raw mask size (W×H): {raw_mask.size}")
    
    # Convert to numpy arrays
    image_np = np.array(raw_image)
    mask_np = np.array(raw_mask)
    
    print(f"   Image numpy shape (H×W×C): {image_np.shape}")
    print(f"   Mask numpy shape: {mask_np.shape}")
    
    # Test different transformations
    transformations = {
        'original': mask_np,
        'transpose': mask_np.T,
        'flip_lr': np.fliplr(mask_np),
        'flip_ud': np.flipud(mask_np),
        'rot90_cw': np.rot90(mask_np, k=-1),  # 90° clockwise
        'rot90_ccw': np.rot90(mask_np, k=1),   # 90° counter-clockwise
        'rot180': np.rot90(mask_np, k=2),      # 180°
    }
    
    print("\n2. Testing different mask transformations...")
    results = {}
    
    for name, transformed_mask in transformations.items():
        print(f"   Testing {name}: shape {transformed_mask.shape}")
        
        # Check if dimensions match image
        if transformed_mask.shape[:2] == image_np.shape[:2]:
            # Generate bounding boxes
            try:
                # Handle different mask formats
                if len(transformed_mask.shape) == 3:
                    if transformed_mask.shape[2] == 4:  # RGBA
                        mask_for_boxes = transformed_mask[:,:,3] if np.any(transformed_mask[:,:,3] < 255) else cv2.cvtColor(transformed_mask[:,:,:3], cv2.COLOR_RGB2GRAY)
                    elif transformed_mask.shape[2] == 3:  # RGB
                        mask_for_boxes = cv2.cvtColor(transformed_mask, cv2.COLOR_RGB2GRAY)
                    else:
                        mask_for_boxes = transformed_mask[:,:,0]
                else:
                    mask_for_boxes = transformed_mask
                
                # Convert to binary
                if mask_for_boxes.dtype != np.uint8:
                    mask_for_boxes = mask_for_boxes.astype(np.uint8)
                
                unique_values = np.unique(mask_for_boxes)
                if len(unique_values) > 2:
                    _, mask_for_boxes = cv2.threshold(mask_for_boxes, 0, 255, cv2.THRESH_BINARY)
                elif len(unique_values) == 2:
                    mask_for_boxes = np.where(mask_for_boxes > 0, 255, 0).astype(np.uint8)
                
                boxes = mask_to_boxes(mask_for_boxes)
                results[name] = {
                    'mask': mask_for_boxes,
                    'boxes': boxes,
                    'valid': True
                }
                print(f"     ✓ Generated {len(boxes)} boxes: {boxes[:1]}")  # Show first box
            except Exception as e:
                print(f"     ✗ Error: {e}")
                results[name] = {'valid': False}
        else:
            print(f"     ✗ Shape mismatch: mask {transformed_mask.shape[:2]} vs image {image_np.shape[:2]}")
            results[name] = {'valid': False}
    
    return image_np, results

def create_visual_comparison(image_np, results):
    """Create a comprehensive visual comparison of all transformations"""
    
    print("\n3. Creating visual comparison...")
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() if v.get('valid', False)}
    
    if not valid_results:
        print("   No valid transformations found!")
        return
    
    # Create subplot grid
    n_results = len(valid_results) + 1  # +1 for original image
    cols = 3
    rows = (n_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    # Plot original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot each transformation
    for idx, (name, result) in enumerate(valid_results.items(), 1):
        if idx >= len(axes):
            break
            
        # Create overlay: image with mask and bounding boxes
        overlay = image_np.copy()
        mask = result['mask']
        boxes = result['boxes']
        
        # Add semi-transparent mask overlay
        mask_colored = np.zeros_like(overlay)
        mask_colored[mask > 0] = [255, 0, 0]  # Red overlay
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        axes[idx].imshow(overlay)
        axes[idx].set_title(f"{name.title()}\nBoxes: {len(boxes)}", fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(valid_results) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("comprehensive_transformation_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return valid_results

def find_best_transformation(image_np, results):
    """Analyze which transformation gives the most reasonable bounding box"""
    
    print("\n4. Analyzing transformation quality...")
    
    valid_results = {k: v for k, v in results.items() if v.get('valid', False)}
    
    if not valid_results:
        return None
    
    scores = {}
    
    for name, result in valid_results.items():
        boxes = result['boxes']
        if not boxes:
            scores[name] = {'score': 0, 'reason': 'No boxes detected'}
            continue
        
        score = 0
        reasons = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            # For cracks, we expect either:
            # 1. Long horizontal cracks (high aspect ratio, width > height)
            # 2. Long vertical cracks (low aspect ratio, height > width)
            
            if aspect_ratio > 2.0:  # Horizontal crack
                score += 10
                reasons.append(f"Good horizontal aspect ratio: {aspect_ratio:.2f}")
            elif aspect_ratio < 0.5:  # Vertical crack
                score += 8  # Slightly lower score as most cracks in dataset seem horizontal
                reasons.append(f"Good vertical aspect ratio: {aspect_ratio:.2f}")
            else:
                score += 2
                reasons.append(f"Square-ish box (aspect ratio: {aspect_ratio:.2f})")
            
            # Bonus for reasonable size (not too small, not too large)
            area = width * height
            image_area = image_np.shape[0] * image_np.shape[1]
            area_ratio = area / image_area
            
            if 0.01 < area_ratio < 0.8:  # Between 1% and 80% of image
                score += 5
                reasons.append(f"Good size (area ratio: {area_ratio:.3f})")
            
        scores[name] = {'score': score, 'reason': '; '.join(reasons)}
    
    # Print analysis
    print("   Transformation scores:")
    for name, data in sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"     {name:12}: {data['score']:2d} - {data['reason']}")
    
    # Find best transformation
    best_name = max(scores.keys(), key=lambda x: scores[x]['score'])
    best_score = scores[best_name]['score']
    
    print(f"\n   🏆 Best transformation: {best_name} (score: {best_score})")
    
    return best_name, valid_results[best_name]

def test_multiple_samples():
    """Test the analysis on multiple samples to see if there's a consistent pattern"""
    
    print("\n5. Testing multiple samples for consistency...")
    
    dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
    test_samples = ["1.jpg", "5.jpg", "10.jpg", "100.jpg", "101.jpg"]
    
    results_summary = {}
    
    for sample in test_samples:
        print(f"\n   Testing {sample}:")
        image_path = os.path.join(dataset_path, "images", sample)
        mask_path = os.path.join(dataset_path, "annotations", sample.replace('.jpg', '.png'))
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"     ✗ Files not found")
            continue
        
        # Quick analysis for this sample
        raw_image = Image.open(image_path)
        raw_mask = Image.open(mask_path)
        
        image_np = np.array(raw_image)
        mask_np = np.array(raw_mask)
        
        # Test key transformations
        transformations = {
            'original': mask_np,
            'transpose': mask_np.T,
            'rot90_cw': np.rot90(mask_np, k=-1),
            'rot90_ccw': np.rot90(mask_np, k=1),
        }
        
        valid_transforms = []
        for name, transformed_mask in transformations.items():
            if transformed_mask.shape[:2] == image_np.shape[:2]:
                valid_transforms.append(name)
        
        results_summary[sample] = valid_transforms
        print(f"     Valid transformations: {valid_transforms}")
    
    print(f"\n   Summary of valid transformations across samples:")
    for sample, transforms in results_summary.items():
        print(f"     {sample:8}: {transforms}")
    
    # Find most consistent transformation
    transform_counts = {}
    for transforms in results_summary.values():
        for transform in transforms:
            transform_counts[transform] = transform_counts.get(transform, 0) + 1
    
    print(f"\n   Transformation frequency:")
    for transform, count in sorted(transform_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     {transform:12}: {count}/{len(results_summary)} samples")
    
    return results_summary

def main():
    print("🚀 Starting detailed coordinate analysis...")
    
    # Analyze the main problematic case
    image_np, results = analyze_image_mask_alignment()
    
    # Create visual comparison
    valid_results = create_visual_comparison(image_np, results)
    
    # Find best transformation
    best_transform, best_result = find_best_transformation(image_np, results)
    
    # Test multiple samples
    multi_sample_results = test_multiple_samples()
    
    print(f"\n{'='*60}")
    print("🎯 CONCLUSIONS:")
    print(f"{'='*60}")
    print(f"1. Best transformation for sample image: {best_transform}")
    print(f"2. Most common valid transformation across samples: {max(transform_counts.items(), key=lambda x: x[1])[0] if 'transform_counts' in locals() else 'Unknown'}")
    print(f"3. Visual analysis saved to: comprehensive_transformation_analysis.png")
    
    return best_transform, best_result

if __name__ == "__main__":
    main() 