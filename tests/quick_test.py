import os
import numpy as np
import matplotlib.pyplot as plt
from utils import mask_to_boxes, load_image_and_mask, visualize_boxes

def quick_test_single_sample(image_name="1.jpg"):
    """Quick test with a single sample from the dataset"""
    
    # Dataset paths
    dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
    image_path = os.path.join(dataset_path, "images", image_name)
    mask_name = image_name.replace('.jpg', '.png')
    mask_path = os.path.join(dataset_path, "annotations", mask_name)
    
    print(f"Testing with sample: {image_name}")
    print(f"Image path: {image_path}")
    print(f"Mask path: {mask_path}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    if not os.path.exists(mask_path):
        print(f"❌ Mask file not found: {mask_path}")
        return
    
    # Load image and mask
    print("\n📁 Loading image and mask...")
    image, mask = load_image_and_mask(image_path, mask_path)
    
    if image is None or mask is None:
        print("❌ Failed to load image or mask")
        return
    
    print(f"✅ Successfully loaded!")
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask unique values: {np.unique(mask)}")
    
    # Generate bounding boxes
    print("\n📦 Generating bounding boxes...")
    boxes = mask_to_boxes(mask)
    print(f"✅ Generated {len(boxes)} bounding boxes")
    
    if len(boxes) > 0:
        print("   Sample boxes:")
        for i, box in enumerate(boxes[:5]):  # Show first 5 boxes
            print(f"     Box {i+1}: {box}")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    try:
        labels = [f"Object {i+1}" for i in range(len(boxes))]
        fig = visualize_boxes(image, boxes, labels)
        
        # Save the result
        output_path = f"quick_test_result_{image_name.replace('.jpg', '.png')}"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()  # Display the result
        
        print(f"✅ Visualization saved as: {output_path}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def compare_multiple_samples(sample_names=None):
    """Compare results across multiple samples"""
    
    if sample_names is None:
        sample_names = ["1.jpg", "5.jpg", "10.jpg"]
    
    print("🔍 Comparing multiple samples...")
    print("=" * 50)
    
    results = []
    
    for sample_name in sample_names:
        print(f"\nTesting {sample_name}:")
        
        # Dataset paths
        dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
        image_path = os.path.join(dataset_path, "images", sample_name)
        mask_name = sample_name.replace('.jpg', '.png')
        mask_path = os.path.join(dataset_path, "annotations", mask_name)
        
        # Check if files exist
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"  ❌ Files not found for {sample_name}")
            continue
        
        # Load and process
        image, mask = load_image_and_mask(image_path, mask_path)
        if image is None or mask is None:
            print(f"  ❌ Failed to load {sample_name}")
            continue
        
        boxes = mask_to_boxes(mask)
        
        result = {
            'name': sample_name,
            'image_shape': image.shape,
            'mask_shape': mask.shape,
            'unique_values': len(np.unique(mask)),
            'num_boxes': len(boxes)
        }
        results.append(result)
        
        print(f"  ✅ {len(boxes)} boxes generated")
    
    # Print comparison table
    print("\n📊 Comparison Results:")
    print("-" * 70)
    print(f"{'Sample':<15} {'Image Shape':<15} {'Mask Values':<12} {'Boxes':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<15} {str(result['image_shape']):<15} {result['unique_values']:<12} {result['num_boxes']:<8}")
    
    return results

def test_torchvision_vs_opencv():
    """Compare torchvision implementation with a simple OpenCV approach"""
    
    print("🔬 Comparing Torchvision vs OpenCV approach...")
    
    # Use a simple sample
    dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
    image_path = os.path.join(dataset_path, "images", "1.jpg")
    mask_path = os.path.join(dataset_path, "annotations", "1.png")
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print("❌ Test files not found")
        return
    
    # Load image and mask
    image, mask = load_image_and_mask(image_path, mask_path)
    if image is None or mask is None:
        print("❌ Failed to load test files")
        return
    
    # Torchvision approach (our current implementation)
    print("\n🔥 Torchvision approach:")
    torchvision_boxes = mask_to_boxes(mask)
    print(f"   Generated {len(torchvision_boxes)} boxes")
    if torchvision_boxes:
        print(f"   Sample: {torchvision_boxes[0]}")
    
    # Simple OpenCV approach for comparison
    print("\n🔧 OpenCV approach (for comparison):")
    import cv2
    
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values > 0]
    
    opencv_boxes = []
    for value in unique_values:
        binary_mask = (mask == value).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            opencv_boxes.append([x, y, x + w, y + h])
    
    print(f"   Generated {len(opencv_boxes)} boxes")
    if opencv_boxes:
        print(f"   Sample: {opencv_boxes[0]}")
    
    # Compare results
    print(f"\n📈 Comparison:")
    print(f"   Torchvision: {len(torchvision_boxes)} boxes")
    print(f"   OpenCV: {len(opencv_boxes)} boxes")
    print(f"   Match: {'✅' if len(torchvision_boxes) == len(opencv_boxes) else '❌'}")

if __name__ == "__main__":
    print("🚀 Quick Testing Script")
    print("=" * 50)
    
    # Test 1: Single sample
    quick_test_single_sample("1.jpg")
    
    print("\n" + "=" * 50)
    
    # Test 2: Multiple samples comparison
    compare_multiple_samples()
    
    print("\n" + "=" * 50)
    
    # Test 3: Implementation comparison
    test_torchvision_vs_opencv()
    
    print("\n🎉 Quick tests completed!") 