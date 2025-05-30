import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
from utils import (
    mask_to_boxes, 
    load_image_and_mask, 
    visualize_boxes, 
    load_dataset, 
    find_mask_file
)

class DatasetTester:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_dir = os.path.join(dataset_path, "images")
        self.annotations_dir = os.path.join(dataset_path, "annotations")
        self.output_dir = os.path.join(dataset_path, "test_outputs")
        
        # Create output directory for test results
        os.makedirs(self.output_dir, exist_ok=True)
        
    def test_dataset_loading(self):
        """Test loading the dataset"""
        print("=" * 50)
        print("Testing Dataset Loading")
        print("=" * 50)
        
        # Test load_dataset function
        image_files = load_dataset(self.dataset_path)
        print(f"Found {len(image_files)} images in dataset")
        
        if len(image_files) > 0:
            print(f"Sample image files: {image_files[:5]}")
            return image_files
        else:
            print("No images found!")
            return []
    
    def test_mask_file_finding(self, image_files):
        """Test finding corresponding mask files"""
        print("\n" + "=" * 50)
        print("Testing Mask File Finding")
        print("=" * 50)
        
        found_masks = 0
        missing_masks = []
        
        for image_file in image_files[:10]:  # Test first 10 images
            mask_file = find_mask_file(image_file, self.annotations_dir)
            if mask_file:
                found_masks += 1
                print(f"✓ {image_file} -> {mask_file}")
            else:
                missing_masks.append(image_file)
                print(f"✗ {image_file} -> No mask found")
        
        print(f"\nFound masks for {found_masks}/{min(10, len(image_files))} images")
        if missing_masks:
            print(f"Missing masks: {missing_masks}")
        
        return found_masks > 0
    
    def test_image_and_mask_loading(self, sample_images=None):
        """Test loading images and masks"""
        print("\n" + "=" * 50)
        print("Testing Image and Mask Loading")
        print("=" * 50)
        
        if sample_images is None:
            sample_images = ["1.jpg", "5.jpg", "10.jpg"]
        
        loaded_samples = []
        
        for image_file in sample_images:
            image_path = os.path.join(self.images_dir, image_file)
            
            # Find corresponding mask
            mask_file = find_mask_file(image_file, self.annotations_dir)
            if not mask_file:
                print(f"✗ No mask found for {image_file}")
                continue
                
            mask_path = os.path.join(self.annotations_dir, mask_file)
            
            # Test loading
            if os.path.exists(image_path):
                image, mask = load_image_and_mask(image_path, mask_path)
                
                if image is not None and mask is not None:
                    print(f"✓ Loaded {image_file}")
                    print(f"  Image shape: {image.shape}")
                    print(f"  Mask shape: {mask.shape}")
                    print(f"  Mask unique values: {np.unique(mask)}")
                    loaded_samples.append((image_file, image, mask))
                else:
                    print(f"✗ Failed to load {image_file}")
            else:
                print(f"✗ Image file not found: {image_file}")
        
        return loaded_samples
    
    def test_box_generation(self, loaded_samples):
        """Test bounding box generation from masks"""
        print("\n" + "=" * 50)
        print("Testing Box Generation")
        print("=" * 50)
        
        box_results = []
        
        for image_file, image, mask in loaded_samples:
            print(f"\nTesting {image_file}:")
            
            # Generate boxes using torchvision
            boxes = mask_to_boxes(mask)
            
            print(f"  Generated {len(boxes)} bounding boxes")
            if boxes:
                print(f"  Sample boxes: {boxes[:3]}")  # Show first 3 boxes
                
                # Validate box coordinates
                valid_boxes = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                        valid_boxes.append(box)
                    else:
                        print(f"  ⚠ Invalid box {i}: {box}")
                
                print(f"  Valid boxes: {len(valid_boxes)}/{len(boxes)}")
                box_results.append((image_file, image, mask, valid_boxes))
            else:
                print(f"  ⚠ No boxes generated for {image_file}")
        
        return box_results
    
    def test_visualization(self, box_results):
        """Test visualization of bounding boxes"""
        print("\n" + "=" * 50)
        print("Testing Visualization")
        print("=" * 50)
        
        for image_file, image, mask, boxes in box_results:
            print(f"\nVisualizing {image_file}:")
            
            try:
                # Create labels for boxes
                labels = [f"Object {i+1}" for i in range(len(boxes))]
                
                # Test visualization
                fig = visualize_boxes(image, boxes, labels)
                
                # Save visualization
                output_path = os.path.join(self.output_dir, f"visualization_{image_file.replace('.jpg', '.png')}")
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  ✓ Saved visualization to {output_path}")
                
                # Also save original image and mask for comparison
                orig_path = os.path.join(self.output_dir, f"original_{image_file}")
                Image.fromarray(image).save(orig_path)
                
                mask_path = os.path.join(self.output_dir, f"mask_{image_file.replace('.jpg', '.png')}")
                Image.fromarray(mask).save(mask_path)
                
            except Exception as e:
                print(f"  ✗ Visualization failed for {image_file}: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n" + "=" * 50)
        print("Testing Edge Cases")
        print("=" * 50)
        
        # Test with empty mask
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        boxes = mask_to_boxes(empty_mask)
        print(f"Empty mask test: Generated {len(boxes)} boxes (expected: 0)")
        
        # Test with single pixel mask
        single_pixel_mask = np.zeros((100, 100), dtype=np.uint8)
        single_pixel_mask[50, 50] = 255
        boxes = mask_to_boxes(single_pixel_mask)
        print(f"Single pixel mask test: Generated {len(boxes)} boxes")
        
        # Test visualization with no boxes
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        try:
            fig = visualize_boxes(test_image, [], [])
            plt.close(fig)
            print("Empty boxes visualization: ✓ Passed")
        except Exception as e:
            print(f"Empty boxes visualization: ✗ Failed - {e}")
    
    def generate_test_report(self, box_results):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 50)
        print("Generating Test Report")
        print("=" * 50)
        
        report = {
            "dataset_path": self.dataset_path,
            "total_samples_tested": len(box_results),
            "results": []
        }
        
        for image_file, image, mask, boxes in box_results:
            result = {
                "image_file": image_file,
                "image_shape": image.shape,
                "mask_shape": mask.shape,
                "unique_mask_values": np.unique(mask).tolist(),
                "num_boxes_generated": len(boxes),
                "boxes": boxes
            }
            report["results"].append(result)
        
        # Save report
        report_path = os.path.join(self.output_dir, "test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Test report saved to: {report_path}")
        
        # Print summary
        total_boxes = sum(len(result["boxes"]) for result in report["results"])
        print(f"\nTest Summary:")
        print(f"- Samples tested: {len(box_results)}")
        print(f"- Total boxes generated: {total_boxes}")
        print(f"- Average boxes per image: {total_boxes/len(box_results):.1f}")
        print(f"- Output directory: {self.output_dir}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("Starting comprehensive dataset testing...")
        
        # Test 1: Dataset loading
        image_files = self.test_dataset_loading()
        if not image_files:
            print("Cannot proceed with tests - no images found")
            return
        
        # Test 2: Mask file finding
        mask_finding_success = self.test_mask_file_finding(image_files)
        if not mask_finding_success:
            print("Cannot proceed with tests - no masks found")
            return
        
        # Test 3: Image and mask loading (test first 5 images)
        sample_images = image_files[:5]
        loaded_samples = self.test_image_and_mask_loading(sample_images)
        
        if not loaded_samples:
            print("Cannot proceed with tests - no samples loaded successfully")
            return
        
        # Test 4: Box generation
        box_results = self.test_box_generation(loaded_samples)
        
        # Test 5: Visualization
        if box_results:
            self.test_visualization(box_results)
        
        # Test 6: Edge cases
        self.test_edge_cases()
        
        # Test 7: Generate report
        if box_results:
            self.generate_test_report(box_results)
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)

def main():
    # Use the absolute path to the dataset
    dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
    
    # Create tester instance
    tester = DatasetTester(dataset_path)
    
    # Run all tests
    tester.run_all_tests()

if __name__ == "__main__":
    main() 