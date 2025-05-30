import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import json
import glob

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import mask_to_boxes

class FullDatasetEvaluator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.results = {}
        
        # Find all images in the dataset
        images_folder = os.path.join(dataset_path, "images")
        annotations_folder = os.path.join(dataset_path, "annotations")
        
        if not os.path.exists(images_folder) or not os.path.exists(annotations_folder):
            raise ValueError(f"Dataset folders not found: {images_folder} or {annotations_folder}")
        
        # Get all jpg files and check if corresponding mask exists
        self.test_samples = []
        image_files = glob.glob(os.path.join(images_folder, "*.jpg"))
        
        for image_file in sorted(image_files):
            image_name = os.path.basename(image_file)
            mask_name = image_name.replace('.jpg', '.png')
            mask_path = os.path.join(annotations_folder, mask_name)
            
            if os.path.exists(mask_path):
                self.test_samples.append(image_name)
        
        print(f"Found {len(self.test_samples)} image-mask pairs in the dataset")
        
    def evaluate_sample(self, sample_name, current_idx, total_count):
        """Evaluate a single sample"""
        
        print(f"\n{'='*80}")
        print(f"🔍 EVALUATING: {sample_name} ({current_idx + 1}/{total_count})")
        print(f"{'='*80}")
        
        # Load image and mask
        image_path = os.path.join(self.dataset_path, "images", sample_name)
        mask_path = os.path.join(self.dataset_path, "annotations", sample_name.replace('.jpg', '.png'))
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"❌ Files not found for {sample_name}")
            return None
            
        raw_image = Image.open(image_path)
        raw_mask = Image.open(mask_path)
        
        image_np = np.array(raw_image)
        mask_np = np.array(raw_mask)
        
        print(f"📏 Image shape: {image_np.shape}")
        print(f"📏 Mask shape: {mask_np.shape}")
        print(f"📊 Progress: {current_idx + 1}/{total_count} ({((current_idx + 1)/total_count)*100:.1f}%)")
        
        # Create transformations
        transformations = {
            'transpose': mask_np.T,
            'rot90_cw': np.rot90(mask_np, k=-1),  # 90° clockwise
            'rot90_ccw': np.rot90(mask_np, k=1),   # 90° counter-clockwise
        }
        
        # Create visualization
        self.create_comparison_plot(image_np, transformations, sample_name, current_idx + 1, total_count)
        
        # Get user input
        return self.get_user_choice(sample_name, current_idx + 1, total_count)
        
    def create_comparison_plot(self, image_np, transformations, sample_name, current_idx, total_count):
        """Create and display comparison plot"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Transformation Comparison: {sample_name} ({current_idx}/{total_count})", 
                    fontsize=16, fontweight='bold')
        
        # Top row: Original image and mask transformations
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Original Image", fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        
        # Show each transformation
        transform_items = list(transformations.items())
        for i, (name, transformed_mask) in enumerate(transform_items):
            axes[0, i + 1].imshow(transformed_mask, cmap='gray')
            axes[0, i + 1].set_title(f"{name.replace('_', ' ').title()}\nShape: {transformed_mask.shape}", fontsize=10)
            axes[0, i + 1].axis('off')
        
        # Bottom row: Image with overlaid masks and bounding boxes
        axes[1, 0].imshow(image_np)
        axes[1, 0].set_title("Reference Image", fontweight='bold', fontsize=12)
        axes[1, 0].axis('off')
        
        transformation_metrics = {}
        
        for i, (name, transformed_mask) in enumerate(transform_items):
            # Create overlay
            overlay = image_np.copy()
            
            # Process mask for box generation
            mask_for_boxes = transformed_mask.copy()
            if mask_for_boxes.dtype != np.uint8:
                mask_for_boxes = mask_for_boxes.astype(np.uint8)
            
            # Convert to binary
            unique_values = np.unique(mask_for_boxes)
            if len(unique_values) > 2:
                _, mask_for_boxes = cv2.threshold(mask_for_boxes, 0, 255, cv2.THRESH_BINARY)
            elif len(unique_values) == 2:
                mask_for_boxes = np.where(mask_for_boxes > 0, 255, 0).astype(np.uint8)
            
            # Generate boxes
            boxes = mask_to_boxes(mask_for_boxes)
            
            # Add semi-transparent red overlay where mask is white
            mask_colored = np.zeros_like(overlay)
            mask_colored[mask_for_boxes > 0] = [255, 0, 0]  # Red
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            # Draw bounding boxes in green
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            axes[1, i + 1].imshow(overlay)
            
            # Calculate metrics
            if boxes:
                box = boxes[0]
                width = box[2] - box[0]
                height = box[3] - box[1]
                aspect_ratio = width / height if height > 0 else 0
                title = f"{name.replace('_', ' ').title()}\nBoxes: {len(boxes)} | AR: {aspect_ratio:.2f}"
                
                transformation_metrics[name] = {
                    'boxes': len(boxes),
                    'aspect_ratio': aspect_ratio,
                    'width': width,
                    'height': height
                }
            else:
                title = f"{name.replace('_', ' ').title()}\nNo boxes detected"
                transformation_metrics[name] = {
                    'boxes': 0,
                    'aspect_ratio': 0,
                    'width': 0,
                    'height': 0
                }
                
            axes[1, i + 1].set_title(title, fontsize=10)
            axes[1, i + 1].axis('off')
        
        # Print metrics to console for reference
        print(f"\n📊 Transformation Metrics for {sample_name}:")
        print("-" * 50)
        for name, metrics in transformation_metrics.items():
            print(f"  {name.replace('_', ' ').title()}: {metrics['boxes']} boxes, AR: {metrics['aspect_ratio']:.2f}")
        
        plt.tight_layout()
        plt.show(block=True)  # This will show the plot and wait for user to close it
        
    def get_user_choice(self, sample_name, current_idx, total_count):
        """Get user's choice via command line"""
        
        print(f"\n🎯 Please evaluate {sample_name} ({current_idx}/{total_count}):")
        print("=" * 50)
        print("Which transformation best aligns the mask with the crack?")
        print()
        print("Options:")
        print("  1. Transpose")
        print("  2. Rot90 Cw (90° clockwise)")
        print("  3. Rot90 Ccw (90° counter-clockwise)")
        print("  s. Skip this image")
        print("  q. Quit evaluation and show results so far")
        print("  r. Show results summary so far")
        print()
        
        while True:
            choice = input("Enter your choice (1/2/3/s/q/r): ").strip().lower()
            
            if choice == '1':
                confidence = self.get_confidence()
                return {'transformation': 'transpose', 'confidence': confidence}
            elif choice == '2':
                confidence = self.get_confidence()
                return {'transformation': 'rot90_cw', 'confidence': confidence}
            elif choice == '3':
                confidence = self.get_confidence()
                return {'transformation': 'rot90_ccw', 'confidence': confidence}
            elif choice == 's':
                return {'transformation': 'skipped', 'confidence': 0}
            elif choice == 'q':
                return None
            elif choice == 'r':
                self.show_partial_results()
                continue
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, s, q, or r.")
                
    def get_confidence(self):
        """Get confidence level from user"""
        print("\nHow confident are you in this choice?")
        print("1 = Very uncertain")
        print("2 = Somewhat uncertain") 
        print("3 = Neutral")
        print("4 = Somewhat confident")
        print("5 = Very confident")
        
        while True:
            try:
                confidence = int(input("Enter confidence (1-5): ").strip())
                if 1 <= confidence <= 5:
                    return confidence
                else:
                    print("❌ Please enter a number between 1 and 5.")
            except ValueError:
                print("❌ Please enter a valid number.")
                
    def show_partial_results(self):
        """Show results so far during evaluation"""
        
        if not self.results:
            print("\n📊 No results recorded yet.")
            return
            
        print(f"\n{'='*50}")
        print("📊 RESULTS SO FAR")
        print(f"{'='*50}")
        
        # Count votes
        vote_counts = {}
        evaluated_count = 0
        
        for sample, result in self.results.items():
            if result['selected_transformation'] != 'skipped':
                transform = result['selected_transformation']
                vote_counts[transform] = vote_counts.get(transform, 0) + 1
                evaluated_count += 1
        
        print(f"Images evaluated: {evaluated_count}/{len(self.results)}")
        print(f"Total images in dataset: {len(self.test_samples)}")
        
        if vote_counts:
            print(f"\nCurrent vote tally:")
            for transform, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / evaluated_count) * 100 if evaluated_count > 0 else 0
                print(f"  {transform.replace('_', ' ').title()}: {count} votes ({percentage:.1f}%)")
        
        print(f"\n{'='*50}")
                
    def run_evaluation(self):
        """Run the complete evaluation"""
        
        print("🚀 FULL DATASET TRANSFORMATION EVALUATION")
        print("=" * 80)
        print(f"Total images to evaluate: {len(self.test_samples)}")
        print()
        print("This tool will show you different mask transformations for EVERY image")
        print("in your dataset and ask you to choose which one best aligns with the crack.")
        print()
        print("Instructions:")
        print("- Look at the bottom row showing overlays")
        print("- Red areas show the mask")
        print("- Green boxes show detected bounding boxes")
        print("- Choose the transformation where the mask/boxes")
        print("  best align with the visible crack in the image")
        print("- You can type 'q' to quit early and see results")
        print("- You can type 'r' to see current results during evaluation")
        print()
        
        # Ask if user wants to start from a specific image
        start_idx = 0
        if len(self.test_samples) > 20:
            start_choice = input(f"Start from beginning? (y/n) - 'n' to specify starting image: ").strip().lower()
            if start_choice == 'n':
                while True:
                    try:
                        start_idx = int(input(f"Enter starting image index (1-{len(self.test_samples)}): ")) - 1
                        if 0 <= start_idx < len(self.test_samples):
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(self.test_samples)}")
                    except ValueError:
                        print("Please enter a valid number")
        
        print(f"\nStarting evaluation from image {start_idx + 1}...")
        input("Press Enter to start...")
        
        for i in range(start_idx, len(self.test_samples)):
            sample = self.test_samples[i]
            
            result = self.evaluate_sample(sample, i, len(self.test_samples))
            
            if result is None:  # User quit
                print(f"\n👋 Evaluation stopped by user at image {i + 1}/{len(self.test_samples)}.")
                break
                
            self.results[sample] = {
                'selected_transformation': result['transformation'],
                'confidence': result['confidence'],
                'timestamp': self.get_timestamp(),
                'image_index': i + 1
            }
            
            print(f"✅ Recorded: {sample} -> {result['transformation']} (confidence: {result['confidence']})")
            
            # Show progress update every 10 images
            if (i + 1) % 10 == 0:
                self.show_partial_results()
        
        # Show final results
        self.show_final_results()
        
    def get_timestamp(self):
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
        
    def show_final_results(self):
        """Display final comprehensive results"""
        
        if not self.results:
            print("\n📊 No results to show.")
            return
            
        print(f"\n{'='*80}")
        print("🎯 FINAL EVALUATION RESULTS")
        print(f"{'='*80}")
        
        # Count votes
        vote_counts = {}
        confidence_weighted_votes = {}
        skipped_count = 0
        evaluated_count = 0
        
        for sample, result in self.results.items():
            if result['selected_transformation'] == 'skipped':
                skipped_count += 1
                continue
                
            evaluated_count += 1
            transform = result['selected_transformation']
            confidence = result['confidence']
            
            vote_counts[transform] = vote_counts.get(transform, 0) + 1
            confidence_weighted_votes[transform] = confidence_weighted_votes.get(transform, 0) + confidence
        
        # Summary stats
        print(f"📊 Evaluation Summary:")
        print(f"   Total images in dataset: {len(self.test_samples)}")
        print(f"   Images evaluated: {len(self.results)}")
        print(f"   Images with valid selection: {evaluated_count}")
        print(f"   Images skipped: {skipped_count}")
        print(f"   Completion rate: {(len(self.results)/len(self.test_samples))*100:.1f}%")
        
        if not vote_counts:
            print("\n❌ No valid evaluations to analyze.")
            return
        
        # Show vote summary
        print(f"\n📊 Vote Summary:")
        print("-" * 40)
        for transform, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / evaluated_count) * 100
            print(f"  {transform.replace('_', ' ').title()}: {count} votes ({percentage:.1f}%)")
        
        # Show confidence-weighted results
        print(f"\n📈 Confidence-Weighted Scores:")
        print("-" * 40)
        for transform, weighted_count in sorted(confidence_weighted_votes.items(), key=lambda x: x[1], reverse=True):
            avg_confidence = weighted_count / vote_counts.get(transform, 1)
            print(f"  {transform.replace('_', ' ').title()}: {weighted_count:.1f} (avg confidence: {avg_confidence:.1f})")
        
        # Determine winner
        winner = max(vote_counts.keys(), key=lambda k: confidence_weighted_votes.get(k, 0))
        winner_votes = vote_counts[winner]
        winner_percentage = (winner_votes / evaluated_count) * 100
        winner_confidence = confidence_weighted_votes[winner] / winner_votes
        
        print(f"\n🏆 RECOMMENDED TRANSFORMATION: {winner.replace('_', ' ').title()}")
        print(f"   Total votes: {winner_votes}/{evaluated_count} ({winner_percentage:.1f}%)")
        print(f"   Average confidence: {winner_confidence:.1f}/5")
        print(f"   Weighted score: {confidence_weighted_votes[winner]:.1f}")
        
        # Save comprehensive results
        comprehensive_results = {
            'evaluation_summary': {
                'total_images': len(self.test_samples),
                'images_evaluated': len(self.results),
                'valid_evaluations': evaluated_count,
                'skipped': skipped_count,
                'completion_rate': (len(self.results)/len(self.test_samples))*100
            },
            'recommended_transformation': winner,
            'vote_counts': vote_counts,
            'confidence_weighted_votes': confidence_weighted_votes,
            'detailed_results': self.results,
            'all_images_in_dataset': self.test_samples
        }
        
        # Save to files
        with open("full_dataset_evaluation_results.json", 'w') as f:
            json.dump(comprehensive_results, f, indent=4)
            
        print(f"\n💾 Comprehensive results saved to:")
        print(f"   - full_dataset_evaluation_results.json")
        
        print(f"\n{'='*80}")
        print("🎉 Full dataset evaluation completed!")
        print(f"Based on your evaluation of {evaluated_count} images,")
        print(f"the recommended transformation is: {winner.replace('_', ' ').title()}")
        print(f"{'='*80}")

def main():
    """Main function"""
    dataset_path = r"C:\Users\ASUS TUF\Desktop\Annotators\seg-to-box\dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        return
        
    try:
        evaluator = FullDatasetEvaluator(dataset_path)
        evaluator.run_evaluation()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main() 