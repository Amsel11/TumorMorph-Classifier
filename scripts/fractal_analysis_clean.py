import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from scipy import ndimage
from scipy.stats import linregress

class BrainTumorFractalAnalyzer:
    def __init__(self, apply_morphological_correction=True, verbose=False):
        self.apply_correction = apply_morphological_correction
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_segmentation(self, seg_path):
        """Load BraTS segmentation file and return as PyTorch tensor."""
        nii = nib.load(seg_path)
        seg_data = nii.get_fdata()
        return torch.from_numpy(seg_data).float().to(self.device)
    
    def extract_tight_bounding_box(self, binary_volume, padding=1):
        if torch.sum(binary_volume) == 0:
            return binary_volume
        
        # Ensure device consistency
        device = binary_volume.device
        
        # Find coordinates of non-zero voxels
        nonzero_coords = torch.nonzero(binary_volume)
        
        # Get min and max coordinates in each dimension
        min_coords = torch.min(nonzero_coords, dim=0)[0]
        max_coords = torch.max(nonzero_coords, dim=0)[0]
        
        # Add padding but ensure we don't go out of bounds
        min_coords = torch.clamp(min_coords - padding, min=0)
        max_coords = torch.clamp(max_coords + padding, max=torch.tensor(binary_volume.shape, device=device)-1)
        
        # Extract the tight bounding box
        tight_volume = binary_volume[
            min_coords[0]:max_coords[0]+1,
            min_coords[1]:max_coords[1]+1, 
            min_coords[2]:max_coords[2]+1
        ]
        
        return tight_volume
    
    def morphological_correction(self, binary_volume):
        """Apply morphological operations to reduce segmentation artifacts."""
        if torch.sum(binary_volume) == 0:
            return binary_volume
        
        # Store original device
        original_device = binary_volume.device
        
        volume_np = binary_volume.cpu().numpy()
        
        # Fill holes and apply closing to reduce fragmentation
        filled = ndimage.binary_fill_holes(volume_np)
        structure = ndimage.generate_binary_structure(3, 1)
        closed = ndimage.binary_closing(filled, structure=structure, iterations=2)
        
        # Keep largest connected component
        labeled, num_features = ndimage.label(closed)
        if num_features > 1:
            component_sizes = ndimage.sum(closed, labeled, range(1, num_features + 1))
            largest_component = np.argmax(component_sizes) + 1
            final = (labeled == largest_component)
        else:
            final = closed
        
        # Return to original device
        return torch.from_numpy(final.astype(np.float32)).to(original_device)
    
    def generate_box_sizes(self, volume_shape, max_boxes=15):
        """Generate adaptive box sizes for optimal scale coverage."""
        min_dim = min(volume_shape)
        max_box_size = min_dim // 3
        
        box_sizes = []
        
        # Fine scale boxes
        for size in [1, 2, 3, 4, 5]:
            if size <= max_box_size:
                box_sizes.append(size)
        
        # Coarse scale boxes with exponential spacing
        current = 6
        while current <= max_box_size and len(box_sizes) < max_boxes:
            box_sizes.append(current)
            if current < 10:
                current += 2
            elif current < 20:
                current += 3
            else:
                current = int(current * 1.3)
        
        return box_sizes[:max_boxes]
    
    def box_counting_3d(self, binary_volume, box_sizes):
        """Perform 3D box-counting for fractal dimension calculation."""
        counts = []
        dims = binary_volume.shape
        
        for box_size in box_sizes:
            count = 0
            
            for i in range(0, dims[0], box_size):
                for j in range(0, dims[1], box_size):
                    for k in range(0, dims[2], box_size):
                        box = binary_volume[
                            i:min(i+box_size, dims[0]),
                            j:min(j+box_size, dims[1]), 
                            k:min(k+box_size, dims[2])
                        ]
                        
                        if torch.sum(box) > 0:
                            count += 1
            
            counts.append(count)
        
        return np.array(counts)
    
    def calculate_fractal_dimension(self, binary_volume, region_name=""):
        """Calculate fractal dimension using box-counting method."""
        if torch.sum(binary_volume) == 0:
            return None, None, None, None
        
        # Extract tight bounding box
        original_shape = binary_volume.shape
        tight_volume = self.extract_tight_bounding_box(binary_volume)
        
        # Apply morphological correction for necrotic regions
        if self.apply_correction and region_name.lower() == "necrotic":
            tight_volume = self.morphological_correction(tight_volume)
        
        if self.verbose:
            original_voxels = torch.sum(binary_volume).item()
            tight_voxels = torch.sum(tight_volume).item()
            volume_reduction = np.prod(original_shape) / np.prod(tight_volume.shape)
            print(f"{region_name}: Volume {original_shape} -> {tight_volume.shape} "
                  f"({volume_reduction:.1f}x reduction, {tight_voxels:,} voxels)")
        
        # Generate box sizes and perform counting
        box_sizes = self.generate_box_sizes(tight_volume.shape)
        
        if len(box_sizes) < 4:
            if self.verbose:
                print(f"{region_name}: Insufficient box sizes for analysis")
            return None, None, None, None
        
        counts = self.box_counting_3d(tight_volume, box_sizes)
        
        # Filter valid data points
        valid_indices = counts > 0
        if np.sum(valid_indices) < 4:
            if self.verbose:
                print(f"{region_name}: Insufficient valid data points")
            return None, None, None, None
        
        valid_counts = counts[valid_indices]
        valid_sizes = np.array(box_sizes)[valid_indices]
        
        # Linear regression in log-log space
        log_sizes = np.log(valid_sizes)
        log_counts = np.log(valid_counts)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_counts)
        
        fractal_dimension = -slope
        r_squared = r_value**2
        
        if self.verbose:
            scale_range = max(valid_sizes) / min(valid_sizes)
            print(f"{region_name}: FD = {fractal_dimension:.4f}, R² = {r_squared:.4f}, "
                  f"Scale range = {scale_range:.1f}:1, Data points = {len(valid_sizes)}")
        
        return fractal_dimension, valid_sizes, valid_counts, r_squared
    
    def extract_tumor_regions(self, segmentation):
        """Extract tumor regions from BraTS segmentation (labels: 1=necrotic, 2=edema, 4=enhancing)."""
        regions = {
            'necrotic': (segmentation == 1).float(),
            'edema': (segmentation == 2).float(), 
            'enhancing': (segmentation == 3).float()
        }
        return regions
    
    def analyze_patient(self, seg_path, patient_name):
        """Perform complete fractal analysis for a patient."""
        print(f"Analyzing patient: {patient_name}")
        
        
        # Load segmentation
        segmentation = self.load_segmentation(seg_path)
        unique_labels = np.unique(segmentation.cpu().numpy())
        print(f"Available labels in segmentation: {unique_labels}")
        
        if self.verbose:
            print(f"Segmentation shape: {segmentation.shape}")
            print(f"Device: {self.device}")
        
        # Extract regions
        regions = self.extract_tumor_regions(segmentation)
        
        # Calculate fractal dimensions
        results = {}
        
        for region_name, region_volume in regions.items():
            if torch.sum(region_volume) == 0:
                if self.verbose:
                    print(f"No {region_name} region found")
                results[region_name] = {
                    'fractal_dim': None, 'sizes': None, 'counts': None, 'r_squared': None
                }
                continue
            
            fractal_dim, sizes, counts, r_squared = self.calculate_fractal_dimension(
                region_volume, region_name
            )
            # Also compute and print voxel count for this region
            voxel_count = int(torch.sum(region_volume).item())
            print(f"{region_name.capitalize()} region voxel count: {voxel_count}")
            results[region_name] = {
                'fractal_dim': fractal_dim,
                'sizes': sizes,
                'counts': counts,
                'r_squared': r_squared
            }
        
        return results
    
    def validate_biological_order(self, results):
        """Check if fractal dimensions follow expected biological order."""
        expected_order = ['necrotic', 'edema', 'enhancing']
        measured_values = {}
        
        for region in expected_order:
            if region in results and results[region]['fractal_dim'] is not None:
                measured_values[region] = results[region]['fractal_dim']
        
        if len(measured_values) < 2:
            return None, None
        
        # Check order
        ordered_regions = list(measured_values.keys())
        ordered_values = [measured_values[r] for r in ordered_regions]
        is_correct_order = all(ordered_values[i] <= ordered_values[i+1] 
                             for i in range(len(ordered_values)-1))
        
        return is_correct_order, measured_values
    
    def plot_results(self, results, output_path=None):
        """Generate fractal dimension plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'necrotic': '#E74C3C', 'edema': '#3498DB', 'enhancing': '#2ECC71'}
        
        for region, color in colors.items():
            if region in results and results[region]['fractal_dim'] is not None:
                sizes = results[region]['sizes']
                counts = results[region]['counts']
                fractal_dim = results[region]['fractal_dim']
                r_squared = results[region]['r_squared']
                
                # Data points
                ax.loglog(sizes, counts, 'o', color=color, markersize=8, alpha=0.7,
                         label=f'{region.capitalize()}')
                
                # Fitted line
                log_sizes = np.log(sizes)
                log_counts = np.log(counts)
                A = np.vstack([log_sizes, np.ones(len(log_sizes))]).T
                slope, intercept = np.linalg.lstsq(A, log_counts, rcond=None)[0]
                
                fitted_counts = np.exp(slope * log_sizes + intercept)
                ax.loglog(sizes, fitted_counts, '--', color=color, linewidth=2, alpha=0.8,
                         label=f'{region.capitalize()} (FD={fractal_dim:.3f}, R²={r_squared:.3f})')
        
        ax.set_xlabel('Box Size (voxels)', fontsize=12)
        ax.set_ylabel('Number of Occupied Boxes', fontsize=12)
        ax.set_title('3D Fractal Dimension Analysis - Brain Tumor Regions', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def save_results(self, results, patient_name, seg_path, output_dir):
        """Save analysis results to JSON and text files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate biological order
        is_correct_order, measured_values = self.validate_biological_order(results)
        
        # Prepare data for saving
        log_data = {
            'patient_name': patient_name,
            'seg_path': seg_path,
            'timestamp': datetime.now().isoformat(),
            'morphological_correction_applied': self.apply_correction,
            'fractal_dimensions': {},
            'raw_data': {},
            'biological_validation': {
                'is_correct_order': is_correct_order,
                'measured_values': measured_values
            }
        }
        
        # Process results
        for region, data in results.items():
            if data['fractal_dim'] is not None:
                log_data['fractal_dimensions'][region] = float(data['fractal_dim'])
                log_data['raw_data'][region] = {
                    'box_sizes': data['sizes'].tolist() if data['sizes'] is not None else None,
                    'box_counts': data['counts'].tolist() if data['counts'] is not None else None,
                    'r_squared': float(data['r_squared']) if data['r_squared'] is not None else None
                }
            else:
                log_data['fractal_dimensions'][region] = None
                log_data['raw_data'][region] = None
        
        # Save JSON
        json_path = os.path.join(output_dir, f'{patient_name}_fractal_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Save text summary
        txt_path = os.path.join(output_dir, f'{patient_name}_fractal_summary.txt')
        with open(txt_path, 'w') as f:
            f.write(f"3D Fractal Dimension Analysis\n")
            f.write(f"Patient: {patient_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Morphological correction: {'Applied' if self.apply_correction else 'Not applied'}\n\n")
            
            f.write("Results:\n")
            f.write("-" * 40 + "\n")
            for region in ['necrotic', 'edema', 'enhancing']:
                if region in results and results[region]['fractal_dim'] is not None:
                    fd = results[region]['fractal_dim']
                    r2 = results[region]['r_squared']
                    f.write(f"{region.capitalize():12}: {fd:.4f} (R² = {r2:.4f})\n")
                else:
                    f.write(f"{region.capitalize():12}: Not calculated\n")
            
            if is_correct_order is not None:
                f.write(f"\nBiological Order Validation:\n")
                f.write("-" * 40 + "\n")
                if is_correct_order:
                    f.write("Order follows expected pattern: necrotic < edema < enhancing\n")
                else:
                    f.write("Order deviates from typical pattern (may indicate unusual pathology)\n")
        
        return json_path, txt_path

def main():
    # Configuration
    seg_path = "/gpfs/data/oermannlab/users/schula12/Morphology/Lumiere/segmentations_swin/swin_Patient-001_week-044.nii.gz"
    patient_name = "Patient-001"
    output_dir = "fractal_results"
    
    # Initialize analyzer
    analyzer = BrainTumorFractalAnalyzer(
        apply_morphological_correction=True,
        verbose=True
    )
    
    # Perform analysis
    results = analyzer.analyze_patient(seg_path, patient_name)
    
    # Display results
    print("\nFractal Dimension Results:")
    print("-" * 40)
    for region in ['necrotic', 'edema', 'enhancing']:
        if region in results and results[region]['fractal_dim'] is not None:
            fd = results[region]['fractal_dim']
            r2 = results[region]['r_squared']
            print(f"{region.capitalize():12}: {fd:.4f} (R² = {r2:.4f})")
        else:
            print(f"{region.capitalize():12}: Not calculated")
    
    # Biological validation
    is_correct, measured_values = analyzer.validate_biological_order(results)
    if is_correct is not None:
        print(f"\nBiological Order: {'Correct' if is_correct else 'Atypical'}")
        if not is_correct:
            print("Note: Atypical order may indicate unusual tumor pathology")
    
    # Save results
    json_path, txt_path = analyzer.save_results(results, patient_name, seg_path, output_dir)
    
    # Generate plot
    plot_path = os.path.join(output_dir, f'{patient_name}_fractal_plot.png')
    analyzer.plot_results(results, plot_path)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"- JSON: {json_path}")
    print(f"- Summary: {txt_path}")
    print(f"- Plot: {plot_path}")

if __name__ == "__main__":
    main() 