# Enhanced TumorMorphology Class with Integrated Lesion Analysis
# ==============================================================
# This integrates lesion analysis directly into the TumorMorphology class
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_dir, 'scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
import numpy as np
import torch
from scipy.stats import linregress
from scipy import ndimage
from typing import Dict, List, Optional, Tuple, Union
import nibabel as nib
from fractal_analysis_clean import BrainTumorFractalAnalyzer

class TumorMorphology:
    """Enhanced TumorMorphology class with integrated lesion analysis"""
    
    def __init__(self, use_gpu=True, label_remap=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.label_remap = label_remap or {}
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
        
        # Lesion analysis parameters
        self.lesion_thresholds = {
            1: 100,  # enhancing
            2: 50,   # necrotic  
            3: 200   # edema
        }
        
        # Create fractal analyzer
        self.fractal_analyzer = BrainTumorFractalAnalyzer(
            apply_morphological_correction=True,
            verbose=True
        )

    
    # ==========================================================================
    # EXISTING MORPHOLOGY METHODS (keeping your original functionality)
    # ==========================================================================
    def remap_labels(self, seg):
        seg_remapped = seg.copy()
        for old_label, new_label in self.label_remap.items():
            seg_remapped[seg == old_label] = new_label
        return seg_remapped

    def box_count_2d_gpu(self, data_tensor, box_size):
        """GPU 2D box counting (existing method)"""
        H, W = data_tensor.shape
        n_boxes_h = H // box_size
        n_boxes_w = W // box_size
        
        if n_boxes_h == 0 or n_boxes_w == 0:
            return 0, []
        
        cropped = data_tensor[:n_boxes_h*box_size, :n_boxes_w*box_size]
        reshaped = cropped.view(n_boxes_h, box_size, n_boxes_w, box_size)
        boxes = reshaped.permute(0, 2, 1, 3)
        box_sums = boxes.sum(dim=(2, 3))
        
        non_empty_boxes = (box_sums > 0).sum().item()
        masses = box_sums.cpu().numpy().flatten()
        
        return non_empty_boxes, masses
    
    def box_count_3d_gpu(self, data_tensor, box_size):
        """GPU 3D box counting (existing method)"""
        D, H, W = data_tensor.shape
        n_boxes_d = D // box_size
        n_boxes_h = H // box_size  
        n_boxes_w = W // box_size
        
        if n_boxes_d == 0 or n_boxes_h == 0 or n_boxes_w == 0:
            return 0, []
        
        cropped = data_tensor[:n_boxes_d*box_size, :n_boxes_h*box_size, :n_boxes_w*box_size]
        reshaped = cropped.view(n_boxes_d, box_size, n_boxes_h, box_size, n_boxes_w, box_size)
        boxes = reshaped.permute(0, 2, 4, 1, 3, 5)
        box_sums = boxes.sum(dim=(3, 4, 5))
        
        non_empty_boxes = (box_sums > 0).sum().item()
        masses = box_sums.cpu().numpy().flatten()
        
        return non_empty_boxes, masses
    
    def calculate_robust_lacunarity(self, masses):
        """Improved lacunarity calculation with numerical stability"""
        if len(masses) == 0:
            return None
        
        # Remove zeros and very small values
        masses_clean = masses[masses > 1e-6]
        
        if len(masses_clean) == 0:
            return None
        
        mean_mass = np.mean(masses_clean)
        var_mass = np.var(masses_clean)
        
        # Check for numerical stability
        if mean_mass < 1e-6:
            return None
        
        lacunarity = (var_mass / (mean_mass ** 2)) + 1
        
        # Sanity check - lacunarity should be reasonable
        if lacunarity > 50:  # More reasonable threshold
            return None
        
        return lacunarity
    
    def box_counting_3d_gpu(self, binary_volume):
        """GPU-accelerated 3D fractal dimension and improved lacunarity"""
        if not np.any(binary_volume):
            return None, None
        
        if self.use_gpu:
            data_tensor = torch.from_numpy(binary_volume.astype(np.float32)).to(self.device)
        else:
            data_tensor = torch.from_numpy(binary_volume.astype(np.float32))
        
        box_sizes = [2, 3, 4, 6, 8]
        counts = []
        all_masses = []
        valid_sizes = []
        
        for box_size in box_sizes:
            if box_size >= min(data_tensor.shape):
                break
            
            if self.use_gpu:
                count, masses = self.box_count_3d_gpu(data_tensor, box_size)
            else:
                count, masses = self._box_count_3d_cpu(binary_volume, box_size)
            
            if count > 0:
                counts.append(count)
                all_masses.extend(masses)
                valid_sizes.append(box_size)
        
        if len(counts) < 3:
            return None, None
        
        # Fractal dimension
        log_sizes = np.log(1.0 / np.array(valid_sizes))
        log_counts = np.log(counts)
        slope, _, _, _, _ = linregress(log_sizes, log_counts)
        fractal_dim = max(1.0, min(3.0, slope))
        
        # Improved lacunarity
        lacunarity = self.calculate_robust_lacunarity(np.array(all_masses))
        
        return fractal_dim, lacunarity
    
    # ==========================================================================
    # NEW INTEGRATED LESION ANALYSIS METHODS
    # ==========================================================================
    
    def analyze_connected_components_gpu(self, mask_3d: np.ndarray, min_lesion_size: int = 100) -> Dict:
        """
        GPU-accelerated connected component analysis for lesion detection
        
        Args:
            mask_3d: 3D binary mask
            min_lesion_size: Minimum voxel count to consider as a lesion
        
        Returns:
            Dict with lesion analysis results
        """
        if mask_3d is None or not np.any(mask_3d):
            return {
                'lesion_count': 0,
                'lesion_count_filtered': 0,
                'lesion_volumes': [],
                'lesion_volumes_filtered': [],
                'artifacts_removed': 0,
                'total_volume': 0,
                'mean_lesion_size': 0,
                'max_lesion_size': 0,
                'min_lesion_size': 0
            }
        
        # Find connected components using scipy (CPU-based but efficient)
        labeled_array, num_components = ndimage.label(mask_3d)
        
        if num_components == 0:
            return {
                'lesion_count': 0,
                'lesion_count_filtered': 0,
                'lesion_volumes': [],
                'lesion_volumes_filtered': [],
                'artifacts_removed': 0,
                'total_volume': np.sum(mask_3d),
                'mean_lesion_size': 0,
                'max_lesion_size': 0,
                'min_lesion_size': 0
            }
        
        # Calculate volume of each component
        lesion_volumes = []
        for i in range(1, num_components + 1):
            lesion_volume = np.sum(labeled_array == i)
            lesion_volumes.append(lesion_volume)
        
        # Filter by minimum size
        lesion_volumes_filtered = [v for v in lesion_volumes if v >= min_lesion_size]
        artifacts_removed = len(lesion_volumes) - len(lesion_volumes_filtered)
        
        # Statistics
        if lesion_volumes_filtered:
            mean_size = np.mean(lesion_volumes_filtered)
            max_size = np.max(lesion_volumes_filtered)
            min_size = np.min(lesion_volumes_filtered)
        else:
            mean_size = max_size = min_size = 0
        
        return {
            'lesion_count': num_components,
            'lesion_count_filtered': len(lesion_volumes_filtered),
            'lesion_volumes': lesion_volumes,
            'lesion_volumes_filtered': lesion_volumes_filtered,
            'artifacts_removed': artifacts_removed,
            'total_volume': np.sum(mask_3d),
            'mean_lesion_size': mean_size,
            'max_lesion_size': max_size,
            'min_lesion_size': min_size
        }
    
    def analyze_lesion_changes(self, baseline_seg: np.ndarray, followup_seg: np.ndarray, 
                              labels: List[int] = [1, 2, 3]) -> Dict:
        """
        Analyze lesion changes between baseline and followup scans
        
        Args:
            baseline_seg: 3D baseline segmentation
            followup_seg: 3D followup segmentation  
            labels: Tissue labels to analyze [1=enhancing, 2=necrotic, 3=edema]
        
        Returns:
            Dict with lesion change analysis per region
        """
        
        if baseline_seg is None or followup_seg is None:
            return {'error': 'Missing segmentation data'}
        
        label_names = {1: "enhancing", 2: "necrotic", 3: "edema"}
        
        lesion_changes = {
            'per_region': {},
            'summary': {
                'total_new_lesions': 0,
                'total_lost_lesions': 0,
                'net_lesion_change': 0,
                'regions_with_new_lesions': [],
                'regions_with_lost_lesions': [],
                'most_affected_region': None,
                'progression_severity': 'stable'
            }
        }
        
        total_new_lesions = 0
        total_lost_lesions = 0
        region_activity = {}
        
        for label in labels:
            region_name = label_names.get(label, f"label_{label}")
            threshold = self.lesion_thresholds.get(label, 100)
            
            # Extract region masks
            baseline_mask = (baseline_seg == label).astype(bool)
            followup_mask = (followup_seg == label).astype(bool)
            
            # Analyze lesions at both timepoints
            baseline_analysis = self.analyze_connected_components_gpu(baseline_mask, threshold)
            followup_analysis = self.analyze_connected_components_gpu(followup_mask, threshold)
            
            # Calculate changes
            baseline_count = baseline_analysis['lesion_count_filtered']
            followup_count = followup_analysis['lesion_count_filtered']
            new_lesions = max(0, followup_count - baseline_count)
            lost_lesions = max(0, baseline_count - followup_count)
            net_change = followup_count - baseline_count
            
            # Store detailed analysis
            lesion_changes['per_region'][region_name] = {
                'baseline_lesion_count': baseline_count,
                'followup_lesion_count': followup_count,
                'new_lesions': new_lesions,
                'lost_lesions': lost_lesions,
                'net_lesion_change': net_change,
                'baseline_mean_size': baseline_analysis['mean_lesion_size'],
                'followup_mean_size': followup_analysis['mean_lesion_size'],
                'baseline_max_size': baseline_analysis['max_lesion_size'],
                'followup_max_size': followup_analysis['max_lesion_size'],
                'baseline_artifacts_removed': baseline_analysis['artifacts_removed'],
                'followup_artifacts_removed': followup_analysis['artifacts_removed'],
                'threshold_used': threshold,
                'new_lesion_detected': new_lesions > 0,
                'lesion_loss_detected': lost_lesions > 0
            }
            
            # Track totals and activity
            total_new_lesions += new_lesions
            total_lost_lesions += lost_lesions
            region_activity[region_name] = new_lesions + lost_lesions
            
            if new_lesions > 0:
                lesion_changes['summary']['regions_with_new_lesions'].append(region_name)
            if lost_lesions > 0:
                lesion_changes['summary']['regions_with_lost_lesions'].append(region_name)
        
        # Summary calculations
        lesion_changes['summary']['total_new_lesions'] = total_new_lesions
        lesion_changes['summary']['total_lost_lesions'] = total_lost_lesions
        lesion_changes['summary']['net_lesion_change'] = total_new_lesions - total_lost_lesions
        
        # Most affected region
        if region_activity:
            most_affected = max(region_activity, key=region_activity.get)
            if region_activity[most_affected] > 0:
                lesion_changes['summary']['most_affected_region'] = most_affected
        
        # Progression severity assessment
        if total_new_lesions == 0 and total_lost_lesions == 0:
            severity = 'stable'
        elif lesion_changes['per_region'].get('enhancing', {}).get('new_lesions', 0) > 0:
            severity = 'severe_progression'  # New enhancing lesions = severe
        elif total_new_lesions >= 3:
            severity = 'severe_progression'
        elif total_new_lesions >= 1:
            severity = 'moderate_progression'
        elif total_lost_lesions >= 3:
            severity = 'significant_response'
        elif total_lost_lesions >= 1:
            severity = 'mild_response'
        else:
            severity = 'stable'
        
        lesion_changes['summary']['progression_severity'] = severity
        
        return lesion_changes
    
    def calculate_lesion_morphology(self, mask_3d: np.ndarray, min_lesion_size: int = 100) -> Dict:
        """
        Calculate morphological features for individual lesions
        
        Args:
            mask_3d: 3D binary mask
            min_lesion_size: Minimum lesion size to analyze
            
        Returns:
            Dict with per-lesion morphological features
        """
        lesion_analysis = self.analyze_connected_components_gpu(mask_3d, min_lesion_size)
        
        if lesion_analysis['lesion_count_filtered'] == 0:
            return {
                'lesion_count': 0,
                'lesion_morphology': [],
                'aggregate_stats': {}
            }
        
        # Find connected components
        labeled_array, _ = ndimage.label(mask_3d)
        
        lesion_morphologies = []
        
        for i in range(1, lesion_analysis['lesion_count'] + 1):
            lesion_mask = (labeled_array == i).astype(bool)
            lesion_volume = np.sum(lesion_mask)
            
            if lesion_volume < min_lesion_size:
                continue
            
            # Calculate morphological features for this lesion
            lesion_morph = {
                'lesion_id': i,
                'volume': lesion_volume,
                'surface_area': self.calculate_surface_area(lesion_mask),
                'compactness': None,
                'fractal_3d': None,
                'lacunarity_3d': None
            }
            
            # Compactness
            if lesion_morph['surface_area'] > 0:
                lesion_morph['compactness'] = self.calculate_compactness(
                    lesion_volume, lesion_morph['surface_area']
                )
            
            # Fractal analysis (only for larger lesions due to computational cost)
            # Fractal analysis (only for larger lesions due to computational cost)
            if lesion_volume > 1000:
                tensor_lesion = torch.tensor(lesion_mask).float().to(self.device)
                fd_3d, _, _, _ = self.fractal_analyzer.calculate_fractal_dimension(tensor_lesion, region_name="lesion")
                lac_3d = self.calculate_robust_lacunarity(lesion_mask.flatten())

                lesion_morph['fractal_3d'] = fd_3d
                lesion_morph['lacunarity_3d'] = lac_3d

            
            lesion_morphologies.append(lesion_morph)
        
        # Aggregate statistics
        if lesion_morphologies:
            volumes = [l['volume'] for l in lesion_morphologies]
            compactnesses = [l['compactness'] for l in lesion_morphologies if l['compactness'] is not None]
            fractals = [l['fractal_3d'] for l in lesion_morphologies if l['fractal_3d'] is not None]
            
            aggregate_stats = {
                'mean_volume': np.mean(volumes),
                'std_volume': np.std(volumes),
                'mean_compactness': np.mean(compactnesses) if compactnesses else None,
                'std_compactness': np.std(compactnesses) if compactnesses else None,
                'mean_fractal_3d': np.mean(fractals) if fractals else None,
                'std_fractal_3d': np.std(fractals) if fractals else None
            }
        else:
            aggregate_stats = {}
        
        return {
            'lesion_count': len(lesion_morphologies),
            'lesion_morphology': lesion_morphologies,
            'aggregate_stats': aggregate_stats
        }
    
    # ==========================================================================
    # EXISTING METHODS (keeping original functionality)
    # ==========================================================================
    
    def calculate_surface_area(self, binary_volume):
        """Calculate surface area using marching cubes algorithm"""
        from skimage.measure import marching_cubes
        
        if not np.any(binary_volume):
            return 0
        
        try:
            # Generate mesh using marching cubes
            verts, faces, normals, values = marching_cubes(
                binary_volume.astype(float), 
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
            
            # Calculate area of each triangle face
            areas = []
            for face in faces:
                v0, v1, v2 = verts[face]
                # Cross product for triangle area
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                areas.append(area)
            
            return np.sum(areas)
        
        except Exception:
            # Fallback to simple face counting if marching cubes fails
            return np.sum(binary_volume)  # Rough    approximation
    
    def calculate_compactness(self, volume, surface_area):
        """Calculate compactness (sphericity) (existing method)"""
        if volume == 0 or surface_area == 0:
            return None
        return (36 * np.pi * volume**2) / (surface_area**3)
    
    # ==========================================================================
    # UNIFIED ANALYSIS METHODS
    # ==========================================================================
    
    def calculate_complete_morphology_with_lesions(self, seg1: np.ndarray, seg2: np.ndarray, 
                                                labels: List[int] = [1, 2, 3]) -> Dict:
        """
        Complete morphological analysis including lesion analysis

        Args:
            seg1: Baseline segmentation
            seg2: Follow-up segmentation
            labels: Labels to analyze

        Returns:
            Dictionary with morphological and lesion analysis results
        """

        # ✅ 1. Apply label remapping to both segmentations if specified
   


        results = {
            'morphology': {
                'volumes': {'seg1': {}, 'seg2': {}},
                'surface_areas': {'seg1': {}, 'seg2': {}},
                'compactness': {'seg1': {}, 'seg2': {}},
                'fractal_3d': {'seg1': {}, 'seg2': {}},
                'lacunarity_3d': {'seg1': {}, 'seg2': {}}
            },
            'lesion_analysis': {},
            'lesion_changes': {}
        }

        # ✅ 2. Traditional morphological & lesion analysis
        for label in labels:
            for seg_name, seg in [('seg1', seg1), ('seg2', seg2)]:
                mask = (seg == label).astype(bool)

                if np.any(mask):
                    # Volume
                    volume = int(np.sum(mask))
                    results['morphology']['volumes'][seg_name][f'label_{label}'] = volume

                    # Surface area
                    surface_area = self.calculate_surface_area(mask)
                    results['morphology']['surface_areas'][seg_name][f'label_{label}'] = surface_area

                    # Fractal dimension
                    tensor_mask = torch.tensor(mask).float().to(self.device)
                    fd_3d, _, _, _ = self.fractal_analyzer.calculate_fractal_dimension(
                        tensor_mask, region_name=f'label_{label}'
                    )

                    # Lacunarity
                    lacunarity_3d = self.calculate_robust_lacunarity(mask.flatten())

                    results['morphology']['fractal_3d'][seg_name][f'label_{label}'] = fd_3d
                    results['morphology']['lacunarity_3d'][seg_name][f'label_{label}'] = lacunarity_3d

                    # Compactness
                    compactness = self.calculate_compactness(volume, surface_area)
                    results['morphology']['compactness'][seg_name][f'label_{label}'] = compactness

                    # Lesion morphology
                    lesion_morph = self.calculate_lesion_morphology(mask)
                    results['lesion_analysis'][f'{seg_name}_label_{label}'] = lesion_morph

        # ✅ 3. Lesion change analysis
        results['lesion_changes'] = self.analyze_lesion_changes(seg1, seg2, labels)

        return results


# ==========================================================================
# USAGE EXAMPLE
# ==========================================================================
def compute_fractal_by_label(seg, device, fractal_analyzer, labels=[1, 2, 3], seg_name='seg1'):
    """Quick full-region FD calculator (no lesion separation)"""
    print(f"\n🧠 Fractal Dimensions for {seg_name}:")

    def to_scalar(x, name='value'):
        try:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                if x.size == 1:
                    return x.item()
                elif x.ndim == 1:
                    print(f"  ⚠️ {name} is 1D array with size {x.size}, taking first element")
                    return float(x[0])
                else:
                    raise ValueError(f"{name} has unexpected shape: {x.shape}")
            return float(x)
        except Exception as e:
            raise ValueError(f"Failed to convert {name} to scalar: {e}")

    for label in labels:
        mask = (seg == label)
        if np.any(mask):
            # Print bounding box and reduction info for this label
            nonzero_coords = np.array(mask.nonzero())
            if nonzero_coords.shape[1] == 0:
                bbox_shape = (0, 0, 0)
                reduction = 0.0
            else:
                bbox_shape = tuple(np.ptp(nonzero_coords, axis=1) + 1)
                reduction = np.prod(mask.shape) / np.prod(bbox_shape)
            print(
                f"label_{label}: Volume {mask.shape} -> {bbox_shape} "
                f"({reduction:.1f}x reduction, {np.count_nonzero(mask):,} voxels)"
            )
            try:
                tensor_mask = torch.tensor(mask).float().to(device)
                fd, r2, scale_range, _ = fractal_analyzer.calculate_fractal_dimension(
                    tensor_mask, region_name=f'label_{label}'
                )

                # 🩺 Robust scalar conversion
                fd = to_scalar(fd, name='FD')
                r2 = to_scalar(r2, name='R²')
                scale_range = to_scalar(scale_range, name='Scale range')

                print(f"  Label {label}: FD = {fd:.4f}, R² = {r2:.4f}, Scale range = {scale_range}")
            except Exception as e:
                print(f"  Label {label}: ❌ Error computing FD — {e}")
        else:
            print(f"  Label {label}: ⚠️ Empty region")



def example_enhanced_usage():
    """Example of using enhanced TumorMorphology with integrated lesion analysis"""
    
    # Create enhanced morphology analyzer
    label_remap = {1:2, 2:3, 3:1}
    morphology = TumorMorphology(use_gpu=True, label_remap=label_remap)
    
    
    # Load segmentations (example)
    baseline_seg = nib.load('/gpfs/data/oermannlab/users/schula12/Morphology/Lumiere/segmentations_swin/swin_Patient-001_week-000-2.nii.gz').get_fdata().astype(int)
    print('I am here')
    followup_seg = nib.load('/gpfs/data/oermannlab/users/schula12/Morphology/Lumiere/segmentations_swin/swin_Patient-001_week-044.nii.gz').get_fdata().astype(int)
    # Fractal-only analysis per label (NO lesion separation)
    if hasattr(morphology, 'label_remap') and morphology.label_remap:
        print(f"🔁 Applying label remapping: {morphology.label_remap}")
        baseline_seg = morphology.remap_labels(baseline_seg)
        followup_seg = morphology.remap_labels(followup_seg)
    compute_fractal_by_label(baseline_seg, morphology.device, morphology.fractal_analyzer, seg_name='Baseline')
    compute_fractal_by_label(followup_seg, morphology.device, morphology.fractal_analyzer, seg_name='Follow-up')
    # Complete analysis with integrated lesion analysis
    results = morphology.calculate_complete_morphology_with_lesions(baseline_seg, followup_seg)
    
    # Access different types of results
    lesion_changes = results['lesion_changes']
    morphology_features = results['morphology']
    per_lesion_analysis = results['lesion_analysis']

    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    print("\n🔍 Lesion Changes:")
    pp.pprint(lesion_changes)

    print("\n📐 Morphology Features:")
    pp.pprint(morphology_features)

    print("\n🧠 Per Lesion Analysis:")
    pp.pprint(per_lesion_analysis)

    
    print("""
    ✅ Enhanced TumorMorphology now includes:
    
    1. 🔬 Integrated Lesion Analysis:
       - Connected component analysis with GPU optimization
       - Lesion counting and size statistics
       - Lesion change tracking between timepoints
       - Per-lesion morphological features
    
    2. 🎯 Improved Existing Features:
       - Robust lacunarity calculation (handles extreme values)
       - Better error handling and numerical stability
       - Unified analysis interface
    
    3. 📊 Comprehensive Output:
       - Traditional morphological features
       - Lesion-level analysis
       - Change analysis between timepoints
       - Summary statistics and progression severity
    """)

if __name__ == "__main__":
    example_enhanced_usage()