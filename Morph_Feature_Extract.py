import numpy as np
import nibabel as nib
from MorphFeatureClass import TumorMorphology

def extract_morph_features(seg1, seg2, features=["fractal_3d"], level="per_region", labels=[1, 2, 3]):
    morph = TumorMorphology(use_gpu=True)
    results = morph.calculate_complete_morphology_with_lesions(seg1, seg2, labels)

    feature_dict = {}

    # Flatten per-region level features
    if level == "per_region":
        for seg_key in ["seg1", "seg2"]:
            for label in labels:
                label_key = f"label_{label}"
                for feat in features:
                    val = results["morphology"].get(feat, {}).get(seg_key, {}).get(label_key, None)
                    feature_dict[f"{seg_key}_{label_key}_{feat}"] = val

    # Flatten whole tumor (labels merged)
    elif level == "whole":
        full_mask1 = np.isin(seg1, labels)
        full_mask2 = np.isin(seg2, labels)

        # Calculate on combined mask
        for seg_name, mask in [("seg1", full_mask1), ("seg2", full_mask2)]:
            if np.any(mask):
                if "fractal_3d" in features or "lacunarity_3d" in features:
                    fd, lac = morph.box_counting_3d_gpu(mask)
                    if "fractal_3d" in features:
                        feature_dict[f"{seg_name}_whole_fractal_3d"] = fd
                    if "lacunarity_3d" in features:
                        feature_dict[f"{seg_name}_whole_lacunarity_3d"] = lac

                if "volume" in features:
                    feature_dict[f"{seg_name}_whole_volume"] = int(np.sum(mask))

                if "surface_area" in features:
                    sa = morph.calculate_surface_area(mask)
                    feature_dict[f"{seg_name}_whole_surface_area"] = sa

                if "compactness" in features:
                    volume = np.sum(mask)
                    sa = morph.calculate_surface_area(mask)
                    comp = morph.calculate_compactness(volume, sa)
                    feature_dict[f"{seg_name}_whole_compactness"] = comp

    # Flatten per-lesion aggregates
    elif level == "per_lesion":
        for seg_key in ["seg1", "seg2"]:
            for label in labels:
                key = f"{seg_key}_label_{label}"
                lesion_stats = results["lesion_analysis"].get(key, {})
                aggr = lesion_stats.get("aggregate_stats", {})
                for feat in features:
                    val = aggr.get(f"mean_{feat}", None)
                    feature_dict[f"{key}_mean_{feat}"] = val

                # Count
                if "lesion_count" in features:
                    feature_dict[f"{key}_lesion_count"] = lesion_stats.get("lesion_count", 0)

    else:
        raise ValueError(f"Invalid level '{level}' — must be one of: 'per_region', 'whole', 'per_lesion'.")

    return feature_dict


if __name__ == "__main__":
    baseline_seg = nib.load('/gpfs/data/oermannlab/users/schula12/Morphology/Lumiere/segmentations_swin/swin_Patient-001_week-000-2.nii.gz').get_fdata().astype(int)
    followup_seg = nib.load('/gpfs/data/oermannlab/users/schula12/Morphology/Lumiere/segmentations_swin/swin_Patient-001_week-044.nii.gz').get_fdata().astype(int)

    features = extract_morph_features(
        baseline_seg,
        followup_seg,
        features=["fractal_3d", "volume", "lesion_count"],
        level="per_lesion",  # or "per_region", or "whole"
        labels=[1, 2]  # e.g. enhancing and necrotic only
    )

    import pprint
    pprint.pprint(features)
