import pandas as pd
import numpy as np
import pathlib
from scipy.stats import kruskal, spearmanr


def encode_categorical_columns(df, columns):
    encoding_maps = {}
    for col in columns:
        clean_col_name = col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
        new_col_name = f"{clean_col_name}_numerical"

        unique_vals = df[col].dropna().unique()
        encoding_dict = {val: idx for idx, val in enumerate(unique_vals)}
        encoding_maps[col] = encoding_dict
        df[new_col_name] = df[col].map(encoding_dict)

    return df, encoding_maps


def decode_column(encoded_values, mapping):
    reverse_map = {v: k for k, v in mapping.items()}
    return [reverse_map[val] for val in encoded_values]


def check_existing_morph_features(df, verbose=True):
    import re

    morph_feature_keywords = {
        'volumes': ['_volume', 'volume_label_'],
        'surface_areas': ['surface_area'],
        'compactness': ['compactness'],
        'fractal_3d': ['fractal_3d'],
        'lacunarity_3d': ['lacunarity_3d'],
        'lesion_count': ['lesion_count'],
        'lesion_stats': ['mean_lesion_size', 'max_lesion_size', 'min_lesion_size'],
        'lesion_changes': ['new_lesions', 'lost_lesions', 'net_lesion_change'],
        'lesion_flags': ['new_lesion_detected', 'lesion_loss_detected'],
        'progression_severity': ['progression_severity'],
        'aggregate_lesion_stats': ['aggregate_stats'],
    }

    found = {}
    for category, patterns in morph_feature_keywords.items():
        found[category] = [
            col for col in df.columns
            if any(re.search(pattern, col, re.IGNORECASE) for pattern in patterns)
        ]

    if verbose:
        print("\U0001F4CA Morphological Features Present in DataFrame:\n")
        for category, columns in found.items():
            if columns:
                print(f"\U0001F539 {category} ({len(columns)}):")
                for col in sorted(columns):
                    print(f"   - {col}")
                print()

    return found


def get_available_morph_features(df, include_regions=True):
    import re
    pattern = re.compile(r"(fractal|volume|compactness|surface_area|lacunarity|lesion_count|mean_lesion_size|max_lesion_size|min_lesion_size)_(baseline|followup)_(whole|label_\d+)")
    feature_region_map = {}
    existing_deltas = set()

    for col in df.columns:
        if col.endswith('_change') or col.endswith('_volatility'):
            base = col.rsplit('_', 1)[0]  # remove _change or _volatility
            existing_deltas.add(base)
            continue

        match = pattern.fullmatch(col)
        if match:
            feature, _, region = match.groups()
            if feature not in feature_region_map:
                feature_region_map[feature] = set()
            feature_region_map[feature].add(region)

    return list(feature_region_map.keys()), feature_region_map, existing_deltas

def get_all_morph_feature_columns(df, include_deltas=True):
    """
    Returns all columns related to morphological features:
    - baseline/followup morph features (from get_available_morph_features)
    - optionally: change and volatility columns
    """
    features, region_map, _ = get_available_morph_features(df)
    all_cols = []

    # Add baseline/followup columns
    for feat in features:
        for region in region_map[feat]:
            for tp in ['baseline', 'followup']:
                col = f"{feat}_{tp}_{region}"
                if col in df.columns:
                    all_cols.append(col)

    if include_deltas:
        delta_cols = [col for col in df.columns if col.endswith("_change") or col.endswith("_volatility")]
        all_cols.extend(delta_cols)

    return sorted(all_cols)

def compute_morph_features(df, features_to_compute, per_region=True, labels=[1, 2, 3], verbose=False):
    from MorphFeatureClass import TumorMorphology
    import nibabel as nib
    from tqdm import tqdm

    morphology = TumorMorphology()
    all_rows = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = row['patient_id']
        case_id = row['case_id']
        key = (patient_id, case_id)
        if key not in all_rows:
            all_rows[key] = {}

        for timepoint, path_col in [('baseline', 'baseline_swin_path'), ('followup', 'followup_swin_path')]:
            seg_path = pathlib.Path(row[path_col])
            try:
                seg = nib.load(seg_path).get_fdata().astype(int)

                if "whole" in features_to_compute or not per_region:
                    mask = (seg > 0)
                    res = _compute_features_for_mask(mask, features_to_compute, morphology)
                    for feat_name, val in res.items():
                        colname = f"{feat_name}_{timepoint}_whole"
                        all_rows[key][colname] = val

                if per_region:
                    for label in labels:
                        mask = (seg == label)
                        res = _compute_features_for_mask(mask, features_to_compute, morphology)
                        for feat_name, val in res.items():
                            colname = f"{feat_name}_{timepoint}_label_{label}"
                            all_rows[key][colname] = val

            except Exception as e:
                print(f"❌ Error for {patient_id} {case_id} {timepoint}: {e}")

    formatted = []
    for (patient_id, case_id), features in all_rows.items():
        row = {'patient_id': patient_id, 'case_id': case_id}
        row.update(features)
        formatted.append(row)

    return pd.DataFrame(formatted)


def _compute_features_for_mask(mask, features_to_compute, morphology):
    res = {}
    if not np.any(mask):
        for feat in features_to_compute:
            res[feat] = None
        return res

    if "volume" in features_to_compute:
        res["volume"] = int(mask.sum())

    if "surface_area" in features_to_compute:
        sa = morphology.calculate_surface_area(mask)
        res["surface_area"] = sa
    else:
        sa = None

    if "compactness" in features_to_compute:
        if sa is None:
            sa = morphology.calculate_surface_area(mask)
        res["compactness"] = morphology.calculate_compactness(mask.sum(), sa)

    if "fractal" in features_to_compute:
        fractal, _ = morphology.box_counting_3d_gpu(mask)
        res["fractal"] = fractal

    if "lacunarity" in features_to_compute:
        _, lac = morphology.box_counting_3d_gpu(mask)
        res["lacunarity"] = lac

    if "lesion" in features_to_compute:
        lesion_stats = morphology.analyze_connected_components_gpu(mask)
        res["lesion_count"] = lesion_stats["lesion_count_filtered"]
        res["mean_lesion_size"] = lesion_stats["mean_lesion_size"]
        res["max_lesion_size"] = lesion_stats["max_lesion_size"]

    return res


def add_volume_change_features(df, features, baseline_prefix="baseline_swin_", followup_prefix="followup_swin_"):
    for feat in features:
        # Support feature names that already include region (like "fractal_whole")
        base_col = f"{feat}_baseline"
        follow_col = f"{feat}_followup"

        # If that fails, fall back to prefix mode
        if base_col not in df.columns or follow_col not in df.columns:
            base_col = baseline_prefix + feat
            follow_col = followup_prefix + feat

        change_col = f"{feat}_change"
        volatility_col = f"{feat}_volatility"
        print(f"For feature '{feat}', the following columns in {base_col} and {follow_col} are being created: '{change_col}' (difference between follow-up and baseline) and '{volatility_col}' (change per week, i.e., change divided by time difference in weeks).")

        df[change_col] = df[follow_col] - df[base_col]
        df[volatility_col] = df[change_col] / df["time_difference_weeks"].replace(0, np.nan)

    df = df.dropna(subset=[f"{feat}_volatility" for feat in features])

    return df


def statistical_analysis_of_changes(df, feature_change_col, response_col="response"):
    groups = [df[df[response_col] == r][feature_change_col] for r in sorted(df[response_col].unique())]
    stat, p = kruskal(*groups)
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    eta_squared = (stat - k + 1) / (n_total - k)

    print(f"\U0001F4CA Kruskal-Wallis test for {feature_change_col}:")
    print(f"  p-value = {p:.5f}")
    print(f"  Effect size η2 ≈ {eta_squared:.3f}")

    corr, p_corr = spearmanr(df[feature_change_col], df[response_col])
    print(f"\U0001F4C8 Spearman correlation:")
    print(f"  ρ = {corr:.3f}, p = {p_corr:.5f}")

def compute_morph_deltas(df, feature_region_map, existing_deltas=None):
    """
    Computes change and volatility for features organized by region.

    Parameters:
    - df: pandas DataFrame
    - feature_region_map: dict mapping feature names to region sets
        e.g., {'fractal': {'whole', 'label_1'}, 'surface_area': {'whole'}}
    - existing_deltas: optional set of already computed base names to skip

    Returns:
    - df: with added *_change and *_volatility columns
    """
    for feat, regions in feature_region_map.items():
        for region in regions:
            base = f"{feat}_{region}"
            base_col = f"{feat}_baseline_{region}"
            follow_col = f"{feat}_followup_{region}"
            change_col = f"{base}_change"
            volatility_col = f"{base}_volatility"

            if existing_deltas and base in existing_deltas:
                continue
            if base_col not in df.columns or follow_col not in df.columns:
                print(f"⚠️ Skipping {base}: missing columns.")
                continue

            df[change_col] = df[follow_col] - df[base_col]
            df[volatility_col] = df[change_col] / df["time_difference_weeks"].replace(0, np.nan)

    return df
