#!/usr/bin/env python3
"""
SwinUNETR Segmentation Inference for LUMIERE Dataset - FIXED VERSION
Handles files that have .gz extension but are not actually compressed
"""

import torch
import os
import json
import numpy as np
import nibabel as nib
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset
from monai import transforms
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from functools import partial
from pathlib import Path
import logging
import shutil
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedLoadImaged(transforms.LoadImaged):
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Handle both single files and lists of files
                files = d[key] if isinstance(d[key], list) else [d[key]]
                fixed_files = []
                temp_files_to_cleanup = []
                
                for filepath in files:
                    if filepath.endswith('.gz'):
                        try:
                            # First try loading normally
                            nib.load(filepath)
                            fixed_files.append(filepath)
                        except Exception as e:
                            if "not a gzip file" in str(e):
                                # Create temporary file without .gz extension
                                temp_fd, temp_path = tempfile.mkstemp(suffix='.nii')
                                os.close(temp_fd)  # Close the file descriptor
                                
                                # Copy the file content
                                shutil.copy2(filepath, temp_path)
                                fixed_files.append(temp_path)
                                temp_files_to_cleanup.append(temp_path)
                                
                                logger.info(f"Fixed file extension issue for {os.path.basename(filepath)}")
                            else:
                                raise e
                    else:
                        fixed_files.append(filepath)
                
                # Update the data dictionary
                d[key] = fixed_files if isinstance(d[key], list) else fixed_files[0]
                
                # Store temp files for cleanup (in the data dict for later access)
                if temp_files_to_cleanup:
                    if '_temp_files' not in d:
                        d['_temp_files'] = []
                    d['_temp_files'].extend(temp_files_to_cleanup)
        
        # Call parent class
        result = super().__call__(d)
        
        # Clean up temporary files
        if '_temp_files' in result:
            for temp_file in result['_temp_files']:
                try:
                    os.remove(temp_file)
                except:
                    pass  # Ignore cleanup errors
            del result['_temp_files']
        
        return result

def setup_model(model_path, device):
    """Initialize and load SwinUNETR model"""
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def get_image_paths(lumiere_path, patient_id, week_id):
    """Get image file paths following nnUNet naming convention"""
    base_name = f"Patient-{patient_id}_week-{week_id}"
    
    # Based on your ls output, files are directly in subdirectories under images/
    # Each patient-week has its own directory
    image_dir = os.path.join(lumiere_path, "images", base_name)
    
    if not os.path.exists(image_dir):
        logger.warning(f"Image directory does not exist: {image_dir}")
        return [], ""
    
    # nnUNet format files within the patient directory
    image_paths = [
        os.path.join(image_dir, f"{base_name}_0003.nii.gz"),  # FLAIR
        os.path.join(image_dir, f"{base_name}_0001.nii.gz"),  # T1ce
        os.path.join(image_dir, f"{base_name}_0000.nii.gz"),  # T1
        os.path.join(image_dir, f"{base_name}_0002.nii.gz"),  # T2
    ]
    
    # Check if all image files exist
    missing_files = []
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            missing_files.append(f"_{i:04d}.nii.gz")
    
    if missing_files:
        logger.warning(f"Missing image files for {base_name}: {missing_files}")
        return [], ""
    
    # Check segmentation file
    label_path = os.path.join(lumiere_path, "segmentations", f"{base_name}.nii.gz")
    
    if not os.path.exists(label_path):
        logger.warning(f"Segmentation file does not exist: {label_path}")
        return [], ""
    
    logger.info(f"✓ All files found for {base_name}")
    return image_paths, label_path

def run_inference(model, image_paths, label_path, device):
    """Run SwinUNETR inference on a single case"""
    
    # Setup transforms with our fixed LoadImaged
    test_transform = transforms.Compose([
        FixedLoadImaged(keys=["image", "label"]),  # Use our custom loader
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    # Create test data
    test_files = [{
        "image": image_paths,
        "label": label_path,
    }]
    
    test_ds = Dataset(data=test_files, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    # Setup inference
    roi = [128, 128, 128]
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi,
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )
    
    with torch.no_grad():
        for batch_data in test_loader:
            image = batch_data["image"].to(device)
            
            # Run inference
            prob = torch.sigmoid(model_inferer(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            
            # Convert to BraTS label format
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2  # Peritumoral edema
            seg_out[seg[0] == 1] = 1  # Necrotic core
            seg_out[seg[2] == 1] = 3  # Enhancing tumor
            
            return seg_out, batch_data["label"][0].numpy()

def calculate_dice_score(pred, gt):
    """Calculate Dice score between prediction and ground truth"""
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Convert to tensor and add batch dimension
    pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
    gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0)
    
    # Convert to one-hot encoding
    pred_onehot = torch.zeros((1, 4, *pred.shape))
    gt_onehot = torch.zeros((1, 4, *gt.shape))
    
    for i in range(4):
        pred_onehot[0, i] = (pred_tensor[0, 0] == i).float()
        gt_onehot[0, i] = (gt_tensor[0, 0] == i).float()
    
    dice_score = dice_metric(pred_onehot, gt_onehot)
    return float(dice_score.mean())

def process_patients(lumiere_path, model_path, patients_json_path, registration_meta_path):
    """Main processing function"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading SwinUNETR model...")
    model = setup_model(model_path, device)
    
    # Load existing data
    with open(patients_json_path, 'r') as f:
        patients_data = json.load(f)
    
    with open(registration_meta_path, 'r') as f:
        registration_data = json.load(f)
    
    # Create output directories
    swin_seg_dir = os.path.join(lumiere_path, "segmentations_swin_1")
    os.makedirs(swin_seg_dir, exist_ok=True)
    
    # Track processed scans to avoid duplicates
    processed_scans = set()
    
    # Initialize new data structures
    patients_swin = {}
    registration_swin = {}
    
    total_cases = 0
    successful_cases = 0
    
    # Process all patients
    patient_items = list(patients_data.items())  # Process all patients
    
    for patient_key, patient_cases in patient_items:
        logger.info(f"Processing {patient_key}...")
        
        patients_swin[patient_key] = {}
        
        for case_key, case_data in patient_cases.items():
            total_cases += 1
            logger.info(f"  Processing {case_key}...")
            
            try:
                # Copy original case data
                new_case_data = case_data.copy()
                
                # Process baseline
                baseline_path = case_data["baseline"]
                patient_id = baseline_path.split("Patient-")[1].split("_")[0]
                week_id = baseline_path.split("_week-")[1]
                
                baseline_key = f"Patient-{patient_id}_week-{week_id}"
                
                if baseline_key not in processed_scans:
                    logger.info(f"    Processing baseline: {baseline_key}")
                    
                    image_paths, label_path = get_image_paths(lumiere_path, patient_id, week_id)
                    
                    # Check if all files exist
                    all_exist = (len(image_paths) > 0 and 
                               all(os.path.exists(p) for p in image_paths) and 
                               os.path.exists(label_path))
                    
                    if all_exist:
                        # Run inference
                        pred_seg, gt_seg = run_inference(model, image_paths, label_path, device)
                        
                        # Save prediction
                        swin_filename = f"swin_Patient-{patient_id}_week-{week_id}.nii.gz"
                        swin_path = os.path.join(swin_seg_dir, swin_filename)
                        
                        # Load original to get affine and header (handle the .gz issue)
                        try:
                            original_img = nib.load(label_path)
                        except:
                            # If label also has .gz issue, create temp file
                            temp_fd, temp_path = tempfile.mkstemp(suffix='.nii')
                            os.close(temp_fd)
                            shutil.copy2(label_path, temp_path)
                            original_img = nib.load(temp_path)
                            os.remove(temp_path)
                        
                        pred_img = nib.Nifti1Image(pred_seg, original_img.affine, original_img.header)
                        nib.save(pred_img, swin_path)
                        
                        # Calculate Dice score
                        dice_score = calculate_dice_score(pred_seg, gt_seg[0])  # gt_seg has channel dim
                        
                        # Update registration metadata
                        patient_reg_key = f"Patient-{patient_id}"
                        week_reg_key = f"week-{week_id}"
                        
                        if patient_reg_key not in registration_swin:
                            registration_swin[patient_reg_key] = {}
                        
                        if patient_reg_key in registration_data and week_reg_key in registration_data[patient_reg_key]:
                            # Copy original registration data
                            registration_swin[patient_reg_key][week_reg_key] = registration_data[patient_reg_key][week_reg_key].copy()
                            # Add SwinUNETR data
                            registration_swin[patient_reg_key][week_reg_key]["swin_file"] = swin_path
                            registration_swin[patient_reg_key][week_reg_key]["swin_dice"] = dice_score
                        
                        processed_scans.add(baseline_key)
                        logger.info(f"    ✓ Baseline processed. Dice: {dice_score:.4f}")
                    else:
                        logger.warning(f"    ✗ Missing files for baseline {baseline_key}")
                        continue
                
                # Add baseline SwinUNETR path
                new_case_data["baseline_seg_swin"] = f"./segmentations_swin/swin_Patient-{patient_id}_week-{week_id}.nii.gz"
                
                # Process followup
                followup_path = case_data["followup"]
                patient_id_fu = followup_path.split("Patient-")[1].split("_")[0]
                week_id_fu = followup_path.split("_week-")[1]
                
                followup_key = f"Patient-{patient_id_fu}_week-{week_id_fu}"
                
                if followup_key not in processed_scans:
                    logger.info(f"    Processing followup: {followup_key}")
                    
                    image_paths_fu, label_path_fu = get_image_paths(lumiere_path, patient_id_fu, week_id_fu)
                    
                    # Check if all files exist
                    all_exist_fu = (len(image_paths_fu) > 0 and 
                                  all(os.path.exists(p) for p in image_paths_fu) and 
                                  os.path.exists(label_path_fu))
                    
                    if all_exist_fu:
                        # Run inference
                        pred_seg_fu, gt_seg_fu = run_inference(model, image_paths_fu, label_path_fu, device)
                        
                        # Save prediction
                        swin_filename_fu = f"swin_Patient-{patient_id_fu}_week-{week_id_fu}.nii.gz"
                        swin_path_fu = os.path.join(swin_seg_dir, swin_filename_fu)
                        
                        # Load original to get affine and header
                        try:
                            original_img_fu = nib.load(label_path_fu)
                        except:
                            # If label also has .gz issue, create temp file
                            temp_fd, temp_path = tempfile.mkstemp(suffix='.nii')
                            os.close(temp_fd)
                            shutil.copy2(label_path_fu, temp_path)
                            original_img_fu = nib.load(temp_path)
                            os.remove(temp_path)
                        
                        pred_img_fu = nib.Nifti1Image(pred_seg_fu, original_img_fu.affine, original_img_fu.header)
                        nib.save(pred_img_fu, swin_path_fu)
                        
                        # Calculate Dice score
                        dice_score_fu = calculate_dice_score(pred_seg_fu, gt_seg_fu[0])
                        
                        # Update registration metadata
                        patient_reg_key_fu = f"Patient-{patient_id_fu}"
                        week_reg_key_fu = f"week-{week_id_fu}"
                        
                        if patient_reg_key_fu not in registration_swin:
                            registration_swin[patient_reg_key_fu] = {}
                        
                        if patient_reg_key_fu in registration_data and week_reg_key_fu in registration_data[patient_reg_key_fu]:
                            # Copy original registration data
                            registration_swin[patient_reg_key_fu][week_reg_key_fu] = registration_data[patient_reg_key_fu][week_reg_key_fu].copy()
                            # Add SwinUNETR data
                            registration_swin[patient_reg_key_fu][week_reg_key_fu]["swin_file"] = swin_path_fu
                            registration_swin[patient_reg_key_fu][week_reg_key_fu]["swin_dice"] = dice_score_fu
                        
                        processed_scans.add(followup_key)
                        logger.info(f"    ✓ Followup processed. Dice: {dice_score_fu:.4f}")
                    else:
                        logger.warning(f"    ✗ Missing files for followup {followup_key}")
                        continue
                
                # Add followup SwinUNETR path
                new_case_data["followup_seg_swin"] = f"./segmentations_swin/swin_Patient-{patient_id_fu}_week-{week_id_fu}.nii.gz"
                
                # Add to new data structure
                patients_swin[patient_key][case_key] = new_case_data
                successful_cases += 1
                
            except Exception as e:
                logger.error(f"    ✗ Error processing {patient_key}/{case_key}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save updated JSON files
    patients_swin_path = os.path.join(lumiere_path, "patients_swin_full.json")
    registration_swin_path = os.path.join(lumiere_path, "registration_meta_swin_full.json")
    
    with open(patients_swin_path, 'w') as f:
        json.dump(patients_swin, f, indent=4)
    
    with open(registration_swin_path, 'w') as f:
        json.dump(registration_swin, f, indent=4)
    
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"Total cases: {total_cases}")
    logger.info(f"Successful cases: {successful_cases}")
    logger.info(f"Unique scans processed: {len(processed_scans)}")
    logger.info(f"Output files:")
    logger.info(f"  - {patients_swin_path}")
    logger.info(f"  - {registration_swin_path}")
    logger.info(f"  - Segmentations in: {swin_seg_dir}")

if __name__ == "__main__":
    # Configuration
    LUMIERE_PATH = "/gpfs/data/oermannlab/users/schula12/Morphology/Lumiere"
    MODEL_PATH = "/gpfs/data/oermannlab/users/schula12/fold0_f48_ep300_4gpu_dice0_8854/model.pt"
    PATIENTS_JSON = os.path.join(LUMIERE_PATH, "patients.json")
    REGISTRATION_META = os.path.join(LUMIERE_PATH, "registration_meta.json")
    
    # Verify input files exist
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        exit(1)
    
    if not os.path.exists(PATIENTS_JSON):
        logger.error(f"Patients JSON not found: {PATIENTS_JSON}")
        exit(1)
    
    if not os.path.exists(REGISTRATION_META):
        logger.error(f"Registration metadata not found: {REGISTRATION_META}")
        exit(1)
    
    # Run processing
    process_patients(LUMIERE_PATH, MODEL_PATH, PATIENTS_JSON, REGISTRATION_META)