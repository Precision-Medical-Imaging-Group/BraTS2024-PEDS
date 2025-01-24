from pathlib import Path
import numpy as np
import nibabel as nib
import os

from concurrent.futures import ThreadPoolExecutor

def convert_npz_mednext(npz_path, pkl_path, save_nifti=False, nifti_dir=Path('./tmp_prob_nifti'), suffix='mednext'):
    npz = np.load(npz_path, allow_pickle=True)
    pkl = np.load(pkl_path, allow_pickle=True)
    prob = npz[npz.files[0]].astype(np.float32)
    bbox = pkl['crop_bbox']
    shape_original_before_cropping = pkl['original_size_of_raw_data']
    out_list = []
    for i in range(0, prob.shape[0]):
        if bbox is not None:    
            seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.float16)
            for c in range(3):
                bbox[c][1] = np.min((bbox[c][0] + prob[i].shape[c], shape_original_before_cropping[c]))
            seg_old_size[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = prob[i]
        else:
            seg_old_size = prob[i]
        out = np.swapaxes(seg_old_size, 0, 2)
        out_list.append(out)
        if save_nifti:
            if not nifti_dir.exists():
                nifti_dir.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(out.astype(np.float32), affine=np.eye(4)), 
                nifti_dir / f"{npz_path.name.split('.npz')[0]}_{i}_{suffix}.nii.gz"
            )
    
    return np.stack(out_list, axis=0).astype(np.float32)

def convert_npz_nnunet(npz_path, save_nifti=False, nifti_dir=Path('./tmp_prob_nifti'), suffix='nnunet'):
    npz = np.load(npz_path, allow_pickle=True)
    prob = npz[npz.files[0]].astype(np.float32)
    out_list = []
    for i in range(0, prob.shape[0]):
        out = np.swapaxes(prob[i], 0, 2)
        out_list.append(out)
        if save_nifti:
            if not nifti_dir.exists():
                nifti_dir.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(out.astype(np.float32), affine=np.eye(4)), 
                nifti_dir / f"{npz_path.name.split('.npz')[0]}_{i}_{suffix}.nii.gz"
            )
    
    return np.stack(out_list, axis=0).astype(np.float32)

def convert_npz_swinunetr(npz_path, save_nifti=False, nifti_dir=Path('./tmp_prob_nifti'), suffix='swinunetr'):
    npz = np.load(npz_path, allow_pickle=True)
    prob = npz[npz.files[0]].astype(np.float32)
    for i in range(0, prob.shape[0]):
        if save_nifti:
            if not nifti_dir.exists():
                nifti_dir.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(prob[i].astype(np.float32), affine=np.eye(4)), 
                nifti_dir / f"{npz_path.name.split('.npz')[0]}_{i}_{suffix}.nii.gz"
            )
    
    return prob.astype(np.float32)

def ped_ensemble(
    swinunetr_npz_path_list, 
    nnunet_npz_path_list,
    mednext_npz_path_list,
    mednext_pkl_path_list,
    ensembled_path, 
    input_img,
    weights=[1.0, 1.0, 1.0]
):
    if not ensembled_path.exists():
        ensembled_path.mkdir(parents=True)
        
    case = nnunet_npz_path_list[0].name.split('.npz')[0]
    print(f"Ensemble {case}")
        
    if os.path.exists(ensembled_path / f"{case}.nii.gz"):
        return ensembled_path / f"{case}.nii.gz"

    # Helper functions to compute probabilities in parallel
    def compute_swinunetr_probs():
        prob = convert_npz_swinunetr(swinunetr_npz_path_list[0])
        for i in range(1, len(swinunetr_npz_path_list)):
            prob += convert_npz_swinunetr(swinunetr_npz_path_list[i])
        return prob / len(swinunetr_npz_path_list)

    def compute_nnunet_probs():
        prob = convert_npz_nnunet(nnunet_npz_path_list[0])
        for i in range(1, len(nnunet_npz_path_list)):
            prob += convert_npz_nnunet(nnunet_npz_path_list[i])
        return prob / len(nnunet_npz_path_list)

    def compute_mednext_probs():
        prob = convert_npz_mednext(mednext_npz_path_list[0], mednext_pkl_path_list[0])
        for i in range(1, len(mednext_npz_path_list)):
            prob += convert_npz_mednext(mednext_npz_path_list[i], mednext_pkl_path_list[i])
        return prob / len(mednext_npz_path_list)

    # Use ThreadPoolExecutor to parallelize the probability computations
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_swinunetr = executor.submit(compute_swinunetr_probs)
        future_nnunet = executor.submit(compute_nnunet_probs)
        future_mednext = executor.submit(compute_mednext_probs)

        # Wait for results
        prob_swinunetr = future_swinunetr.result()
        print(f"Probabilities SwinUNETR: {prob_swinunetr.shape}")

        prob_nnunet = future_nnunet.result()
        print(f"Probabilities nnUNet: {prob_nnunet.shape}")

        prob_mednext = future_mednext.result()
        print(f"Probabilities MedNeXt: {prob_mednext.shape}")
    
    prob = weights[0] * prob_swinunetr + weights[1] * prob_nnunet + weights[2] * prob_mednext
    prob /= sum(weights)
    
    seg = np.argmax(prob, axis=0)
    print(f"Seg: {seg.shape}")

    # save seg
    img = nib.load(input_img)
    nib.save(nib.Nifti1Image(seg.astype(np.int8), img.affine), ensembled_path / f"{case}.nii.gz")
    return ensembled_path / f"{case}.nii.gz"

def batch_ped_ensemble(
    swinunetr_pred_dirs, 
    nnunet_pred_dirs,
    mednext_pred_dirs,
    input_img_dir, 
    ensembled_dir, 
    weights=[1.0, 1.0, 1.0],
    cv=False
):
    if not ensembled_dir.exists():
        ensembled_dir.mkdir(parents=True)
        
    if cv:
        
        # Get files inside each of the nnunet_pred_dirs items
        cases = [[case_path.name[:-7] for case_path in pred_dir.iterdir() if str(case_path).endswith(".nii.gz")] for pred_dir in nnunet_pred_dirs]
        
        for f in range(len(cases)):
            for case in cases[f]:
                
                # swinunetr_npz_path_list = [swinunetr_pred_dirs[f] / f"{case}-t1n.npz"]
                swinunetr_npz_path_list = [swinunetr_pred_dirs[f] / f"{case}.npz"]
                nnunet_npz_path_list = [nnunet_pred_dirs[f] / f"{case}.npz"]
                mednext_npz_path_list = [mednext_pred_dirs[f] / f"{case}.npz"]
                mednext_pkl_path_list = [mednext_pred_dirs[f] / f"{case}.pkl"]

                saved_path = ped_ensemble(
                    swinunetr_npz_path_list, 
                    nnunet_npz_path_list, 
                    mednext_npz_path_list, 
                    mednext_pkl_path_list, 
                    ensembled_dir, 
                    input_img_dir / case / f"{case}-t1n.nii.gz", 
                    weights=weights
                )
                print(f"Saved {saved_path}")
        
    else:
    
        cases = [case_path.name for case_path in input_img_dir.iterdir() if case_path.is_dir()]

        for case in cases:
            swinunetr_npz_path_list = [pred / f"{case}-t1n.npz" for pred in swinunetr_pred_dirs]
            nnunet_npz_path_list = [pred / f"{case}.npz" for pred in nnunet_pred_dirs]
            mednext_npz_path_list = [pred / f"{case}.npz" for pred in mednext_pred_dirs]
            mednext_pkl_path_list = [pred / f"{case}.pkl" for pred in mednext_pred_dirs]

            saved_path = ped_ensemble(
                swinunetr_npz_path_list, 
                nnunet_npz_path_list, 
                mednext_npz_path_list, 
                mednext_pkl_path_list, 
                ensembled_dir, 
                input_img_dir / case / f"{case}-t1n.nii.gz", 
                weights=weights
            )
            print(f"Saved {saved_path}")

# # usage
# root = Path('C:/Users/zhifa/Develop')
# infer_dir = root / 'BraTS2024' / 'output'
# task = 'ped_stratified'
# val_dir = root / 'BraTS2024-Data' / 'brats2024_ped_val'

# dest_dir = root / 'BraTS2024' / 'output' / task 
# swinunetr_pred_dirs = [dest_dir / f'swinunetr_e650_f{i}_label' for i in range(5)]
# nnunet_pred_dirs = [dest_dir / f'nnunet_e200_f{i}_label' for i in range(5)]
# mednext_pred_dirs = [dest_dir / f'mednext_e200_f{i}_label' for i in range(5)]
# input_img_dir = val_dir
# ensembled_dir = root / 'BraTS2024' / 'submission' / task / 'ped_weighted_ensemble_label'
# weights = [0.330911177, 0.330839468, 0.338249355]

# batch_ped_ensemble(
#     swinunetr_pred_dirs, 
#     nnunet_pred_dirs, 
#     mednext_pred_dirs, 
#     input_img_dir, 
#     ensembled_dir, 
#     weights=weights
# )

def main_cv():
    
    swinunetr_pred_path = Path("/home/v363/v363397/media/output_cv/ped_stratified")
    swinunetr_pred_dirs = [swinunetr_pred_path / f'swinunetr_e650_f{i}_b1p4' for i in [0,1,2]]+[swinunetr_pred_path / f'swinunetr_e1000_f{i}_b1p4' for i in [3,4]]

    nnunet_pred_path = Path("/home/v363/v363397/media/nnUNet_results/Dataset021_BraTS2024-PED/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres")
    nnunet_pred_dirs = [nnunet_pred_path / f'fold_{i}' / 'validation' for i in range(5)]
    
    mednext_pred_path = Path("/home/v363/v363397/media/nnUNet_trained_models/nnUNet/3d_fullres/Task021_BraTS2024-PEDs/nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit__nnUNetPlansv2.1_trgSp_1x1x1")
    mednext_pred_dirs = [mednext_pred_path / f'fold_{i}' / 'validation_raw' for i in range(5)]
    
    input_img_dir = Path("/home/v363/v363397/stay/brats2024/data/MICCAI-BraTS2024-PED/BraTS-PEDs2024_Training")
    ensembled_dir = Path("/home/v363/v363397/stay/brats2024/Task099a_postprocessed_cv/PEDs/ensembled_preds_cv")
    
    weights = [0.330911177, 0.330839468, 0.338249355]

    batch_ped_ensemble(
        swinunetr_pred_dirs, 
        nnunet_pred_dirs, 
        mednext_pred_dirs, 
        input_img_dir, 
        ensembled_dir, 
        weights=weights,
        cv=True
    )
    
def main_val():
    
    swinunetr_pred_path = Path("/home/v363/v363397/media/output_val/ped_stratified")
    swinunetr_pred_dirs = [swinunetr_pred_path / f'swinunetr_e650_f{i}_b1p4' for i in [0,1,2]]+[swinunetr_pred_path / f'swinunetr_e1000_f{i}_b1p4' for i in [3,4]]  # Paths not updated
    
    nnunet_pred_path = Path("/home/v363/v363397/stay/brats2024/Task012a_BraTS2024-PED_nnUNet_SS/6_predict/predicted/folds_independent")
    nnunet_pred_dirs = [nnunet_pred_path / f'fold_{i}' / "nnUNetTrainer_200epochs/3d_fullres" for i in range(5)]
    
    mednext_pred_path = Path("/home/v363/v363397/stay/brats2024/Task012b_BraTS2024-PED_MedNeXt_SS/6_predict/predicted/folds_independent")
    mednext_pred_dirs = [mednext_pred_path / f'fold_{i}' / "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit/3d_fullres" for i in range(5)]
    
    input_img_dir = Path("/home/v363/v363397/stay/brats2024/data/MICCAI-BraTS2024-PED/BraTS_Validation_Data_backup")
    ensembled_dir = Path("/home/v363/v363397/stay/brats2024/Task099b_postprocessed_val/PEDs/ensembled_preds_val")
    
    weights = [0.330911177, 0.330839468, 0.338249355]

    batch_ped_ensemble(
        swinunetr_pred_dirs, 
        nnunet_pred_dirs, 
        mednext_pred_dirs, 
        input_img_dir, 
        ensembled_dir, 
        weights=weights
    )
    
# usage
if __name__ == '__main__':
    # main_cv()
    main_val()