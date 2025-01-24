import os
import subprocess
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import math

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_dataset_info(challenge_name: str):
    '''
    Returns the corresponding dataset name based on the provided challenge name.

    Parameters:
    challenge_name (str): Name of the challenge.

    Returns:
    str: Corresponding dataset name.
    
    Raises:
    Exception: If the challenge name is not compatible.
    '''

    if challenge_name=="BraTS-PED":
        dataset_name = "Task021_BraTS2024-PEDs"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans="nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    elif challenge_name=="BraTS-SSA":
        dataset_name = "Task022_BraTS2024-SSA"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_150epochs_StratifiedSplit"
        plans="nnUNetPlans_pretrained_SSA"
        configuration = "3d_fullres"
    elif challenge_name=="BraTS-GLI":
        dataset_name = "Task023_BraTS2024-GLI"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans="nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    elif challenge_name=="BraTS-MEN-RT":
        dataset_name = "Task024_BraTS2024-MEN-RT"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans="nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    elif challenge_name=="BraTS-MET":
        dataset_name = "Task026_BraTS2024-MET"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans="nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    else:
        raise Exception("Challenge name not compatible.")
    
    return dataset_name, trainer, plans, configuration

def set_env_paths(input_path, path):
    #raw_path = maybe_make_dir(path / 'nnUNet_raw')
    preprocessed = maybe_make_dir(Path('/tmp/') / 'nnUNet_preprocessed')
    #results =  maybe_make_dir(path / 'nnUNet_results')
    return f"nnUNet_raw_data_base='{input_path}' nnUNet_preprocessed='{preprocessed}' RESULTS_FOLDER='/tmp/'"

def run_infer_mednext(input_folder: str, output_folder: str,  challenge_name: str, name,  folds=[0,1,2,3,4], save_npz=True, ensemble=True)->list:
    """_summary_

    Args:
        input_folder (str): input folder
        output_folder (str): folder where output is to be stored
        challenge_name (str): task name for which inference is done
        name (str): file name
        folds (list, optional): describe which folds to run. Defaults to [0,1,2,3,4].
        save_npz (bool, optional): whether to save the probabilities. Defaults to True.

    Returns:
        list: a list of the npz files from each fold
    """
    
    # Check challenge name and get dataset name
    dataset_name, trainer, plans, configuration = get_dataset_info(challenge_name)
    env_set = set_env_paths(input_folder, output_folder)

    if ensemble:
        output_folder_fold = os.path.join(output_folder,"ens")
        print(f"Running nnUnet inference with all folds (ensemble)..")
        cmd = f"{env_set} mednextv1_predict -i '{input_folder}' -o '{output_folder_fold}' -t '{dataset_name}' -m '{configuration}' -tr '{trainer}' -p '{plans}'"
        if(save_npz):
            cmd+=" --save_npz"
        subprocess.run(cmd, shell=True)

        return [Path(os.path.join(output_folder_fold, name+'.npz'))], [Path(os.path.join(output_folder_fold, name+'.pkl'))]
    else:
        npz_path_list = []
        pkl_path_list = []
        def run_subprocess_mednext(fold):
            output_folder_fold = os.path.join(output_folder, f"fold_{fold}")
            print(f"Running nnU-Net inference for fold {fold}")
            cmd = f"{env_set} mednextv1_predict -i '{input_folder}' -o '{output_folder_fold}' -t '{dataset_name}' -m '{configuration}' -tr '{trainer}' -p '{plans}' -f '{fold}'"
            if(save_npz):
                cmd+=" --save_npz"
            subprocess.run(cmd, shell=True) 

        with ThreadPoolExecutor(max_workers=math.ceil(len(folds)/2)) as executor:
            executor.map(run_subprocess_mednext, folds)
        for fold in tqdm(folds):
            output_folder_fold = os.path.join(output_folder, f"fold_{fold}")
            npz_path_list.append(Path(os.path.join(output_folder_fold, name+'.npz')))
            pkl_path_list.append(Path(os.path.join(output_folder_fold, name+'.pkl')))

        return npz_path_list, pkl_path_list