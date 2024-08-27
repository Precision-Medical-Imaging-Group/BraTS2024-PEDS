import tempfile
import os
import shutil
from pathlib import Path
from nnunet.install_model import install_model_from_zip
from ensembler.ped_weighted_ensemble import ped_ensemble
from nnunet.runner import run_infer_nnunet
from mednext.runner import run_infer_mednext
from swinunetr.runner import run_infer_swinunetr
from pp_cluster.infer import get_cluster, get_cluster_artifacts
from radiomics.feature_extraction_v2 import extract_all, save_json, load_json
from postproc.postprocess_cc import remove_small_component
from postproc.postprocess_lblredef import label_redefinition

TASK = 'BraTS-PED'
#read the weights_cv.json
json_path = Path('./ensembler/weights_cv.json')
ENSEMBLE_WEIGHTS = load_json(json_path)[TASK]
NAME_MAPPER = load_json(Path('./weights/name.json'))
CLUSTER_ARTIFACT = get_cluster_artifacts(TASK)
THRESHOLD_FILE_CC = Path('./postproc/cc_thresholds_cv.json')
THRESHOLD_FILE_LBLDEF = Path('./postproc/lblredef_thresholds_cv.json')

CONSTANTS = {
    'pyradiomics_paramfile':'./radiomics/params.yaml',
    'nnunet_model_path':'./weights/BraTS2024_PEDs_nnunetv2_model.zip',
    'mednext_model_path':'./weights/BraTS2024_PEDs_MedNeXt_model.zip',
    'swinunetr_model_path':'./weights/swinunetr_peds_trunc.zip',
}

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return Path(path)

def lbl_redefination(task, threshold_file, input_dir, df_radiomics, out_dir):
    # file from input_dir to out_dir
    for file in input_dir.iterdir():
        shutil.copy(file, out_dir)

def postprocess_single(input_dir, seg_dir, out_dir):
    case_name = input_dir.name
    # create dilated (or not) segmentation on the largest region
    seg_path = seg_dir / f"{case_name}.nii.gz"
    out_path = maybe_make_dir(out_dir) / f"{case_name}.nii.gz"

    
    # compute radiomics
    df_radiomics= extract_all(CONSTANTS['pyradiomics_paramfile'], input_dir.parent, case_name, seg_path, 1, region='wt', tmpp='/tmp/', seg_suffix='', sequences=['-t1n', '-t1c', '-t2w', '-t2f'])
    # find cluster
    cluster = get_cluster(df_radiomics, CLUSTER_ARTIFACT)[0]
    df_radiomics['cluster'] = int(cluster)
    
    # remove disconnected regions
    removed_cc_dir = maybe_make_dir(seg_dir / 'remove_cc')
    remove_small_component(TASK, THRESHOLD_FILE_CC, seg_dir, df_radiomics, removed_cc_dir)
    # TODO: label redefination
    label_redefinition(TASK, THRESHOLD_FILE_LBLDEF, removed_cc_dir,df_radiomics, out_dir)
    return out_path

def infer_single(input_path, out_dir):
    """do inference on a single folder

    Args:
        input_path (path): input folder, where the 4 nii.gz are stored
        out_dir (path): out folder, where the seg.nii.gz is to be stored
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        print(f'storing artifacts in tmp dir {temp_dir}')
        input_folder_raw = maybe_make_dir(temp_dir/ 'inp')
        name = input_path.name
        print(name)
        shutil.copytree(input_path, input_folder_raw / name)
        # only for the test
        #os.remove(input_folder_raw / name/ f'{name}-seg.nii.gz')
        for key, val in NAME_MAPPER.items():
            os.rename(input_folder_raw / name/ f'{name}{key}', input_folder_raw / name/ f'{name}{val}')
            one_image = input_folder_raw / name/ f'{name}{val}'
        
        nnunet_npz_path_list = run_infer_nnunet(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'nnunet'), TASK, name, ensemble=False)
        
        mednext_npz_path_list, mednext_pkl_path_list = run_infer_mednext(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'mednext'), TASK, name, ensemble=False)
        swinunetr_npz_path_list = run_infer_swinunetr(Path(input_path), maybe_make_dir(temp_dir/ 'swinunetr'), TASK) # run infer segresnet here
        # for testing
        ensemble_folder =  maybe_make_dir(input_folder_raw / 'ensemble')
        ensembled_pred_nii_path = ped_ensemble(swinunetr_npz_path_list, nnunet_npz_path_list, mednext_npz_path_list, mednext_pkl_path_list, ensemble_folder, one_image, weights=[ENSEMBLE_WEIGHTS['SwinUNETR'], ENSEMBLE_WEIGHTS['nnUNetv2'], ENSEMBLE_WEIGHTS['MedNeXt']])
        radiomics = postprocess_single(input_path, ensemble_folder, out_dir)


def batch_processor(input_folder, output_folder):
    for input_path in Path(input_folder).iterdir():
        infer_single(input_path, Path(output_folder))

def setup_model_weights():
    install_model_from_zip(CONSTANTS['nnunet_model_path'])
    install_model_from_zip(CONSTANTS['swinunetr_model_path'])
    install_model_from_zip(CONSTANTS['mednext_model_path'], mednext=True)
    
def batch_postprocess(input_folder, seg_folder, output_folder):
    l_radiomics = []
    l_path = [x for x in input_folder.iterdir() if x.is_dir()]
    for input_path in l_path:
        # infer_single(input_path, output_folder)
        radiomics = postprocess_single(input_path, seg_folder, output_folder)
        l_radiomics.append(radiomics)
    save_json(output_folder / 'val_radiomics.json', l_radiomics) 
    return len(list(Path(input_folder).iterdir()))


if __name__ == "__main__":
    setup_model_weights()
    batch_processor('./ins', './outs')