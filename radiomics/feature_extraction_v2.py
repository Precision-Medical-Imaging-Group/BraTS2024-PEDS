from pathlib import Path
from typing import Any, List, Tuple, Union, Optional
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import cc3d
from scipy.ndimage import binary_dilation, generate_binary_structure
import shutil
import warnings


def load_json(path: Path) -> Any:
    """
    Loads a JSON file from the specified path.

    Args:
        path: A Path representing the file path.

    Returns:
        The data loaded from the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """
    Saves data to a JSON file at the specified path.

    Args:
        path: A Path representing the file path.
        data: The data to be serialized and saved.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_jsonl(path: Path) -> List[Any]:
    """
    Loads a JSONL file (JSON lines) from the specified path.

    Args:
        path: A Path representing the file path.

    Returns:
        A list of data loaded from the JSONL file.
    """
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, data: List[Any]) -> None:
    """
    Saves data to a JSONL file at the specified path.

    Args:
        path: A Path representing the file path.
        data: A list of data to be serialized and saved.
    """
    with open(path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')


def maybe_make_dir(path: str) -> Path:
    """
    Creates a directory at the specified path if it does not exist.

    Args:
        path: A string representing the directory path.

    Returns:
        A Path object for the created or existing directory.
    """
    dir = Path(path)
    if not dir.exists():
        dir.mkdir(parents=True)
    return dir


def maybe_remove_dir(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.exists() and path_obj.is_dir():
        try:
            shutil.rmtree(path_obj)
            print(f"Directory {path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {path}: {e}")
    return path_obj
    

def extract_feature(paramfile, data_path, seg_path, pid, seq, region='wt', tmpp='.'):
    img_path = data_path / f"{pid}{seq}.nii.gz"
    tmp_path = maybe_make_dir(tmpp) / f"{pid}{seq}.json"
    cmd = f"pyradiomics {str(img_path)} {str(seg_path)} -o {str(tmp_path)} -f json --param {str(paramfile)}"
    os.system(cmd)
    # load features
    new_dict = {'StudyID': pid}
    dt = load_json(tmp_path)
    tmp_path.unlink()
    # rename features
    old_dict = dt[0]
    for k in old_dict.keys():
        if k.startswith('original_shape_'):
            new_k = k.replace('original_shape_', f'{region}_shape_', 1)
            new_dict[new_k] = old_dict[k]
        elif k.startswith('original_'):
            new_k = k.replace('original_', f'{region}{seq.replace("-","_")}_', 1)
            new_dict[new_k] = old_dict[k]

    return new_dict


def extract_case(paramfile, data_path, seg_path, pid, region='wt', tmpp='.', sequences=['-t1n', '-t1c', '-t2w', '-t2f']):
    # sequences = ['t1n', 't1c', 't2w', 't2f']
    new_dict = {}
    for i, seq in enumerate(sequences):
        feature = extract_feature(paramfile, data_path, seg_path, pid, seq, region, tmpp)
        # if seq in ['-t1n','_t1n'] or len(sequences)==1:
        if i == 0:
            for k in feature.keys():
                new_dict[k] = feature[k]
        else:
            for k in feature.keys():
                if not (k.startswith('StudyID') or k.startswith(f'{region}_shape')):
                    new_dict[k] = feature[k]
    return new_dict


def create_dilation(seg_path, out_path, dilation_factor=3, region='wt'):
    img_obj = nib.load(seg_path)
    img_data = img_obj.get_fdata()
    if region == 'wt':
        binary_seg = np.where(img_data > 0, 1, 0)
    else:
        warnings.warn(f"Invalid region: {region} Compute whole tumor instead.")
        binary_seg = np.where(img_data > 0, 1, 0)
    # dilation_struct = generate_binary_structure(3, 1)
    # dilated_seg = binary_dilation(binary_seg, structure=dilation_struct, iterations=dilation_factor)

    labels_out, n = cc3d.connected_components(binary_seg, connectivity=26, return_N=True)
    vol_max = 0
    label_max = 0
    # vol_s = []
    for i in range(n):
        tmp = np.where(labels_out == i+1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol > vol_max:
            vol_max = vol
            label_max = i+1
        # vol_s.append(vol)

    dilated_seg = np.where(labels_out == label_max, 1, 0)
    seg_obj = nib.Nifti1Image(dilated_seg.astype(np.int8), img_obj.affine)
    nib.save(seg_obj, out_path)


def extract_all(paramfile, data_path, case_, seg_path, dilation_factor=3, region='wt', tmpp='.', seg_suffix='-seg', sequences=['-t1n', '-t1c', '-t2w', '-t2f']):
    cases = [data_path / case_]
    
    features = []
    t0 = time.time()
    for i, case in enumerate(cases):
        dilated_seg_path = maybe_make_dir(tmpp) / f"{case.name}_{region}_dilated.nii.gz"
        #seg_path = data_path / case / f"{case}{seg_suffix}.nii.gz"
        create_dilation(seg_path, dilated_seg_path, dilation_factor, region)

        features.append(extract_case(paramfile, data_path / case, dilated_seg_path, case.name, region, tmpp, sequences))
        dilated_seg_path.unlink()

        t1 = (time.time() - t0) / 60.0
        print(f"{i+1:04d} {case} extraction time: {t1:.1f} min")
    #save_json(output_path, features)
    return pd.DataFrame(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-d", "--dilation", type=int)
    parser.add_argument("-r", "--region", type=str)
    parser.add_argument("-p", "--param", type=str)
    parser.add_argument("-t", "--tmpp", default='./tmp', type=str)
    parser.add_argument("-s", "--seg-suffix", default='-seg', type=str)
    parser.add_argument("-seq", "--sequences", nargs='+', default=['-t1n', '-t1c', '-t2w', '-t2f'])
    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    
    maybe_make_dir(args.tmpp)

    extract_all(
        args.param, 
        Path(args.input_dir),
        args.output, 
        dilation_factor=args.dilation, 
        region=args.region,
        tmpp=args.tmpp,
        seg_suffix=args.seg_suffix,
        sequences=args.sequences)
    
    maybe_remove_dir(args.tmpp)
