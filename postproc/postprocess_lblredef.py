import pandas as pd
import glob
import os
import nibabel as nib
import numpy as np
import json
import argparse

LABEL_MAPPING_FACTORY = {
    "BraTS-PED": {
        1: "ET",
        2: "NET",
        3: "CC",
        4: "ED"
    },
    "BraTS-SSA": {
        1: "NCR",
        2: "ED",
        3: "ET"
    },
    "BraTS-MEN-RT": {
        1: "GTV"
    },
    "BraTS-MET": {
        1: "NETC",
        2: "SNFH",
        3: "ET"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='BraTS2024 CC Postprocessing.')
    parser.add_argument('--challenge_name', type=str, help='The name of the challenge (e.g., BraTS-PED, BraTS-MET)')
    parser.add_argument('--input_folder_pred', type=str, help='The input folder containing the predictions')
    parser.add_argument('--output_folder_pp_cc', type=str, help='The output folder to save the postprocessed predictions')
    parser.add_argument('--thresholds_file', type=str, help='The JSON file containing the thresholds')
    parser.add_argument('--clusters_file', type=str, help='The JSON file containing the clusters')

    return parser.parse_args()

def get_ratio_labels_wt(seg, labels=[1,2,3,4]):
    selected_voxels=0
    for l in labels:
        selected_voxels += np.sum(seg==l)
    wt_voxels = np.sum(seg!=0)  # everything but background
    if(wt_voxels==0):
        return 1
    
    return selected_voxels/wt_voxels

def postprocess_lblredef(img, thresholds_dict, label_mapping):
    # Make a copy of pred_orig
    pred = img.copy()
    # Iterate over each of the labels
    for label_name, th_dict in thresholds_dict.items():
        label_number = int(label_name.split("_")[-1])
        label_name = label_mapping[label_number]
        th = th_dict["th"]
        redefine_to = th_dict["redefine_to"]
        print(f"Label {label_name} - value {label_number} - Redefinition to {redefine_to} - Applying threshold {th}")
        # Get ratio with respect to the whole tumor
        ratio = get_ratio_labels_wt(pred, labels=[label_number,])
        # Conditioned redefinition
        if ratio < th:
            pred = np.where(pred == label_number, redefine_to, pred)
    # Return the postprocessed image
    return pred

def postprocess_batch(input_files, output_folder, thresholds_dict, label_mapping):
    # for each file, apply the postprocessing
    for f in input_files:
        print(f"Processing file {f}")
        save_path = os.path.join(output_folder, os.path.basename(f))
        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping.")
            continue
        # read the file
        pred_orig = nib.load(f).get_fdata()
        # Apply postprocessing
        pred = postprocess_lblredef(pred_orig, thresholds_dict, label_mapping)
        # Save the new prediction on a temp dir
        print(save_path)
        # Save the prediction
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(nib.Nifti1Image(pred, nib.load(f).affine), save_path)
        
def get_thresholds_task(challenge_name, input_file):
    with open(input_file, 'r') as f:
        thresholds = json.load(f)
    if challenge_name not in thresholds:
        raise ValueError(f"Challenge {challenge_name} not found in the thresholds JSON file.")
    return thresholds[challenge_name]

def get_thresholds_cluster(thresholds, cluster_name):
    if cluster_name not in thresholds:
        raise ValueError(f"Cluster {cluster_name} not found in the thresholds JSON file.")
    return thresholds[cluster_name]

def get_files(input_folder_pred):
    files_pred = sorted(glob.glob(os.path.join(input_folder_pred, "*.nii.gz")))
    print(f"Found {len(files_pred)} files to be processed.")
    return files_pred

def get_cluster_files(cluster_assignment, c, files_pred):
    cluster_ids = [e["StudyID"] for e in cluster_assignment if e["cluster"] == c]
    cluster_files_pred = [f for f in files_pred if os.path.basename(f).replace(".nii.gz","") in cluster_ids]
    print(f"Cluster {c} contains {len(cluster_files_pred)} files.")
    return cluster_files_pred

def read_cluster_assignment(clusters_json):
    with open(clusters_json, "r") as f:
        cluster_assignment = json.load(f)
    cluster_assignment = [{key: value for key, value in e.items() if key in ["StudyID", "cluster"]} for e in cluster_assignment]
    cluster_array = np.unique([e["cluster"] for e in cluster_assignment])
    print(f"Found {len(cluster_array)} clusters: {sorted(cluster_array)}.")
    return cluster_assignment, sorted(cluster_array)
def read_cluster_assignment_df(clusterdf):
    cluster_assignment = clusterdf.to_dict(orient="records")
    cluster_assignment = [{key: value for key, value in e.items() if key in ["StudyID", "cluster"]} for e in cluster_assignment]
    cluster_array = np.unique([e["cluster"] for e in cluster_assignment])
    print(f"Found {len(cluster_array)} clusters: {sorted(cluster_array)}.")
    return cluster_assignment, sorted(cluster_array)

def label_redefinition(challenge_name, thresholds_file, input_folder_pred,clustersdf, output_folder_pp_cc):
    label_mapping = LABEL_MAPPING_FACTORY[challenge_name]
    thresholds = get_thresholds_task(challenge_name, thresholds_file)
    files_pred = get_files(input_folder_pred)
    cluster_assignment, cluster_array = read_cluster_assignment_df(clustersdf)
    for cluster in cluster_array:
        cluster_files_pred = get_cluster_files(cluster_assignment, cluster, files_pred)
        thresholds_cluster = get_thresholds_cluster(thresholds, f"cluster_{cluster}")
        postprocess_batch(cluster_files_pred, output_folder_pp_cc, thresholds_cluster, label_mapping)

    return output_folder_pp_cc


def main():
    args = parse_args()
    args.label_mapping = LABEL_MAPPING_FACTORY[args.challenge_name]
    
    thresholds = get_thresholds_task(args.challenge_name, args.thresholds_file)
    files_pred = get_files(args.input_folder_pred)
    cluster_assignment, cluster_array = read_cluster_assignment(args.clusters_file)

    for cluster in cluster_array:
        cluster_files_pred = get_cluster_files(cluster_assignment, cluster, files_pred)
        thresholds_cluster = get_thresholds_cluster(thresholds, f"cluster_{cluster}")
        postprocess_batch(cluster_files_pred, args.output_folder_pp_cc, thresholds_cluster, args.label_mapping)

if __name__ == "__main__":
    main()