from pathlib import Path
import pandas as pd
import pickle
import argparse
import json


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_cluster(radiomics_df:pd.DataFrame, custer_arftifacts:dict)->list:
    normalizer = custer_arftifacts['normalizer']
    normal_values = normalizer.transform(radiomics_df.iloc[:, 1:].values)
    pca = custer_arftifacts['pca']
    scores_pca = pca.transform(normal_values)
    kmeans = custer_arftifacts['kmeans']
    # get the cluster assignment
    cluster_assignment = kmeans.predict(scores_pca)
    return cluster_assignment
def get_cluster_artifacts(task:str)->dict:
    if task == 'BraTS-SSA':
        pkl_path = 'kmeans-cluster-artifacts/SSA_cluster.pkl'
    elif task == 'BraTS-PED':
        pkl_path = 'kmeans-cluster-artifacts/PEDS_cluster.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
# code to test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", type=str)
    parser.add_argument("-o", "--output_json", type=str)
    parser.add_argument("-c", "--cluster_pickle", type=str)
    args = parser.parse_args()

    # df_radiomics = pd.read_csv('radiomics.csv')
    data = load_json(args.input_json) 
    df_radiomics = pd.DataFrame(data)
    with open(args.cluster_pickle, 'rb') as f:
        cluster_artifacts = pickle.load(f)
    cluster_assignment = get_cluster(df_radiomics, cluster_artifacts)

    for case, cluster in zip(data, cluster_assignment):
        case['cluster'] = int(cluster)
    save_json(args.output_json, data)

# python pp_cluster/infer.py -i /home/abhijeet/Code/BraTS2024/datalist/radiomics/BraTS2024-GLI-training_data1_v2-radiomics.json -o datalist/pp_cluster_assignment/GLI.json -c kmeans-cluster-aftifacts/GLI_cluster.pkl
