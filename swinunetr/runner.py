import subprocess
from pathlib import Path
import torch

def run_infer_swinunetr(input_path: Path, output_folder: Path, challenge, folds=[0,1,2,3,4])->list:
    """runner that helps run swinunter based on a model path and 
    returns path to the probability npz

    Args:
        input_path (Path): path to the folder containing the 4 input files 
        output_path (Path): path to folder to store teh npz file
        model_folder_pth (Path): path where model weights are stored

    Returns:
        path: path to the npz file
    """
    pretrained_name = 'best_model.pt'
    npz_folder_list = []
    
    for fold in folds:
        if fold in [3,4]:
            e = 1000
        else:
            e = 650
        pretrained_path =  Path(f"/tmp/swinunetr_e{e}_f{fold}_b1p4/")
        args = torch.load(pretrained_path / pretrained_name)['args']
        output_dir = output_folder / f"swinunetr3d_f{fold}"
        npz_folder_list.append(output_dir)
        cmd = 'python swinunetr/inference.py'
        cmd = ' '.join((cmd, f"--datadir='{str(input_path)}'"))
        cmd = ' '.join((cmd, f"--exp_path='{str(output_dir)}'"))
        cmd = ' '.join((cmd, f'--roi_x={args.roi_x}'))
        cmd = ' '.join((cmd, f'--roi_y={args.roi_y}'))
        cmd = ' '.join((cmd, f'--roi_z={args.roi_z}'))
        cmd = ' '.join((cmd, f'--in_channels={args.in_channels}'))
        cmd = ' '.join((cmd, f'--out_channels={args.out_channels}'))
        cmd = ' '.join((cmd, '--spatial_dims=3'))
        cmd = ' '.join((cmd, '--use_checkpoint'))
        cmd = ' '.join((cmd, '--feature_size=48'))
        cmd = ' '.join((cmd, '--infer_overlap=0.625'))
        cmd = ' '.join((cmd, '--cacherate=1.0'))
        cmd = ' '.join((cmd, '--workers=0'))
        cmd = ' '.join((cmd, f'--pretrained_model_name={pretrained_name}'))
        cmd = ' '.join((cmd, f'--pretrained_dir={str(pretrained_path)}'))
        cmd = ' '.join((cmd, '--pred_label'))
        print(cmd)
        subprocess.run(cmd, shell=True)  # Executes the command in the shell
    
    npz_path_list = [f / f"{input_path.name}-t1n.npz" for f in npz_folder_list]
    return npz_path_list