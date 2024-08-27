import logging
from pathlib import Path
import time
import torch
import numpy as np
from typing import Any, List, Tuple, Union, Optional
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations
from monai.data import decollate_batch
import nibabel as nib
from functools import partial
import argparse
import json

from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    MapTransform,
    OneOf,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandFlipd,
    RandAdjustContrastd,
    RandSimulateLowResolutiond,
    ToTensord
)

from monai.data import DataLoader, CacheDataset, decollate_batch
# from datasets_utils import get_loader

parser = argparse.ArgumentParser(description='Swin UNETR segmentation inference')
parser.add_argument('--datadir', default='/dataset/dataset0/', type=str, help='dataset directory')
parser.add_argument('--exp_path', default='test1', type=str, help='experiment output path')
parser.add_argument('--jsonlist', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--fold', default=1, type=int, help='data fold')
parser.add_argument('--pretrained_model_name', default='model.pt', type=str, help='pretrained model name')
parser.add_argument('--feature_size', default=48, type=int, help='feature size')
parser.add_argument('--infer_overlap', default=0.6, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--posrate', default=1.0, type=float, help='positive label rate')
parser.add_argument('--negrate', default=1.0, type=float, help='negative label rate')
parser.add_argument('--nsamples', default=1, type=int, help='number of croped samples')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--cacherate', default=1.0, type=float, help='cache data rate')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
parser.add_argument('--pretrained_dir', default='./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/', type=str,
                    help='pretrained checkpoint directory')
parser.add_argument('--pred_label', action='store_true', help='predict labels or regions')


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


class ConvertToMultiChannelBasedOnBratsPEDClassesd(MapTransform):
    """
    Convert labels to multi channels based on new brats 2023 classes:
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1, 2, 3, 4 to construct WT
            result.append(torch.logical_or(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3), d[key] == 4))
            # merge label 1, 2, 3 to construct TC
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # merge label 1, 2 to construct NET
            result.append(torch.logical_or(d[key] == 1, d[key] == 2))
            # label 1 is ET
            result.append(d[key] == 1)
            d[key] = torch.stack(result, axis=0).float()
        return d    


def get_loader(args):
    """
    Load data sets for training, validation and testing from json files
    """
    data_root = Path(args.datadir)
    #data_root = Path(args.data_dir)
    channel_order= ['-t1n.nii.gz', '-t1c.nii.gz','-t2w.nii.gz','-t2f.nii.gz']
    img_paths = [f"{data_root.name}{c}" for c in channel_order]

    # val_data = json_data['validation']
    val_data = [{'image': img_paths}]
    #val_data = load_json(args.jsonlist)['validation']

    # add data root to json file lists 
    for i in range(0, len(val_data)):
        val_data[i]['label'] = ""
        for j in range(0, len(val_data[i]['image'])):
            val_data[i]['image'][j] = str(data_root / val_data[i]['image'][j])
        
    val_transform = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
    val_ds = CacheDataset(
        data=val_data, 
        transform=val_transform, 
        cache_rate=args.cacherate, 
        num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )

    return val_loader
    

def main():
    time0 = time.time()
    args = parser.parse_args()
    output_directory = Path(args.exp_path)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
        
    logging.basicConfig(
        filename=output_directory / 'infer.log', 
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    val_loader = get_loader(args)    
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = Path(pretrained_dir) / model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint
    )

    model_dict = torch.load(pretrained_pth, map_location=device)['model']
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )
    
    post_trans = Activations(sigmoid=not args.pred_label, softmax=args.pred_label)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            image = batch["image"].cuda()
            affine = batch['image_meta_dict']['original_affine'][0].numpy()
            filepath = Path(batch['image_meta_dict']['filename_or_obj'][0])
            img_name = filepath.name.split('.nii.gz')[0]
            output_pred =  model_inferer_test(image)
            logging.info(f"Inference on case {img_name}")
            logging.info(f"Label-wise: {args.pred_label}")
            logging.info(f"Image shape: {image.shape}")
            logging.info(f"Prediction shape: {output_pred.shape}")
            
            prob = [post_trans(i) for i in decollate_batch(output_pred)]
            prob_np = prob[0].detach().cpu().numpy()
            logging.info(f"Probmap shape: {prob_np.shape}")
            np.savez(output_directory / f"{img_name}.npz", probabilities=prob_np)
            # save integer masks 
            if args.pred_label:
                seg_out = np.argmax(prob_np, axis=0)
            else:
                seg = (prob_np > 0.5).astype(np.int8)
                seg_out = np.zeros_like(seg[0])
                seg_out = np.where(seg[0] == 1, 4, 0)
                seg_out = np.where((seg[1] == 1) & (seg_out == 4), 3, seg_out)
                seg_out = np.where((seg[2] == 1) & (seg_out == 3), 2, seg_out)
                seg_out = np.where((seg[3] == 1) & (seg_out == 2), 1, seg_out)
                # seg_out[seg[3] == 1] = 4
                # seg_out[seg[0] == 1] = 1
                # seg_out[seg[2] == 1] = 3
            
            nib.save(nib.Nifti1Image(seg_out.astype(np.int8), affine),
                    output_directory / f"{img_name}.nii.gz")
                
            logging.info(f"Seg shape: {seg_out.shape}")
                 
        logging.info(f"Finished inference! {int(time.time() - time0)} s")


if __name__ == '__main__':
    main()