import os
from multiprocessing import Pool

import argparse
import nibabel as nib

from CCMetrics.CC_base import CCDiceMetric
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # Import process_map for progress bars with multiprocessing

def process_file(args):
    gt_file, cache_dir = args
    metric = CCDiceMetric(use_caching=True, caching_dir=cache_dir)
    y = nib.load(gt_file).get_fdata()
    metric.cache_datapoint(y)

def main():
    parser = argparse.ArgumentParser(description='Cache data')
    parser.add_argument('--gt', type=str, help='Path to the directory containing the ground truth nii.gz images')
    parser.add_argument('--cache_dir', type=str, help='Path to the directory where the cache files will be stored')
    parser.add_argument('--nof_workers', type=int, default=1, help='Number of workers to use for parallel processing')
    
    args = parser.parse_args()
    
    assert args.gt is not None, 'Please provide the path to the ground truth images'
    assert os.path.exists(args.gt), 'The path to the ground truth images does not exist'
    assert args.cache_dir is not None, 'Please provide the path to the cache directory'
    
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir) 
        print(f"Created cache directory at {args.cache_dir}")

    identified_gt_files = [x for x in os.listdir(args.gt) if x.endswith('.nii.gz')]
    full_path_gt_files = [os.path.join(args.gt, x) for x in identified_gt_files]

    print(f"Found {len(identified_gt_files)} ground truth files in directory {args.gt}")
    print(f"Identified files look like this: {identified_gt_files[:5]}")

    if args.nof_workers > 1:
        # Use process_map for parallel processing with progress bar
        process_map(
            process_file, 
            [(gt_file, args.cache_dir) for gt_file in full_path_gt_files], 
            max_workers=args.nof_workers, 
            desc="Processing files", 
            unit="file"
        )
    else:
        # For single-worker case, use regular tqdm
        metric = CCDiceMetric(use_caching=True, caching_dir=args.cache_dir)
        for gt_file in tqdm(full_path_gt_files, desc="Processing files", unit="file"):
            y = nib.load(gt_file).get_fdata()
            metric.cache_datapoint(y)
        
if __name__ == '__main__':
    main()