import os
import sys
import json
import numpy as np
from glob import glob

log_dir = sys.argv[1]
seg_results_name = "segmentation_result.json"
seg_key_name = "mIoU"
det_results_name = "metrics_summary.json"
det_key_name = "mean_ap"


def read_results(log_dir, filename, key):
    results = []
    for file in glob(os.path.join(log_dir, "**", filename), recursive=True):
        print(f"Reading results from {file}.")
        with open(file, 'r') as f:
            res = json.load(f)
        results.append(res[key])
    return np.mean(results)


map = read_results(log_dir, det_results_name, det_key_name)
print(f"Your mAP: {map * 100:.4f}%\n")

miou = read_results(log_dir, seg_results_name, seg_key_name)
print(f"Your mIoU: {miou * 100:.4f}%\n")
