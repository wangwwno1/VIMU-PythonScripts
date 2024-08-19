import os
from os.path import join
import argparse

from joblib import Parallel, delayed

import pandas as pd

from utils.evaluation_helpers import evaluate_performance

parser = argparse.ArgumentParser(description="Automated System Evaluation")
parser.add_argument('log_dir', metavar='LOG_DIR', action='store',
                    help='path to flight log folder, will recursive collect all logs under this folder, ignore symlink')
parser.add_argument('--export_dir', action='store', default=None,
                    help='path to export the analysis reports')


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    export_dir = arg_parser.export_dir
    if export_dir is None:
        export_dir = arg_parser.log_dir

    log_files = []
    for (abs_path, _, file_names) in os.walk(arg_parser.log_dir):
        log_files.extend([os.path.join(abs_path, f) for f in file_names if f.endswith('.ulg')])

    para_module = Parallel(n_jobs=-1, verbose=100)

    def para_func(log_file, *args, **kwargs):
        try:
            return evaluate_performance(log_file, *args, **kwargs)
        except Exception as e:
            print(f"Got exception for Log: {log_file}")
            print(e)
            return None


    result = para_module(delayed(para_func)(log_file, deviation_threshold=3.0, detect_timeout_ms=5000)
                         for log_file in log_files)

    result = [v for v in result if v is not None]

    if len(result) > 0:
        os.makedirs(export_dir, exist_ok=True)
        full_log_info, full_detection_report = zip(*result)
        full_detection_report = [report for report in full_detection_report if report is not None]
        if len(full_log_info) > 0:
            full_log_info = pd.concat(full_log_info, axis=0, ignore_index=True)
            full_log_info.to_csv(join(export_dir, 'log_metas.csv'), index=False)
        if len(full_detection_report) > 0:
            full_detection_report = pd.concat(full_detection_report, axis=0, ignore_index=True)
            full_detection_report.to_csv(join(export_dir, 'detection_reports.csv'), index=False)
