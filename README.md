# **Automated & Analysis Scripts for VIMU**

This repository holds supplementary scripts for VIMU, a PBAR-enhanced [PX4](http://px4.io/) flight control solution for drones. It contains several python scripts for automated evaluation and data analysis. Before using these scripts, you must first install the **enhanced PX4 autopilot** and the **development environments**.

Clone the source code from the repo:

```bash
git clone https://github.com/wangwwno1/VIMU-PythonScripts.git VIMU-PythonScripts
```

Extract `data.7z` to folder `VIMU-PythonScripts`, then install the following dependencies.

## **Install ROS1 with MAVROS**

1. Refer to the guide (1.1~1.5) for ROS1 installation:
    - Melodic (For Ubuntu 18.04): https://wiki.ros.org/melodic/Installation/Ubuntu
    - Noetic (For Ubuntu 20.04): https://wiki.ros.org/noetic/Installation/Ubuntu
2. Run the following command to install MAVROS.
    - For Ubuntu 18.04: `sudo apt-get install ros-melodic-mavros ros-melodic-mavros-extras`
    - For Ubuntu 20.04: `sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras`
3. Then install **GeographicLib** datasets by running the `install_geographiclib_datasets.sh` script:
    
    ```bash
    wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
    ./install_geographiclib_datasets.sh
    ```
    

## **Install Python Dependencies**

Run the following command to install the required python packages.

```bash
pip install -r ./requirements_pip.txt
```

## Script Usage

### Collect Flight Logs with `collect_flight_logs.py`
 1. Automate collect flight logs for various baselines and test cases.
 2. **WARN: Colleting thousands of flight logs would last for hours to days. Run a small-scale test first to ensure all settings are properly configured before the full-scale experiments.** 
 3. In `BASELINE_FIRMWARE_PATH` , specify the path to the enhanced PX4-Autopilot
 4. Specify the number of trials:
     1. `NUM_NORMAL_TRIAL` for `identification`, `validation` and `complex_maneuver/no_attack_and_detection`
     2. `NUM_ATTACK_TRIAL` for other cases
 5. Speficy the test cases to collect in the `TEST_CASES` and `*_TEST_CASE` variables.
     1. You can specify multiple test cases or baselines, the script will run them sequentially.
 6. Execute the scripts to collect flight logs.
### Examine flight logs with `evaluate_solution.py`
1. Evaluate detection and recovery performance per test case
2. Run the following command with path to the test case folder with flight logs.

    ```bash
    python3 evaluate_solution.py "data/evaluation/path/to/testcase"
    ```
        
3. The script generate two files (`detection_reports.csv` and `log_metas.csv`) in that folder. 
4. Use `log_metas.csv` to remove faulty logs that have `attack_not_found=True`
5. Rerun the script again to ensure all flight logs are valid.
### Prepare for ROC and param selection: `calculate_error_ratios.py` and `calculate_test_ratios.py`
1. Generate intermediate data for detector parameter selection & ROC curves
2. Set and run `calculate_error_ratios.py` first to obtain `error_ratios.pkl` 
    1. Set baseline to process in `BASELINE_NAMES` 
    2. Specify test cases to process in `*_TEST_CASE`
    3. Run the script to generate `error_ratios.pkl` under the test case folder.
3. Execute `calculate_test_ratios.py` to obtain Threshold-FPR/TPR data at various settings.
    1. Set baseline to process in `SELECTED_BASELINES`
    2. Specify test cases to process in `*_TEST_CASE`
    3. (Optional) Modify the parameter in `DETECTOR_PARAMS` if wish to test other detector parameter settings.
    4. Run the script to generate TPR/FPR-Threshold data for various detector in  `TEST_CASE_FOLDER/param_selection/*_threshold_per_flight.csv`
### Group detection and recovery metrics with `extract_perf_metrics.py`
1. Set baseline to process in `BASELINE_NAMES` 
2. Specify test cases to process in `*_TEST_CASE`
3. Move related `detection_reports.csv` and `log_metas.csv` to the specified location, organized as `data/BASELINE_NAMES/SCENE_NAME/TEST_CASE_NAME/files`
4. Run the script to generate the aggregated file for figure.

### Aggregate ROC data with `validate_detector.py`
1. See documentation for `extract_perf_metrics.py`

### Plot various figures (scripts starts with `plot_`)
1. Specify the `CSV_PATH` to the report csv file
2. Specify the filename of figure
3. Run the python script to generate the figure file.