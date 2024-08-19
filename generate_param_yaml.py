from copy import deepcopy

import numpy as np

from utils.mav_utils import AttackType, StealthyType

# Use customized profile, log when armed until disarmed
LOG_PARAMS = dict(SDLOG_PROFILE=0, SDLOG_MODE=0)

CMD_PARAMS = dict(
    # COM_POS_FS_DELAY=5,       # 500s, use to suppress commander failsafe alarm
    CBRK_VELPOSERR=0,      # 201607 will disable position failsafe for all flight mode
    MIS_DIST_1WP=2000.0,
    MIS_DIST_WPS=2000.0	# Increase distance between waypoints to accomodate our mission setting.
)

EKF2_PARAMS = dict(
    EKF2_AID_MASK=1,        # 1 for GPS fusion only
    # EKF2_NOAID_TOUT=int(500 * 1e6)  # 500s
)
# jMAVSim (SITL) running at fixed 250 Hz
IMU_PARAMS = dict(IMU_GYRO_RATEMAX=250, IMU_INTEG_RATE=250)

# jMAVSim Quadcopter model
# For SAVIOR, set VM_MOTOR_TAU as zero
VEHICLE_MODEL_PARAMS = dict(
    VM_MASS=0.8, VM_THR_FACTOR=4.0, VM_DRAG_FACTOR=0.05,
    VM_INERTIA_XX=0.005,
    VM_INERTIA_XY=0.0, VM_INERTIA_YY=0.005,
    VM_INERTIA_XZ=0.0, VM_INERTIA_YZ=0.0, VM_INERTIA_ZZ=0.009,
    VM_MOTOR_TAU=0.005, VM_ANG_ACC_NOISE=0.075,
    EKF2_MCOEF=0.0, EKF2_BCOEF_X=45.5, EKF2_BCOEF_Y=45.5,
    IV_IMU_DELAY_US=500000, VIMU_PREDICT_US=20000
)

GPS_P_NOISE = 0.5
GPS_V_NOISE = 0.3
ACC_NOISE = 0.35
GYR_NOISE = 0.1
SENSOR_NOISE_PARAMS = dict(
    IV_ACC_NOISE=ACC_NOISE,
    IV_GYR_NOISE=GYR_NOISE,
    EKF2_GPS_P_NOISE=GPS_P_NOISE,
    EKF2_GPS_V_NOISE=GPS_V_NOISE,
    EKF2_BARO_NOISE=3.5,
    EKF2_MAG_NOISE=5.0e-2
)

CUSUM_DETECTOR_PARAMS = dict(
    IV_GPS_P_CSUM_H=3.0, IV_GPS_P_MSHIFT=0.5,
    IV_GPS_V_CSUM_H=3.5, IV_GPS_V_MSHIFT=1.0,
    IV_BARO_CSUM_H=3.0, IV_BARO_MSHIFT=0.25,
    IV_MAG_CSUM_H=3.0, IV_MAG_MSHIFT=0.25,
    IV_ACC_CSUM_H=3.0, IV_ACC_MSHIFT=1.0,
    IV_GYR_CSUM_H=3.0, IV_GYR_MSHIFT=0.5,
)
CUSUM_DETECTOR_PARAMS_NO_BUFFER = CUSUM_DETECTOR_PARAMS
CUSUM_DETECTOR_PARAMS_VIMU = deepcopy(CUSUM_DETECTOR_PARAMS)
DISABLE_CUSUM_PARAMS = {k: 0.0 for k in CUSUM_DETECTOR_PARAMS.keys() if k.endswith('_CSUM_H')}

EWMA_DETECTOR_PARAMS = dict(
    IV_GPS_P_EMA_H=0.45, IV_GPS_P_ALPHA=0.01, IV_GPS_P_EMA_CAP=0.85,
    IV_GPS_V_EMA_H=0.5, IV_GPS_V_ALPHA=0.01, IV_GPS_V_EMA_CAP=1.1,
    IV_MAG_EMA_H=0.3, IV_MAG_ALPHA=0.01, IV_MAG_EMA_CAP=0.52,
    IV_BARO_EMA_H=0.15, IV_BARO_ALPHA=0.05, IV_BARO_EMA_CAP=0.52,
    IV_ACC_EMA_H=0.95, IV_ACC_ALPHA=0.01, IV_ACC_EMA_CAP=1.1,
    IV_GYR_EMA_H=0.235, IV_GYR_ALPHA=0.01, IV_GYR_EMA_CAP=0.85
)
DISABLE_EWMA_PARAMS = {k: 0.0 for k in EWMA_DETECTOR_PARAMS.keys() if k.endswith('_EMA_H')}

L1TW_DETECTOR_PARAMS = dict(
    IV_GPS_P_TWIN_H=1.15, IV_GPS_P_RST_CNT=10, IV_GPS_P_CD_CNT=10,
    IV_GPS_V_TWIN_H=3.10, IV_GPS_V_RST_CNT=10, IV_GPS_V_CD_CNT=10,
    IV_BARO_TWIN_H=0.1, IV_BARO_RST_CNT=10, IV_BARO_CD_CNT=10,
    IV_MAG_TWIN_H=0.35, IV_MAG_RST_CNT=10, IV_MAG_CD_CNT=10,
    IV_ACC_TWIN_H=5.20, IV_ACC_RST_CNT=10, IV_ACC_CD_CNT=10,
    IV_GYR_TWIN_H=0.464, IV_GYR_RST_CNT=10, IV_GYR_CD_CNT=10,
)
DISABLE_L1TW_PARAMS = {}
for k in L1TW_DETECTOR_PARAMS.keys():
    if k.endswith('_TWIN_H'):
        DISABLE_L1TW_PARAMS[k] = 0.0
    if k.endswith('CD_CNT'):
        L1TW_DETECTOR_PARAMS[k] = int(L1TW_DETECTOR_PARAMS[k] * 2.5)

DISABLE_L1TW_PARAMS = {k: 0.0 for k in L1TW_DETECTOR_PARAMS.keys() if k.endswith('_TWIN_H')}

L2TW_DETECTOR_PARAMS = dict(
    IV_GPS_P_TWIN_H=1.402, IV_GPS_P_RST_CNT=10, IV_GPS_P_CD_CNT=10,
    IV_GPS_V_TWIN_H=4.42, IV_GPS_V_RST_CNT=10, IV_GPS_V_CD_CNT=10,
    IV_BARO_TWIN_H=0.02, IV_BARO_RST_CNT=10, IV_BARO_CD_CNT=10,
    IV_MAG_TWIN_H=0.20, IV_MAG_RST_CNT=10, IV_MAG_CD_CNT=10,
    IV_ACC_TWIN_H=30.25, IV_ACC_RST_CNT=10, IV_ACC_CD_CNT=10,
    IV_GYR_TWIN_H=0.65, IV_GYR_RST_CNT=20, IV_GYR_CD_CNT=20,
)
for k in L2TW_DETECTOR_PARAMS.keys():
    if k.endswith('CD_CNT'):
        L2TW_DETECTOR_PARAMS[k] = int(L2TW_DETECTOR_PARAMS[k] * 2.5)
DISABLE_L2TW_PARAMS = DISABLE_L1TW_PARAMS

DEFAULT_DET_PARAMS = dict(**CUSUM_DETECTOR_PARAMS, **EWMA_DETECTOR_PARAMS, **L1TW_DETECTOR_PARAMS)
# Set default control limit to 0.0
DEFAULT_DET_PARAMS.update({k: 0.0 for k in DEFAULT_DET_PARAMS.keys() if k.endswith('_H')})
# CUSUM MShift to 1.0
DEFAULT_DET_PARAMS.update({k: 1.0 for k in DEFAULT_DET_PARAMS.keys() if k.endswith('_MSHIFT')})
# EWMA Alpha to 1.0
DEFAULT_DET_PARAMS.update({k: 1.0 for k in DEFAULT_DET_PARAMS.keys() if k.endswith('_ALPHA')})
# EWMA CAP to 1.0
DEFAULT_DET_PARAMS.update({k: 1.0 for k in DEFAULT_DET_PARAMS.keys() if k.endswith('_EMA_CAP')})
# TimeWindow Reset Count and Cool Down to 1 (int)
DEFAULT_DET_PARAMS.update({k: 1 for k in DEFAULT_DET_PARAMS.keys() if k.endswith('_RST_CNT') or k.endswith('_CD_CNT')})

# Single Sensor Attack - Overt & Stealthy
GPS_P_OVERT_BIAS = [0.5, 0.75, 1.0, 1.25, 1.5, 20.0]
# GPS_V_OVERT_BIAS = [0.75, 1.0, 1.1, 1.2, 2.0, 3.0]  # Unused
# ACCEL_OVERT_BIAS = [0.8, 0.9, 1.0, 1.1, 2.0, 3.0]  # Unused
ATK_GYR_BIAS_STD = [0.4, 0.6, 0.8, 1.0, 3.0, 6.0]
# CUAV V5+ equips BMI055, ICM-20689, ICM-20602 IMUs
#   Model        Max Amp. (rad/s)   Induced Freq. (Hz)
#   BMI055       Unknown            Unknown
#   ICM-20689    1.899              205.9
#   ICM-20602    0.927              19.7
ICM20689_MAX_ATK_GYR_AMP_RAD_S = 1.899
ICM20689_ATK_GYR_FREQ_HZ = 205.9
ICM20602_MAX_ATK_AMP_RAD_S = 0.927
ICM20602_ATK_GYR_FREQ_HZ = 19.7
DEFAULT_ATK_GYR_FREQ_HZ = 250.0  # SITL Frequency
DEFAULT_ATK_GYR_PHASE = 0.0
GYRO_OVERT_FREQ_HZ = [
    ICM20602_ATK_GYR_FREQ_HZ,
    ICM20689_ATK_GYR_FREQ_HZ,
    DEFAULT_ATK_GYR_FREQ_HZ
]
IMU_01_FLAG = (1 << 0) | (1 << 1)
IMU_012_FLAG = IMU_01_FLAG | (1 << 2)
BARO_01_FLAG = MAG_01_FLAG = IMU_01_FLAG

# Sensor Attack Parameter Section - Default Params
DEFAULT_ATK_PARAMS = dict(
    IV_DEBUG_LOG=1,
    ATK_APPLY_TYPE=0, ATK_STEALTH_TYPE=0, ATK_COUNTDOWN_MS=5000,
    IV_DELAY_MASK=AttackType.NoAttack, IV_TTD_DELAY_MS=0
)
# GPS Overt Bias Attacks
GPS_ATK_PARAMS = dict(
    ATK_GPS_P_CLS=0, ATK_GPS_P_IV=0.01, ATK_GPS_P_RATE=1.0, ATK_GPS_P_CAP=10.0, ATK_GPS_P_HDG=0.0, ATK_GPS_P_PITCH=0.0,
    ATK_GPS_V_CLS=0, ATK_GPS_V_IV=0.01, ATK_GPS_V_RATE=1.0, ATK_GPS_V_CAP=10.0, ATK_GPS_V_HDG=0.0, ATK_GPS_V_PITCH=0.0
)
# IMU Overt Bias Attacks
IMU_ATK_PARAMS = dict(
    ATK_MULTI_IMU=0,
    ATK_GYR_AMP=0.0, ATK_GYR_FREQ=DEFAULT_ATK_GYR_FREQ_HZ, ATK_GYR_PHASE=DEFAULT_ATK_GYR_PHASE,
    ATK_ACC_BIAS=0.0
)
MAG_ATK_PARAMS = dict(ATK_MULTI_MAG=0)
BARO_ATK_PARAMS = dict(ATK_MULTI_BARO=0)

DEFAULT_ATK_PARAMS.update(**GPS_ATK_PARAMS, **IMU_ATK_PARAMS, **MAG_ATK_PARAMS,
                          **BARO_ATK_PARAMS)

STEALTHY_TYPES = [
    StealthyType.NoStealthy,
    StealthyType.CumulativeSum,
    StealthyType.CumulativeSum | StealthyType.ExponentialMovingAverage,
    StealthyType.TimeWindow
]


if __name__ == '__main__':
    import os
    import posixpath as path
    import yaml

    def get_default_attack_params(atk_apply_type: int, atk_stealth_type: int = StealthyType.NoStealthy) -> dict:
        result = dict(ATK_APPLY_TYPE=atk_apply_type, ATK_STEALTH_TYPE=atk_stealth_type)

        if atk_apply_type & (AttackType.Gyroscope | AttackType.Accelerometer):
            result.setdefault('ATK_MULTI_IMU', IMU_012_FLAG)

        if atk_apply_type & AttackType.Barometer:
            result.setdefault('ATK_MULTI_BARO', BARO_01_FLAG)

        extra_params = dict()
        # Ensure no duplicate extra params
        if atk_apply_type & AttackType.Gyroscope:
            extra_params = dict(
                ATK_GYR_AMP=0.6,
                ATK_GYR_FREQ=DEFAULT_ATK_GYR_FREQ_HZ,
                ATK_GYR_PHASE=DEFAULT_ATK_GYR_PHASE,
                **extra_params
            )

        if atk_apply_type & AttackType.Accelerometer:
            extra_params = dict(ATK_ACC_BIAS=5.0, **extra_params)

        if atk_apply_type & AttackType.GpsPosition:
            extra_params = dict(ATK_GPS_P_IV=10.0, ATK_GPS_P_CAP=10.0, **extra_params)

        if atk_apply_type & AttackType.GpsVelocity:
            extra_params = dict(ATK_GPS_V_IV=10.0, ATK_GPS_V_CAP=10.0, **extra_params)

        result = dict(**result, **extra_params)
        return result


    def export_attack_params(dir_root, param_list):
        if len(param_list) == 0:
            return

        os.makedirs(dir_root, exist_ok=True)
        for idx, param_dict in enumerate(param_list):
            with open(path.join(dir_root, f'attack_params_{idx}.yaml'), 'w') as f:
                yaml.dump(param_dict, f)


    def export_firmware_params(folder: str, default_params: dict, detector_params: dict):
        os.makedirs(folder, exist_ok=True)
        default_params = default_params.copy()
        for k in detector_params.keys():
            if k in DEFAULT_DET_PARAMS:
                default_params[k] = DEFAULT_DET_PARAMS[k]
        with open(path.join(folder, 'default_params.yaml'), 'w') as f:
            yaml.dump(default_params, f)
        with open(path.join(folder, 'detector_params.yaml'), 'w') as f:
            yaml.dump(detector_params, f)

    EXPORT_FOLDER = 'data/parameter_yaml'
    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    # Solution related params
    SOL_PARAM_ROOT = path.join(EXPORT_FOLDER, 'solution')

    DEFAULT_PARAMS = dict(**LOG_PARAMS, **IMU_PARAMS, **CMD_PARAMS, **EKF2_PARAMS, **SENSOR_NOISE_PARAMS, **DEFAULT_ATK_PARAMS)

    # Control Invariant - CCS18
    export_firmware_params(path.join(SOL_PARAM_ROOT, 'control_invariant'), DEFAULT_PARAMS, L2TW_DETECTOR_PARAMS)

    # Software Sensor - RAID20
    # With compensation
    export_firmware_params(path.join(SOL_PARAM_ROOT, 'software_sensor_sup'), DEFAULT_PARAMS, L1TW_DETECTOR_PARAMS)

    # SAVIOR - USENIX20 + Our Buffer
    savior_params = DEFAULT_PARAMS.copy()
    savior_params.update(**VEHICLE_MODEL_PARAMS)
    savior_params['VM_MOTOR_TAU'] = 0.0
    export_firmware_params(path.join(SOL_PARAM_ROOT, 'savior_with_buffer'), savior_params, CUSUM_DETECTOR_PARAMS)

    # SAVIOR - USENIX20
    savior_params = savior_params.copy()
    savior_params.update(IV_IMU_DELAY_US=4000, VIMU_PREDICT_US=4000)
    export_firmware_params(path.join(SOL_PARAM_ROOT, 'savior'), savior_params, CUSUM_DETECTOR_PARAMS)

    # VirtualIMU - Ours solution
    vimu_params = DEFAULT_PARAMS.copy()
    vimu_params.update(**VEHICLE_MODEL_PARAMS)
    vimu_params.update(VM_LEN_SCALE_X=0.233345, VM_LEN_SCALE_Y=0.233345, VM_LEN_SCALE_Z=1.0)
    vimu_detector_params = dict(**CUSUM_DETECTOR_PARAMS, **EWMA_DETECTOR_PARAMS)
    export_firmware_params(path.join(SOL_PARAM_ROOT, 'virtual_imu'), vimu_params, vimu_detector_params)

    # VirtualIMU - Ours solution but without delay buffer
    vimu_params = DEFAULT_PARAMS.copy()
    vimu_params.update(**VEHICLE_MODEL_PARAMS)
    vimu_params.update(IV_IMU_DELAY_US=4000, VIMU_PREDICT_US=4000)
    vimu_params.update(VM_LEN_SCALE_X=0.233345, VM_LEN_SCALE_Y=0.233345, VM_LEN_SCALE_Z=1.0)
    vimu_detector_params = dict(**CUSUM_DETECTOR_PARAMS, **EWMA_DETECTOR_PARAMS)
    export_firmware_params(path.join(SOL_PARAM_ROOT, 'virtual_imu_no_buffer'), vimu_params, vimu_detector_params)

    # Virtual-IMU with CUSUM Detector - Alternative version with detector from SAVIOR
    vimu_params = DEFAULT_PARAMS.copy()
    vimu_params.update(**VEHICLE_MODEL_PARAMS)
    vimu_detector_params = dict(**CUSUM_DETECTOR_PARAMS)
    export_firmware_params(path.join(SOL_PARAM_ROOT, 'virtual_imu_cusum'), vimu_params, vimu_detector_params)

    # Attack Related Params
    ATTACK_PARAM_ROOT = path.join(EXPORT_FOLDER, 'attack')

    # Default (No Attack & Detector)
    export_attack_params(path.join(ATTACK_PARAM_ROOT, 'no_attack_or_detection'), [DEFAULT_ATK_PARAMS])

    # Gps overt attack
    gps_overt_attacks = []
    for bias_std in GPS_P_OVERT_BIAS:
        attack_params = get_default_attack_params(AttackType.GpsPosition)
        for k in ['ATK_GPS_P_IV', 'ATK_GPS_P_CAP']:
            attack_params[k] = bias_std * GPS_P_NOISE  # noqa
        gps_overt_attacks.append(attack_params)
    export_attack_params(path.join(ATTACK_PARAM_ROOT, 'gps_overt_attack'), gps_overt_attacks)

    # GPS Joint Attack
    export_attack_params(path.join(ATTACK_PARAM_ROOT, 'gps_joint_attack'),
                         [get_default_attack_params(AttackType.GpsJointPVAttack)])

    # Gyroscope overt attack
    for induced_freq in GYRO_OVERT_FREQ_HZ:
        gyro_attack_params = []
        for bias_in_std in ATK_GYR_BIAS_STD:
            attack_param_dict = get_default_attack_params(AttackType.Gyroscope)
            attack_param_dict.update(
                ATK_GYR_AMP=bias_in_std * GYR_NOISE,
                ATK_GYR_FREQ=induced_freq
            )
            gyro_attack_params.append(attack_param_dict)

        suffix = "default"
        if induced_freq == ICM20602_ATK_GYR_FREQ_HZ:
            suffix = "icm20602"
            attack_param_dict = get_default_attack_params(AttackType.Gyroscope)
            attack_param_dict.update(ATK_GYR_AMP=ICM20602_MAX_ATK_AMP_RAD_S, ATK_GYR_FREQ=induced_freq)
            gyro_attack_params.append(attack_param_dict)
        elif induced_freq == ICM20689_ATK_GYR_FREQ_HZ:
            suffix = "icm20689"
            attack_param_dict = get_default_attack_params(AttackType.Gyroscope)
            attack_param_dict.update(ATK_GYR_AMP=ICM20689_MAX_ATK_GYR_AMP_RAD_S, ATK_GYR_FREQ=induced_freq)
            gyro_attack_params.append(attack_param_dict)

        export_attack_params(path.join(ATTACK_PARAM_ROOT, f'gyro_overt_attack_{suffix}'), gyro_attack_params)

        # Triple-Module Redundancy Test Only Attack 2 of 3 IMUs
        for d in gyro_attack_params:
            d['ATK_MULTI_IMU'] = IMU_01_FLAG
        export_attack_params(path.join(ATTACK_PARAM_ROOT, f'tmr_test_{suffix}'), gyro_attack_params)

    # Stealthy Attacks
    gyro_stealthy_attacks = []
    bound_fac = 0.99
    for stealth_type in STEALTHY_TYPES:
        max_deviations = []
        if stealth_type & StealthyType.ExponentialMovingAverage:
            max_deviations.append(bound_fac * EWMA_DETECTOR_PARAMS['IV_GYR_EMA_H'])
            # fixme remove 0.3 setting
            max_deviations.append(0.3)
        elif stealth_type & StealthyType.CumulativeSum:
            max_deviations.append(bound_fac * CUSUM_DETECTOR_PARAMS['IV_GYR_MSHIFT'])
            if not np.isclose(CUSUM_DETECTOR_PARAMS['IV_GYR_MSHIFT'], CUSUM_DETECTOR_PARAMS_NO_BUFFER['IV_GYR_MSHIFT']):
                max_deviations.append(bound_fac * CUSUM_DETECTOR_PARAMS_NO_BUFFER['IV_GYR_MSHIFT'])
        elif stealth_type & StealthyType.TimeWindow:
            max_deviations.append(bound_fac * L2TW_DETECTOR_PARAMS['IV_GYR_TWIN_H'] ** 0.5)
            # fixme restore l1tw setting - we remove this setting because l1tw is close to cusum
            # max_deviations.append(bound_fac * L1TW_DETECTOR_PARAMS['IV_GYR_TWIN_H'])
        else:
            continue

        default_params = get_default_attack_params(AttackType.Gyroscope, stealth_type)
        default_params.pop('ATK_GYR_AMP')
        for max_dev in max_deviations:
            gyro_stealthy_attacks.append(dict(ATK_GYR_AMP=max_dev * GYR_NOISE, **default_params))


    export_attack_params(path.join(ATTACK_PARAM_ROOT, 'gyro_stealthy_attack_default'), gyro_stealthy_attacks)

    gps_stealthy_attacks = []
    bound_fac = 0.99
    for stealth_type in STEALTHY_TYPES:
        max_deviations = []
        if stealth_type & StealthyType.ExponentialMovingAverage:
            max_deviations.append(bound_fac * EWMA_DETECTOR_PARAMS['IV_GPS_P_EMA_H'])
        elif stealth_type & StealthyType.CumulativeSum:
            max_deviations.append(bound_fac * CUSUM_DETECTOR_PARAMS['IV_GPS_P_MSHIFT'])
            if not np.isclose(CUSUM_DETECTOR_PARAMS['IV_GPS_P_MSHIFT'], CUSUM_DETECTOR_PARAMS_NO_BUFFER['IV_GPS_P_MSHIFT']):
                max_deviations.append(bound_fac * CUSUM_DETECTOR_PARAMS_NO_BUFFER['IV_GPS_P_MSHIFT'])
        elif stealth_type & StealthyType.TimeWindow:
            max_deviations.append(bound_fac * min(L2TW_DETECTOR_PARAMS['IV_GPS_P_TWIN_H'] ** 0.5, L1TW_DETECTOR_PARAMS['IV_GPS_P_TWIN_H']))
            # fixme restore l1tw setting - we remove this setting because l1tw is close to cusum
            # max_deviations.append(bound_fac * L1TW_DETECTOR_PARAMS['IV_GYR_TWIN_H'])
        else:
            continue

        default_params = get_default_attack_params(AttackType.GpsPosition, stealth_type)
        default_params.pop('ATK_GPS_P_IV')

        for max_dev in max_deviations:
            gps_stealthy_attacks.append(dict(ATK_GPS_P_IV=max_dev * GPS_P_NOISE, **default_params))
    export_attack_params(path.join(ATTACK_PARAM_ROOT, 'gps_stealthy_attack_default'), gps_stealthy_attacks)

    # TTD vs. Recovery Duration Test
    TTD_GYRO_DURATION_PARAMS = []
    TTD = [8, 12, 20, 30, 40, 50, 80, 120, 160, 200]
    for ttd_time in TTD:
        d = get_default_attack_params(AttackType.Gyroscope)
        d.update(IV_DELAY_MASK=AttackType.Gyroscope, IV_TTD_DELAY_MS=ttd_time)
        TTD_GYRO_DURATION_PARAMS.append(d)
    export_attack_params(path.join(ATTACK_PARAM_ROOT, 'ttd_test'), TTD_GYRO_DURATION_PARAMS)

    # Recovery test with full detector-recovery scheme
    RECOVERY_TEST_PARAMS = [
        get_default_attack_params(attack_type)
        for attack_type in (
            AttackType.Barometer,
            AttackType.GpsPosition,
            AttackType.Gyroscope,
            AttackType.GpsPosition | AttackType.Barometer,
            AttackType.GpsPosition | AttackType.Gyroscope,
            AttackType.GpsPosition | AttackType.Barometer | AttackType.Gyroscope,
            AttackType.Magnetometer | AttackType.Accelerometer | AttackType.Gyroscope
        )
    ]
    export_attack_params(path.join(ATTACK_PARAM_ROOT, 'recovery_test'), RECOVERY_TEST_PARAMS)
