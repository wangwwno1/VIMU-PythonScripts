import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = ['euler_to_quaternion', 'quaternion_to_euler',
           'quaternion_multiply', 'quaternion_raw_multiply',
           'inverse_quaternion']


# For Position, Velocity, Acceleration, Angular Velocity, Torque & Thrust Force
def vector3_to_vector4(xyz: np.ndarray, min_length=1., mag_scale=None, mag_bias=0.0):
    if mag_scale is None:
        mag_scale = 1
    length = np.linalg.norm(xyz, axis=-1, keepdims=True).clip(min_length)
    normed_vector = xyz / length
    scaled_magnitude = np.log10(length) * mag_scale + mag_bias

    return np.concatenate((normed_vector, scaled_magnitude), axis=-1)


def euler_to_quaternion(euler_angle_in_rad):
    """
    :param euler_angle_in_rad: Euler attitude, unit: rad
    :return: Quaternion attitude with axis order w i j k
    """
    # Scipy put the real part at last index
    # But in PX4 and ArduPilot the real part should at first index
    quaternion_in_ijkw = R.from_euler('xyz', euler_angle_in_rad).as_quat()
    quaternion_in_wijk = quaternion_in_ijkw.take([3, 0, 1, 2], axis=-1)
    return quaternion_in_wijk


def quaternion_to_euler(quaternion_in_wijk):
    quaternion_in_ijkw = quaternion_in_wijk.take([1, 2, 3, 0], axis=-1)
    euler_angle_in_rad = R.from_quat(quaternion_in_ijkw).as_euler('xyz')
    return euler_angle_in_rad


def quaternion_multiply(self, other):
    result = quaternion_raw_multiply(self, other)
    result = np.stack(result, axis=-1)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def quaternion_raw_multiply(self, other):
    q, p = self, other
    q = np.expand_dims(q, axis=0).swapaxes(0, -1).squeeze(-1)
    p = np.expand_dims(p, axis=0).swapaxes(0, -1).squeeze(-1)
    result = [q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
              q[1] * p[0] + q[0] * p[1] - q[3] * p[2] + q[2] * p[3],
              q[2] * p[0] + q[3] * p[1] + q[0] * p[2] - q[1] * p[3],
              q[3] * p[0] - q[2] * p[1] + q[1] * p[2] + q[0] * p[3]]
    return result


def inverse_quaternion(quaternion_in_wijk):
    vec_length = np.linalg.norm(quaternion_in_wijk, axis=-1, keepdims=True)
    return quaternion_in_wijk * np.asarray([1, -1, -1, -1]) / vec_length


def motor_to_mixer_output(normed_actuator_output: np.ndarray, reverse_mixer: np.ndarray):
    """
    Calculate the torque and thrust relative to iris airframe.
    :param normed_actuator_output: actuator output between -1.0 ~ 1.0, with zero as no thrust, one as max thrust
    :param reverse_mixer: (N, 6) matrix with N as motor numbers
    # TODO Add TWR adjustment
    # :param estimated_hover_thrust: hover_thrust estimated by autopilot, use to scale TWR
    # twr_scale = estimated_hover_thrust / STD_HOVER_THRUST
    :return: Three-axis torques and thrust with both shape as (Batch, 3)
    """
    torque_thrust_value = np.matmul(normed_actuator_output, reverse_mixer)
    torque_value = np.take(torque_thrust_value, [0, 1, 2], axis=-1)
    thrust_value = np.take(torque_thrust_value, [3, 4, 5], axis=-1)
    return torque_value, thrust_value
