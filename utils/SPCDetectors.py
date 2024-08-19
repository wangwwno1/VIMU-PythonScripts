from collections import deque

import numpy as np


class CUSUM:
    def __init__(self, control_limit, mean_shift):
        if control_limit <= 0.:
            raise ValueError("control_limit should greater than 0")

        if mean_shift <= 0.:
            raise ValueError("mean_shift should greater than 0")

        if control_limit < mean_shift:
            raise ValueError("control_limit should be greater than mean_shift")

        self.control_limit = control_limit
        self.mean_shift = mean_shift
        self.cumulative_sum = None

    def validate(self, val: np.ndarray) -> None:
        if self.cumulative_sum is None or np.shape(self.cumulative_sum) != np.shape(val):
            self.cumulative_sum = np.zeros_like(val)

        self.cumulative_sum += np.abs(val) - self.mean_shift
        self.cumulative_sum = np.maximum(self.cumulative_sum, 0.)

    @property
    def test_ratio(self) -> float:
        return self.test_value / self.control_limit

    @property
    def test_ratios(self):
        return self.test_values / self.control_limit

    @property
    def test_value(self):
        return np.max(self.test_values)

    @property
    def test_values(self):
        if self.cumulative_sum is None:
            return 0.
        return self.cumulative_sum


class EWMA:
    def __init__(self, control_limit, alpha, cap=np.inf):
        if control_limit <= 0.:
            raise ValueError("control_limit should greater than 0")

        if alpha <= 0. or alpha >= 1:
            raise ValueError("alpha should between 0 and 1")

        if cap <= 0.:
            raise ValueError("cap should greater than 0")

        if control_limit >= cap:
            raise ValueError("control_limit should be lesser than cap")

        self.control_limit = control_limit
        self.alpha = np.clip(alpha, 0., 1.)
        self.cap = cap
        self.moving_average = None

    def validate(self, val: np.ndarray) -> None:
        if self.moving_average is None or np.shape(self.moving_average) != np.shape(val):
            self.moving_average = np.zeros_like(val)

        if np.isfinite(self.cap) and self.cap >= 0.:
            val = np.clip(val, -self.cap, +self.cap)

        self.moving_average = self.alpha * val + (1. - self.alpha) * self.moving_average

    @property
    def test_ratio(self) -> float:
        return self.test_value / self.control_limit

    @property
    def test_ratios(self):
        return self.test_values / self.control_limit

    @property
    def test_values(self):
        if self.moving_average is None:
            return 0.
        return np.abs(self.moving_average)

    @property
    def test_value(self):
        return np.max(self.test_values)


class L1TW:
    # Note: use for param search only, DO NOT USE THIS IMPLEMENTATION FOR FIRMWARE!
    def __init__(self, control_limit, time_window):
        assert isinstance(time_window, int) and time_window > 0,\
            ValueError("time_window should be an integer and is greater than zero")

        self.control_limit = control_limit
        self.time_window = time_window
        self.deque = deque(maxlen=time_window)
        self.error_deque = deque(maxlen=time_window)
        self.window_sum = None

    def validate(self, val: np.ndarray) -> None:
        if self.window_sum is None:
            self.window_sum = np.zeros_like(val)

        self.deque.append(val)

        offset = 0.0
        if len(self.error_deque) > 0:
            # Use last window mean error as offset
            offset = self.error_deque[0]

        window_array = np.asarray(self.deque)
        if len(self.deque) == self.time_window:
            self.error_deque.append(np.mean(window_array, axis=0))

        self.calculate_window_sum(window_array - offset)

    def calculate_window_sum(self, window_array: np.ndarray):
        self.window_sum = np.sum(np.abs(window_array), axis=0)

    @property
    def test_ratio(self) -> float:
        return self.test_value / self.control_limit

    @property
    def test_ratios(self):
        return self.test_values / self.control_limit

    @property
    def test_value(self):
        return np.max(self.test_values)

    @property
    def test_values(self):
        if self.window_sum is None:
            return 0.
        return np.abs(self.window_sum)


class L2TW(L1TW):
    def calculate_window_sum(self, window_array: np.ndarray):
        window_cusum = np.cumsum(np.square(window_array), axis=0)
        sample_counts = np.arange(1, window_cusum.shape[0] + 1)
        sample_counts = sample_counts.reshape([-1] + [1] * (window_cusum.ndim - 1))
        max_average = np.max(window_cusum / sample_counts, axis=0)
        self.window_sum = max_average
