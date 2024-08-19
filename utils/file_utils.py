from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd


__all__ = ['ParameterSet', 'ExperimentParamSet', 'GPSSpoofingParams',
           'SensorData', 'StateData', 'ErrorRatioData', 'TestCaseConfig']


@dataclass()
class ParameterSet(OrderedDict):
    """
    An OrderedDict that can call its key by class attribute.
    Examples:
        @dataclass
        class SomeOutput(ParameterSet):
            OutputA: float
            OutputB: int = None

        output = SomeOutput(1.0)
        print(output.OutputA)  # will print 1.0, equivalent to od['OutputA']
    """
    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        # assert all(
        #     field.default is None for field in class_fields[1:]
        # ), f"{self.__class__.__name__} should not have more than one required field."

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any, ...]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class GPSSpoofingParams(ParameterSet):
    SENS_GPS_DEV_IV: float = None
    SENS_GPS_DEV_ROR: float = None
    SENS_GPS_DEV_MAX: float = None
    SENS_GPS_DEV_HDG: float = None
    SENS_GPS_DEV_AOA: float = None


@dataclass
class ExperimentParamSet(ParameterSet):
    PRE_FLIGHT: dict = None
    AT_FLIGHT_BEGIN: dict = None
    AT_FLIGHT_END: dict = None
    POST_FLIGHT: dict = None


@dataclass
class SensorData(ParameterSet):
    gps: list = None
    barometer: list = None
    magnetometer: list = None
    accelerometer: list = None
    gyroscope: list = None


@dataclass
class ErrorRatioData(ParameterSet):
    gps_position: list = None
    gps_velocity: list = None
    barometer: list = None
    magnetometer: list = None
    accelerometer: list = None
    gyroscope: list = None


@dataclass
class StateData(ParameterSet):
    position: pd.DataFrame = None
    velocity: pd.DataFrame = None
    attitude: pd.DataFrame = None
    angular_rate: pd.DataFrame = None


@dataclass()
class TestCaseConfig(ParameterSet):
    firmware_path: str
    mission_file_path: str
    topic_file_path: str
    log_export_path: str
    init_param_path: str
    detector_param_path: Optional[str] = None
    attack_param_dir: Optional[str] = None
    attack_waypoint: Optional[int] = None
    max_deviation: Optional[float] = None


@dataclass()
class RotorModelParam(ParameterSet):
    propeller_position: np.ndarray
    propeller_axis: np.ndarray
    spin_direction: np.ndarray
