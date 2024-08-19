from enum import Enum, EnumMeta, unique


__all__ = ['AttackType', 'StealthyType']


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class EnumDirectValueMeta(EnumMeta):
    def __getattribute__(cls, name):
        try:
            value = super().__getattribute__(name)
            if isinstance(value, cls):
                value = value.value
            return value
        except AttributeError as err:
            if hasattr(cls, '_missing_'):
                cls._missing_(name)
            else:
                raise err


@unique
class AttackType(ExplicitEnum, metaclass=EnumDirectValueMeta):
    NoAttack: int = 0
    Gyroscope: int = 1
    Accelerometer: int = (1 << 1)
    GpsPosition: int = (1 << 2)
    GpsVelocity: int = (1 << 3)
    Barometer: int = (1 << 4)
    Magnetometer: int = (1 << 5)
    GpsJointPVAttack: int = (1 << 6)


@unique
class StealthyType(ExplicitEnum, metaclass=EnumDirectValueMeta):
    NoStealthy: int = 0
    CumulativeSum: int = (1 << 0)
    ExponentialMovingAverage: int = (1 << 1)
    TimeWindow: int = (1 << 2)
