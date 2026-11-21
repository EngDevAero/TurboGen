"""All Unit convertions."""

import numpy as np
from typing import cast

from .constants import Array1D, Array2D, FloatArrayPair, FloatOrArray, Coords3D


class UnitConverter:
    """
    Unit convertion, call only right after user inputs or right
    before displaying values.
    """

    @staticmethod
    def deg_to_rad(deg: FloatArrayPair) -> FloatArrayPair:
        return cast(FloatArrayPair, np.radians(deg))

    @staticmethod
    def rad_to_deg(rad: FloatArrayPair) -> FloatArrayPair:
        return cast(FloatArrayPair, np.degrees(rad))

    @staticmethod
    def axial_to_m(x: Array1D, r: Array1D) -> Array1D:
        dx, dr = np.diff(x), np.diff(r)
        dm = np.hypot(dx, dr)
        return np.concatenate([[0.0], np.cumsum(dm)])

    @staticmethod
    def m_to_m_prime(m: Array1D, r: Array1D) -> Array1D:
        r_mid = (r[1:] + r[:-1]) / 2
        m_prime_list = np.cumsum(np.diff(m) / r_mid)
        return np.concatenate([[0.0], m_prime_list])

    @staticmethod
    def axial_to_m_prime(x: Array1D, r: Array1D) -> Array1D:
        dm = UnitConverter.axial_to_m(x, r)
        return UnitConverter.m_to_m_prime(m=dm, r=r)


class CoordinateConverter:
    """
    For cylindrical to cartesian convertion, the x-axis and z-axis is swapped
    so that the rotation axis is on z (comment for most tubomachinery
    templates).
    """

    @staticmethod
    def cylindrical_to_cartesian_x(
        r: FloatOrArray, theta: FloatOrArray
    ) -> FloatOrArray:
        return r * np.sin(theta)

    @staticmethod
    def cylindrical_to_cartesian_y(
        r: FloatOrArray, theta: FloatOrArray
    ) -> FloatOrArray:
        return r * np.cos(theta)

    @staticmethod
    def cylindrical_to_cartesian_z(x: FloatOrArray) -> FloatOrArray:
        return x

    @staticmethod
    def cylindrical_to_cartesian_coordinates(
        x: FloatOrArray, r: FloatOrArray, theta: FloatOrArray
    ) -> Coords3D:
        cart_x = CoordinateConverter.cylindrical_to_cartesian_x(r=r, theta=theta)
        cart_y = CoordinateConverter.cylindrical_to_cartesian_y(r=r, theta=theta)
        cart_z = CoordinateConverter.cylindrical_to_cartesian_z(x)
        return cart_x, cart_y, cart_z
