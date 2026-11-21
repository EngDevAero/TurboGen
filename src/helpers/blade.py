"""Blade geometry and angle distributions functions."""

from scipy.interpolate import CubicSpline, PchipInterpolator
from typing import Literal
import numpy as np
import numpy.typing as npt

from .curves import Curves
from .constants import (
    Array1D,
    Array1DList,
    Array2D,
    ImpellerDataOrdered,
    ImpellerDataValOrdered,
)
from .validations import is_arrays_resolution_match


class Blade:
    """
    Parameters
    ----------
    res: Resolution for all impeller contours.

    Attributes
    ----------
    self.res: Resolution for all impeller contours.
    """

    def __init__(self, res: int):
        self.res = res

    @staticmethod
    def constant_beta_distribution(
        m_prime_arr: Array1D, beta: float = np.radians(20)
    ) -> Array1D:
        """Returns an array for the constant beta distribution values across M-Prime."""
        return np.full_like(m_prime_arr, beta)

    @staticmethod
    def theta_distribution_from_beta(
        m_prime_arr: Array1D,
        beta_arr: Array1D,
        theta_start: float,
    ) -> Array1D:
        """
        Compute all the theta values from beta, the calculated theta values represents
        the blade camber.

        Parameters
        ----------
        m_prime_arr:     A list of M-prime vaules.
        beta_arr:  Beta distribution across M-prime (rad).
        theta_start: Theta at M-prime = 0 (rad).
        """
        res = len(beta_arr)
        is_arrays_resolution_match(m_prime_arr, res=res, raise_err=True)

        theta_dtrib = []
        for i in range(res):
            theta = theta_start + np.tan(beta_arr[i]) * m_prime_arr[i]
            theta_dtrib.append(theta)
        return np.array(theta_dtrib)

    @staticmethod
    def constant_blade_thickness_contour(m_prime_arr: Array1D, t_b: float) -> Array1D:
        """Returns an array for the constant blade thickness values across M-Prime."""
        return np.full_like(m_prime_arr, t_b)

    @staticmethod
    def base_SS_profile(
        m_prime_arr: Array1D,
        camber: Array1D,
        r: Array1D,
        t_b_dtrib: Array1D,
    ) -> Array1D:
        """
        Compute the coordinates for the base profile of the suction surface.

        Parameters
        ----------
        m_prime_arr:     M-prime vaules from LE to TE.
        theta_dtrib: Theta distribution across M-prime (rad).
        r:           Radius values array of the component (m).
        t_b_dtrib:   Blade thickness distribution across M-prime (m).
        """
        t_b_dtrib_from_camber = t_b_dtrib / (2 * r)
        return Curves.offset_curve(
            m_prime_arr, camber, offset_dtrib=t_b_dtrib_from_camber
        )[1]

    @staticmethod
    def base_PS_profile(
        m_prime_arr: Array1D,
        camber: Array1D,
        r: Array1D,
        t_b_dtrib: Array1D,
    ) -> Array1D:
        """
        Compute the coordinates for the base profile of the Pressure surface where
        the LE and TE are NOT TRIMMED.

        Parameters
        ----------
        m_prime_arr:     M-prime vaules from LE to TE.
        camber: Theta distribution across M-prime (rad).
        r:           Radius values array of the component (m).
        t_b_dtrib:   Blade thickness distribution across M-prime (m).
        """
        t_b_dtrib_from_camber = -t_b_dtrib / (2 * r)
        return Curves.offset_curve(
            m_prime_arr, camber, offset_dtrib=t_b_dtrib_from_camber
        )[1]

    @staticmethod
    def trim_LE_coordinates(
        i_pt: int,
        SS: Array1D,
        PS: Array1D,
        camber: Array1D,
        m_prime_arr: Array1D,
    ) -> tuple[Array1D, Array1D]:
        """
        Compute the theta values of the leading edge of the base blade profile on both
        the suction surface and pressure surface, returns only the modified leading ss
        and ps theta values.

        Parameters
        ----------
        i_pt:        The point of M-Prime from the LE.
        SS:          Suction surface profile (M-Prime based).
        PS:          Pressure surface profile (M-Prime based).
        t_b_dtrib:   Blade thickness distribution (M based).
        camber: Theta distribution (M-Prime based).
        m_prime_arr:     M-Prime from LE to TE.
        """
        is_arrays_resolution_match(SS, PS, camber, m_prime_arr, raise_err=True)
        m_prime_R = m_prime_arr[i_pt] - m_prime_arr[0]
        t_SS = SS[i_pt] - camber[i_pt]
        t_PS = PS[i_pt] - camber[i_pt]

        LE_SS = Curves.ellipse_tip_coordinates(
            x=m_prime_arr[:i_pt],
            y=camber[:i_pt],
            R=m_prime_R,
            R2=t_SS,
            tip_loc="front",
        )
        LE_PS = Curves.ellipse_tip_coordinates(
            x=m_prime_arr[:i_pt],
            y=camber[:i_pt],
            R=m_prime_R,
            R2=t_PS,
            tip_loc="front",
        )
        return LE_SS, LE_PS

    @staticmethod
    def trim_TE_coordinates(
        i_pt: int,
        SS: Array1D,
        PS: Array1D,
        camber: Array1D,
        m_prime_arr: Array1D,
    ) -> tuple[Array1D, Array1D]:
        """
        Compute the theta values of the trailing edge of the base blade profile on both
        the suction surface and pressure surface, returns only the modified trailing ss
        and ps theta values.

        Parameters
        ----------
        i_pt:        The point of M-Prime from the TE.
        SS:          Suction surface profile (M-Prime based).
        PS:          Pressure surface profile (M-Prime based).
        theta_dtrib: Theta distribution (M-Prime based).
        m_prime_arr:     M-Prime from LE to TE.
        """
        is_arrays_resolution_match(SS, PS, camber, m_prime_arr, raise_err=True)
        i_from_start = len(m_prime_arr) - i_pt
        m_prime_R = m_prime_arr[-1] - m_prime_arr[i_from_start - 1]
        t_SS = SS[i_from_start - 1] - camber[i_from_start - 1]
        t_PS = PS[i_from_start - 1] - camber[i_from_start - 1]

        TE_SS = Curves.ellipse_tip_coordinates(
            x=m_prime_arr[i_from_start:],
            y=camber[i_from_start:],
            R=m_prime_R,
            R2=t_SS,
            tip_loc="end",
        )
        TE_PS = Curves.ellipse_tip_coordinates(
            x=m_prime_arr[i_from_start:],
            y=camber[i_from_start:],
            R=m_prime_R,
            R2=t_PS,
            tip_loc="end",
        )
        return TE_SS, TE_PS

    @staticmethod
    def trim_LE(
        LE_SS: Array1D,
        LE_PS: Array1D,
        SS: Array1D,
        PS: Array1D,
    ):
        """
        Replace the leading edge theta values with the trimmed theta values
        on both suction and pressure surface.
        """
        is_arrays_resolution_match(LE_SS, LE_PS, raise_err=True)
        i_pt = len(LE_SS)
        SS = np.concatenate((LE_SS, SS[i_pt:]))
        PS = np.concatenate((LE_PS, PS[i_pt:]))
        return SS, PS

    @staticmethod
    def trim_TE(
        TE_SS: Array1D,
        TE_PS: Array1D,
        SS: Array1D,
        PS: Array1D,
    ):
        """
        Replace the trailing edge theta values with the trimmed theta values
        on both suction and pressure surface.
        """
        is_arrays_resolution_match(TE_SS, TE_PS, raise_err=True)
        i_from_start = len(SS) - len(TE_SS)
        SS = np.concatenate((SS[:i_from_start], TE_SS))
        PS = np.concatenate((PS[:i_from_start], TE_PS))
        return SS, PS

    @staticmethod
    def sort_blade_surface_lists_by_increasing_span(
        span_frac_arr: Array1D,
        axial_arr_span: Array1DList,
        radial_arr_span: Array1DList,
        camber_arr_span: Array1DList,
        periodic_low_arr_span: Array1DList,
        periodic_high_arr_span: Array1DList,
        SS_arr_span: Array1DList,
        PS_arr_span: Array1DList,
    ) -> ImpellerDataValOrdered:
        """
        Sort the blade surface profile list and relevant blade datas (optional)
        into increasing span fraction order.

        THE SURFACE LISTS MUST MATCH IT'S CORRESPONDING SPAN FRACTION LIST
        IN-ORDER TO SORT THE LISTS CORRECTLY.

        Parameters
        ----------
        span_frac_arr:          List of span fraction values.
        axial_arr_list:         Axial values across span (m).
        radial_arr_list:        Radial values across span (m).
        camber_arr_list:        Camber theta vales across span (rad).
        periodic_low_arr_span:  Periodic boundary negatively away from the camber.
        periodic_high_arr_span: Periodic boundary positvely away from the camber.
        SS_list:                SS theta values across span (rad).
        PS_list:                PS theta values across span (rad).
        """
        is_arrays_resolution_match(
            axial_arr_span,
            radial_arr_span,
            camber_arr_span,
            periodic_low_arr_span,
            periodic_high_arr_span,
            SS_arr_span,
            PS_arr_span,
            raise_err=True,
        )

        order = np.argsort(span_frac_arr)
        span_ordered = np.array(span_frac_arr, dtype=float)[order]
        axial_ordered = np.stack(axial_arr_span)[order]
        radial_ordered = np.stack(radial_arr_span)[order]
        camber_ordered = np.stack(camber_arr_span)[order]
        periodic_low_ordered = np.stack(periodic_low_arr_span)[order]
        periodic_high_ordered = np.stack(periodic_high_arr_span)[order]
        SS_ordered = np.stack(SS_arr_span)[order]
        PS_ordered = np.stack(PS_arr_span)[order]
        return (
            span_ordered,
            axial_ordered,
            radial_ordered,
            camber_ordered,
            periodic_low_ordered,
            periodic_high_ordered,
            SS_ordered,
            PS_ordered,
        )

    @staticmethod
    def spanwise_density(span_arr: Array1D, n_pts: int) -> Array1D:
        """Create the density fraction array shared by every spline."""
        return np.linspace(span_arr[0], span_arr[-1], n_pts)

    @staticmethod
    def spline_across_span(
        span_dens: Array1D,
        span_ordered: Array1D,
        values_ordered: Array2D,
        n_pts: int,
        method: Literal["cubic", "pchip"],
    ) -> Array2D:
        """
        Construct splines across the span direction connecting all SS and PS
        blade profile smoothly.

        Parameters
        ----------
        span_dens:     Span length of every span section (m).
        ss_list:        List of suction surface theta values (rad).
        ps_list:        List of pressure surface theta values (rad).
        span_frac_list: List of span fraction values.
        n_pts:          Set number of points of the spline.
        method:         Either CubicSpline or PchipInterpolator.
        """
        if method == "cubic" and len(span_ordered) < 3:
            raise ValueError("Cubic spline needs at least 3 span fractions")
        if np.any(np.diff(span_ordered) <= 0):
            raise ValueError("'span_sort' must be strictly increasing")

        Interp = CubicSpline if method == "cubic" else PchipInterpolator
        return Interp(span_ordered, values_ordered, axis=0)(span_dens)
