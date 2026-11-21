"""
Contain all codes during the design process of the impeller, including
meridional design constains, streamline contours, and blade surface
contours.
"""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .initial_setup import InitialMeridionalPoints, InitialBladeValues, InitialLayers
from src.helpers import (
    Curves,
    MeridionalPlots,
    BladePlots2D,
    Blade,
    UnitConverter,
    is_arrays_resolution_match,
)
from src.helpers.constants import (
    Point,
    Array1D,
    ImpellerDesignData,
    ImpellerDesignDict,
    Coords2D,
    CurveSettings,
    UnpackedCurveSettings,
    Plot2D,
    Plot2DList,
    AxesFigure2D,
    Coords3D,
)


class ImpellerDesign:
    """
    Parameters
    ----------
    init_pts:    All initial setup points for hub/shroud.
    init_blade:  All initial setup values for impeller blade.
    theta_start: Theta at M-prime = 0 (rad).
    res:         Resolution for all impeller contours.

    Attributes
    ----------
    self._init_pts:   All initial setup points for hub/shroud.
    self._init_blade: All initial setup values for impeller blade.
    self.pt_h1:       x and r values of hub inlet location (m).
    self.pt_h2:       x and r values of hub outlet location (m).
    self.pt_s1:       x and r values of shroud inlet location (m).
    self.pt_s2:       x and r values of shroud outlet location (m).
    self.wrap_ang:    Nominal wrap angle from LE to TE (rad).
    self.t_b:         Blade thickness (m).
    self.n_blades:    Number of blades.
    self.d_periodic:  The theta value from the camber to periodic boundary (rad).
    self.theta_start: Theta at M-prime = 0 (rad).
    self.res:         Resolution (number of points) for all impeller contours.
    self.blade:       Impeller blade class.
    self.imp_design:  Dictionary containing data about impeller contour lines.
    self.hub_wall:    x, r values of the hub wall (m).
    self.shroud_wall: x, r values of the shroud wall (m).
    """

    def __init__(
        self,
        init_pts: InitialMeridionalPoints,
        init_blade: InitialBladeValues,
        theta_start: float = 0,
        res: int = 500,
    ):
        # Initial setup values
        self._init_pts = init_pts
        self._init_blade = init_blade

        # Impeller meridional profile
        self.pt_h1 = self._init_pts.pt_h1
        self.pt_h2 = self._init_pts.pt_h2
        self.pt_s1 = self._init_pts.pt_s1
        self.pt_s2 = self._init_pts.pt_s2

        # Impeller blade
        self.wrap_ang = self._init_blade.wrap_ang
        self.t_b = self._init_blade.t_B
        self.n_blades = self._init_blade.n_blades
        self.d_periodic = np.pi / self.n_blades

        self.theta_start = theta_start
        self.res = res

        self.blade = Blade(res=self.res)

        self.imp_design: ImpellerDesignDict = {}
        self.hub_wall: Coords3D = (np.array(0), np.array(0), np.array(0))
        self.shroud_wall: Coords3D = (np.array(0), np.array(0), np.array(0))

    def compute_profiles_at_span_fractions(
        self, span_frac_list: list = [0, 0.25, 0.5, 0.75, 1]
    ) -> None:
        """Compute all the values for the meridional profile."""
        # Streamlines at the set span fractions
        plots = []
        coords_2d_list = []
        for span_frac in span_frac_list:
            coords_2d_list.append(
                self.compute_impeller_data_at_span_fraction(span_frac)
            )

    def compute_impeller_data_at_span_fraction(
        self, span_frac: float, i_pt_LE: int = 13, i_pt_TE: int = 50
    ) -> Coords2D:
        """
        The main function for calculating all the component data that is required
        for it's construction, store all the component data in this instance
        attribute dictionary self.imp_design to avoid calculating the same
        data again.

        Parameters
        ----------
        span_frac: Current span fraction.
        i_pt_LE:   Number of points from LE, useful for LE trimming.
        i_pt_TE:   Number of points from TE, useful for TE trimming.
        """

        (x, r), curve_settings = self.estimate_streamline_coordinates(span_frac)

        # M-Prime (LE to TE)
        m_prime_arr = UnitConverter.axial_to_m_prime(x=x, r=r)

        # Angle distribution
        beta_arr = self.blade.constant_beta_distribution(m_prime_arr)
        camber = self.blade.theta_distribution_from_beta(
            m_prime_arr=m_prime_arr, beta_arr=beta_arr, theta_start=self.theta_start
        )
        periodic_low = camber - self.d_periodic
        periodic_high = camber + self.d_periodic

        # Thickness distribution
        t_b_dtrib = self.blade.constant_blade_thickness_contour(m_prime_arr, self.t_b)

        # Base blade surface profile
        base_SS = self.blade.base_SS_profile(
            m_prime_arr=m_prime_arr, camber=camber, r=r, t_b_dtrib=t_b_dtrib
        )
        base_PS = self.blade.base_PS_profile(
            m_prime_arr=m_prime_arr, camber=camber, r=r, t_b_dtrib=t_b_dtrib
        )

        # Trim blade LE and TE
        LE_SS, LE_PS = self.blade.trim_LE_coordinates(
            i_pt=i_pt_LE,
            SS=base_SS,
            PS=base_PS,
            camber=camber,
            m_prime_arr=m_prime_arr,
        )
        TE_SS, TE_PS = self.blade.trim_TE_coordinates(
            i_pt=i_pt_TE,
            SS=base_SS,
            PS=base_PS,
            camber=camber,
            m_prime_arr=m_prime_arr,
        )

        self._update_impeller_dictionary_at_span_fraction(
            span_frac=span_frac,
            axial=x,
            m_prime_arr=m_prime_arr,
            radial=r,
            beta_arr=beta_arr,
            camber=camber,
            periodic_low=periodic_low,
            periodic_high=periodic_high,
            t_b_dtrib=t_b_dtrib,
            base_SS=base_SS,
            base_PS=base_PS,
            LE_SS=LE_SS,
            LE_PS=LE_PS,
            TE_SS=TE_SS,
            TE_PS=TE_PS,
            curve_settings=curve_settings,
        )
        return x, r

    def estimate_streamline_coordinates(
        self, span_frac: float
    ) -> tuple[Coords2D, CurveSettings]:
        """
        Estimate the control points for the three point cubic spline and
        generate the coordinates for the spline.
        """

        def _interpolate_point(span_frac, pt_h: Point, pt_s: Point):
            pt_h_arr, pt_s_arr = np.array(pt_h, dtype=float), np.array(
                pt_s, dtype=float
            )
            x, r = span_frac * (pt_s_arr - pt_h_arr) + pt_h_arr
            return float(x), float(r)

        # Starting and ending point of the streamline from span fraction
        pt1 = _interpolate_point(span_frac, pt_h=self.pt_h1, pt_s=self.pt_s1)
        pt3 = _interpolate_point(span_frac, pt_h=self.pt_h2, pt_s=self.pt_s2)

        # Estimate control points and generate contour coordinates
        curve_settings = Curves.estimate_three_point_spline_inputs(pt1=pt1, pt3=pt3)
        return (
            Curves.three_point_cubic_spline_coordinates(
                pt1=curve_settings["pt1"],
                pt2=curve_settings["pt2"],
                pt3=curve_settings["pt3"],
                alpha1=curve_settings["alpha1"],
                alpha3=curve_settings["alpha3"],
                phi=curve_settings["phi"],
                n_pts=self.res,
            ),
            curve_settings,
        )

    def _update_impeller_dictionary_at_span_fraction(
        self,
        span_frac: float,
        axial: Array1D,
        m_prime_arr: Array1D,
        radial: Array1D,
        beta_arr: Array1D,
        camber: Array1D,
        periodic_low: Array1D,
        periodic_high: Array1D,
        t_b_dtrib: Array1D,
        base_SS: Array1D,
        base_PS: Array1D,
        LE_SS: Array1D,
        LE_PS: Array1D,
        TE_SS: Array1D,
        TE_PS: Array1D,
        curve_settings: CurveSettings,
    ) -> None:
        """
        Add/overwrite the impeller data at the specfied span fraction in this instance
        attribute dictionary self.imp_design.

        Parameters
        ----------
        span_frac:      Current span fraction.
        axial:          Axial from LE to TE (m).
        m_prime_arr:        M-Prime from LE to TE.
        radial:         Radius across the axial direction (m).
        beta_arr:     Beta distribution across M-Prime (rad).
        camber:         Camber line's theta values across M-Prime (rad).
        t_b_dtrib:      Blade thickness distribution (m).
        base_SS:        Base SS profile (rad).
        base_PS:        Base SS profile (rad).
        LE_SS:          SS theta values at the LE section (rad).
        LE_PS:          PS theta values at the LE section (rad).
        TE_SS:          SS theta values at the TE section (rad).
        TE_PS:          PS theta values at the TE section (rad).
        curve_settings: Control points, alpha and phi for recreating the curve.
        """
        # Entire blade profile resolution
        is_arrays_resolution_match(
            axial,
            m_prime_arr,
            radial,
            beta_arr,
            camber,
            t_b_dtrib,
            base_SS,
            base_PS,
            raise_err=True,
        )

        # Trim edges resolution
        is_arrays_resolution_match(LE_SS, LE_PS, raise_err=True)
        is_arrays_resolution_match(TE_SS, TE_PS, raise_err=True)

        # Save design data
        self.imp_design[str(span_frac)] = {
            "axial": axial,
            "m_prime_arr": m_prime_arr,
            "radial": radial,
            "beta_arr": beta_arr,
            "camber": camber,
            "periodic_low": periodic_low,
            "periodic_high": periodic_high,
            "t_b_dtrib": t_b_dtrib,
            "base_SS": base_SS,
            "base_PS": base_PS,
            "LE_SS": LE_SS,
            "LE_PS": LE_PS,
            "TE_SS": TE_SS,
            "TE_PS": TE_PS,
            "curve_settings": curve_settings,
        }

    def set_hub_wall_streamwise_coordinates(self):
        """
        The hub wall streamwise contour is same as the streamline contour at
        span fraction at 0.
        """
        x_arr = self.imp_design["0"]["axial"]
        r_arr = self.imp_design["0"]["radial"]
        theta_arr = (
            self.imp_design["0"]["periodic_high"] - self.imp_design["0"]["periodic_low"]
        )
        self.hub_wall = x_arr, r_arr, theta_arr

    def set_shroud_wall_streamwise_coordinates(self, d_clearance: float | None = None):
        """
        If clearance is specfied, the shroud wall will be offset from the streamline
        contour at span fraction at 1.
        """
        x_arr = self.imp_design["1"]["axial"]
        r_arr = self.imp_design["1"]["radial"]
        theta_arr = (
            self.imp_design["1"]["periodic_high"] - self.imp_design["0"]["periodic_low"]
        )

        if not d_clearance:
            self.shroud_wall = x_arr, r_arr, theta_arr
        else:
            x_offset_arr, r_offset_arr = Curves.offset_curve(
                x=x_arr, y=r_arr, offset_dis=d_clearance
            )
            self.shroud_wall = x_offset_arr, r_offset_arr, theta_arr

    def plot_all_meridional_profiles(self) -> None:
        """
        Plot all the streamlines at span fraction profile stored in
        self.imp_design.
        """
        fig, ax = MeridionalPlots.create_meridional_figure()

        for key, value in self.imp_design.items():
            MeridionalPlots.plot_meridional_contour(
                x_arr=value["axial"], r_arr=value["radial"], span_frac=key, ax=ax
            )
        self._plot_meridional_profile_boundary(ax)
        ax.grid()
        fig.legend()

    def _plot_meridional_profile_boundary(self, ax: AxesFigure2D) -> None:
        """Plot the boundary of the meridional profile."""
        # Boundary lines
        MeridionalPlots.plot_meridional_component_axis(
            x_arr=self.imp_design["0"]["axial"], ax=ax
        )
        MeridionalPlots.plot_inlet_normal(self.pt_h1, self.pt_s1, ax=ax)
        MeridionalPlots.plot_outlet_normal(self.pt_h2, self.pt_s2, ax=ax)

    def plot_angle_distribution_at_span_fraction(self, span_frac: float) -> None:
        """
        Plot the beta distribution and the blade camber across M-prime (LE to TE)
        at the specfied span fraction using the data stored in self.imp_design.
        """
        fig, ax = BladePlots2D.create_angle_distribution_figure()

        span_frac_str = str(span_frac)
        imp_design = self.imp_design[span_frac_str]
        BladePlots2D.plot_beta_distribution_at_span_fraction(
            m_prime_arr=imp_design["m_prime_arr"],
            beta_arr=imp_design["beta_arr"],
            span_frac=span_frac_str,
            ax=ax,
        )
        BladePlots2D.plot_blade_camber_at_span_fraction(
            m_prime_arr=imp_design["m_prime_arr"],
            camber=imp_design["camber"],
            span_frac=span_frac_str,
            ax=ax,
        )
        ax.grid()
        fig.legend()

    def plot_blade_profile_at_span_fraction(
        self,
        span_frac: float,
    ) -> Plot2DList | None:
        """
        Plot the theta distribution for blade camber and surface across M-prime (LE to TE)
        at the specfied span fraction using the data stored in self.imp_design.

        The applied trimmed edges is only used for plotting and won't overwrite the
        theta values of the base blade surface profiles in self.imp_design.
        """
        fig, ax = BladePlots2D.create_blade_profile_figure()

        span_frac_str = str(span_frac)
        imp_design = self.imp_design[span_frac_str]
        SS, PS = self.blade.trim_LE(
            LE_SS=imp_design["LE_SS"],
            LE_PS=imp_design["LE_PS"],
            SS=imp_design["base_SS"],
            PS=imp_design["base_PS"],
        )
        SS, PS = self.blade.trim_TE(
            TE_SS=imp_design["TE_SS"], TE_PS=imp_design["TE_PS"], SS=SS, PS=PS
        )

        BladePlots2D.plot_blade_profile_at_span_fraction(
            m_prime_arr=imp_design["m_prime_arr"],
            camber=imp_design["camber"],
            base_SS=imp_design["base_SS"],
            base_PS=imp_design["base_PS"],
            SS=SS,
            PS=PS,
            span_frac=span_frac_str,
            ax=ax,
        )

        ax.grid()
        fig.legend()
