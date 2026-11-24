"""
Contain all codes during the design process of the impeller, including
meridional design constains, streamline contours, and blade surface
contours.
"""

from typing import Literal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .initial_setup import InitialMeridionalPoints, InitialBladeValues
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
    ImpellerBladeDict,
    Coords2D,
    CurveSettings,
    UnpackedCurveSettings,
    Plot2D,
    Plot2DList,
    AxesFigure2D,
    Coords3D,
    ImpBdaryDmainDict,
)


class ImpellerDesign:
    """
    Parameters
    ----------
    init_pts:       All initial setup points for hub/shroud.
    l_inlet:        Inlet length (m).
    l_outlet:       Outlet length (m).
    init_blade:     All initial setup values for impeller blade.
    theta_start:    Theta at M-prime = 0 (rad).
    shroud_clrance: Shroud clearance (m).
    res:            Resolution for all impeller contours.

    Attributes
    ----------
    self.pt_h1:          x and r values of hub inlet location (m).
    self.pt_h2:          x and r values of hub outlet location (m).
    self.pt_s1:          x and r values of shroud inlet location (m).
    self.pt_s2:          x and r values of shroud outlet location (m).
    self.t_b:            Constant blade thickness (m).
    self.n_blades:       Number of blades.
    self.theta_start:    Theta at LE (rad).
    self.res:            Resolution (number of points) for all impeller contours.
    self.shroud_clrance: Shroud clearance (m).
    self.d_periodic:     Theta value from the camber to periodic boundary (rad).
    self.blade_class:    Impeller blade class.
    self.blade:          Impeller blade design values.
    self.blade_pass_dmain:     Boundary surface values.
    self.inlet_dmain:    Outlet domain values.
    self.outlet_dmain:   Outlet domain values.
    """

    def __init__(
        self,
        init_pts: InitialMeridionalPoints,
        init_blade: InitialBladeValues,
        l_inlet: float,
        l_outlet: float,
        theta_start: float = 0,
        shroud_clrance: float = 0,
        res: int = 500,
        inlet_dmain_res: int = 5,
        outlet_dmain_res: int = 10,
    ):
        # Impeller meridional profile
        self.pt_h1 = init_pts.pt_h1
        self.pt_h2 = init_pts.pt_h2
        self.pt_s1 = init_pts.pt_s1
        self.pt_s2 = init_pts.pt_s2

        # Impeller blade
        self.t_b = init_blade.t_B
        self.n_blades = init_blade.n_blades
        self.theta_start = theta_start
        self.res = res
        self.inlet_dmain_res = inlet_dmain_res
        self.outlet_dmain_res = outlet_dmain_res
        self.shroud_clrance = shroud_clrance
        self.d_periodic = np.pi / self.n_blades

        # Impeller Inlet/Outlet length
        self.l_inlet = l_inlet
        self.l_outlet = l_outlet

        self.blade: ImpellerBladeDict = {}
        self.blade_pass_dmain: ImpBdaryDmainDict = {}
        self.inlet_dmain: ImpBdaryDmainDict = {}
        self.outlet_dmain: ImpBdaryDmainDict = {}

    def blade_values_spanwise(
        self,
        span_frac_list: list = [0, 0.25, 0.5, 0.75, 1],
        i_pt_LE: int | None = None,
        i_pt_TE: int | None = None,
    ) -> None:
        """Compute all the values for the meridional profile."""
        # Streamlines at the set span fractions
        for span_frac in span_frac_list:
            (x_arr, r_arr), curve_settings = self._get_streamline_coordinates(span_frac)
            self._blade_values(
                x_arr=x_arr,
                r_arr=r_arr,
                curve_settings=curve_settings,
                span_frac=str(span_frac),
            )

    def _blade_values(
        self,
        x_arr: Array1D,
        r_arr: Array1D,
        span_frac: str,
        curve_settings: CurveSettings | None = None,
        i_pt_LE: int = 13,
        i_pt_TE: int = 50,
    ) -> None:
        """
        The main function for calculating all the required values for constructing
        the blade. The calculated values will be stored in self.blade to avoid
        calculating the same data again.

        Parameters
        ----------
        x_arr:          Axial values from LE to TE (m).
        r_arr:          Radius values from LE to TE (m).
        span_frac:      Current span fraction.
        curve_settings: Curve settings for recreating the curve.
        i_pt_LE:        Number of points from LE, useful for LE trimming.
        i_pt_TE:        Number of points from TE, useful for TE trimming.
        """
        # M-Prime (LE to TE)
        m_prime_arr = UnitConverter.axial_to_m_prime(x=x_arr, r=r_arr)

        # Angle distribution
        beta_arr = Blade.get_constant_beta_distribution(m_prime_arr)
        camber = Blade.get_theta_distribution_from_beta(
            m_prime_arr=m_prime_arr, beta_arr=beta_arr, theta_start=self.theta_start
        )
        periodic_low = camber - self.d_periodic
        periodic_high = camber + self.d_periodic

        # Thickness distribution
        t_b_dtrib = Blade.get_constant_blade_thickness_distribution(
            m_prime_arr, self.t_b
        )

        # Base blade surface profile
        base_SS = Blade.get_base_SS(
            m_prime_arr=m_prime_arr, camber=camber, r_arr=r_arr, t_b_dtrib=t_b_dtrib
        )
        base_PS = Blade.get_base_PS(
            m_prime_arr=m_prime_arr, camber=camber, r_arr=r_arr, t_b_dtrib=t_b_dtrib
        )

        # Trim blade LE and TE
        LE_SS, LE_PS = Blade.get_trimmed_LE(
            i_pt=i_pt_LE,
            SS=base_SS,
            PS=base_PS,
            camber=camber,
            m_prime_arr=m_prime_arr,
        )
        TE_SS, TE_PS = Blade.get_trimmed_TE(
            i_pt=i_pt_TE,
            SS=base_SS,
            PS=base_PS,
            camber=camber,
            m_prime_arr=m_prime_arr,
        )

        # Update self.blade dictionary
        self._add_blade_data(
            span_frac=span_frac,
            x_arr=x_arr,
            r_arr=r_arr,
            m_prime_arr=m_prime_arr,
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

    def _get_streamline_coordinates(
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

    def _add_blade_data(
        self,
        span_frac: str,
        x_arr: Array1D,
        r_arr: Array1D,
        m_prime_arr: Array1D,
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
        curve_settings: CurveSettings | None,
    ) -> None:
        """
        Add/overwrite the impeller data at the specfied span fraction in this instance
        attribute dictionary self.blade.

        Parameters
        ----------
        span_frac:      Current span fraction.
        x_arr:          Axial values from LE to TE (m).
        r_arr:          Radius values from LE to TE (m).
        m_prime_arr:    M-Prime from LE to TE.
        beta_arr:       Beta distribution from LE to TE (rad).
        camber:         Camber line's theta values from LE to TE (rad).
        periodic_low:   Theta values from LE to TE where it's half way between current
                        and pervious camber (rad).
        periodic_high:  Theta values from LE to TE where it's half way between current
                        and next camber (rad).
        t_b_dtrib:      Blade thickness from LE to TE (m).
        base_SS:        Theta values from LE to TE of the base SS profile (rad).
        base_PS:        Theta values from LE to TE of the base SS profile (rad).
        LE_SS:          Theta values at the LE section of SS (rad).
        LE_PS:          Theta values at the LE section of PS (rad).
        TE_SS:          Theta values at the TE section of SS (rad).
        TE_PS:          Theta values at the TE section of PS (rad).
        curve_settings: Control points, alpha and phi for recreating the curve.
        """
        # Entire blade profile resolution
        is_arrays_resolution_match(
            x_arr,
            r_arr,
            m_prime_arr,
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

        # Add design values
        self.blade[span_frac] = {
            "x_arr": x_arr,
            "r_arr": r_arr,
            "m_prime_arr": m_prime_arr,
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
            "curve_settings": curve_settings if curve_settings else None,
        }

    def blade_passage_domain_values(self):
        """
        The hub wall streamwise contour is same as the streamline contour at
        span fraction at 0.

        If clearance is specfied, the shroud wall will be offset from the streamline
        contour at span fraction at 1.

        The blade tip streamwise contour is same as the streamline contour at
        span fraction at 1.
        """

        def _get_blade_passage_domain_values(surf: Literal["0", "1"]):
            x_arr = self.blade[surf]["x_arr"]
            r_arr = self.blade[surf]["r_arr"]
            mid_pt_arr = self.blade[surf]["camber"]
            periodic_low = self.blade[surf]["periodic_high"]
            periodic_high = self.blade[surf]["periodic_low"]
            return x_arr, r_arr, mid_pt_arr, periodic_low, periodic_high

        x_h_arr, r_h_arr, mid_pt_h_arr, periodic_low_h, periodic_high_h = (
            _get_blade_passage_domain_values("0")
        )
        x_s_arr, r_s_arr, mid_pt_s_arr, periodic_low_s, periodic_high_s = (
            _get_blade_passage_domain_values("1")
        )

        self._add_boundary_domain_values(
            dmain_dict=self.blade_pass_dmain,
            surf="hub_wall",
            x_arr=x_h_arr,
            r_arr=r_h_arr,
            mid_pt_arr=mid_pt_h_arr,
            periodic_low=periodic_low_h,
            periodic_high=periodic_high_h,
        )

        if not self.shroud_clrance:
            self._add_boundary_domain_values(
                dmain_dict=self.blade_pass_dmain,
                surf="shroud_wall",
                x_arr=x_s_arr,
                r_arr=r_s_arr,
                mid_pt_arr=mid_pt_s_arr,
                periodic_low=periodic_low_s,
                periodic_high=periodic_high_s,
            )
        else:
            x_offset_arr, r_offset_arr = Curves.offset_curve(
                x=x_s_arr, y=r_s_arr, offset_dis=self.shroud_clrance
            )
            self._add_boundary_domain_values(
                dmain_dict=self.blade_pass_dmain,
                surf="shroud_wall",
                x_arr=x_offset_arr,
                r_arr=r_offset_arr,
                mid_pt_arr=mid_pt_s_arr,
                periodic_low=periodic_low_s,
                periodic_high=periodic_high_s,
            )

        self._add_boundary_domain_values(
            dmain_dict=self.blade_pass_dmain,
            surf="tip_surf",
            x_arr=x_s_arr,
            r_arr=r_s_arr,
            mid_pt_arr=mid_pt_s_arr,
            periodic_low=periodic_low_s,
            periodic_high=periodic_high_s,
        )

    def inlet_domain_values(self):
        """
        This function assumes the inlet domain extends from the blade inlet in the
        axial direction.
        """

        def _get_inlet_domain_values(
            surf: Literal["hub_wall", "shroud_wall", "tip_surf"],
        ):
            # Axial
            x_1 = np.array(self.blade_pass_dmain[surf]["x_arr"])[0]
            x = x_1 - self.l_inlet
            x_arr = np.linspace(x, x_1, self.inlet_dmain_res)

            # Radial
            r_1 = np.array(self.blade_pass_dmain[surf]["r_arr"])[0]
            r_arr = np.linspace(r_1, r_1, self.inlet_dmain_res)

            # Theta
            mid_pt = np.array(self.blade_pass_dmain[surf]["mid_pt_arr"])[0]
            mid_pt_arr = np.linspace(mid_pt, mid_pt, self.inlet_dmain_res)
            periodic_low = mid_pt_arr - self.d_periodic
            periodic_high = mid_pt_arr + self.d_periodic

            return x_arr, r_arr, mid_pt_arr, periodic_low, periodic_high

        x_h_arr, r_h_arr, mid_pt_h_arr, periodic_low_h, periodic_high_h = (
            _get_inlet_domain_values("hub_wall")
        )
        x_s_arr, r_s_arr, mid_pt_s_arr, periodic_low_s, periodic_high_s = (
            _get_inlet_domain_values("shroud_wall")
        )
        x_t_arr, r_t_arr, mid_pt_t_arr, periodic_low_t, periodic_high_t = (
            _get_inlet_domain_values("tip_surf")
        )

        self._add_boundary_domain_values(
            dmain_dict=self.inlet_dmain,
            surf="hub_wall",
            x_arr=x_h_arr,
            r_arr=r_h_arr,
            mid_pt_arr=mid_pt_h_arr,
            periodic_low=periodic_low_h,
            periodic_high=periodic_high_h,
        )
        self._add_boundary_domain_values(
            dmain_dict=self.inlet_dmain,
            surf="shroud_wall",
            x_arr=x_s_arr,
            r_arr=r_s_arr,
            mid_pt_arr=mid_pt_s_arr,
            periodic_low=periodic_low_s,
            periodic_high=periodic_high_s,
        )
        self._add_boundary_domain_values(
            dmain_dict=self.inlet_dmain,
            surf="tip_surf",
            x_arr=x_t_arr,
            r_arr=r_t_arr,
            mid_pt_arr=mid_pt_t_arr,
            periodic_low=periodic_low_t,
            periodic_high=periodic_high_t,
        )

    def outlet_domain_values(self):
        """
        This function assumes the outlet domain extends from the blade outlet in the
        radial direction.
        """

        def _get_outlet_domain_values(
            surf: Literal["hub_wall", "shroud_wall", "tip_surf"],
        ):
            # Axial
            x_2 = np.array(self.blade_pass_dmain[surf]["x_arr"])[-1]
            x_arr = np.linspace(x_2, x_2, self.outlet_dmain_res)

            # Radial
            r_2 = np.array(self.blade_pass_dmain[surf]["r_arr"])[-1]
            r = r_2 + self.l_outlet
            r_arr = np.linspace(r_2, r, self.outlet_dmain_res)

            # Theta
            mid_pt = np.array(self.blade_pass_dmain[surf]["mid_pt_arr"])[-1]
            mid_pt_arr = np.linspace(mid_pt, mid_pt, self.outlet_dmain_res)
            periodic_low = mid_pt_arr - self.d_periodic
            periodic_high = mid_pt_arr + self.d_periodic

            return x_arr, r_arr, mid_pt_arr, periodic_low, periodic_high

        x_h_arr, r_h_arr, mid_pt_h_arr, periodic_low_h, periodic_high_h = (
            _get_outlet_domain_values("hub_wall")
        )
        x_s_arr, r_s_arr, mid_pt_s_arr, periodic_low_s, periodic_high_s = (
            _get_outlet_domain_values("shroud_wall")
        )
        x_t_arr, r_t_arr, mid_pt_t_arr, periodic_low_t, periodic_high_t = (
            _get_outlet_domain_values("tip_surf")
        )

        self._add_boundary_domain_values(
            dmain_dict=self.outlet_dmain,
            surf="hub_wall",
            x_arr=x_h_arr,
            r_arr=r_h_arr,
            mid_pt_arr=mid_pt_h_arr,
            periodic_low=periodic_low_h,
            periodic_high=periodic_high_h,
        )
        self._add_boundary_domain_values(
            dmain_dict=self.outlet_dmain,
            surf="shroud_wall",
            x_arr=x_s_arr,
            r_arr=r_s_arr,
            mid_pt_arr=mid_pt_s_arr,
            periodic_low=periodic_low_s,
            periodic_high=periodic_high_s,
        )
        self._add_boundary_domain_values(
            dmain_dict=self.outlet_dmain,
            surf="tip_surf",
            x_arr=x_t_arr,
            r_arr=r_t_arr,
            mid_pt_arr=mid_pt_t_arr,
            periodic_low=periodic_low_t,
            periodic_high=periodic_high_t,
        )

    def _add_boundary_domain_values(
        self,
        dmain_dict,
        surf: str,
        x_arr: Array1D,
        r_arr: Array1D,
        mid_pt_arr: Array1D,
        periodic_low: Array1D,
        periodic_high: Array1D,
    ):
        """
        Parameters
        ----------
        surf:          Surface name.
        x_arr:         Axial values from LE to TE (m).
        r_arr:         Radius values from LE to TE (m).
        mid_pt_arr:    Theta values at the mid point from LE to TE (rad).
        periodic_low:  Theta values from LE to TE where it's half way between current
                       and pervious camber (rad).
        periodic_high: Theta values from LE to TE where it's half way between current
                       and next camber (rad).
        """
        dmain_dict[surf] = {
            "x_arr": x_arr,
            "r_arr": r_arr,
            "mid_pt_arr": mid_pt_arr,
            "periodic_low": periodic_low,
            "periodic_high": periodic_high,
        }

    def plot_meridional_profiles(self) -> None:
        fig, ax = MeridionalPlots.create_meridional_figure()

        for key, value in self.blade.items():
            if key == "1":
                MeridionalPlots.plot_meridional_contour(
                    x_arr=value["x_arr"],
                    r_arr=value["r_arr"],
                    ax=ax,
                    linestyle="solid",
                    linewidth=1,
                    color="black",
                )
            elif key == "0":
                continue
            else:
                MeridionalPlots.plot_meridional_contour(
                    x_arr=value["x_arr"],
                    r_arr=value["r_arr"],
                    ax=ax,
                    linestyle="solid",
                    linewidth=1,
                    color="grey",
                )

        # Hub wall
        MeridionalPlots.plot_meridional_contour(
            x_arr=self.inlet_dmain["hub_wall"]["x_arr"],
            r_arr=self.inlet_dmain["hub_wall"]["r_arr"],
            ax=ax,
            linestyle="solid",
            linewidth=1.5,
            color="black",
        )
        MeridionalPlots.plot_meridional_contour(
            x_arr=self.blade_pass_dmain["hub_wall"]["x_arr"],
            r_arr=self.blade_pass_dmain["hub_wall"]["r_arr"],
            ax=ax,
            linestyle="solid",
            linewidth=1.5,
            color="black",
        )
        MeridionalPlots.plot_meridional_contour(
            x_arr=self.outlet_dmain["hub_wall"]["x_arr"],
            r_arr=self.outlet_dmain["hub_wall"]["r_arr"],
            ax=ax,
            linestyle="solid",
            linewidth=1.5,
            color="black",
        )

        # Shroud wall
        MeridionalPlots.plot_meridional_contour(
            x_arr=self.inlet_dmain["shroud_wall"]["x_arr"],
            r_arr=self.inlet_dmain["shroud_wall"]["r_arr"],
            ax=ax,
            linestyle="solid",
            linewidth=1.5,
            color="black",
        )
        MeridionalPlots.plot_meridional_contour(
            x_arr=self.blade_pass_dmain["shroud_wall"]["x_arr"],
            r_arr=self.blade_pass_dmain["shroud_wall"]["r_arr"],
            ax=ax,
            linestyle="solid",
            linewidth=1.5,
            color="black",
        )
        MeridionalPlots.plot_meridional_contour(
            x_arr=self.outlet_dmain["shroud_wall"]["x_arr"],
            r_arr=self.outlet_dmain["shroud_wall"]["r_arr"],
            ax=ax,
            linestyle="solid",
            linewidth=1.5,
            color="black",
        )

        self._plot_meridional_profile_boundary(ax)
        ax.grid()

    def _plot_meridional_profile_boundary(self, ax: AxesFigure2D) -> None:
        MeridionalPlots.plot_meridional_component_axis(
            x_arr=self.blade["0"]["x_arr"], ax=ax
        )
        MeridionalPlots.plot_inlet_normal(self.pt_h1, self.pt_s1, ax=ax)
        MeridionalPlots.plot_outlet_normal(self.pt_h2, self.pt_s2, ax=ax)

    def plot_angle_distribution(self, span_frac: float) -> None:
        fig, ax = BladePlots2D.create_angle_distribution_figure()
        span_frac_str = str(span_frac)
        imp_design = self.blade[span_frac_str]
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

    def plot_blade_profile(
        self,
        span_frac: float,
    ) -> Plot2DList | None:
        fig, ax = BladePlots2D.create_blade_profile_figure()
        span_frac_str = str(span_frac)
        imp_design = self.blade[span_frac_str]
        SS, PS = Blade.trim_LE(
            LE_SS=imp_design["LE_SS"],
            LE_PS=imp_design["LE_PS"],
            SS=imp_design["base_SS"],
            PS=imp_design["base_PS"],
        )
        SS, PS = Blade.trim_TE(
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
