"""
Contains all functions for calculating the coordinates and plotting the 3D
impeller.

The generation process requires importing the design data from the impeller
design process's instance dictionary, and some pre-processing before it can
be used for the calculation of the 3D Cartesian geometry.
"""

from inspect import stack
from math import radians
from typing import Literal, Sequence
from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np

from .design import ImpellerDesign
from .initial_setup import InitialMeridionalPoints, InitialBladeValues
from src.helpers import (
    MeridionalPlots,
    BladePlots2D,
    Blade,
    CoordinateConverter,
    BladePlots3D,
    is_arrays_resolution_match,
)
from src.helpers.constants import (
    ImpellerBladeData,
    ImpellerBladeDict,
    RotationDirect,
    ImpCoords2D,
    FullImp3D,
    ImpDesignVals,
    Array1D,
    Array1DList,
    Array2D,
    Coords2D,
    Coords3D,
)


class ImpellerGeneration:
    """
    Parameters
    ----------
    impeller_dict: Impeller design dictionary.

    Attributes
    ----------
    self.grid_dens:       Grid density of the 3D geometry.
    self.theta_direct:    Rotational direction of theta.
    self.imp_design:      Impeller design class.
    self.d_periodic:      Theta value from the camber to periodic boundary (rad).
    self.n_blades:        Number of blades of the impeller.
    self.bdary_surf:      Boundary surfaces.
    self.imp_design_vals: Imported design data from the impeller design process instance's
                          dictionary with LE and TE trimming applied, that is ready for
                          pre-processing prior the 3D geometry generation.
    self.imp_gen_val:     Processed and ordered cylindrical coordinate values that is ready
                          for the converstion to Cartesian coordinate values.
    self.imp_3d:          3D cartesian coordinates of the blade.
    self.imp_2d:          2D coordinates of the impeller.
    self.hub_wall:        3D coordinates of the hub wall.
    self.shroud_wall:     3D coordinates of the shroud wall.
    self.tip_surface:     3D coordinates of the tip surface.
    """

    def __init__(
        self,
        grid_dens: int,
        theta_direct: RotationDirect,
        imp_design: ImpellerDesign,
    ):
        self.grid_dens = grid_dens
        self.theta_direct = theta_direct

        # From ImpellerDesign class
        self.imp_design = imp_design
        self.d_periodic = self.imp_design.d_periodic
        self.n_blades = self.imp_design.n_blades
        self.bdary_surf = self.imp_design.bdary_surf

        self.imp_design_vals: ImpDesignVals = {}
        self._load_blade_design_values(self.imp_design.blade)
        self.imp_gen_val: ImpCoords2D = {}
        self.imp_3d: FullImp3D = {}

    def _load_blade_design_values(self, blade_design_val: ImpellerBladeDict) -> None:
        """
        Apply blade edges trims and load all the necessary data from the
        impeller design dictionary to this instance attribute dictionary
        self.imp_design_vals.

        ONLY CALL ONCE at the start.
        """
        exclude_keys = (
            "base_SS",
            "base_PS",
            "LE_SS",
            "LE_PS",
            "TE_SS",
            "TE_PS",
            "curve_settings",
        )
        for key, value in blade_design_val.items():
            SS, PS = self._apply_LE_TE_trims(value)
            if key in exclude_keys:
                continue
            self.imp_design_vals[key] = {
                "x_arr": value["x_arr"],
                "r_arr": value["r_arr"],
                "m_prime_arr": value["m_prime_arr"],
                "beta_arr": value["beta_arr"],
                "camber": value["camber"],
                "periodic_low": value["periodic_low"],
                "periodic_high": value["periodic_high"],
                "t_b_dtrib": value["t_b_dtrib"],
                "SS": SS,
                "PS": PS,
            }

    def _apply_LE_TE_trims(self, value: ImpellerBladeData) -> tuple[Array1D, Array1D]:
        """
        Create blade surface profile for SS and PS and store them into self.imp_design_vals.

        ONLY CALL ONCE in self._load_impeller_data().
        """
        SS, PS = Blade.trim_LE(
            LE_SS=value["LE_SS"],
            LE_PS=value["LE_PS"],
            SS=value["base_SS"],
            PS=value["base_PS"],
        )
        SS, PS = Blade.trim_TE(
            TE_SS=value["TE_SS"],
            TE_PS=value["TE_PS"],
            SS=SS,
            PS=PS,
        )
        return SS, PS

    def spline_spanwise(
        self,
        method: Literal["cubic", "pchip"] = "cubic",
    ) -> None:
        """
        Create spline across the span of the blade so the blade surface is smooth.

        method: Either CubicSpline or PchipInterpolator.
        """
        span_ordered = self.imp_2d["span_dens"]
        span_dens = Blade.get_spanwise_density(span_ordered, self.grid_dens)

        def _spline(name: str) -> Array2D:
            return Blade.get_spline_across_span(
                span_dens=span_dens,
                span_ordered=span_ordered,
                values_ordered=self.imp_2d[name],
                n_pts=self.grid_dens,
                method=method,
            )

        self.imp_gen_val["span_dens"] = span_dens
        self.imp_gen_val["x_arr_ordered"] = _spline("x_arr_ordered")
        self.imp_gen_val["r_arr_ordered"] = _spline("r_arr_ordered")
        self.imp_gen_val["camber_ordered"] = _spline("camber_ordered")
        self.imp_gen_val["periodic_low_ordered"] = _spline("periodic_low_ordered")
        self.imp_gen_val["periodic_high_ordered"] = _spline("periodic_high_ordered")
        self.imp_gen_val["SS_ordered"] = _spline("SS_ordered")
        self.imp_gen_val["PS_ordered"] = _spline("PS_ordered")

    def pre_process_design_values(self) -> None:
        """
        Retrieve all the values from created blade profile across span and
        sort value arrays by increasing span.
        """
        # Append to value list for every span fraction
        span_frac_list: list[float] = []
        x_arr_span: Array1DList = []
        r_arr_span: Array1DList = []
        camber_span: Array1DList = []
        periodic_low_span: Array1DList = []
        periodic_high_span: Array1DList = []
        SS_span: Array1DList = []
        PS_span: Array1DList = []

        for key, value in sorted(
            self.imp_design_vals.items(), key=lambda kv: float(kv[0])
        ):
            span_frac_list.append(float(key))
            x_arr_span.append(value["x_arr"])
            r_arr_span.append(value["r_arr"])
            camber_span.append(value["camber"])
            periodic_low_span.append(value["periodic_low"])
            periodic_high_span.append(value["periodic_high"])
            SS_span.append(value["SS"])
            PS_span.append(value["PS"])
        span_frac_arr = np.array(span_frac_list, dtype=float)

        self.imp_2d = Blade.sort_blade_surface_lists_by_increasing_span(
            span_frac_arr=span_frac_arr,
            x_arr_span=x_arr_span,
            r_arr_span=r_arr_span,
            camber_span=camber_span,
            periodic_low_span=periodic_low_span,
            periodic_high_span=periodic_high_span,
            SS_span=SS_span,
            PS_span=PS_span,
        )

    def boundary_surfaces(self):
        """Calculate the cylindrical coordinates for both the hub and shroud wall."""
        self.hub_wall = self._get_boundary_surface_coordinates("hub_wall")
        self.tip_surface = self._get_boundary_surface_coordinates("tip_surface")
        self.shroud_wall = self._get_boundary_surface_coordinates("shroud_wall")

    def _get_boundary_surface_coordinates(
        self, surface: Literal["hub_wall", "tip_surface", "shroud_wall"]
    ) -> Coords3D:
        """
        Calculate the cylindrical coordinates of the end wall surface that is
        set at span fraction.
        """
        # Retrieve coordinates at span fraction
        high = self.bdary_surf[surface]["periodic_high"]
        low = self.bdary_surf[surface]["periodic_low"]
        x_arr = self.bdary_surf[surface]["x_arr"]
        r_arr = self.bdary_surf[surface]["r_arr"]

        t = np.linspace(0, 1, self.grid_dens)[:, None]
        theta = low[None, :] + t * (high - low)[None, :]
        x = np.broadcast_to(x_arr[None, :], theta.shape)
        r = np.broadcast_to(r_arr[None, :], theta.shape)
        return x, r, theta

    def convert_to_cartesian_3d(self) -> None:
        """
        Convert the cylindrical coordinates of the impeller to 3D Cartesian, and
        store values in self.imp_3d.
        """
        for n_blade in range(self.n_blades):
            self.imp_3d[str(n_blade)] = {}
            pitch = n_blade * self.d_periodic * 2
            self._blade_surface_to_cartesian_3D(n_blade, pitch)
            self._blade_end_walls_to_cartesian_3D(n_blade, pitch)
            self._blade_tip_to_cartesian_3D(n_blade, pitch)
            self._periodic_to_cartesian_3D(n_blade, pitch)
            self._shroud_periodic_to_cartesian_3D(n_blade, pitch)

    def _blade_surface_to_cartesian_3D(self, n, pitch):
        self.imp_3d[str(n)]["SS"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct * (self.imp_gen_val["SS_ordered"] + pitch),
            )
        )
        self.imp_3d[str(n)]["PS"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct * (self.imp_gen_val["PS_ordered"] + pitch),
            )
        )

    def _blade_tip_to_cartesian_3D(self, n, pitch):
        self.imp_3d[str(n)]["tip_surface"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.tip_surface[0],
                r=self.tip_surface[1],
                theta=self.theta_direct * (self.tip_surface[2] + pitch),
            )
        )

    def _blade_end_walls_to_cartesian_3D(self, n, pitch):
        self.imp_3d[str(n)]["hub_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.hub_wall[0],
                r=self.hub_wall[1],
                theta=self.theta_direct * (self.hub_wall[2] + pitch),
            )
        )
        self.imp_3d[str(n)]["shroud_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.shroud_wall[0],
                r=self.shroud_wall[1],
                theta=self.theta_direct * (self.shroud_wall[2] + pitch),
            )
        )

    def _periodic_to_cartesian_3D(self, n, pitch):
        self.imp_3d[str(n)]["periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct
                * (self.imp_gen_val["periodic_low_ordered"] + pitch),
            )
        )
        self.imp_3d[str(n)]["periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct
                * (self.imp_gen_val["periodic_high_ordered"] + pitch),
            )
        )

    def _shroud_periodic_to_cartesian_3D(self, n_blade, pitch):
        def _get_shroud_periodic_surface(
            periodic_loc: Literal["periodic_low", "periodic_high"],
        ) -> Coords3D:
            """
            The shroud periodic surface starts from the edge of the periodic surface
            at blade tip radius extending to the shroud edge parrallel to it on either
            sides.
            """
            i = 0 if periodic_loc == "periodic_low" else -1

            x_arr = np.stack(
                (
                    np.array(self.tip_surface[0])[i],
                    np.array(self.shroud_wall[0])[i],
                )
            )
            r_arr = np.stack(
                (
                    np.array(self.tip_surface[1])[i],
                    np.array(self.shroud_wall[1])[i],
                )
            )
            theta_arr = self.theta_direct * np.stack(
                (
                    np.array(self.tip_surface[2])[i],
                    np.array(self.shroud_wall[2])[i],
                )
            )
            return x_arr, r_arr, theta_arr

        x_low, r_low, theta_low = _get_shroud_periodic_surface("periodic_low")
        x_high, r_high, theta_high = _get_shroud_periodic_surface("periodic_high")

        self.imp_3d[str(n_blade)]["shroud_periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=x_low,
                r=r_low,
                theta=theta_low + pitch,
            )
        )
        self.imp_3d[str(n_blade)]["shroud_periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=x_high,
                r=r_high,
                theta=theta_high + pitch,
            )
        )

    def plot_spline_spanwise(
        self,
        res: int,
        stream_frac_list: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> None:
        """
        Plot every span spline from hub to tip at the specified stream
        location.

        Parameters
        ----------
        ss_dens:         Array of SS theta values (rad).
        ps_dens:         Array of PS theta values (rad).
        dens:            Array of distance values on the span direction.
        res:              The resolution of the blade profiles.
        stream_frac_list: Sequence of stream fraction for plotting.
        """
        fig, ax = BladePlots2D.create_span_profile_figure()
        BladePlots2D.plot_blade_span_spline(
            dens=self.imp_gen_val["span_dens"],
            SS_dens=self.imp_gen_val["SS_ordered"],
            PS_dens=self.imp_gen_val["PS_ordered"],
            span=self.imp_2d["span_dens"],
            SS=self.imp_2d["SS_ordered"],
            PS=self.imp_2d["PS_ordered"],
            res=res,
            stream_frac_list=stream_frac_list,
            ax=ax,
        )
        ax.grid()
        fig.legend()

    def plot_impeller(
        self,
        axes_lim: float,
        single_passage: bool = False,
        SS: bool = True,
        PS: bool = True,
        tip_surface: bool = True,
        hub_wall: bool = True,
        shroud_wall: bool = True,
        periodic_low: bool = True,
        periodic_high: bool = True,
        shroud_periodic_low: bool = True,
        shroud_periodic_high: bool = True,
    ):
        """Plot the impeller will optional toogle to display/hide surfaces."""
        fig, ax = BladePlots3D.create_figure(axes_lim)

        for n in range(self.n_blades):
            if SS:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["SS"],
                    ax=ax,
                    linewidth=0.5,
                    color="blue",
                )

            if PS:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["PS"],
                    ax=ax,
                    linewidth=0.5,
                    color="blue",
                )

            if tip_surface:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["tip_surface"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )

            if hub_wall:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["hub_wall"],
                    ax=ax,
                    linewidth=0.5,
                    color="darkblue",
                )

            if shroud_wall:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_wall"],
                    ax=ax,
                    linewidth=0.5,
                    color="darkblue",
                )

            if periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="green",
                )

            if periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="green",
                )

            if shroud_periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="green",
                )

            if shroud_periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="green",
                )

            if single_passage:
                break

        # Set initial view depending on the theta direction
        if self.theta_direct == 1:
            ax.view_init(elev=-120, azim=204, roll=-110)
        else:
            ax.view_init(elev=-60, azim=168, roll=-80)
