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
    # --- Set ups ---
    self.grid_dens:    Grid density of the 3D geometry.
    self.theta_direct: Rotational direction of theta.
    # --- Design ---
    self.imp_design: Impeller design class.
    self.d_periodic: Theta value from the camber to periodic boundary (rad).
    self.n_blades:   Number of blades of the impeller.
    # --- Component ---
    self.imp_design_vals: Imported design data from the impeller design process instance's
                          dictionary with LE and TE trimming applied, that is ready for
                          pre-processing prior the 3D geometry generation.
    self.imp_gen_val:     Processed and ordered cylindrical coordinate values that is ready
                          for the converstion to Cartesian coordinate values.
    self.imp_3d:          3D cartesian coordinates of the blade.
    self.imp_2d:          2D coordinates of the impeller.
    # --- Domains ---
    self.blade_pass_dmain: Blade passage domain.
    self.inlet_dmain:      Inlet domain.
    self.outlet_dmain:     Outlet domain.
    # --- Wall surfaces ---
    self.hub_wall:    3D coordinates of the hub wall.
    self.shroud_wall: 3D coordinates of the shroud wall.
    self.tip_surf:    3D coordinates of the tip surface.
    # --- Inlet domain ---
    self.inlet_shroud_wall: 3D coordinates of the gap between blade tip and shroud wall at the
                            inlet of the inlet domain.
    self.inlet_tip_surf:    3D coordinates of the surface on blade tip at the inlet of the inlet domain.
    self.inlet_periodic:    (x, r, theta, theta) values of the inlet domain's periodic surfaces.
    # --- Outlet domain ---
    self.outlet_hub_wall:    3D coordinates of the hub wall at the outlet domain.
    self.outlet_shroud_wall: 3D coordinates of the shroud wall at the outlet domain.
    self.outlet_tip_surf:    3D coordinates of the surface on blade tip at the outlet of the outlet domain.
    self.outlet_periodic:    (x, r, theta, theta) values of the outlet domain's periodic surfaces.
    # --- Inlet/Outlet domains inlet/outlet surfaces ---
    self.inlet_in_surf:   3D coordinates of the hub to tip surface at the inlet of the inlet domain.
    self.inlet_out_surf:  3D coordinates of the hub to tip surface at the outlet of the inlet domain.
    self.outlet_in_surf:  3D coordinates of the hub to tip surface at the inlet of the outlet domain.
    self.outlet_out_surf: 3D coordinates of the hub to tip surface at the outlet of the outlet domain.
    # --- Inlet/Outlet domains shroud inlet/outlet surfaces ---
    self.shroud_inlet_in_surf:   3D coordinates of the tip to shroud surface at the inlet of the inlet domain.
    self.shroud_inlet_out_surf:  3D coordinates of the tip to shroud surface at the outlet of the inlet domain.
    self.shroud_outlet_in_surf:  3D coordinates of the tip to shroud surface at the inlet of the outlet domain.
    self.shroud_outlet_out_surf: 3D coordinates of the tip to shroud surface at the outlet of the outlet domain.
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
        self.blade_pass_dmain = self.imp_design.blade_pass_dmain
        self.inlet_dmain = self.imp_design.inlet_dmain
        self.outlet_dmain = self.imp_design.outlet_dmain

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

    def domain_boundaries(self):
        """Calculate the cylindrical coordinates for both the hub and shroud wall."""
        # Blade passage domain
        self.hub_wall = self._get_domain_boundary_coordinates(
            self.blade_pass_dmain, "hub_wall"
        )
        self.shroud_wall = self._get_domain_boundary_coordinates(
            self.blade_pass_dmain, "shroud_wall"
        )
        self.tip_surf = self._get_domain_boundary_coordinates(
            self.blade_pass_dmain, "tip_surf"
        )

        # Inlet domain
        self.inlet_hub_wall = self._get_domain_boundary_coordinates(
            self.inlet_dmain, "hub_wall"
        )
        self.inlet_shroud_wall = self._get_domain_boundary_coordinates(
            self.inlet_dmain, "shroud_wall"
        )
        self.inlet_tip_surf = self._get_domain_boundary_coordinates(
            self.inlet_dmain, "tip_surf"
        )
        self.inlet_periodic = self._get_extended_domain_values(
            self.inlet_dmain, "inlet"
        )

        # Outlet domain
        self.outlet_hub_wall = self._get_domain_boundary_coordinates(
            self.outlet_dmain, "hub_wall"
        )
        self.outlet_shroud_wall = self._get_domain_boundary_coordinates(
            self.outlet_dmain, "shroud_wall"
        )
        self.outlet_tip_surf = self._get_domain_boundary_coordinates(
            self.outlet_dmain, "tip_surf"
        )
        self.outlet_periodic = self._get_extended_domain_values(
            self.outlet_dmain, "outlet"
        )

    def _get_domain_boundary_coordinates(
        self, dmain_dict, surf: Literal["hub_wall", "tip_surf", "shroud_wall"]
    ) -> Coords3D:
        """
        Calculate the cylindrical coordinates of the end wall surface that is
        set at span fraction.
        """
        # Retrieve coordinates at span fraction
        x_arr = dmain_dict[surf]["x_arr"]
        r_arr = dmain_dict[surf]["r_arr"]
        periodic_low = dmain_dict[surf]["periodic_low"]
        periodic_high = dmain_dict[surf]["periodic_high"]

        t = np.linspace(0, 1, self.grid_dens)[:, None]
        theta = periodic_low[None, :] + t * (periodic_high - periodic_low)[None, :]
        x = np.broadcast_to(x_arr[None, :], theta.shape)
        r = np.broadcast_to(r_arr[None, :], theta.shape)
        return x, r, theta

    def _get_extended_domain_values(
        self,
        dmain_dict,
        dmain: Literal["inlet", "outlet"],
    ):
        """
        Calculate x, r, periodic_low, and periodic_high values at either inlet or outlet domain
        for the creation of the periodic surfaces at both domains.
        """
        col = 0 if dmain == "inlet" else -1
        x_blade = self.imp_gen_val["x_arr_ordered"][:, col]
        r_blade = self.imp_gen_val["r_arr_ordered"][:, col]
        blade_periodic_low = self.imp_gen_val["periodic_low_ordered"][:, col]
        blade_periodic_high = self.imp_gen_val["periodic_high_ordered"][:, col]

        x_h_dmain = dmain_dict["hub_wall"]["x_arr"][col]
        r_h_dmain = dmain_dict["hub_wall"]["r_arr"][col]
        hub_periodic_low = dmain_dict["hub_wall"]["periodic_low"][col]
        hub_periodic_high = dmain_dict["hub_wall"]["periodic_high"][col]

        x_t_dmain = dmain_dict["tip_surf"]["x_arr"][col]
        r_t_dmain = dmain_dict["tip_surf"]["r_arr"][col]
        tip_periodic_low = dmain_dict["tip_surf"]["periodic_low"][col]
        tip_periodic_high = dmain_dict["tip_surf"]["periodic_high"][col]

        span = self.imp_gen_val["span_dens"]
        x_pts = x_h_dmain + span * (x_t_dmain - x_h_dmain)
        r_pts = r_h_dmain + span * (r_t_dmain - r_h_dmain)
        periodic_low_pts = hub_periodic_low + span * (
            tip_periodic_low - hub_periodic_low
        )
        periodic_high_pts = hub_periodic_high + span * (
            tip_periodic_high - hub_periodic_high
        )

        res = (
            self.imp_design.inlet_dmain_res
            if dmain == "inlet"
            else self.imp_design.outlet_dmain_res
        )
        res = np.linspace(0, 1, res)
        x_ext = x_blade[:, None] + res[None, :] * (x_pts - x_blade)[:, None]
        r_ext = r_blade[:, None] + res[None, :] * (r_pts - r_blade)[:, None]
        periodic_low_ext = (
            blade_periodic_low[:, None]
            + res[None, :] * (periodic_low_pts - blade_periodic_low)[:, None]
        )
        periodic_high_ext = (
            blade_periodic_high[:, None]
            + res[None, :] * (periodic_high_pts - blade_periodic_high)[:, None]
        )
        return x_ext, r_ext, periodic_low_ext, periodic_high_ext

    def inlet_outlet_surfaces(self):
        """Calculate the inlet/outlet surfaces of the inlet/outlet domains."""

        def _get_row_values(dmain_dict, col_loc: Literal["first", "last"]) -> Coords3D:
            col = 0 if col_loc == "first" else -1
            x_n, r_n, theta_n = (a[:, col] for a in np.array(dmain_dict))
            return x_n, r_n, theta_n

        def _get_coordinates(
            hub_dmain_dict,
            tip_dmain_dict,
            col_loc: Literal["first", "last"],
            grid_dens: int,
        ):
            x_in_h, r_in_h, theta_in_h = _get_row_values(hub_dmain_dict, col_loc)
            x_in_t, r_in_t, theta_in_t = _get_row_values(tip_dmain_dict, col_loc)

            x_inlet_in = []
            r_inlet_in = []
            theta_inlet_in = []
            for i in range(len(np.array(x_in_h))):
                x_inlet_in.append(
                    np.linspace(np.array(x_in_h)[i], np.array(x_in_t)[i], grid_dens)
                )
                r_inlet_in.append(
                    np.linspace(np.array(r_in_h)[i], np.array(r_in_t)[i], grid_dens)
                )
                theta_inlet_in.append(
                    np.linspace(
                        np.array(theta_in_h)[i], np.array(theta_in_t)[i], grid_dens
                    )
                )
            return np.array(x_inlet_in), np.array(r_inlet_in), np.array(theta_inlet_in)

        # Inlet/Outlet surfaces
        x_inlet_in, r_inlet_in, theta_inlet_in = _get_coordinates(
            self.inlet_hub_wall, self.inlet_tip_surf, "first", self.grid_dens
        )
        x_inlet_out, r_inlet_out, theta_inlet_out = _get_coordinates(
            self.inlet_hub_wall, self.inlet_tip_surf, "last", self.grid_dens
        )
        x_outlet_in, r_outlet_in, theta_outlet_in = _get_coordinates(
            self.outlet_hub_wall, self.outlet_tip_surf, "first", self.grid_dens
        )
        x_outlet_out, r_outlet_out, theta_outlet_out = _get_coordinates(
            self.outlet_hub_wall, self.outlet_tip_surf, "last", self.grid_dens
        )
        self.inlet_in_surf = (
            x_inlet_in,
            r_inlet_in,
            theta_inlet_in,
        )
        self.inlet_out_surf = (
            x_inlet_out,
            r_inlet_out,
            theta_inlet_out,
        )
        self.outlet_in_surf = (
            x_outlet_in,
            r_outlet_in,
            theta_outlet_in,
        )
        self.outlet_out_surf = (
            x_outlet_out,
            r_outlet_out,
            theta_outlet_out,
        )

        # Shroud inlet/outlet surfaces
        x_shroud_inlet_in, r_shroud_inlet_in, theta_shroud_inlet_in = _get_coordinates(
            self.inlet_shroud_wall, self.inlet_tip_surf, "first", 2
        )
        x_shroud_inlet_out, r_shroud_inlet_out, theta_shroud_inlet_out = (
            _get_coordinates(self.inlet_shroud_wall, self.inlet_tip_surf, "last", 2)
        )
        x_shroud_outlet_in, r_shroud_outlet_in, theta_shroud_outlet_in = (
            _get_coordinates(self.outlet_shroud_wall, self.outlet_tip_surf, "first", 2)
        )
        x_shroud_outlet_out, r_shroud_outlet_out, theta_shroud_outlet_out = (
            _get_coordinates(self.outlet_shroud_wall, self.outlet_tip_surf, "last", 2)
        )
        self.shroud_inlet_in_surf = (
            x_shroud_inlet_in,
            r_shroud_inlet_in,
            theta_shroud_inlet_in,
        )
        self.shroud_inlet_out_surf = (
            x_shroud_inlet_out,
            r_shroud_inlet_out,
            theta_shroud_inlet_out,
        )
        self.shroud_outlet_in_surf = (
            x_shroud_outlet_in,
            r_shroud_outlet_in,
            theta_shroud_outlet_in,
        )
        self.shroud_outlet_out_surf = (
            x_shroud_outlet_out,
            r_shroud_outlet_out,
            theta_shroud_outlet_out,
        )

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
            self._inlet_outlet_surfaces_to_cartesian_3D(n_blade, pitch)

    def _blade_surface_to_cartesian_3D(self, n_blade, pitch):
        self.imp_3d[str(n_blade)]["SS"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct * (self.imp_gen_val["SS_ordered"] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["PS"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct * (self.imp_gen_val["PS_ordered"] + pitch),
            )
        )

    def _blade_tip_to_cartesian_3D(self, n_blade, pitch):
        self.imp_3d[str(n_blade)]["tip_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.tip_surf[0],
                r=self.tip_surf[1],
                theta=self.theta_direct * (self.tip_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["inlet_tip_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.inlet_tip_surf[0],
                r=self.inlet_tip_surf[1],
                theta=self.theta_direct * (self.inlet_tip_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["outlet_tip_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.outlet_tip_surf[0],
                r=self.outlet_tip_surf[1],
                theta=self.theta_direct * (self.outlet_tip_surf[2] + pitch),
            )
        )

    def _blade_end_walls_to_cartesian_3D(self, n_blade, pitch):
        self.imp_3d[str(n_blade)]["hub_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.hub_wall[0],
                r=self.hub_wall[1],
                theta=self.theta_direct * (self.hub_wall[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["shroud_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.shroud_wall[0],
                r=self.shroud_wall[1],
                theta=self.theta_direct * (self.shroud_wall[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["inlet_hub_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.inlet_hub_wall[0],
                r=self.inlet_hub_wall[1],
                theta=self.theta_direct * (self.inlet_hub_wall[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["inlet_shroud_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.inlet_shroud_wall[0],
                r=self.inlet_shroud_wall[1],
                theta=self.theta_direct * (self.inlet_shroud_wall[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["outlet_hub_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.outlet_hub_wall[0],
                r=self.outlet_hub_wall[1],
                theta=self.theta_direct * (self.outlet_hub_wall[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["outlet_shroud_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.outlet_shroud_wall[0],
                r=self.outlet_shroud_wall[1],
                theta=self.theta_direct * (self.outlet_shroud_wall[2] + pitch),
            )
        )

    def _inlet_outlet_surfaces_to_cartesian_3D(self, n_blade: int, pitch: float):
        # Inlet/outlet surfaces
        self.imp_3d[str(n_blade)]["inlet_in_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.inlet_in_surf[0],
                r=self.inlet_in_surf[1],
                theta=self.theta_direct * (self.inlet_in_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["inlet_out_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.inlet_out_surf[0],
                r=self.inlet_out_surf[1],
                theta=self.theta_direct * (self.inlet_out_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["outlet_in_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.outlet_in_surf[0],
                r=self.outlet_in_surf[1],
                theta=self.theta_direct * (self.outlet_in_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["outlet_out_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.outlet_out_surf[0],
                r=self.outlet_out_surf[1],
                theta=self.theta_direct * (self.outlet_out_surf[2] + pitch),
            )
        )

        # Shroud inlet/outlet surfaces
        self.imp_3d[str(n_blade)]["shroud_inlet_in_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.shroud_inlet_in_surf[0],
                r=self.shroud_inlet_in_surf[1],
                theta=self.theta_direct * (self.shroud_inlet_in_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["shroud_inlet_out_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.shroud_inlet_out_surf[0],
                r=self.shroud_inlet_out_surf[1],
                theta=self.theta_direct * (self.shroud_inlet_out_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["shroud_outlet_in_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.shroud_outlet_in_surf[0],
                r=self.shroud_outlet_in_surf[1],
                theta=self.theta_direct * (self.shroud_outlet_in_surf[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["shroud_outlet_out_surf"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.shroud_outlet_out_surf[0],
                r=self.shroud_outlet_out_surf[1],
                theta=self.theta_direct * (self.shroud_outlet_out_surf[2] + pitch),
            )
        )

    def _periodic_to_cartesian_3D(self, n_blade: int, pitch: float):
        self.imp_3d[str(n_blade)]["periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct
                * (self.imp_gen_val["periodic_low_ordered"] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["x_arr_ordered"],
                r=self.imp_gen_val["r_arr_ordered"],
                theta=self.theta_direct
                * (self.imp_gen_val["periodic_high_ordered"] + pitch),
            )
        )

        self.imp_3d[str(n_blade)]["inlet_periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.inlet_periodic[0],
                r=self.inlet_periodic[1],
                theta=self.theta_direct * (self.inlet_periodic[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["inlet_periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.inlet_periodic[0],
                r=self.inlet_periodic[1],
                theta=self.theta_direct * (self.inlet_periodic[3] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["outlet_periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.outlet_periodic[0],
                r=self.outlet_periodic[1],
                theta=self.theta_direct * (self.outlet_periodic[2] + pitch),
            )
        )
        self.imp_3d[str(n_blade)]["outlet_periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.outlet_periodic[0],
                r=self.outlet_periodic[1],
                theta=self.theta_direct * (self.outlet_periodic[3] + pitch),
            )
        )

    def _shroud_periodic_to_cartesian_3D(self, n_blade, pitch):
        def _get_shroud_periodic_surface(
            tip_surf_dict,
            shroud_wall_dict,
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
                    np.array(tip_surf_dict[0])[i],
                    np.array(shroud_wall_dict[0])[i],
                )
            )
            r_arr = np.stack(
                (
                    np.array(tip_surf_dict[1])[i],
                    np.array(shroud_wall_dict[1])[i],
                )
            )
            theta_arr = self.theta_direct * np.stack(
                (
                    np.array(tip_surf_dict[2])[i],
                    np.array(shroud_wall_dict[2])[i],
                )
            )
            return x_arr, r_arr, theta_arr

        # Blade passage domain
        x_low, r_low, theta_low = _get_shroud_periodic_surface(
            tip_surf_dict=self.tip_surf,
            shroud_wall_dict=self.shroud_wall,
            periodic_loc="periodic_low",
        )
        x_high, r_high, theta_high = _get_shroud_periodic_surface(
            tip_surf_dict=self.tip_surf,
            shroud_wall_dict=self.shroud_wall,
            periodic_loc="periodic_high",
        )

        # Inlet domain
        x_in_low, r_in_low, theta_in_low = _get_shroud_periodic_surface(
            tip_surf_dict=self.inlet_tip_surf,
            shroud_wall_dict=self.inlet_shroud_wall,
            periodic_loc="periodic_low",
        )
        x_in_high, r_in_high, theta_in_high = _get_shroud_periodic_surface(
            tip_surf_dict=self.inlet_tip_surf,
            shroud_wall_dict=self.inlet_shroud_wall,
            periodic_loc="periodic_high",
        )

        # Outlet domain
        x_out_low, r_out_low, theta_out_low = _get_shroud_periodic_surface(
            tip_surf_dict=self.outlet_tip_surf,
            shroud_wall_dict=self.outlet_shroud_wall,
            periodic_loc="periodic_low",
        )
        x_out_high, r_out_high, theta_out_high = _get_shroud_periodic_surface(
            tip_surf_dict=self.outlet_tip_surf,
            shroud_wall_dict=self.outlet_shroud_wall,
            periodic_loc="periodic_high",
        )

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
        self.imp_3d[str(n_blade)]["inlet_shroud_periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=x_in_low,
                r=r_in_low,
                theta=theta_in_low + pitch,
            )
        )
        self.imp_3d[str(n_blade)]["inlet_shroud_periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=x_in_high,
                r=r_in_high,
                theta=theta_in_high + pitch,
            )
        )
        self.imp_3d[str(n_blade)]["outlet_shroud_periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=x_out_low,
                r=r_out_low,
                theta=theta_out_low + pitch,
            )
        )
        self.imp_3d[str(n_blade)]["outlet_shroud_periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=x_out_high,
                r=r_out_high,
                theta=theta_out_high + pitch,
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
        # --- Inlet surfaces ---
        inlet_in_surf: bool = True,
        inlet_out_surf: bool = True,
        # --- Outlet surfaces ---
        outlet_in_surf: bool = True,
        outlet_out_surf: bool = True,
        # --- Shroud inlet surfaces ---
        shroud_inlet_in_surf: bool = True,
        shroud_inlet_out_surf: bool = True,
        # --- Shroud outlet surfaces ---
        shroud_outlet_in_surf: bool = True,
        shroud_outlet_out_surf: bool = True,
        # --- Blade surfaces ---
        SS: bool = True,
        PS: bool = True,
        # --- Hub walls ---
        hub_wall: bool = True,
        inlet_hub_wall: bool = True,
        outlet_hub_wall: bool = True,
        # --- Shroud walls ---
        shroud_wall: bool = True,
        inlet_shroud_wall: bool = True,
        outlet_shroud_wall: bool = True,
        # --- Tip surfaces ---
        tip_surf: bool = True,
        inlet_tip_surf: bool = True,
        outlet_tip_surf: bool = True,
        # --- Periodic lows ---
        periodic_low: bool = True,
        inlet_periodic_low: bool = True,
        outlet_periodic_low: bool = True,
        # --- Periodic highs ---
        periodic_high: bool = True,
        inlet_periodic_high: bool = True,
        outlet_periodic_high: bool = True,
        # --- Shroud periodic lows ---
        shroud_periodic_low: bool = True,
        inlet_shroud_periodic_low: bool = True,
        outlet_shroud_periodic_low: bool = True,
        # --- Shroud periodic highss ---
        shroud_periodic_high: bool = True,
        inlet_shroud_periodic_high: bool = True,
        outlet_shroud_periodic_high: bool = True,
    ):
        """Plot the impeller will optional toogle to display/hide surfaces."""
        fig, ax = BladePlots3D.create_figure(axes_lim)

        for n in range(self.n_blades):
            # Inlet/Outlet surfaces
            if inlet_in_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_in_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )
            if inlet_out_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_out_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )
            if outlet_in_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_in_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )
            if outlet_out_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_out_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )

            # Shroud inlet/outlet surfaces
            if shroud_inlet_in_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_inlet_in_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )
            if shroud_inlet_out_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_inlet_out_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )
            if shroud_outlet_in_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_outlet_in_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )
            if shroud_outlet_out_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_outlet_out_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )

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

            if tip_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["tip_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )

            if inlet_tip_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_tip_surf"],
                    ax=ax,
                    linewidth=0.5,
                    color="cyan",
                )

            if outlet_tip_surf:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_tip_surf"],
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

            if inlet_hub_wall:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_hub_wall"],
                    ax=ax,
                    linewidth=0.5,
                    color="darkblue",
                )

            if inlet_shroud_wall:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_shroud_wall"],
                    ax=ax,
                    linewidth=0.5,
                    color="darkblue",
                )

            if outlet_hub_wall:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_hub_wall"],
                    ax=ax,
                    linewidth=0.5,
                    color="darkblue",
                )

            if outlet_shroud_wall:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_shroud_wall"],
                    ax=ax,
                    linewidth=0.5,
                    color="darkblue",
                )

            if periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )
            if inlet_periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )
            if outlet_periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )
            if inlet_periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )
            if outlet_periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if shroud_periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if shroud_periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["shroud_periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if inlet_shroud_periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_shroud_periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if inlet_shroud_periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["inlet_shroud_periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if outlet_shroud_periodic_low:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_shroud_periodic_low"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if outlet_shroud_periodic_high:
                BladePlots3D.plot_surface(
                    cart_coords_3d=self.imp_3d[str(n)]["outlet_shroud_periodic_high"],
                    ax=ax,
                    linewidth=0.5,
                    color="lightgreen",
                )

            if single_passage:
                break

        # Set initial view depending on the theta direction
        if self.theta_direct == 1:
            ax.view_init(elev=-120, azim=204, roll=-110)
        else:
            ax.view_init(elev=-60, azim=168, roll=-80)
