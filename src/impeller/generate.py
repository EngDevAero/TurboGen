"""
Contains all functions for generating the 3D impeller. The generation process requires
importing the design data from the impeller design process's instance dictionary, and
some pre-processing before it can be used to calculation the 3D Cartesian geometry.
"""

from math import radians
from typing import Literal, Sequence
from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np

from .design import ImpellerDesign
from .initial_setup import InitialMeridionalPoints, InitialBladeValues, InitialLayers
from src.helpers import (
    MeridionalPlots,
    BladePlots2D,
    Blade,
    CoordinateConverter,
    BladePlots3D,
    is_arrays_resolution_match,
)
from src.helpers.constants import (
    RotationDirect,
    ImpellerDesignData,
    ImpellerDataOrdered,
    ComponentDict3D,
    ImpellerDesignDict,
    ImpellerDataDict,
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
    self.n_blades:    Number of blade of the impeller.
    self.d_periodic:  The theta value from the camber to periodic boundary (rad).
    self.imp_design: Imported design data from the impeller design process instance's
                      dictionary with LE and TE trimming applied, that is ready for
                      pre-processing prior the 3D geometry generation.
    self.imp_gen_val: Processed and ordered cylindrical coordinate values that is ready
                      for the converstion to Cartesian coordinate values.
    self.imp_cart_3d: 3D cartesian coordinates of the blade.
    self.hub_wall:    x, r values of the hub wall (m).
    self.shroud_wall: x, r values of the shroud wall (m).
    """

    def __init__(
        self,
        n_blades: int,
        d_periodic: float,
        rot_direct: RotationDirect,
        imp_design: ImpellerDesignDict,
        hub_wall: Coords3D,
        shroud_wall: Coords3D,
    ):
        self.rot_direct = rot_direct
        self.imp_design: ImpellerDataDict = {}
        self._load_impeller_design_data(imp_design)
        self.n_blades = n_blades
        self.d_periodic = d_periodic
        self.imp_gen_val: ImpellerDataOrdered = {}
        self.imp_cart_3d: ComponentDict3D = {}
        self.hub_wall: Coords3D = hub_wall
        self.shroud_wall: Coords3D = shroud_wall

    def _load_impeller_design_data(self, imp_design: ImpellerDesignDict) -> None:
        """
        Apply blade edges trims and load all the necessary data from the
        impeller design dictionary to this instance attribute dictionary
        self.imp_design.

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
        for key, value in imp_design.items():
            SS, PS = self._apply_LE_TE_trims(value)
            if key in exclude_keys:
                continue
            self.imp_design[key] = {
                "axial": value["axial"],
                "m_prime_arr": value["m_prime_arr"],
                "radial": value["radial"],
                "beta_arr": value["beta_arr"],
                "camber": value["camber"],
                "periodic_low": value["periodic_low"],
                "periodic_high": value["periodic_high"],
                "t_b_dtrib": value["t_b_dtrib"],
                "SS": SS,
                "PS": PS,
            }

    def _apply_LE_TE_trims(self, value: ImpellerDesignData) -> tuple[Array1D, Array1D]:
        """
        Create blade surface profile for SS and PS and store
        them into self.imp_design.

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

    def pre_process_data(self) -> ImpellerDataOrdered:
        """
        Retrieve all the values from created blade profile across span and
        sort value arrays by increasing span.
        """
        span_frac_list: list[float] = []
        axial_arr_span: Array1DList = []
        radial_arr_span: Array1DList = []
        camber_arr_span: Array1DList = []
        periodic_low_arr_span: Array1DList = []
        periodic_high_arr_span: Array1DList = []
        SS_arr_span: Array1DList = []
        PS_arr_span: Array1DList = []

        # Append to value list for every span fraction
        for key, value in sorted(self.imp_design.items(), key=lambda kv: float(kv[0])):
            span_frac_list.append(float(key))
            axial_arr_span.append(value["axial"])
            radial_arr_span.append(value["radial"])
            camber_arr_span.append(value["camber"])
            periodic_low_arr_span.append(value["periodic_low"])
            periodic_high_arr_span.append(value["periodic_high"])
            SS_arr_span.append(value["SS"])
            PS_arr_span.append(value["PS"])

        span_frac_arr = np.array(span_frac_list, dtype=float)

        (
            span_ordered,
            axial_ordered,
            radial_ordered,
            camber_ordered,
            periodic_low_ordered,
            periodic_high_ordered,
            SS_ordered,
            PS_ordered,
        ) = Blade.sort_blade_surface_lists_by_increasing_span(
            span_frac_arr=span_frac_arr,
            axial_arr_span=axial_arr_span,
            radial_arr_span=radial_arr_span,
            camber_arr_span=camber_arr_span,
            periodic_low_arr_span=periodic_low_arr_span,
            periodic_high_arr_span=periodic_high_arr_span,
            SS_arr_span=SS_arr_span,
            PS_arr_span=PS_arr_span,
        )
        return {
            "span_dens": span_ordered,
            "axial_ordered": axial_ordered,
            "radial_ordered": radial_ordered,
            "camber_ordered": camber_ordered,
            "periodic_low_ordered": periodic_low_ordered,
            "periodic_high_ordered": periodic_high_ordered,
            "SS_ordered": SS_ordered,
            "PS_ordered": PS_ordered,
        }

    def compute_spline_values_across_span(
        self,
        n_pts: int,
        val_dict: ImpellerDataOrdered,
        method: Literal["cubic", "pchip"] = "cubic",
    ) -> None:
        """
        Create spline across the span of the blade so the blade surface is smooth.

        n_pts:  Set number of points of the spline.
        method: Either CubicSpline or PchipInterpolator.
        """
        span_ordered = val_dict["span_dens"]
        span_dens = Blade.spanwise_density(span_ordered, n_pts)

        def _spline(name: str) -> Array2D:
            return Blade.spline_across_span(
                span_dens=span_dens,
                span_ordered=span_ordered,
                values_ordered=val_dict[name],
                n_pts=n_pts,
                method=method,
            )

        self.imp_gen_val["span_dens"] = span_dens
        self.imp_gen_val["axial_ordered"] = _spline("axial_ordered")
        self.imp_gen_val["radial_ordered"] = _spline("radial_ordered")
        self.imp_gen_val["camber_ordered"] = _spline("camber_ordered")
        self.imp_gen_val["periodic_low_ordered"] = _spline("periodic_low_ordered")
        self.imp_gen_val["periodic_high_ordered"] = _spline("periodic_high_ordered")
        self.imp_gen_val["SS_ordered"] = _spline("SS_ordered")
        self.imp_gen_val["PS_ordered"] = _spline("PS_ordered")

        high = self.imp_design["0"]["periodic_high"]
        low = self.imp_design["0"]["periodic_low"]
        x_arr = self.imp_design["0"]["axial"]
        r_arr = self.imp_design["0"]["radial"]
        t = np.linspace(0, 1, n_pts)[:, None]
        theta = low[None, :] + t * (high - low)[None, :]
        x = np.broadcast_to(x_arr[None, :], theta.shape)
        r = np.broadcast_to(r_arr[None, :], theta.shape)
        self.hub_wall = x, r, theta

        high = self.imp_design["1"]["periodic_high"]
        low = self.imp_design["1"]["periodic_low"]
        x_arr = self.imp_design["1"]["axial"]
        r_arr = self.imp_design["1"]["radial"]
        t = np.linspace(0, 1, n_pts)[:, None]
        theta = low[None, :] + t * (high - low)[None, :]
        x = np.broadcast_to(x_arr[None, :], theta.shape)
        r = np.broadcast_to(r_arr[None, :], theta.shape)
        self.shroud_wall = x, r, theta

    def plot_spline_values(
        self,
        val_dict: ImpellerDataOrdered,
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
            span=val_dict["span_dens"],
            SS=val_dict["SS_ordered"],
            PS=val_dict["PS_ordered"],
            res=res,
            stream_frac_list=stream_frac_list,
            ax=ax,
        )
        ax.grid()
        fig.legend()

    def compute_impeller_3d_coordinates(self) -> None:
        """Calculate the 3D coordinates for SS, PS, camber."""
        self.imp_cart_3d["camber"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["axial_ordered"],
                r=self.imp_gen_val["radial_ordered"],
                theta=self.rot_direct * self.imp_gen_val["camber_ordered"],
            )
        )

        self.imp_cart_3d["SS"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["axial_ordered"],
                r=self.imp_gen_val["radial_ordered"],
                theta=self.rot_direct * self.imp_gen_val["SS_ordered"],
            )
        )
        self.imp_cart_3d["PS"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["axial_ordered"],
                r=self.imp_gen_val["radial_ordered"],
                theta=self.rot_direct * self.imp_gen_val["PS_ordered"],
            )
        )

        self.imp_cart_3d["periodic_low"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["axial_ordered"],
                r=self.imp_gen_val["radial_ordered"],
                theta=self.rot_direct * self.imp_gen_val["periodic_low_ordered"],
            )
        )
        self.imp_cart_3d["periodic_high"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["axial_ordered"],
                r=self.imp_gen_val["radial_ordered"],
                theta=self.rot_direct * self.imp_gen_val["periodic_high_ordered"],
            )
        )

        self.imp_cart_3d["hub_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.hub_wall[0],
                r=self.hub_wall[1],
                theta=self.rot_direct * self.hub_wall[2]
            )
        )
        self.imp_cart_3d["shroud_wall"] = (
            CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.shroud_wall[0],
                r=self.shroud_wall[1],
                theta=self.rot_direct * self.shroud_wall[2]
            )
        )

    def plot_SS_PS_3D(self):
        """Plot impeller blade surfaces."""
        fig, ax = BladePlots3D.create_figure_3d()
        BladePlots3D.plot_blade_surface(cart_coords_3d=self.imp_cart_3d["SS"], ax=ax)
        BladePlots3D.plot_blade_surface(cart_coords_3d=self.imp_cart_3d["PS"], ax=ax)

        BladePlots3D.plot_blade_surface(
            cart_coords_3d=self.imp_cart_3d["periodic_low"], ax=ax
        )
        BladePlots3D.plot_blade_surface(
            cart_coords_3d=self.imp_cart_3d["periodic_high"], ax=ax
        )

        BladePlots3D.plot_blade_surface(
            cart_coords_3d=self.imp_cart_3d["hub_wall"], ax=ax
        )
        BladePlots3D.plot_blade_surface(
            cart_coords_3d=self.imp_cart_3d["shroud_wall"], ax=ax
        )

        if self.rot_direct == 1:
            ax.view_init(elev=-120, azim=204, roll=-110)
        else:
            ax.view_init(elev=-60, azim=168, roll=-80)

    def plot_full_impeller(self, axes_lim: float):
        """Plot the entire impeller by stacking the blades."""
        fig, ax = BladePlots3D.create_figure_3d(axes_lim)
        for n in range(self.n_blades):
            pitch = n * self.d_periodic * 2
            theta_SS_arr = self.rot_direct * self.imp_gen_val["SS_ordered"] + pitch
            theta_PS_arr = self.rot_direct * self.imp_gen_val["PS_ordered"] + pitch
            SS = CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["axial_ordered"],
                r=self.imp_gen_val["radial_ordered"],
                theta=theta_SS_arr,
            )
            PS = CoordinateConverter.cylindrical_to_cartesian_coordinates(
                x=self.imp_gen_val["axial_ordered"],
                r=self.imp_gen_val["radial_ordered"],
                theta=theta_PS_arr,
            )

            BladePlots3D.plot_blade_surface(cart_coords_3d=SS, ax=ax)
            BladePlots3D.plot_blade_surface(cart_coords_3d=PS, ax=ax)
            ax.view_init(elev=-60, azim=140, roll=-53)


# Design inputs
init_pts = InitialMeridionalPoints(
    pt_h1=(0.0, 0.02),
    pt_h2=(0.04, 0.065),
    pt_s1=(0.0, 0.04),
    pt_s2=(0.032, 0.065),
)
init_blade = InitialBladeValues(wrap_ang=np.radians(80), t_B=0.003, n_blades=8)

# Calculate 3D axes length
f = 1.3
xmax = max(init_pts.pt_h2[1], init_pts.pt_s2[1])
zmax = max(init_pts.pt_h2[0], init_pts.pt_s2[0])
zmin = min(init_pts.pt_h1[0], init_pts.pt_s1[0])
xlim = xmax * 2 * f
zlim = (zmax - zmin) * f
max_lim = max(abs(xlim), abs(zlim))


# Design
imp_design = ImpellerDesign(init_pts=init_pts, init_blade=init_blade)
imp_design.compute_profiles_at_span_fractions()
imp_design.set_hub_wall_streamwise_coordinates()
imp_design.set_shroud_wall_streamwise_coordinates()
plot_meridional = imp_design.plot_all_meridional_profiles()
imp_design.plot_angle_distribution_at_span_fraction(0.5)
imp_design.plot_blade_profile_at_span_fraction(0.5)

# Generate
imp_gen = ImpellerGeneration(
    n_blades=imp_design.n_blades,
    d_periodic=imp_design.d_periodic,
    rot_direct=-1,
    imp_design=imp_design.imp_design,
    hub_wall=imp_design.hub_wall,
    shroud_wall=imp_design.shroud_wall,
)
val_dict = imp_gen.pre_process_data()
imp_gen.compute_spline_values_across_span(n_pts=30, val_dict=val_dict)
imp_gen.plot_spline_values(val_dict=val_dict, res=imp_design.res)
imp_gen.compute_impeller_3d_coordinates()
imp_gen.plot_SS_PS_3D()
imp_gen.plot_full_impeller(max_lim)

plt.show()
