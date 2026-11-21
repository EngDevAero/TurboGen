"""Contains all plot functions."""

from matplotlib import pyplot as plt
from typing import Literal, Sequence
from matplotlib.lines import lineStyles
import numpy as np

from .constants import (
    Array1D,
    Array2D,
    Array2DList,
    Point,
    Figure2D,
    AxesFigure2D,
    FigComponents2D,
    Figure3D,
    AxesFigure3D,
    Coords3D,
    FigComponents3D,
)
from .converters import UnitConverter


class MeridionalPlots:
    """Meridional plots function and controls."""

    @staticmethod
    def create_meridional_figure() -> FigComponents2D:
        fig, ax = plt.subplots()
        ax.axis("equal")
        ax.set_xlabel("x-axial (m)")
        ax.set_ylabel("radius (m)")
        return fig, ax

    @staticmethod
    def plot_meridional_contour(
        x_arr: Array1D,
        r_arr: Array1D,
        span_frac: str,
        ax: AxesFigure2D,
    ) -> None:
        """
        Plot the meridional contour (x, r) at a specified span fraction.

        Parameters
        ----------
        x_arr:     Axial values (m).
        r_arr:     Radial values (m).
        span_frac: Span fraction.
        """
        label = f"Span fraction = {span_frac}"
        if span_frac in ("0", "1"):
            ax.plot(
                x_arr,
                r_arr,
                linestyle="solid",
                linewidth=1.5,
                color="black",
                label=label,
            )
        else:
            ax.plot(
                x_arr, r_arr, linestyle="solid", linewidth=1, color="grey", label=label
            )

    @staticmethod
    def plot_meridional_component_axis(
        x_arr: Array1D,
        ax: AxesFigure2D,
    ) -> None:
        axis_r = np.linspace(0, 0, len(x_arr))
        ax.plot(x_arr, axis_r, linestyle="dashdot", linewidth=1, color="black")

    @staticmethod
    def plot_inlet_normal(
        pt_h1: Point,
        pt_s1: Point,
        ax: AxesFigure2D,
    ) -> None:
        x = [pt_h1[0], pt_s1[0]]
        r = [pt_h1[1], pt_s1[1]]
        ax.plot(x, r, linestyle="solid", linewidth=1, color="black")

    @staticmethod
    def plot_outlet_normal(
        pt_h2: Point,
        pt_s2: Point,
        ax: AxesFigure2D,
    ) -> None:
        x = [pt_h2[0], pt_s2[0]]
        r = [pt_h2[1], pt_s2[1]]
        ax.plot(x, r, linestyle="solid", linewidth=1, color="black")


class BladePlots2D:
    """Blade related plots (Angle distributions and profiles)."""

    @staticmethod
    def create_angle_distribution_figure() -> FigComponents2D:
        fig, ax = plt.subplots()
        ax.set_xlabel("M-Prime (LE to TE)")
        ax.set_ylabel("Angle (deg)")
        return fig, ax

    @staticmethod
    def create_blade_profile_figure() -> FigComponents2D:
        fig, ax = plt.subplots()
        ax.set_xlabel("M-Prime (LE to TE)")
        ax.set_ylabel("Theta (deg)")
        return fig, ax

    @staticmethod
    def create_span_profile_figure() -> FigComponents2D:
        fig, ax = plt.subplots()
        ax.set_xlabel("Span fraction")
        ax.set_ylabel("Theta (deg)")
        return fig, ax

    @staticmethod
    def plot_beta_distribution_at_span_fraction(
        m_prime_arr: Array1D,
        beta_arr: Array1D,
        span_frac: str,
        ax: AxesFigure2D,
        disp_beta_unit: Literal["deg", "rad"] = "deg",
        disp_chord_unit: Literal["m_prime"] = "m_prime",
    ) -> None:
        """
        Plot the beta distribution (M-Prime, beta) at a specified span fraction.

        Parameters
        ----------
        m_prime_arr:   M-Prime from LE to TE.
        beta:      Beta values across M-Prime (rad).
        span_frac: Span fraction.
        """
        # Unit convertion
        if disp_beta_unit == "deg":
            beta_arr = UnitConverter.rad_to_deg(rad=beta_arr)

        label = f"Beta at span fraction = {span_frac}"
        ax.plot(m_prime_arr, beta_arr, "cyan", label=label)

    @staticmethod
    def plot_blade_camber_at_span_fraction(
        m_prime_arr: Array1D,
        camber: Array1D,
        span_frac: str,
        ax: AxesFigure2D,
        disp_theta_unit: Literal["deg", "rad"] = "deg",
        disp_chord_unit: Literal["m_prime"] = "m_prime",
    ) -> None:
        """
        Plot the Blade camber (M-Prime, theta) at a specified span fraction.

        Parameters
        ----------
        m_prime_arr:   M-Prime from LE to TE.
        theta:     Theta values across M-Prime (rad).
        span_frac: Span fraction.
        """
        # Unit convertion
        if disp_theta_unit == "deg":
            camber = UnitConverter.rad_to_deg(rad=camber)

        label = f"Blade camber at span fraction = {span_frac}"
        ax.plot(m_prime_arr, camber, "b--", label=label)

    @staticmethod
    def plot_blade_profile_at_span_fraction(
        m_prime_arr: Array1D,
        camber: Array1D,
        base_SS: Array1D,
        base_PS: Array1D,
        SS: Array1D,
        PS: Array1D,
        span_frac: str,
        ax: AxesFigure2D,
        disp_ang_unit: Literal["deg", "rad"] = "deg",
        disp_chord_unit: Literal["m_prime"] = "m_prime",
    ) -> None:
        """
        Plot the blade profile at a specified span fraction.

        Parameters
        ----------
        m_prime_arr:     M-Prime from LE to TE.
        camber: Theta distribution of the blade camber across M-Prime (deg).
        base_SS:     Base suction surface theta values across M-Prime (deg).
        base_PS:     Base pressure surface theta values across M-Prime (deg).
        SS:          Suction surface of the blade.
        PS:          Pressure surface of the blade.
        span_frac:   Span fraction.
        """
        if disp_ang_unit == "deg":
            camber = UnitConverter.rad_to_deg(rad=camber)
            base_SS = UnitConverter.rad_to_deg(rad=base_SS)
            base_PS = UnitConverter.rad_to_deg(rad=base_PS)
            SS = UnitConverter.rad_to_deg(rad=SS)
            PS = UnitConverter.rad_to_deg(rad=PS)

        # Labels
        label_camber = f"Blade camber at span fraction = {span_frac}"
        label_base_SS = f"Suction surface base profile at span fraction = {span_frac}"
        label_base_PS = f"Pressure surface base profile at span fraction = {span_frac}"
        label_SS = f"Suction surface at span fraction = {span_frac}"
        label_PS = f"Pressure surface at span fraction = {span_frac}"

        # Plots
        ax.plot(m_prime_arr, camber, "k:", label=label_camber)
        ax.plot(m_prime_arr, base_SS, "r:", label=label_base_SS)
        ax.plot(m_prime_arr, base_PS, "b:", label=label_base_PS)
        ax.plot(m_prime_arr, SS, "r", label=label_SS)
        ax.plot(m_prime_arr, PS, "b", label=label_PS)

    @staticmethod
    def plot_blade_span_spline(
        dens: Array1D,
        SS_dens: Array2D,
        PS_dens: Array2D,
        span: Array1D,
        SS: Array2D,
        PS: Array2D,
        res: int,
        stream_frac_list: Sequence[float],
        ax: AxesFigure2D,
        disp_ang_unit: Literal["deg", "rad"] = "deg",
    ) -> None:
        if disp_ang_unit == "deg":
            SS_dens = UnitConverter.rad_to_deg(rad=SS_dens)
            PS_dens = UnitConverter.rad_to_deg(rad=PS_dens)
            SS = UnitConverter.rad_to_deg(rad=SS)
            PS = UnitConverter.rad_to_deg(rad=PS)

        marker_SS = "s"
        marker_PS = "^"
        for k, stream_frac in enumerate(stream_frac_list):
            color = f"C{k}"
            label_SS = f"Suction surface at {stream_frac} streamline"
            label_PS = f"Pressure surface at {stream_frac} streamline"
            no_label = "_nolegend_"
            i = int(round(stream_frac * (res - 1)))  # Point on the streamline

            # Spline curves
            ax.plot(dens, SS_dens[:, i], linestyle="-", color=color, label=label_SS)
            ax.plot(dens, PS_dens[:, i], linestyle="--", color=color, label=label_PS)

            # Original sections
            ax.plot(
                span,
                SS[:, i],
                linestyle="",
                marker=marker_SS,
                color=color,
                label=no_label,
            )
            ax.plot(
                span,
                PS[:, i],
                linestyle="",
                marker=marker_PS,
                color=color,
                label=no_label,
            )

        label_og_SS = "Suction surface of the original blade profile"
        label_og_PS = "Pressure surface of the original blade profile"
        ax.plot([], [], linestyle="", marker=marker_SS, color="k", label=label_og_SS)
        ax.plot([], [], linestyle="", marker=marker_PS, color="k", label=label_og_PS)


class BladePlots3D:
    @staticmethod
    def create_figure_3d(axes_lim: float | None = None) -> FigComponents3D:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        # Equal axes
        if axes_lim:
            lim = -axes_lim / 2, axes_lim / 2
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_zlim(lim)
        return fig, ax

    @staticmethod
    def plot_blade_surface(cart_coords_3d: Coords3D, ax: AxesFigure3D):
        ax.set_box_aspect((1, 1, 1))
        x, y, z = cart_coords_3d
        ax.plot_wireframe(x, y, z, linewidth=0.3)
