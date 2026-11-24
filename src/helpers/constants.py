"""
Contains constants for all modules.

DO NOT IMPORT ANY LOCAL MODULES INTO THIS FILE.
"""

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, TypeVar

import numpy as np
from numpy.linalg import matrix_power
import numpy.typing as npt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.lines as mlines

# Design inputs
RotationDirect = Literal[1, -1]

# Coordinate points
Point = tuple[float, float]
Points = list[Point]

# Float/Arrays
Array1D = npt.NDArray[np.float64]
Array1DList = list[Array1D]
Array2D = npt.NDArray[np.float64]
Array2DList = list[Array2D]
Array3D = npt.NDArray[np.float64]
FloatOrArray = float | Array1D | Array2D
FloatArrayPair = TypeVar("FloatArrayPair", float, Array1D, Array1DList, Array2D)

# 2D plots
Coords2D = tuple[Array1D, Array1D]  # 2D coordinates
Plot2D = list[mlines.Line2D]  # 2D plot
Plot2DList = list[Plot2D]  # List of 2D plots
Figure2D = Figure
AxesFigure2D = Axes
FigComponents2D = tuple[Figure2D, AxesFigure2D]

# 3D plots
Coords3D = tuple[FloatOrArray, FloatOrArray, FloatOrArray]
Figure3D = Figure
AxesFigure3D = Axes3D
FigComponents3D = tuple[Figure3D, AxesFigure3D]


class CurveSettings(TypedDict):
    pt1: Point
    pt2: Point
    pt3: Point
    alpha1: float
    alpha3: float
    phi: float


class ImpellerBladeData(TypedDict):
    x_arr: Array1D
    r_arr: Array1D
    m_prime_arr: Array1D
    beta_arr: Array1D
    camber: Array1D
    periodic_low: Array1D
    periodic_high: Array1D
    t_b_dtrib: Array1D
    base_SS: Array1D
    base_PS: Array1D
    LE_SS: Array1D
    LE_PS: Array1D
    TE_SS: Array1D
    TE_PS: Array1D
    curve_settings: CurveSettings | None


class ImpBdaryDmainData(TypedDict):
    x_arr: Array1D
    r_arr: Array1D
    mid_pt_arr: Array1D
    periodic_low: Array1D
    periodic_high: Array1D


class ImpDesignValsPerSpan(TypedDict):
    x_arr: Array1D
    r_arr: Array1D
    m_prime_arr: Array1D
    beta_arr: Array1D
    camber: Array1D
    periodic_low: Array1D
    periodic_high: Array1D
    t_b_dtrib: Array1D
    SS: Array1D
    PS: Array1D


class ImpCoords2D(TypedDict):
    span_dens: Array1D
    x_arr_ordered: Array2D
    r_arr_ordered: Array2D
    camber_ordered: Array2D
    periodic_low_ordered: Array2D
    periodic_high_ordered: Array2D
    SS_ordered: Array2D
    PS_ordered: Array2D


class ImpCoords3D(TypedDict):
    # --- Inlet surfaces ---
    inlet_in_surf: Coords3D
    inlet_out_surf: Coords3D
    # --- Outlet surfaces ---
    outlet_in_surf: Coords3D
    outlet_out_surf: Coords3D
    # --- Shroud inlet surfaces ---
    shroud_inlet_in_surf: Coords3D
    shroud_inlet_out_surf: Coords3D
    # --- Shroud outlet surfaces ---
    shroud_outlet_in_surf: Coords3D
    shroud_outlet_out_surf: Coords3D
    # --- Blade surfaces ---
    SS: Coords3D
    PS: Coords3D
    # --- Hub walls ---
    hub_wall: Coords3D
    inlet_hub_wall: Coords3D
    outlet_hub_wall: Coords3D
    # --- Shroud walls ---
    shroud_wall: Coords3D
    inlet_shroud_wall: Coords3D
    outlet_shroud_wall: Coords3D
    # --- Tip surfaces ---
    tip_surf: Coords3D
    inlet_tip_surf: Coords3D
    outlet_tip_surf: Coords3D
    # --- Periodic lows ---
    periodic_low: Coords3D
    inlet_periodic_low: Coords3D
    outlet_periodic_low: Coords3D
    # --- Periodic highs ---
    periodic_high: Coords3D
    inlet_periodic_high: Coords3D
    outlet_periodic_high: Coords3D
    # --- Shroud periodic lows ---
    shroud_periodic_low: Coords3D
    inlet_shroud_periodic_low: Coords3D
    outlet_shroud_periodic_low: Coords3D
    # --- Shroud periodic highss ---
    shroud_periodic_high: Coords3D
    inlet_shroud_periodic_high: Coords3D
    outlet_shroud_periodic_high: Coords3D


ImpellerBladeDict = dict[str, ImpellerBladeData]
ImpBdaryDmainDict = dict[str, ImpBdaryDmainData]
ImpDesignVals = dict[str, ImpDesignValsPerSpan]
ImpellerDataValOrdered = tuple[
    Array1D, Array2D, Array2D, Array2D, Array2D, Array2D, Array2D, Array2D
]
FullImp3D = dict[str, ImpCoords3D]


UnpackedCurveSettings = tuple[Point, Point, Point, float, float, float]
