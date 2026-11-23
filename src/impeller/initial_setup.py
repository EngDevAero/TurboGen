from dataclasses import dataclass


@dataclass
class InitialMeridionalPoints:
    """
    Parameters
    ----------
    pt_h1: x and y Coordinates of hub inlet location (m).
    pt_h2: x and y Coordinates of hub outlet location (m).
    pt_s1: x and y Coordinates of shroud inlet location (m).
    pt_s2: x and y Coordinates of shroud outlet location (m).
    """

    pt_h1: tuple[float, float]
    pt_h2: tuple[float, float]
    pt_s1: tuple[float, float]
    pt_s2: tuple[float, float]


@dataclass
class InitialBladeValues:
    """
    Parameters
    ----------
    wrap_ang: Nominal wrap angle from LE to TE (rad).
    t_B:      Blade thickness (m).
    n_blades: Number of blades.
    """

    t_B: float
    n_blades: int

@dataclass
class PlotSettings:
    blade_profile_res: int = 500
    grid_dens: int = 30
