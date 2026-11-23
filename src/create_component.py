import matplotlib.pyplot as plt
import numpy as np

from src.impeller import (
    InitialMeridionalPoints,
    InitialBladeValues,
    ImpellerDesign,
    ImpellerGeneration,
    PlotSettings,
)
from src.helpers import BladePlots3D

# Design inputs
init_pts = InitialMeridionalPoints(
    pt_h1=(0.0, 0.02),
    pt_h2=(0.04, 0.065),
    pt_s1=(0.0, 0.04),
    pt_s2=(0.032, 0.065),
)
init_blade = InitialBladeValues(t_B=0.003, n_blades=11)
plt_settings = PlotSettings(blade_profile_res=500, grid_dens=30)
shroud_clrance = 0.001
theta_direct = -1
span_frac_list = [0, 0.25, 0.5, 0.75, 1]
i_pt_LE = 13
i_pt_TE = 50


# Design
def compute_boundary_data_values(imp_design: ImpellerDesign):
    imp_design.hub_wall_values()
    imp_design.shroud_wall_values()
    imp_design.blade_tip_values()


imp_design = ImpellerDesign(
    init_pts=init_pts, init_blade=init_blade, shroud_clrance=shroud_clrance
)
imp_design.blade_values_spanwise(span_frac_list, i_pt_LE=i_pt_LE, i_pt_TE=i_pt_TE)
compute_boundary_data_values(imp_design)

# Generate 3D geometry
imp_gen = ImpellerGeneration(
    grid_dens=plt_settings.grid_dens, theta_direct=theta_direct, imp_design=imp_design
)
imp_gen.pre_process_design_values()
imp_gen.spline_spanwise()
imp_gen.boundary_surfaces()
imp_gen.convert_to_cartesian_3d()

# Plots
axes_lim = BladePlots3D.get_axes_limit(pts=init_pts)
# imp_design.plot_meridional_profiles()
# imp_design.plot_angle_distribution(0.5)
# imp_design.plot_blade_profile(0.5)
# imp_gen.plot_spline_spanwise(res=imp_design.res)
imp_gen.plot_impeller(
    axes_lim,
    single_passage=True,
    # SS=False,
    # PS=False,
    # tip_surface=False,
    # hub_wall=False,
    # shroud_wall=False,
    # periodic_low=False,
    # periodic_high=False,
    # shroud_periodic_low=False,
    # shroud_periodic_high=False,
)
imp_gen.plot_impeller(
    axes_lim,
    # SS=False,
    # PS=False,
    tip_surface=False,
    # hub_wall=False,
    shroud_wall=False,
    periodic_low=False,
    periodic_high=False,
    shroud_periodic_low=False,
    shroud_periodic_high=False,
)
plt.show()
