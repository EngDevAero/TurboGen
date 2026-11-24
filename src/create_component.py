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
plt_settings = PlotSettings(blade_profile_res=500, grid_dens=10)
shroud_clrance = 0.001
theta_direct = -1
span_frac_list = [0, 0.25, 0.5, 0.75, 1]
i_pt_LE = 13
i_pt_TE = 50


# Design
def boundary_domain_values(imp_design: ImpellerDesign):
    imp_design.blade_passage_domain_values()
    imp_design.inlet_domain_values()
    imp_design.outlet_domain_values()


imp_design = ImpellerDesign(
    init_pts=init_pts,
    init_blade=init_blade,
    shroud_clrance=shroud_clrance,
    l_inlet=0.01,
    l_outlet=0.02,
)
imp_design.blade_values_spanwise(span_frac_list, i_pt_LE=i_pt_LE, i_pt_TE=i_pt_TE)
boundary_domain_values(imp_design)

# Generate 3D geometry
imp_gen = ImpellerGeneration(
    grid_dens=plt_settings.grid_dens, theta_direct=theta_direct, imp_design=imp_design
)
imp_gen.pre_process_design_values()
imp_gen.spline_spanwise()
imp_gen.domain_boundaries()
imp_gen.inlet_outlet_surfaces()
imp_gen.convert_to_cartesian_3d()

# Plots
axes_lim = BladePlots3D.get_axes_limit(pts=init_pts)
imp_design.plot_meridional_profiles()
# imp_design.plot_angle_distribution(0.5)
# imp_design.plot_blade_profile(0.5)
# imp_gen.plot_spline_spanwise(res=imp_design.res)
imp_gen.plot_impeller(
    axes_lim,
    single_passage=True,
    # --- Inlet surfaces ---
    # inlet_in_surf=False,
    # inlet_out_surf=False,
    # --- Outlet surfaces ---
    # outlet_in_surf=False,
    # outlet_out_surf=False,
    # --- Shroud inlet surfaces ---
    # shroud_inlet_in_surf=False,
    # shroud_inlet_out_surf=False,
    # --- Shroud outlet surfaces ---
    # shroud_outlet_in_surf=False,
    # shroud_outlet_out_surf=False,
    # --- Blade surfaces ---
    # SS=False,
    # PS=False,
    # --- Hub walls ---
    # hub_wall=False,
    # inlet_hub_wall=False,
    # outlet_hub_wall=False,
    # --- Shroud walls ---
    # shroud_wall=False,
    # inlet_shroud_wall=False,
    # outlet_shroud_wall=False,
    # --- Tip surfaces ---
    # tip_surf=False,
    # inlet_tip_surf=False,
    # outlet_tip_surf=False,
    # --- Periodic lows ---
    # periodic_low=False,
    # inlet_periodic_low=False,
    # outlet_periodic_low=False,
    # --- Periodic highs ---
    # periodic_high=False,
    # inlet_periodic_high=False,
    # outlet_periodic_high=False,
    # --- Shroud periodic lows ---
    # shroud_periodic_low=False,
    # inlet_shroud_periodic_low=False,
    # outlet_shroud_periodic_low=False,
    # --- Shroud periodic highss ---
    # shroud_periodic_high=False,
    # inlet_shroud_periodic_high=False,
    # outlet_shroud_periodic_high=False,
)
imp_gen.plot_impeller(
    axes_lim,
    # --- Inlet surfaces ---
    inlet_in_surf=False,
    inlet_out_surf=False,
    # --- Outlet surfaces ---
    outlet_in_surf=False,
    outlet_out_surf=False,
    # --- Shroud inlet surfaces ---
    shroud_inlet_in_surf=False,
    shroud_inlet_out_surf=False,
    # --- Shroud outlet surfaces ---
    shroud_outlet_in_surf=False,
    shroud_outlet_out_surf=False,
    # --- Blade surfaces ---
    # SS=False,
    # PS=False,
    # --- Hub walls ---
    # hub_wall=False,
    # inlet_hub_wall=False,
    # outlet_hub_wall=False,
    # --- Shroud walls ---
    shroud_wall=False,
    inlet_shroud_wall=False,
    outlet_shroud_wall=False,
    # --- Tip surfaces ---
    tip_surf=False,
    inlet_tip_surf=False,
    outlet_tip_surf=False,
    # --- Periodic lows ---
    periodic_low=False,
    inlet_periodic_low=False,
    outlet_periodic_low=False,
    # --- Periodic highs ---
    periodic_high=False,
    inlet_periodic_high=False,
    outlet_periodic_high=False,
    # --- Shroud periodic lows ---
    shroud_periodic_low=False,
    inlet_shroud_periodic_low=False,
    outlet_shroud_periodic_low=False,
    # --- Shroud periodic highss ---
    shroud_periodic_high=False,
    inlet_shroud_periodic_high=False,
    outlet_shroud_periodic_high=False,
)
plt.show()
