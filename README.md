# TurboGen

A Turbomachineary design tool that generates the full component in 3D. The geometry
can be displayed as a single passage or as the full component.

## Features

### Impeller

- **Meridional profile:** Hub and shroud inlet/outlet points.
- **Blade aerofoil profile:** Constant blade angle (beta), thickness distribution, and number of blades.
- **LE and TE trimming:** Adjust the elliptical tips by setting the number of points it will take effect from the edge.

### Plots
- Meridional profile with streamlines and inlet/outlet normals.
- Beta distribution and camber in the streamwise direction.
- 2D blade profile at a chosen span fraction.
- Spanwise spline check at chosen stream fractions.
- 3D impeller, as a single passage or the full component, with each surface (SS, PS tip, hub, shroud, periodic) toggle on or off.

## Roadmap
- Extended inlet and outlet domains.
- Customisable blade angle and thickness distributions.
- `STL` and `STEP` export.
