"""Contains all the equations for creating custom curves."""

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, BSpline
from typing import Literal

from .constants import Point, Points, Array1D, Coords2D, CurveSettings
from .validations import is_arrays_resolution_match


class Curves:

    @staticmethod
    def curve_normal(x: Array1D, y: Array1D):
        """Find the normal of the x, y curve."""
        dx = np.gradient(x)
        dy = np.gradient(y)

        l = np.hypot(dx, dy)
        nx = -dy / l
        ny = dx / l
        return nx, ny

    @staticmethod
    def offset_curve(
        x: Array1D,
        y: Array1D,
        offset_dis: float | None = None,
        offset_dtrib: Array1D | None = None,
    ) -> Coords2D:
        """Offsets the x, y curve with from normal."""
        if offset_dtrib is not None:
            is_arrays_resolution_match(x, y, offset_dtrib, raise_err=True)
        elif offset_dis:
            is_arrays_resolution_match(x, y, raise_err=True)
            offset_dtrib = np.linspace(offset_dis, offset_dis, len(x))
        else:
            raise ValueError(
                "Offset distance or distribution must be provided to offset curve."
            )

        nx, ny = Curves.curve_normal(x, y)
        return x + offset_dtrib * nx, y + offset_dtrib * ny

    @staticmethod
    def circular_arc_coordinates(
        pt1: Point,
        pt2: Point,
        centre: Point,
        r: float,
        n_pts: int,
        phi: float = np.pi / 4,
    ) -> Coords2D:
        """
        Generates coordinates for a circular arc with even spacing.

        Parameters
        ----------
        p1:     Starting point of the arc (x, y).
        p2:     Ending point of the arc (x, y).
        centre: Centre of the arc (x, y).
        r:      Radius of the arc (m).
        n_pts:  Number of intervals for the spline.
        phi:    Rotation of coordinates (rad).
        """
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        pt1_rot = Curves._rotate_forward(pt1, sin_phi=sin_phi, cos_phi=cos_phi)
        pt2_rot = Curves._rotate_forward(pt2, sin_phi=sin_phi, cos_phi=cos_phi)

        x1, y1 = pt1_rot[0], pt1_rot[1]
        x2, y2 = pt2_rot[0], pt2_rot[1]
        x_c, y_c = centre[0], centre[1]

        # Angle changes across the curve
        ang_s = np.arctan2((y1 - y_c), (x1 - x_c))
        ang_e = np.arctan2((y2 - y_c), (x2 - x_c))
        angles = np.linspace(ang_s, ang_e, n_pts)

        # Calculate x, y coordinates
        x = x_c + r * np.cos(angles)
        y = y_c + r * np.sin(angles)

        # Inverse rotation back to original
        x_og = x * cos_phi - y * sin_phi
        y_og = x * sin_phi + y * cos_phi

        return x_og, y_og

    @staticmethod
    def three_point_cubic_spline_coordinates(
        pt1: Point,
        pt2: Point,
        pt3: Point,
        alpha1: float,
        alpha3: float,
        n_pts: int,
        phi: float = np.pi / 4,
    ) -> Coords2D:
        """
        Generates the coordinates for a cubic spline with evenly spaced intervals.

        Parameters
        ----------
        pt1:    The first control coordinate (x, y).
        pt2:    The second control coordinate (x, y).
        pt3:    The third control coordinate (x, y).
        alpha1: Starting angle of the spline (rad).
        alpha3: Ending angle of the spline (rad).
        n_pts:  Number of intervals for the spline.
        phi:    Rotation of coordinates (rad).
        """
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        pt1_rot = Curves._rotate_forward(pt1, sin_phi=sin_phi, cos_phi=cos_phi)
        pt2_rot = Curves._rotate_forward(pt2, sin_phi=sin_phi, cos_phi=cos_phi)
        pt3_rot = Curves._rotate_forward(pt3, sin_phi=sin_phi, cos_phi=cos_phi)

        x = np.array([pt1_rot[0], pt2_rot[0], pt3_rot[0]])
        y = np.array([pt1_rot[1], pt2_rot[1], pt3_rot[1]])

        y1_prime = np.tan(alpha1 - phi)
        y3_prime = np.tan(alpha3 - phi)

        c_spline = CubicSpline(x, y, bc_type=((1, y1_prime), (1, y3_prime)))

        # Evenly spaced spline with 'n_points' points.
        x_eq = np.linspace(x[0], x[-1], n_pts)
        y_eq = c_spline(x_eq)

        # Inverse rotation back to original
        x_eq_og = x_eq * cos_phi - y_eq * sin_phi
        y_eq_og = x_eq * sin_phi + y_eq * cos_phi

        return x_eq_og, y_eq_og

    @staticmethod
    def estimate_three_point_spline_inputs(
        pt1,
        pt3,
        alpha1: float = 0,
        alpha3: float = np.radians(90),
        round_coner: float = np.sqrt(2) - 1,
    ) -> CurveSettings:
        """
        Returns the default control point for a three point cubic spline:
            - First point of the curve (x,y).
            - Second point of the curve (x,y).
            - Third point of the curve (x,y).
            - Inlet flow angle (rad).
            - Outlet flow angle (rad).
            - The rotation of coordinates (rad).

        Directional vectors:
            For alpha = 0:
                The directional vector will be (1, 0), which means it
                is pointing along +x axial.
            For alpha = pi/2:
                The directional vector will be (0, 1), meaning it is
                pointing along +y radial.

        The equations:
            Extend each point so that they meet:
                pt1 + dis1 * dir_vec1 = pt3 + dis3 * dir_vec3
            Rearrange:
                dis1 * dir_vec1 - dis3 * dir_vec3 = pt3 - pt1
            In matrix form:
                [dir_vec1_x, - dir_vec3_x] [dis1] = pt3_x - pt1_x
                [dir_vec1_y, - dir_vec3_y] [dis3] = pt3_y - pt1_y

        Parameters
        ----------
        pt1:         Starting point of the curve (x, y).
        pt3:         Ending point of the curve (x, y).
        alpha1:      Inlet flow angle (rad).
        alpha3:      Outlet flow angle (rad).
        round_coner: Value for controlling how round the coner would be,
                     round_coner = 0 will be a straight line, a larger
                     value would have a tighter bend.
        """
        pt1_arr, pt3_arr = np.array(pt1), np.array(pt3)
        dir_vec1 = np.array([np.cos(alpha1), np.sin(alpha1)])
        dir_vec3 = np.array([np.cos(alpha3), np.sin(alpha3)])

        # Flip for correct layout: [[x1, x3], [y1, y3]]
        A = np.array([dir_vec1, -dir_vec3]).T

        mid_pt = (pt1_arr + pt3_arr) / 2
        if abs(np.linalg.det(A)) < 1e-12:
            # Determinant is zero, the lines will not meet
            pt2 = mid_pt
        else:
            # The second point will be somewhere between the mid
            # point and the corner.
            dis1, _ = np.linalg.solve(A, pt3_arr - pt1_arr)
            pt_corner = pt1_arr + dis1 * dir_vec1
            pt2 = mid_pt + round_coner * (pt_corner - mid_pt)

        phi = (alpha1 + alpha3) / 2
        return {
            "pt1": pt1,
            "pt2": tuple(pt2),
            "pt3": pt3,
            "alpha1": alpha1,
            "alpha3": alpha3,
            "phi": phi,
        }

    @staticmethod
    def _rotate_forward(
        pt: Point,
        sin_phi: float,
        cos_phi: float,
    ) -> Point:
        """
        Rotates the coordinates to avoid error at pi/2.

        Parameters
        ----------
        pt:      Coordinate (x, y).
        sin_phi: sin(phi) value.
        cos_phi: cos(phi) value.
        """
        x = pt[0]
        y = pt[1]
        x_new = x * cos_phi + y * sin_phi
        y_new = y * cos_phi - x * sin_phi
        return x_new, y_new

    @staticmethod
    def bezier_curve_coordinates(ctrl_pts: Points, n_pts: int) -> Coords2D:
        """
        Generates the coordinates for the Bezier curve from the given control points.

        Control points:
            pt0, pt1, ..., ptn

        Parameters
        ----------
        ctrl_pts: A list of (x, y) control point coordinates of the Bezier curve.
        n_pts:    Number of coordinate points to be generated on the curve.
        """
        pts_arr = np.array(ctrl_pts, dtype=float)
        deg = len(pts_arr) - 1
        knots = np.r_[np.zeros(deg + 1), np.ones(deg + 1)]
        curve = BSpline(knots, pts_arr, deg)
        t = np.linspace(0, 1, n_pts)
        curve_coords = curve(t)
        x, y = curve_coords[:, 0], curve_coords[:, 1]
        return x, y

    # NEEDS CHANGE
    # ------------
    # This function adds a total of two points to the existing curves which will not
    # match the resolution with the other arrays of the compoents (e.g. M-Prime).
    # Modify the function so that it replaces the first point instead.
    @staticmethod
    def add_linear_ends_to_curve(
        alpha_s: float,
        alpha_e: float,
        start_pt: Point,
        end_pt: Point,
        x: Array1D,
        y: Array1D,
        l1: float,
        l2: float,
    ) -> Coords2D:
        """
        Add one point at each end of a curve, for the linear segments.

        Parameters
        ----------
        alpha_s:  Starting angle of the circular arc (rad).
        alpha_e:  Ending angle of the circular arc (rad).
        start_pt: Starting point (x, y).
        end_pt:   Ending point (x, y).
        x:        A array of x-coordinates of the curve.
        y:        A array of y-coordinates of the curve.
        l1:       Length of the linear line at the start.
        l2:       Length of the linear line at the end.
        """
        x_s, y_s = start_pt[0], start_pt[1]
        x_e, y_e = end_pt[0], end_pt[1]

        # Shift coordinates of both ends of the curve
        x1_new, y1_new = x_s + l1 * np.cos(alpha_s), y_s + l1 * np.sin(alpha_s)
        x2_new, y2_new = x_e - l2 * np.cos(alpha_e), y_e - l2 * np.sin(alpha_e)

        if l1 > 0:
            # Add linear line starting coordinates at the start
            x = np.insert(x, 0, x_s)
            y = np.insert(y, 0, y_s)

        if l2 > 0:
            # Add linear line ending coordinates at the end
            x = np.append(x, x_e)
            y = np.append(y, y_e)

        return x, y

    @staticmethod
    def ellipse_tip_coordinates(
        x: Array1D,
        y: Array1D,
        R: float,
        R2: float,
        tip_loc: Literal["front", "end"],
    ) -> Array1D:
        """
        Generate coordinates for just the rounded ellipse tip for blade edges.

        Parameters
        ----------
        x:       An array of x values of the original curve.
        y:       An array of y values of the original curve.
        R:       Semi-axis alon x.
        R2:      Semi-axis along y.
        tip_loc: Set front or end of the curve to modify.
        """
        if tip_loc == "front":
            x_tip = x[0]
        else:
            x_tip = x[-1]
        u = np.clip(np.abs(x - x_tip) / R, 0.0, 1.0)
        return y + R2 * np.sqrt(1.0 - (1.0 - u) ** 2)
