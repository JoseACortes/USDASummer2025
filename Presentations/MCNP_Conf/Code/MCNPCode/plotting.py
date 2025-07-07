import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

z_label_pad = -10

def plot_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, color, alpha=1, label=None, zorder=None):
    # Draw a rectangular box (RPP)
    x = [xmin, xmax]
    y = [ymin, ymax]
    z = [zmin, zmax]
    for s, e in [
        # bottom
        ([x[0], y[0], z[0]], [x[1], y[0], z[0]]),
        ([x[1], y[0], z[0]], [x[1], y[1], z[0]]),
        ([x[1], y[1], z[0]], [x[0], y[1], z[0]]),
        ([x[0], y[1], z[0]], [x[0], y[0], z[0]]),
        # top
        ([x[0], y[0], z[1]], [x[1], y[0], z[1]]),
        ([x[1], y[0], z[1]], [x[1], y[1], z[1]]),
        ([x[1], y[1], z[1]], [x[0], y[1], z[1]]),
        ([x[0], y[1], z[1]], [x[0], y[0], z[1]]),
        # sides
        ([x[0], y[0], z[0]], [x[0], y[0], z[1]]),
        ([x[1], y[0], z[0]], [x[1], y[0], z[1]]),
        ([x[1], y[1], z[0]], [x[1], y[1], z[1]]),
        ([x[0], y[1], z[0]], [x[0], y[1], z[1]])
    ]:
        ax.plot3D(*zip(s, e), color=color, alpha=alpha, zorder=zorder)
    if label:
        ax.text((xmin+xmax)/2, (ymin+ymax)/2, zmin+z_label_pad, label, color=color, zorder=zorder)



def plot_cylinder(ax, base, vec, radius, height, color, alpha=1, label=None, zorder=None):
    # Draw a cylinder (RCC) with flat ends
    x0, y0, z0 = base
    dx, dy, dz = vec
    # Normalize direction vector
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    if length == 0:
        return
    dx, dy, dz = dx/length, dy/length, dz/length
    # Create cylinder along z, then rotate
    z = np.linspace(0, height, 30)
    theta = np.linspace(0, 2*np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    # Build rotation matrix
    v = np.array([dx, dy, dz])
    v0 = np.array([0, 0, 1])
    if not np.allclose(v, v0):
        axis = np.cross(v0, v)
        angle = np.arccos(np.clip(np.dot(v0, v), -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        xyz = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
        xyz_rot = R @ xyz
        x_grid = xyz_rot[0].reshape(x_grid.shape)
        y_grid = xyz_rot[1].reshape(y_grid.shape)
        z_grid = xyz_rot[2].reshape(z_grid.shape)
    x_grid += x0
    y_grid += y0
    z_grid += z0
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)

    # Flat ends
    for zc in [0, height]:
        # Circle in xy-plane
        x_end = radius * np.cos(theta)
        y_end = radius * np.sin(theta)
        z_end = np.full_like(x_end, zc)
        # Rotate
        xyz_end = np.stack([x_end, y_end, z_end])
        if not np.allclose(v, v0):
            xyz_end = R @ xyz_end
        x_end = xyz_end[0] + x0
        y_end = xyz_end[1] + y0
        z_end = xyz_end[2] + z0
        # Use Poly3DCollection for flat ends
        verts = [list(zip(x_end, y_end, z_end))]
        poly = Poly3DCollection(verts, color=color, alpha=alpha, zorder=zorder)
        ax.add_collection3d(poly)

    if label:
        ax.text(x0, y0, z0+z_label_pad-10, label, color=color)

def plot_cone(ax, pos, vec, dir, length, color='red', alpha=1, label=None, zorder=None):
    """
    Draw a cone with end point at pos, pointing in direction vec, with angle from dir
    """
    x0, y0, z0 = pos
    dx, dy, dz = vec
    # Normalize direction vector
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    if norm == 0:
        return
    dx, dy, dz = dx/norm, dy/norm, dz/norm
    # Create cone base circle
    theta = np.linspace(0, 2*np.pi, 30)
    r = length * np.tan(np.radians(dir))
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    z_circle = np.zeros_like(x_circle)
    # Rotate circle to point in direction vec
    v0 = np.array([0, 0, 1])
    v = np.array([dx, dy, dz])
    if not np.allclose(v, v0):
        axis = np.cross(v0, v)
        angle = np.arccos(np.clip(np.dot(v0, v), -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        xyz = np.stack([x_circle.flatten(), y_circle.flatten(), z_circle.flatten()])
        xyz_rot = R @ xyz
        x_circle = xyz_rot[0].reshape(x_circle.shape)
        y_circle = xyz_rot[1].reshape(y_circle.shape)
        z_circle = xyz_rot[2].reshape(z_circle.shape)
    x_circle += x0 + dx * length
    y_circle += y0 + dy * length
    z_circle += z0 + dz * length
    # Draw cone surface
    for i in range(len(x_circle)):
        ax.plot([x0, x_circle[i]], [y0, y_circle[i]], [z0, z_circle[i]], color=color, alpha=alpha, zorder=zorder)
    if label:
        ax.text(x0, y0, (z0 * length / 2)+z_label_pad-10, label, color=color)
    


def plot_detector_and_emitter(ax):
    plot_cylinder(ax, base=(56, -5.0, -1.0), vec=(0.0, 20.3, 0.0), radius=4.5, height=20.3, color='red', alpha=1, label='Detector')
    
def plot_MCNP(ax):
    # Detector (surface 21)
    plot_cylinder(ax, base=(56, -5.0, -1.0), vec=(0.0, 20.3, 0.0), radius=4.5, height=20.3, color='red', alpha=1, label='Detector')

    # Shielding boxes (RPPs)
    plot_box(ax, 19, 29, -7.5, 7.5, -11, 9, color='blue', alpha=1, label='PbPE')
    plot_box(ax, 9, 19, 4, 9, -11, 9, color='cyan', alpha=1)
    plot_box(ax, 9, 19, -9, -4, -11, 9, color='cyan', alpha=1)
    plot_box(ax, 19, 29, 7.5, 12.5, -11, 9, color='cyan', alpha=1)
    plot_box(ax, 19, 29, -12.5, -7.5, -11, 9, color='cyan', alpha=1)
    plot_box(ax, 29, 34, -15, 15, -11, 9, color='cyan', alpha=1)
    plot_box(ax, 9, 19, -4, 4, 4, 9, color='cyan', alpha=1)
    plot_box(ax, -26, 26, 18, 28, -11, 9, color='green', alpha=1, label='BA1')
    plot_box(ax, -26, 26, -28, -18, -11, 9, color='green', alpha=1, label='BA2')
    plot_box(ax, -65, 65, -28, 28, 10, 10.5, color='magenta', alpha=1, label='Al')

    # Wheels (outer treads only for clarity)
    plot_cylinder(ax, base=(-2, 77, 8), vec=(0, 25, 0), radius=29, height=25, color='orange', alpha=0.2, label='Wheel 1')
    plot_cylinder(ax, base=(68, 77, 8), vec=(0, 25, 0), radius=29, height=25, color='orange', alpha=0.2, label='Wheel 2')
    plot_cylinder(ax, base=(-2, -77, 8), vec=(0, -25, 0), radius=29, height=25, color='orange', alpha=0.2, label='Wheel 3')
    plot_cylinder(ax, base=(68, -77, 8), vec=(0, -25, 0), radius=29, height=25, color='orange', alpha=0.2, label='Wheel 4')

    # Soil volume (bounding box)
    # plot_box(ax, -56, 56, -45, 45, 42, 92, color='brown', alpha=1, label='Soil')

def make_voxel_grid(midpoints, intensity):
    """
    Create a voxel grid from midpoints and intensity values.
    """
    # Create a grid of zeros
    grid_shape = (len(np.unique(midpoints[:, 0])),
                  len(np.unique(midpoints[:, 1])),
                  len(np.unique(midpoints[:, 2])))
    voxel_grid = intensity.reshape(grid_shape)
    return voxel_grid

def extra(ax):
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_proj_type('ortho')
    ax.invert_zaxis()

def vox_maker(cloud, midpoints, cell_portion):
    cell_mask = np.zeros(cell_portion.shape, dtype=bool)
    cell_mask[cloud] = True
    cell_mask_vox = make_voxel_grid(midpoints, intensity=cell_mask)
    return cell_mask_vox

def vox_plotter(ax, vox_mask, sides, color='cyan', zorder=2, alpha=.2):
    x_walls, y_walls, z_walls = sides
    # Convert wall lists to numpy arrays
    x = np.array(x_walls)
    y = np.array(y_walls)
    z = np.array(z_walls)
    X, Y, Z = np.meshgrid(x, y, -z+42, indexing='xy')
    ax.voxels(
        X, 
        Y, 
        Z, 
        vox_mask, 
        # facecolors=colors,
        facecolors=color,
        edgecolor='k',
        linewidth=0.1,
        alpha=alpha,
        zorder=zorder,
        # axlim_clip=True
        )   

# make a concave hull around the voxels
def concave_hull(ax, vox_mask, sides, color='cyan', zorder=2, alpha=.2):
    """
    Plot a convex hull (or concave hull if implemented) around the corners of the selected voxels.
    """
    flat_mask = vox_mask.flatten()
    x_walls, y_walls, z_walls = sides
    x = np.array(x_walls)
    y = np.array(y_walls)
    z = np.array(z_walls)

    # Get the indices of the selected voxels
    selected_indices = np.where(flat_mask)[0]
    # Get the shape of the voxel grid
    nx = len(x) - 1
    ny = len(y) - 1
    nz = len(z) - 1

    # Convert flat indices to 3D indices
    indices_3d = np.array(np.unravel_index(selected_indices, (ny, nx, nz), order='C')).T
    # switch x and y to match the original code's order
    indices_3d = indices_3d[:, [1, 0, 2]]  # (i, j, k) -> (j, i, k)
    # print(indices_3d.shape, 'indices_3d shape')

    # For each selected voxel, get its 8 corners
    corners = []
    for idx in indices_3d:
        i, j, k = idx
        voxel_corners = [
            [x[i],     y[j],     -z[k] + 42],
            [x[i+1],   y[j],     -z[k] + 42],
            [x[i],     y[j+1],   -z[k] + 42],
            [x[i+1],   y[j+1],   -z[k] + 42],
            [x[i],     y[j],     -z[k+1] + 42],
            [x[i+1],   y[j],     -z[k+1] + 42],
            [x[i],     y[j+1],   -z[k+1] + 42],
            [x[i+1],   y[j+1],   -z[k+1] + 42],
        ]
        corners.extend(voxel_corners)

    corners = np.unique(np.array(corners), axis=0)

    if len(corners) >= 4:
        hull = ConvexHull(corners)
        faces = [corners[simplex] for simplex in hull.simplices]
        ax_bounds = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        # remove faces that are outside the axis limits
        # Only keep faces that have at least one vertex inside the axis limits
        faces = [
            face for face in faces if np.any(
            (face[:, 0] >= ax_bounds[0][0]) & (face[:, 0] <= ax_bounds[0][1]) &
            (face[:, 1] >= ax_bounds[1][0]) & (face[:, 1] <= ax_bounds[1][1]) &
            (face[:, 2] >= ax_bounds[2][0]) & (face[:, 2] <= ax_bounds[2][1])
            )
        ]
        if len(faces) == 0:
            print("No faces to plot, skipping concave hull.")
            return
        poly3d = Poly3DCollection(
            faces,
            alpha=alpha,
            facecolor=color,
            # edgecolor='k',
            zorder=zorder,
            # linewidths=1,
        )
        ax.add_collection3d(poly3d)