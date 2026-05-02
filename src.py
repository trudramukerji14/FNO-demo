import gpytorch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.path import Path
from gpytorch.kernels import MaternKernel, ScaleKernel
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union

# =============================================================================
# GP Classes
# =============================================================================


class CompositeFieldModel:
    """
    Defines a composite field, made from:
     - A background field
     - A target difference field

    Both fields are GPs, but the target difference field has two additions:
     - Softplus to avoid negative values,
     - Blended with zeros with a exponential weighting kernel centered at the origin.
    """

    def __init__(self, background_gp_model, target_gp_model, blend_radius=0.2):
        self.background_model = background_gp_model
        self.target_model = target_gp_model
        self.blend_radius = blend_radius

    def _blend_weight(self, x):
        dist_from_center = torch.sqrt(torch.sum(x**2, dim=1))
        weight = torch.exp(-0.5 * (dist_from_center / self.blend_radius) ** 2)
        return weight  # (n_points,)

    def compose_models(self, bg_samples, tgt_samples, x):
        # Transform the target field with a softplus to avoid negative values
        tgt_samples = torch.nn.functional.softplus(tgt_samples)

        # Blend with zeros
        weight = self._blend_weight(x)
        zeros_field = torch.zeros_like(bg_samples)
        tgt_weighted_samples = (1 - weight) * zeros_field + weight * tgt_samples

        # Return summed field
        return bg_samples + tgt_weighted_samples

    def __call__(self, x):
        # Returns a distribution-like object with a .sample() method

        # Sample base fields
        bg_dist = self.background_model(x)
        tgt_dist = self.target_model(x)

        def sampler(sample_shape):
            # sample_shape is torch.Size([n_samples])
            n_samples = sample_shape[0] if len(sample_shape) > 0 else 1

            bg_samples = bg_dist.sample(torch.Size([n_samples]))
            tgt_samples = tgt_dist.sample(torch.Size([n_samples]))

            return self.compose_models(bg_samples, tgt_samples, x)

        return SampleOnlyDistribution(sampler)


class SampleOnlyDistribution:
    """
    Minimal distribution-like object that only supports .sample().
    """

    def __init__(self, sampler_fn):
        """
        sampler_fn: function(shape: torch.Size) -> Tensor
        """
        self._sampler_fn = sampler_fn

    def sample(self, sample_shape=torch.Size()):
        return self._sampler_fn(sample_shape)

    # Optional: support rsample for API compatibility
    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape)


class ThreeDimAnisoGP(torch.nn.Module):
    def __init__(
        self,
        mean_value=0.0,
        variance_value=1.0,
        covar_length=0.2,
        max_anisotropy_stretch=3.0,
        anisotropy_strength=1.0,
    ):
        super().__init__()

        # Mean and margninal covariance modules
        self.mean_value = mean_value
        self.covar_module_projection = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module_projection.outputscale = torch.tensor(variance_value)
        self.covar_module_projection.base_kernel.lengthscale = torch.tensor(
            covar_length
        )

        # Define anisotropic projection matrix
        R = random_rotation_3d()
        D = random_stretch(
            max_stretch=max_anisotropy_stretch, anisotropy_strength=anisotropy_strength
        )
        proj = R @ D
        self.register_buffer("projection", proj)

    def forward(self, x):
        proj_x = x.matmul(self.projection)
        covar_x = self.covar_module_projection(proj_x)
        mean_x = torch.full(
            (x.size(0),), self.mean_value, dtype=x.dtype, device=x.device
        )
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MaternGeologyPrior(ThreeDimAnisoGP):
    def __init__(self, mean_value, variance_value, covar_length, nu=1.5):
        # We override the kernel here but keep Will's structure
        super().__init__(mean_value, variance_value, covar_length)
        
        # nu=1.5 is the Matérn 3/2 (once differentiable)
        # ard_num_dims=3 for our 3D voxel grid
        self.covar_module = ScaleKernel(
            MaternKernel(nu=nu, ard_num_dims=3)
        )
        self.covar_module.base_kernel.lengthscale = covar_length
        self.covar_module.outputscale = variance_value

def random_rotation_3d():
    # Generate a random 3D rotation matrix using QR decomposition
    # Ensure right-handedness (det = +1)
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def random_stretch(max_stretch=5.0, anisotropy_strength=1.0):
    # Sample log-stretches, enforce max = 0, exponentiate, scale by max_stretch
    # Anisotropy strength controls how similar the stretches are to each other
    logs = anisotropy_strength * torch.randn(3)
    logs -= logs.max()
    d = torch.exp(logs)
    d *= max_stretch
    return torch.diag(d)


# =============================================================================
# Geometry Functions
# =============================================================================


# Polygon generation functions
def generate_random_polygon(center, radius_range=(10, 25), n_vertices=8):
    """
    Generate a random convex polygon using scipy ConvexHull.

    Parameters
    ----------
    center : tuple
        (x, y) center point for polygon
    radius_range : tuple
        (min_radius, max_radius) for vertex distances from center
    n_vertices : int
        Number of random points to generate for hull computation

    Returns
    -------
    ndarray
        (N, 2) array of polygon vertices in counter-clockwise order
    """
    angles = np.random.uniform(0, 2 * np.pi, n_vertices)
    radii = np.random.uniform(radius_range[0], radius_range[1], n_vertices)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    points = np.column_stack([x, y])

    hull = ConvexHull(points)
    return points[hull.vertices]


def generate_polygon_survey(
    physical_bounds, n_polygons=5, radius_range=(10, 25), padding=15.0, seed=None
):
    """
    Generate multiple random polygons for survey regions.

    Parameters
    ----------
    physical_bounds : dict
        Domain bounds {'x': (xmin, xmax), 'y': (ymin, ymax)}
    n_polygons : int
        Number of polygons to generate
    radius_range : tuple
        (min_radius, max_radius) for polygon sizes
    padding : float
        Padding from domain edges (ensures polygons stay in bounds)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list
        List of polygon vertex arrays [(N1, 2), (N2, 2), ...]
    """
    if seed is not None:
        np.random.seed(seed)

    xmin, xmax = physical_bounds["x"]
    ymin, ymax = physical_bounds["y"]

    polygons = []
    for _ in range(n_polygons):
        # Random center within padded bounds
        cx = np.random.uniform(xmin + padding, xmax - padding)
        cy = np.random.uniform(ymin + padding, ymax - padding)

        # Generate polygon
        poly = generate_random_polygon((cx, cy), radius_range)
        polygons.append(poly)

    return polygons


# Point-in-polygon testing function
def points_in_polygons(points, polygon_list):
    """
    Test if points lie within any of the given polygons.

    Parameters
    ----------
    points : ndarray
        (N, 2) array of test point coordinates
    polygon_list : list
        List of (M_i, 2) polygon vertex arrays

    Returns
    -------
    ndarray
        (N,) boolean mask, True where point is inside any polygon
    """
    mask = np.zeros(len(points), dtype=bool)
    for poly_vertices in polygon_list:
        path = Path(poly_vertices)
        mask |= path.contains_points(points)
    return mask


# Point sampling function
def sample_points_in_polygons(
    physical_bounds, polygon_list, grid_density=64, height=15.0
):
    """
    Create regular grid and filter to points within polygon regions.

    Parameters
    ----------
    physical_bounds : dict
        Domain bounds
    polygon_list : list
        List of polygon vertex arrays
    grid_density : int
        Grid resolution for sampling
    height : float
        Z-coordinate (observation height)

    Returns
    -------
    ndarray
        (N, 3) array of observation points (x, y, z)
    """
    xmin, xmax = physical_bounds["x"]
    ymin, ymax = physical_bounds["y"]

    # Create dense regular grid
    x_grid = np.linspace(xmin, xmax, grid_density)
    y_grid = np.linspace(ymin, ymax, grid_density)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    grid_points_2d = np.column_stack([xx.ravel(), yy.ravel()])

    # Filter to points inside polygons
    inside_mask = points_in_polygons(grid_points_2d, polygon_list)
    filtered_points = grid_points_2d[inside_mask]

    # Add height coordinate
    obs_xyz = np.column_stack(
        [
            filtered_points[:, 0],
            filtered_points[:, 1],
            np.full(len(filtered_points), height),
        ]
    )

    return obs_xyz


# Helper functions for unified polygon boundary plotting
def detect_overlapping_groups(polygon_list):
    """
    Group overlapping polygons into connected components.

    Parameters
    ----------
    polygon_list : list of ndarray
        List of (N, 2) polygon vertex arrays

    Returns
    -------
    list of list of int
        List of groups, where each group is a list of polygon indices
        Example: [[0, 2], [1, 3, 4]] means polygons 0,2 overlap and 1,3,4 overlap
    """
    # Convert numpy arrays to Shapely Polygons
    shapely_polys = [Polygon(poly) for poly in polygon_list]

    # Build graph: nodes are polygon indices, edges connect intersecting polygons
    G = nx.Graph()
    G.add_nodes_from(range(len(shapely_polys)))

    # Check all pairs for overlap (including touching edges)
    for i in range(len(shapely_polys)):
        for j in range(i + 1, len(shapely_polys)):
            # Use intersects() to include touching polygons
            if shapely_polys[i].intersects(shapely_polys[j]):
                G.add_edge(i, j)

    # Find connected components
    components = list(nx.connected_components(G))

    # Convert sets to sorted lists for consistency
    groups = [sorted(list(comp)) for comp in components]

    return groups


def compute_unified_boundaries(polygon_list, groups):
    """
    Compute unified exterior boundaries for each polygon group.

    Parameters
    ----------
    polygon_list : list of ndarray
        List of (N, 2) polygon vertex arrays
    groups : list of list of int
        Groups of overlapping polygon indices

    Returns
    -------
    list of ndarray
        List of (M, 2) boundary coordinate arrays, one per group
    """
    boundaries = []

    for group in groups:
        # Get polygons in this group
        group_polys = [Polygon(polygon_list[i]) for i in group]

        # Compute union
        if len(group_polys) == 1:
            # Single polygon, no union needed
            union = group_polys[0]
        else:
            # Union all polygons in group
            union = unary_union(group_polys)

        # Extract exterior boundary
        # Note: unary_union can return MultiPolygon if there are disjoint pieces
        # But our grouping ensures connectivity, so should be Polygon
        if union.geom_type == "Polygon":
            # Get exterior coordinates as numpy array
            coords = np.array(union.exterior.coords)
            boundaries.append(coords)
        elif union.geom_type == "MultiPolygon":
            # Shouldn't happen with our grouping, but handle gracefully
            # Extract all exterior boundaries
            for poly in union.geoms:
                coords = np.array(poly.exterior.coords)
                boundaries.append(coords)

    return boundaries


# =============================================================================
# Animation Functions
# =============================================================================


def create_survey_animation(
    susc_model_3d,
    fno_model,
    physical_bounds,
    grid_shape,
    n_frames=10,
    fps=2,
    output_filename="survey_animation.gif",
    n_polygons=5,
    radius_range=(10, 25),
    padding=15.0,
    grid_density=64,
    obs_height=15.0,
):
    """
    Create an animation showing different survey geometries across frames.

    Parameters
    ----------
    susc_model_3d : ndarray
        3D susceptibility model (nx, ny, nz)
    fno_model : torch.nn.Module
        Trained FNO model
    physical_bounds : dict
        Domain bounds {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
    grid_shape : tuple
        Shape of 3D grid (nx, ny, nz)
    n_frames : int
        Number of frames (different geometries) in animation
    fps : int or float
        Frames per second for output GIF
    output_filename : str
        Output GIF filename
    n_polygons : int
        Number of polygons per frame
    radius_range : tuple
        (min_radius, max_radius) for polygon sizes
    padding : float
        Padding from domain edges
    grid_density : int
        Grid resolution for sampling observation points
    obs_height : float
        Observation height (z-coordinate)

    Returns
    -------
    str
        Path to saved GIF file
    """
    xmin, xmax = physical_bounds["x"]
    ymin, ymax = physical_bounds["y"]

    print(f"Generating animation with {n_frames} frames...")
    print(f"  Polygons per frame: {n_polygons}")
    print(f"  Output: {output_filename}")
    print(f"  FPS: {fps}")

    # Step 1: Pre-generate all polygon geometries and compute FNO predictions
    frame_data = []

    # Compute FNO prediction on full grid once
    x_input = torch.tensor(susc_model_3d).unsqueeze(0).unsqueeze(0)
    fno_model.eval()
    with torch.no_grad():
        fno_pred_3d = fno_model(x_input)
        fno_pred_grid = fno_pred_3d[..., -1].squeeze().numpy()  # (nx, ny)

    # Set up interpolator
    x_grid = np.linspace(xmin, xmax, grid_shape[0])
    y_grid = np.linspace(ymin, ymax, grid_shape[1])
    interp = RegularGridInterpolator(
        (x_grid, y_grid), fno_pred_grid, bounds_error=False, fill_value=None
    )

    print("  Generating geometries and predictions...")
    for i in range(n_frames):
        # Generate polygon geometry with different seed
        polygon_list = generate_polygon_survey(
            physical_bounds,
            n_polygons=n_polygons,
            radius_range=radius_range,
            padding=padding,
            seed=42 + i,  # Different seed for each frame
        )

        # Sample observation points
        obs_xyz = sample_points_in_polygons(
            physical_bounds, polygon_list, grid_density=grid_density, height=obs_height
        )

        # Interpolate FNO to observation points
        fno_pred_obs = interp(obs_xyz[:, :2])

        # Compute unified boundaries
        groups = detect_overlapping_groups(polygon_list)
        unified_boundaries = compute_unified_boundaries(polygon_list, groups)

        # Create masked visualization grid
        viz_density = 100
        x_viz = np.linspace(xmin, xmax, viz_density)
        y_viz = np.linspace(ymin, ymax, viz_density)
        xx_viz, yy_viz = np.meshgrid(x_viz, y_viz, indexing="ij")
        viz_points_2d = np.column_stack([xx_viz.ravel(), yy_viz.ravel()])

        fno_pred_viz = interp(viz_points_2d).reshape(viz_density, viz_density)
        viz_mask = points_in_polygons(viz_points_2d, polygon_list)
        fno_pred_masked = np.ma.masked_where(
            ~viz_mask.reshape(viz_density, viz_density), fno_pred_viz
        )

        frame_data.append(
            {
                "polygon_list": polygon_list,
                "groups": groups,
                "unified_boundaries": unified_boundaries,
                "obs_xyz": obs_xyz,
                "fno_pred_obs": fno_pred_obs,
                "fno_pred_masked": fno_pred_masked,
            }
        )

        if (i + 1) % 5 == 0 or (i + 1) == n_frames:
            print(f"    Generated {i + 1}/{n_frames} frames")

    # Step 2: Determine global colorbar limits
    all_predictions = np.concatenate([fd["fno_pred_obs"] for fd in frame_data])
    vmin_global = np.nanmin(all_predictions)
    vmax_global = np.nanmax(all_predictions)

    print(f"  Global colorbar range: [{vmin_global:.2f}, {vmax_global:.2f}] nT")

    # Step 3: Create animation
    print("  Creating animation...")
    fig, ax = plt.subplots(figsize=(10, 10))

    def update_frame(frame_idx):
        ax.clear()

        data = frame_data[frame_idx]
        polygon_list = data["polygon_list"]
        groups = data["groups"]
        obs_xyz = data["obs_xyz"]
        fno_pred_masked = data["fno_pred_masked"]

        # Background: full FNO prediction
        ax.imshow(
            fno_pred_grid.T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
            alpha=0.2,
        )

        # Main plot: masked FNO prediction
        ax.imshow(
            fno_pred_masked.T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
            vmin=vmin_global,
            vmax=vmax_global,
        )

        # Draw unified fill regions
        for group in groups:
            group_polys = [Polygon(polygon_list[i]) for i in group]
            union = unary_union(group_polys)

            if union.geom_type == "Polygon":
                patch = MPLPolygon(
                    np.array(union.exterior.coords),
                    facecolor="none",
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.add_patch(patch)
            elif union.geom_type == "MultiPolygon":
                for poly in union.geoms:
                    patch = MPLPolygon(
                        np.array(poly.exterior.coords),
                        facecolor="none",
                        edgecolor="black",
                        linewidth=2,
                        alpha=0.8,
                    )
                    ax.add_patch(patch)

        # Draw observation points
        ax.scatter(
            obs_xyz[:, 0],
            obs_xyz[:, 1],
            c="yellow",
            s=1,
            alpha=0.3,
            edgecolors="none",
        )

        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_title(
            f"Survey Geometry {frame_idx + 1}/{n_frames}\n"
            f"{len(polygon_list)} Polygons, {len(obs_xyz)} Observation Points",
            fontsize=14,
        )
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        return [ax]

    # Create animation
    anim = FuncAnimation(
        fig, update_frame, frames=n_frames, interval=1000 / fps, blit=False
    )

    # Add colorbar (outside the animation loop)
    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global), cmap="RdBu_r"
        ),
        ax=ax,
        label="Magnetic Anomaly (nT)",
    )

    # Save as GIF
    print(f"  Saving animation to {output_filename}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer)
    plt.close(fig)

    print("✓ Animation saved successfully!")
    print(f"  File: {output_filename}")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {n_frames / fps:.1f} seconds")

    return output_filename


def create_gp_samples_animation(
    grid_points_torch,
    grid_shape,
    physical_bounds,
    n_frames=10,
    fps=2,
    output_filename="gp_samples.gif",
    seed_start=0,
):
    """
    Create an animation showing different GP model samples.

    Layout matches Section 1 visualization: 1x3 subplots with X, Y, and Z slices.

    Parameters
    ----------
    grid_points_torch : torch.Tensor
        Grid points for sampling (N, 3) tensor
    grid_shape : tuple
        Shape of 3D grid (nx, ny, nz)
    physical_bounds : dict
        Domain bounds {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
    n_frames : int
        Number of frames (different samples) in animation
    fps : int or float
        Frames per second for output GIF
    output_filename : str
        Output GIF filename
    seed_start : int
        Starting seed for random number generator

    Returns
    -------
    str
        Path to saved GIF file
    """

    xmin, xmax = physical_bounds["x"]
    ymin, ymax = physical_bounds["y"]
    zmin, zmax = physical_bounds["z"]

    print(f"Generating GP animation with {n_frames} frames...")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Output: {output_filename}")
    print(f"  FPS: {fps}")

    # Step 1: Pre-generate all samples
    print("  Generating GP samples...")
    samples = []

    for i in range(n_frames):
        torch.manual_seed(seed_start + i)  # Set seed for reproducibility

        # Create NEW GP model for each frame (different random rotations/stretches)
        background_model_i = ThreeDimAnisoGP(
            mean_value=-4.0, variance_value=0.2, covar_length=160.0
        )
        target_diff_model_i = ThreeDimAnisoGP(
            mean_value=0.0, variance_value=1.0, covar_length=20.0
        )
        gp_model_i = CompositeFieldModel(
            background_model_i,
            target_diff_model_i,
            blend_radius=20.0,
        )

        with torch.no_grad():
            dist = gp_model_i(grid_points_torch)
            sample = dist.sample(torch.Size([1])).squeeze().numpy()

        # Reshape to 3D grid
        sample_3d = sample.reshape(grid_shape)
        samples.append(sample_3d)

        if (i + 1) % 5 == 0 or (i + 1) == n_frames:
            print(f"    Generated {i + 1}/{n_frames} samples")

    # Step 2: Determine global colorbar limits
    samples_array = np.array(samples)
    vmin_global = samples_array.min()
    vmax_global = samples_array.max()

    print(f"  Global colorbar range: [{vmin_global:.4f}, {vmax_global:.4f}]")

    # Step 3: Create animation
    print("  Creating animation...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Add flag to prevent duplicate colorbars
    colorbar_added = {"done": False}

    def update_frame(frame_idx):
        for ax in axes:
            ax.clear()

        model_3d = samples[frame_idx]

        # 1. Constant X-slice (Plane Y-Z)
        ax = axes[0]
        ax.imshow(
            model_3d[grid_shape[0] // 2, :, :].T,
            origin="lower",
            extent=(ymin, ymax, zmin, zmax),
            vmin=vmin_global,
            vmax=vmax_global,
            cmap="viridis",
        )
        ax.set_title("Const. X-slice (Y vs Z)", fontsize=12)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")

        # 2. Constant Y-slice (Plane X-Z)
        ax = axes[1]
        ax.imshow(
            model_3d[:, grid_shape[1] // 2, :].T,
            origin="lower",
            extent=(xmin, xmax, zmin, zmax),
            vmin=vmin_global,
            vmax=vmax_global,
            cmap="viridis",
        )
        ax.set_title("Const. Y-slice (X vs Z)", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

        # 3. Constant Z-slice (Plane X-Y)
        ax = axes[2]
        im3 = ax.imshow(
            model_3d[:, :, grid_shape[2] // 2].T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            vmin=vmin_global,
            vmax=vmax_global,
            cmap="viridis",
        )
        ax.set_title("Const. Z-slice (X vs Y)", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add colorbar (first frame of the animation loop)
        # Position it on the right side of the figure
        if not colorbar_added["done"]:
            plt.colorbar(
                im3,
                ax=axes[2],
                label="Log Susceptibility",
            )
            colorbar_added["done"] = True

        # Add overall title
        fig.suptitle(f"GP Sample {frame_idx + 1}/{n_frames}", fontsize=14, y=0.98)

        plt.tight_layout()

        return list(axes)

    # Create animation
    anim = FuncAnimation(
        fig, update_frame, frames=n_frames, interval=1000 / fps, blit=False
    )

    # Save as GIF
    print(f"  Saving animation to {output_filename}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer)
    plt.close(fig)

    print("✓ Animation saved successfully!")
    print(f"  File: {output_filename}")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {n_frames / fps:.1f} seconds")

    return output_filename


def create_3d_summary_animation(
    susc_models,
    mag_anomalies,
    physical_bounds,
    grid_shape,
    n_frames=15,
    fps=2,
    output_filename="summary_3d.gif",
    obs_height=15.0,
    obs_padding=0.0,
    elev=28,
    azim=-50,
):
    """
    Create an animation showing 3D subsurface susceptibility (as a voxel cloud
    at two threshold levels) with the 2D magnetic anomaly floating above.

    Parameters
    ----------
    susc_models : list of ndarray
        3D susceptibility arrays, each with shape (nx, ny, nz)
    mag_anomalies : list of ndarray
        2D magnetic anomaly arrays, each with shape (nx_obs, ny_obs)
    physical_bounds : dict
        Domain bounds {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
    grid_shape : tuple
        Shape of 3D grid (nx, ny, nz)
    n_frames : int
        Number of animation frames
    fps : int or float
        Frames per second
    output_filename : str
        Output GIF filename
    obs_height : float
        Z-coordinate of observation surface
    obs_padding : float
        Padding applied to observation grid edges
    elev : float
        Elevation angle for 3D view
    azim : float
        Azimuth angle for 3D view

    Returns
    -------
    str
        Path to saved GIF file
    """
    xmin, xmax = physical_bounds["x"]
    ymin, ymax = physical_bounds["y"]
    zmin, zmax = physical_bounds["z"]

    n_frames = min(n_frames, len(susc_models))

    # Upsample the grid for smoother rendering (especially z, which is very coarse)
    zoom_factors = (2, 2, 4)
    up_shape = tuple(int(s * f) for s, f in zip(grid_shape, zoom_factors))

    # Physical coordinate grids for the upsampled volume
    x_up = np.linspace(xmin, xmax, up_shape[0])
    y_up = np.linspace(ymin, ymax, up_shape[1])
    z_up = np.linspace(zmin, zmax, up_shape[2])
    xx_up, yy_up, zz_up = np.meshgrid(x_up, y_up, z_up, indexing="ij")

    print(f"Generating 3D summary animation with {n_frames} frames...")
    print(f"  Upsampled grid: {up_shape}")
    print(f"  Output: {output_filename}")

    # ---- Global limits across all frames ----
    all_susc = np.stack(susc_models[:n_frames])
    level_high = np.percentile(all_susc, 90)
    level_mid = np.percentile(all_susc, 70)

    all_mag = np.stack(mag_anomalies[:n_frames])
    mag_vmin, mag_vmax = float(all_mag.min()), float(all_mag.max())

    print(f"  Threshold levels: mid={level_mid:.4f}, high={level_high:.4f} SI")
    print(f"  Magnetic anomaly range: [{mag_vmin:.1f}, {mag_vmax:.1f}] nT")

    # Colors from viridis (matching susceptibility plots in notebook)
    viridis = cm.viridis
    color_high = viridis(0.85)
    color_mid = viridis(0.40)

    # ---- Pre-compute filtered point clouds for each frame ----
    print("  Computing voxel clouds...")
    frame_data = []
    for i in range(n_frames):
        susc_up = zoom(susc_models[i], zoom_factors, order=1)

        # Intermediate level: between mid and high
        mid_mask = (susc_up >= level_mid) & (susc_up < level_high)
        mid_pts = (
            np.column_stack([xx_up[mid_mask], yy_up[mid_mask], zz_up[mid_mask]])
            if mid_mask.any()
            else None
        )

        # High level: above high
        high_mask = susc_up >= level_high
        high_pts = (
            np.column_stack([xx_up[high_mask], yy_up[high_mask], zz_up[high_mask]])
            if high_mask.any()
            else None
        )

        frame_data.append(
            {"mid_pts": mid_pts, "high_pts": high_pts, "mag_2d": mag_anomalies[i]}
        )
        if (i + 1) % 5 == 0 or (i + 1) == n_frames:
            n_mid = len(mid_pts) if mid_pts is not None else 0
            n_high = len(high_pts) if high_pts is not None else 0
            print(f"    {i + 1}/{n_frames}  ({n_mid} mid pts, {n_high} high pts)")

    # ---- Observation surface mesh ----
    obs_shape = mag_anomalies[0].shape
    obs_xmin, obs_xmax = xmin + obs_padding, xmax - obs_padding
    obs_ymin, obs_ymax = ymin + obs_padding, ymax - obs_padding
    x_obs = np.linspace(obs_xmin, obs_xmax, obs_shape[0])
    y_obs = np.linspace(obs_ymin, obs_ymax, obs_shape[1])
    xx_obs, yy_obs = np.meshgrid(x_obs, y_obs, indexing="ij")
    zz_obs = np.full_like(xx_obs, obs_height)

    mag_norm = Normalize(vmin=mag_vmin, vmax=mag_vmax)
    mag_cmap = cm.RdBu_r

    # ---- Build animation ----
    print("  Rendering frames...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Static colorbar for magnetic anomaly
    sm = cm.ScalarMappable(norm=mag_norm, cmap=mag_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, label="Magnetic Anomaly (nT)")

    def update_frame(idx):
        ax.cla()
        ax.computed_zorder = False  # Use artist draw order, not depth sorting
        data = frame_data[idx]

        # -- Subsurface bounding box wireframe --
        cx = [xmin, xmax]
        cy = [ymin, ymax]
        cz = [zmin, zmax]
        for xi in cx:
            for yi in cy:
                ax.plot([xi, xi], [yi, yi], cz, color="gray", lw=0.5, alpha=0.25)
        for xi in cx:
            for zi in cz:
                ax.plot([xi, xi], cy, [zi, zi], color="gray", lw=0.5, alpha=0.25)
        for yi in cy:
            for zi in cz:
                ax.plot(cx, [yi, yi], [zi, zi], color="gray", lw=0.5, alpha=0.25)

        # -- Intermediate susceptibility (translucent scatter) --
        if data["mid_pts"] is not None:
            pts = data["mid_pts"]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[color_mid],
                s=18,
                alpha=0.08,
                marker="s",
                edgecolors="none",
                depthshade=True,
            )

        # -- High susceptibility (more opaque scatter) --
        if data["high_pts"] is not None:
            pts = data["high_pts"]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[color_high],
                s=30,
                alpha=0.35,
                marker="s",
                edgecolors="none",
                depthshade=True,
            )

        # -- Observation surface border --
        bx = [obs_xmin, obs_xmax, obs_xmax, obs_xmin, obs_xmin]
        by = [obs_ymin, obs_ymin, obs_ymax, obs_ymax, obs_ymin]
        bz = [obs_height] * 5
        ax.plot(bx, by, bz, "k-", lw=1.2, alpha=0.5)

        # -- 2D magnetic anomaly surface (RdBu_r, matching notebook) --
        fcolors = mag_cmap(mag_norm(data["mag_2d"]))
        ax.plot_surface(
            xx_obs, yy_obs, zz_obs, facecolors=fcolors, shade=False, alpha=0.92
        )

        # -- Axis settings --
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, obs_height + 2)
        ax.set_xlabel("X (m)", labelpad=8)
        ax.set_ylabel("Y (m)", labelpad=8)
        ax.set_zlabel("Z (m)", labelpad=4)
        ax.set_box_aspect([1, 1, 0.55])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Sample {idx + 1}/{n_frames}", fontsize=14, pad=12)

        return [ax]

    anim = FuncAnimation(
        fig, update_frame, frames=n_frames, interval=1000 / fps, blit=False
    )

    print(f"  Saving to {output_filename}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer)
    plt.close(fig)

    print(f"  Done! {n_frames} frames, {fps} FPS, {n_frames / fps:.1f}s duration")
    return output_filename
