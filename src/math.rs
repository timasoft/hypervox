use crate::DimMapping;
use crate::expr;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Per-phase timing for grid generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GridTimings {
    pub sign_grid_ms: f64,
    pub voxel_fill_ms: f64,
}

/// Configuration for N-dimensional to 3D spatial mapping.
/// Maps N dimensions (0..ndim) to 3D spatial axes (X, Y, Z).
/// Dimensions not mapped to any axis are held at fixed values.
/// Expression variables: x,y,z (spatial axes) and x0..x{N-1} (dimension coords).
#[derive(Debug)]
pub struct DimConfig {
    pub ndim: usize,
    /// Which dimension index varies along the X axis
    pub x_dim: usize,
    /// Which dimension index varies along the Y axis
    pub y_dim: usize,
    /// Which dimension index varies along the Z axis
    pub z_dim: usize,
    /// Fixed coordinate value for each dimension (used for non-spatial dims)
    pub fixed: Vec<f64>,
    /// Offset of the evaluation window in world units
    pub world_offset: (f64, f64, f64),
}

impl Default for DimConfig {
    fn default() -> Self {
        DimMapping::default().into()
    }
}

/// Generates a voxel grid of size `size^3` from an N-dimensional expression.
/// - `expr_str`: mathematical expression in terms of x,y,z (spatial) and x0..x{N-1} (dims)
/// - `base_color`: 24-bit RGB color (0xRRGGBB). Stored in grid as `base_color + 1`
/// - `world_half_extent`: half the size of the region in world units.
/// - `dim`: N-dimensional mapping config
pub fn generate_voxel_grid(
    size: usize,
    expr_str: &str,
    base_color: u32,
    world_half_extent: f64,
    dim: &DimConfig,
) -> Result<(Vec<u32>, GridTimings), String> {
    if size == 0 {
        return Ok((Vec::new(), GridTimings::default()));
    }

    if dim.x_dim >= dim.ndim || dim.y_dim >= dim.ndim || dim.z_dim >= dim.ndim {
        return Err(format!(
            "Axis mapping out of range: x_dim={}, y_dim={}, z_dim={} with ndim={}",
            dim.x_dim, dim.y_dim, dim.z_dim, dim.ndim
        ));
    }

    let expr = expr::parse(expr_str, dim)?;

    let node_dim = size + 1;
    let node_dim_sq = node_dim * node_dim;
    let size_sq = size * size;

    let mut sign_grid = vec![0i8; node_dim * node_dim_sq];
    let mut voxel_grid = vec![0u32; size * size_sq];

    let step = (world_half_extent * 2.0) / size as f64;
    let packed_color = (base_color & 0xFFFFFF).wrapping_add(1);

    let sign_start = Instant::now();

    #[cfg(target_arch = "wasm32")]
    compute_sign_grid(
        &mut sign_grid,
        &expr,
        node_dim,
        node_dim_sq,
        step,
        world_half_extent,
        dim,
    );

    #[cfg(not(target_arch = "wasm32"))]
    compute_sign_grid_par(
        &mut sign_grid,
        &expr,
        node_dim,
        node_dim_sq,
        step,
        world_half_extent,
        dim,
    );

    let sign_grid_ms = sign_start.elapsed().as_secs_f64() * 1000.0;

    let fill_start = Instant::now();

    for vz in 0..size {
        let base_z = vz * node_dim_sq;
        let voxel_base_z = vz * size_sq;

        for vy in 0..size {
            let base_y = base_z + vy * node_dim;
            let voxel_base_y = voxel_base_z + vy * size;

            for vx in 0..size {
                let base = base_y + vx;

                let s000 = sign_grid[base];
                let s100 = sign_grid[base + 1];
                let s010 = sign_grid[base + node_dim];
                let s110 = sign_grid[base + node_dim + 1];
                let s001 = sign_grid[base + node_dim_sq];
                let s101 = sign_grid[base + node_dim_sq + 1];
                let s011 = sign_grid[base + node_dim_sq + node_dim];
                let s111 = sign_grid[base + node_dim_sq + node_dim + 1];

                let has_pos = s000 >= 0
                    || s100 >= 0
                    || s010 >= 0
                    || s110 >= 0
                    || s001 >= 0
                    || s101 >= 0
                    || s011 >= 0
                    || s111 >= 0;
                let has_neg = s000 < 0
                    || s100 < 0
                    || s010 < 0
                    || s110 < 0
                    || s001 < 0
                    || s101 < 0
                    || s011 < 0
                    || s111 < 0;

                if has_pos && has_neg {
                    voxel_grid[voxel_base_y + vx] = packed_color;
                }
            }
        }
    }

    let voxel_fill_ms = fill_start.elapsed().as_secs_f64() * 1000.0;

    Ok((
        voxel_grid,
        GridTimings {
            sign_grid_ms,
            voxel_fill_ms,
        },
    ))
}

#[inline]
fn eval_sign(val: f64) -> i8 {
    if !val.is_finite() {
        0
    } else if val > 0.0 {
        1
    } else if val < 0.0 {
        -1
    } else {
        0
    }
}

#[inline]
fn init_fixed_vars(dim: &DimConfig) -> Vec<f64> {
    let mut vars = vec![0.0; dim.ndim];
    for (d, v) in vars.iter_mut().enumerate() {
        if d != dim.x_dim && d != dim.y_dim && d != dim.z_dim {
            *v = if d < dim.fixed.len() {
                dim.fixed[d]
            } else {
                0.0
            };
        }
    }

    vars
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn compute_sign_grid(
    sign_grid: &mut [i8],
    expr: &expr::Node,
    node_dim: usize,
    node_dim_sq: usize,
    step: f64,
    world_half_extent: f64,
    dim: &DimConfig,
) {
    let mut vars = init_fixed_vars(dim);

    let mut vars_options: Vec<Option<f64>> = vars.iter().copied().map(Some).collect();
    for idx in [dim.x_dim, dim.y_dim, dim.z_dim] {
        vars_options[idx] = None;
    }

    let mut expr = expr.clone();
    expr.pre_eval(&vars_options);

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    for nz in 0..node_dim {
        let fz = z0 + nz as f64 * step;
        vars[dim.z_dim] = fz;
        for ny in 0..node_dim {
            let fy = y0 + ny as f64 * step;
            vars[dim.y_dim] = fy;
            for nx in 0..node_dim {
                let fx = x0 + nx as f64 * step;
                vars[dim.x_dim] = fx;
                let idx = nx + ny * node_dim + nz * node_dim_sq;
                sign_grid[idx] = eval_sign(expr.eval(&vars));
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn compute_sign_grid_par(
    sign_grid: &mut [i8],
    expr: &expr::Node,
    node_dim: usize,
    node_dim_sq: usize,
    step: f64,
    world_half_extent: f64,
    dim: &DimConfig,
) {
    use rayon::prelude::*;

    let chunk_count = std::cmp::min(num_cpus::get(), node_dim);
    let chunk_size = (node_dim as f64 / chunk_count as f64).ceil() as usize;

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    let base_vars = init_fixed_vars(dim);

    let mut base_vars_options: Vec<Option<f64>> = base_vars.iter().copied().map(Some).collect();
    for idx in [dim.x_dim, dim.y_dim, dim.z_dim] {
        base_vars_options[idx] = None;
    }

    let mut expr = expr.clone();
    expr.pre_eval(&base_vars_options);

    let results: Vec<Vec<(usize, i8)>> = (0..node_dim)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|z_range| {
            let mut vars = base_vars.clone();
            let mut local_signs = Vec::with_capacity(z_range.len() * node_dim * node_dim);

            for &nz in z_range {
                let fz = z0 + nz as f64 * step;
                vars[dim.z_dim] = fz;
                for ny in 0..node_dim {
                    let fy = y0 + ny as f64 * step;
                    vars[dim.y_dim] = fy;
                    for nx in 0..node_dim {
                        let fx = x0 + nx as f64 * step;
                        vars[dim.x_dim] = fx;
                        let sign = eval_sign(expr.eval(&vars));
                        let idx = nx + ny * node_dim + nz * node_dim_sq;
                        local_signs.push((idx, sign));
                    }
                }
            }
            local_signs
        })
        .collect();

    for chunk in results {
        for (idx, sign) in chunk {
            sign_grid[idx] = sign;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_generation() {
        // Sphere of radius 2 in region [-5, 5]
        let dim = DimConfig::default();
        let (grid, _) =
            generate_voxel_grid(32, "x^2 + y^2 + z^2 - 4", 0xFF0000, 5.0, &dim).unwrap();
        let filled = grid.iter().filter(|&&v| v != 0).count();
        assert!(filled > 0 && filled < grid.len());
    }

    #[test]
    fn test_sinusoidal_surface() {
        // Test that sin function works
        let dim = DimConfig::default();
        let (grid, _) =
            generate_voxel_grid(16, "sin(x) + cos(y) + z", 0x00FF00, 8.0, &dim).unwrap();
        let filled = grid.iter().filter(|&&v| v != 0).count();
        // Should generate some voxels, but not all
        assert!(filled > 0 && filled < grid.len());
    }

    #[test]
    fn test_4d_nd_variables() {
        // 4D: x0=sphere radius, x1=x, x2=y, x3=z
        // expression uses x0 (dim 3) as extra, mapped such that X=x0, Y=x1, Z=x2, x3 fixed
        let dim = DimConfig {
            ndim: 4,
            x_dim: 1,
            y_dim: 2,
            z_dim: 3,
            fixed: vec![0.0, 0.0, 0.0, 0.0],
            ..DimConfig::default()
        };
        // x1^2 + x2^2 + x3^2 - x0^2 (sphere with radius x0)
        // with x3 mapped to Z, x0 fixed at 0.0 → radius 0 → no sphere
        let (grid, _) =
            generate_voxel_grid(16, "x1^2 + x2^2 + x3^2 - x0^2", 0x00FF00, 8.0, &dim).unwrap();
        let filled = grid.iter().filter(|&&v| v != 0).count();
        assert_eq!(filled, 0);
    }
}
