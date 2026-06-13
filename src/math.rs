use evalexpr::{
    ContextWithMutableFunctions, ContextWithMutableVariables, DefaultNumericTypes, EvalexprError,
    HashMapContext, Node, Value, build_operator_tree,
};

use crate::DimMapping;

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
) -> Result<Vec<u32>, String> {
    if size == 0 {
        return Ok(Vec::new());
    }

    if dim.x_dim >= dim.ndim || dim.y_dim >= dim.ndim || dim.z_dim >= dim.ndim {
        return Err(format!(
            "Axis mapping out of range: x_dim={}, y_dim={}, z_dim={} with ndim={}",
            dim.x_dim, dim.y_dim, dim.z_dim, dim.ndim
        ));
    }

    let tree: Node = build_operator_tree(expr_str).map_err(|e| format!("Parse error: {e}"))?;

    let node_dim = size + 1;
    let node_dim_sq = node_dim * node_dim;
    let size_sq = size * size;

    let mut sign_grid = vec![0i8; node_dim * node_dim_sq];
    let mut voxel_grid = vec![0u32; size * size_sq];

    let step = (world_half_extent * 2.0) / size as f64;
    let packed_color = (base_color & 0xFFFFFF).wrapping_add(1);

    #[cfg(target_arch = "wasm32")]
    compute_sign_grid(
        &mut sign_grid,
        &tree,
        node_dim,
        node_dim_sq,
        step,
        world_half_extent,
        dim,
    )?;

    #[cfg(not(target_arch = "wasm32"))]
    compute_sign_grid_par(
        &mut sign_grid,
        &tree,
        node_dim,
        node_dim_sq,
        step,
        world_half_extent,
        dim,
    )?;

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

    Ok(voxel_grid)
}

/// Helper: extract a single f64 from function args (handles both direct value and 1-element tuple)
#[inline]
fn get_float_arg(
    args: &Value<DefaultNumericTypes>,
) -> Result<f64, EvalexprError<DefaultNumericTypes>> {
    let val = match args {
        Value::Tuple(t) if t.len() == 1 => &t[0],
        v => v,
    };
    val.as_float()
}

/// Helper: extract two f64 values from function args (expects a 2-element tuple)
#[inline]
fn get_two_float_args(
    args: &Value<DefaultNumericTypes>,
) -> Result<(f64, f64), EvalexprError<DefaultNumericTypes>> {
    let tuple = args.as_tuple()?;
    if tuple.len() != 2 {
        return Err(EvalexprError::wrong_function_argument_amount(
            tuple.len(),
            2,
        ));
    }
    let a = tuple[0].as_float()?;
    let b = tuple[1].as_float()?;
    Ok((a, b))
}

/// Registers common math functions into the evalexpr context.
/// HashMapContext doesn't include built-in functions by default, so we add them manually.
#[inline]
fn register_math_functions(ctx: &mut HashMapContext) -> Result<(), String> {
    use evalexpr::Function;

    // Trigonometric (radians)
    ctx.set_function(
        "sin".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.sin()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "cos".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.cos()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "tan".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.tan()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "asin".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.asin()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "acos".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.acos()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "atan".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.atan()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "atan2".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_two_float_args(args).map(|(y, x)| Value::Float(y.atan2(x)))
        }),
    )
    .map_err(|e| e.to_string())?;

    // Hyperbolic
    ctx.set_function(
        "sinh".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.sinh()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "cosh".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.cosh()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "tanh".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.tanh()))
        }),
    )
    .map_err(|e| e.to_string())?;

    // Exponential/logarithmic
    ctx.set_function(
        "sqrt".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.sqrt()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "cbrt".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.cbrt()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "exp".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.exp()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "ln".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.ln()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "log10".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.log10()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "log2".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.log2()))
        }),
    )
    .map_err(|e| e.to_string())?;

    // Power function (dynamic exponent)
    ctx.set_function(
        "pow".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_two_float_args(args).map(|(base, exp)| Value::Float(base.powf(exp)))
        }),
    )
    .map_err(|e| e.to_string())?;

    // Rounding
    ctx.set_function(
        "floor".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.floor()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "ceil".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.ceil()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "round".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.round()))
        }),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "trunc".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.trunc()))
        }),
    )
    .map_err(|e| e.to_string())?;

    // Other utilities
    ctx.set_function(
        "abs".into(),
        Function::<DefaultNumericTypes>::new(|args| {
            get_float_arg(args).map(|v| Value::Float(v.abs()))
        }),
    )
    .map_err(|e| e.to_string())?;

    // Constants as zero-argument functions
    ctx.set_function(
        "PI".into(),
        Function::<DefaultNumericTypes>::new(|_| Ok(Value::Float(std::f64::consts::PI))),
    )
    .map_err(|e| e.to_string())?;

    ctx.set_function(
        "E".into(),
        Function::<DefaultNumericTypes>::new(|_| Ok(Value::Float(std::f64::consts::E))),
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

fn prepare_context(dim: &DimConfig) -> Result<HashMapContext, String> {
    let mut ctx = HashMapContext::new();
    register_math_functions(&mut ctx)?;

    for d in 0..dim.ndim {
        if d != dim.x_dim && d != dim.y_dim && d != dim.z_dim {
            let val = if d < dim.fixed.len() { dim.fixed[d] } else { 0.0 };
            ctx.set_value(format!("x{d}"), Value::Float(val))
                .map_err(|e| format!("Context error: {e}"))?;
        }
    }

    Ok(ctx)
}

#[inline]
fn eval_sign_at_point(
    tree: &Node,
    ctx: &mut HashMapContext,
    fx: f64,
    fy: f64,
    fz: f64,
    dim: &DimConfig,
) -> Result<i8, String> {
    ctx.set_value("x".into(), Value::Float(fx))
        .map_err(|e| format!("Context error: {e}"))?;
    ctx.set_value("y".into(), Value::Float(fy))
        .map_err(|e| format!("Context error: {e}"))?;
    ctx.set_value("z".into(), Value::Float(fz))
        .map_err(|e| format!("Context error: {e}"))?;

    ctx.set_value(format!("x{}", dim.x_dim), Value::Float(fx))
        .map_err(|e| format!("Context error: {e}"))?;
    ctx.set_value(format!("x{}", dim.y_dim), Value::Float(fy))
        .map_err(|e| format!("Context error: {e}"))?;
    ctx.set_value(format!("x{}", dim.z_dim), Value::Float(fz))
        .map_err(|e| format!("Context error: {e}"))?;

    let val = tree
        .eval_with_context(ctx)
        .map_err(|e| format!("Eval at ({fx},{fy},{fz}): {e}"))?;

    let num = val
        .as_float()
        .map_err(|e| format!("Expected number at ({fx},{fy},{fz}): {e}"))?;

    let sign = if !num.is_finite() {
        0
    } else if num > 0.0 {
        1
    } else if num < 0.0 {
        -1
    } else {
        0
    };

    Ok(sign)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn compute_sign_grid(
    sign_grid: &mut [i8],
    tree: &Node,
    node_dim: usize,
    node_dim_sq: usize,
    step: f64,
    world_half_extent: f64,
    dim: &DimConfig,
) -> Result<(), String> {
    let mut ctx = prepare_context(dim)?;

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    for nz in 0..node_dim {
        let fz = z0 + nz as f64 * step;
        for ny in 0..node_dim {
            let fy = y0 + ny as f64 * step;
            let mut fx = x0;
            for nx in 0..node_dim {
                let idx = nx + ny * node_dim + nz * node_dim_sq;
                sign_grid[idx] = eval_sign_at_point(tree, &mut ctx, fx, fy, fz, dim)?;
                fx += step;
            }
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn compute_sign_grid_par(
    sign_grid: &mut [i8],
    tree: &Node,
    node_dim: usize,
    node_dim_sq: usize,
    step: f64,
    world_half_extent: f64,
    dim: &DimConfig,
) -> Result<(), String> {
    use rayon::prelude::*;

    let chunk_count = std::cmp::min(num_cpus::get(), node_dim);
    let chunk_size = (node_dim as f64 / chunk_count as f64).ceil() as usize;

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    let results: Result<Vec<Vec<(usize, i8)>>, String> = (0..node_dim)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|z_range| {
            let mut ctx = prepare_context(dim)?;

            let mut local_signs = Vec::with_capacity(z_range.len() * node_dim * node_dim);

            for &nz in z_range {
                let fz = z0 + nz as f64 * step;
                for ny in 0..node_dim {
                    let fy = y0 + ny as f64 * step;
                    let mut fx = x0;
                    for nx in 0..node_dim {
                        let sign = eval_sign_at_point(tree, &mut ctx, fx, fy, fz, dim)?;
                        let idx = nx + ny * node_dim + nz * node_dim_sq;
                        local_signs.push((idx, sign));
                        fx += step;
                    }
                }
            }
            Ok(local_signs)
        })
        .collect();

    for chunk in results? {
        for (idx, sign) in chunk {
            sign_grid[idx] = sign;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_generation() {
        // Sphere of radius 2 in region [-5, 5]
        let dim = DimConfig::default();
        let grid = generate_voxel_grid(32, "x^2 + y^2 + z^2 - 4", 0xFF0000, 5.0, &dim).unwrap();
        let filled = grid.iter().filter(|&&v| v != 0).count();
        assert!(filled > 0 && filled < grid.len());
    }

    #[test]
    fn test_sinusoidal_surface() {
        // Test that sin function works
        let dim = DimConfig::default();
        let grid = generate_voxel_grid(16, "sin(x) + cos(y) + z", 0x00FF00, 8.0, &dim).unwrap();
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
        let grid =
            generate_voxel_grid(16, "x1^2 + x2^2 + x3^2 - x0^2", 0x00FF00, 8.0, &dim).unwrap();
        let filled = grid.iter().filter(|&&v| v != 0).count();
        assert_eq!(filled, 0);
    }
}
