use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    mesh::Indices,
    prelude::*,
    render::render_resource::PrimitiveTopology,
    window::{PresentMode, WindowMode, WindowResolution},
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use crate::math::DimConfig;

const REGEN_DEBOUNCE: Duration = Duration::from_millis(300);

#[derive(Default)]
enum RegenRequest {
    #[default]
    None,
    Debounce(Instant),
    Force,
}

#[cfg(target_arch = "wasm32")]
use std::sync::OnceLock;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

mod expr;
mod math;

#[cfg(target_arch = "wasm32")]
static PARALLEL_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(target_arch = "wasm32")]
fn set_parallel_available(val: bool) {
    let _ = PARALLEL_AVAILABLE.set(val);
}

fn parallel_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        *PARALLEL_AVAILABLE.get().unwrap_or(&false)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

/// First value `v` where `v + step == v` due to f64 rounding.
/// Always a power of two: `pow2_ceil(step * 2^53)`.
#[inline]
fn first_bad_offset(step: f64) -> f64 {
    let v = step * (1u64 << f64::MANTISSA_DIGITS) as f64;
    let bits = v.to_bits();
    let mantissa_bits = f64::MANTISSA_DIGITS as u64 - 1;
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    if bits & mantissa_mask == 0 {
        v
    } else {
        f64::from_bits((bits & !mantissa_mask) + (1u64 << mantissa_bits))
    }
}

fn format_drag_value(val: f64) -> String {
    let abs_val = val.abs();
    if abs_val != 0.0 && (abs_val < 1e-6 || abs_val >= 1e6) {
        format!("{val:e}")
    } else {
        let s = format!("{val:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Largest `voxel_size` where `first_bad_offset(v)` stays finite.
/// Bound: `v * 2^53 < f64::MAX` → `v < f64::MAX / 2^53`.
/// One ULP below that threshold to avoid `first_bad_offset` returning ∞.
const MAX_VOXEL_SIZE: f64 = (f64::MAX / (1u64 << f64::MANTISSA_DIGITS) as f64).next_down();

type Vec3Arr = [f32; 3];
type IVec3Arr = [i32; 3];
type CornerArr = [Vec3Arr; 4];

const AMBIENT_OCCLUSION_FACTORS: [f32; 4] = [1.0, 0.75, 0.5, 0.3];
const CAMERA_RADIUS: f32 = 1.5;
const CAMERA_HEIGHT: f32 = 0.8;

#[derive(Resource)]
struct GridConfig {
    size: u32,
    voxel_size: f64,
    voxel_count: usize,
}

#[derive(Resource)]
struct DimMapping {
    ndim: usize,
    x_dim: usize,
    y_dim: usize,
    z_dim: usize,
    fixed: Vec<f64>,
    world_offset: (f64, f64, f64),
}

impl Default for DimMapping {
    fn default() -> Self {
        Self {
            ndim: 3,
            x_dim: 0,
            y_dim: 1,
            z_dim: 2,
            fixed: vec![0.0; 3],
            world_offset: (0.0, 0.0, 0.0),
        }
    }
}

impl From<&DimMapping> for DimConfig {
    fn from(value: &DimMapping) -> Self {
        Self {
            ndim: value.ndim,
            x_dim: value.x_dim,
            y_dim: value.y_dim,
            z_dim: value.z_dim,
            fixed: value.fixed.clone(),
            world_offset: value.world_offset,
        }
    }
}

impl From<DimMapping> for DimConfig {
    fn from(value: DimMapping) -> Self {
        Self {
            ndim: value.ndim,
            x_dim: value.x_dim,
            y_dim: value.y_dim,
            z_dim: value.z_dim,
            fixed: value.fixed,
            world_offset: value.world_offset,
        }
    }
}

#[derive(Resource)]
struct SceneEntities {
    camera: Entity,
    voxel_mesh: Entity,
}

#[derive(Resource)]
struct CameraState {
    angle: f32,
    speed: f32,
    mode: CameraMode,
}

#[derive(PartialEq, Eq)]
enum CameraMode {
    AutoOrbit,
    Manual,
}

#[derive(Resource, Clone, PartialEq, Eq)]
struct ShowAxesPlanes {
    show_axes: bool,
    show_ground_grid: bool,
    show_planes: bool,
}

impl Default for ShowAxesPlanes {
    fn default() -> Self {
        Self {
            show_axes: true,
            show_ground_grid: false,
            show_planes: false,
        }
    }
}

#[derive(Resource)]
struct RegenerateEveryFrame {
    enabled: bool,
}

#[derive(Clone)]
struct ExpressionEntry {
    expr: String,
    color: (u8, u8, u8),
    enabled: bool,
}

#[derive(Resource)]
struct ExpressionConfig {
    entries: Vec<ExpressionEntry>,
}

impl Default for ExpressionConfig {
    fn default() -> Self {
        Self {
            entries: vec![ExpressionEntry {
                expr: "x^2 + z * y - 64.0".into(),
                color: rand::random(),
                enabled: true,
            }],
        }
    }
}

#[derive(Resource, Default)]
struct ExpressionStatus {
    is_valid: bool,
    errors: Vec<String>,
}

#[derive(Resource, Default)]
struct ProfilingData {
    parse_ms: f64,
    sign_grid_ms: f64,
    voxel_fill_ms: f64,
    mesh_build_ms: f64,
    total_ms: f64,
}

struct GenerationTimings {
    parse_ms: f64,
    sign_grid_ms: f64,
    voxel_fill_ms: f64,
}

type MeshData = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<u32>);

const FACE_DEFS: [(Vec3Arr, CornerArr, IVec3Arr); 6] = [
    // +X face
    (
        [1.0, 0.0, 0.0],
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        [1, 0, 0],
    ),
    // -X face
    (
        [-1.0, 0.0, 0.0],
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [-1, 0, 0],
    ),
    // +Y face
    (
        [0.0, 1.0, 0.0],
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [0, 1, 0],
    ),
    // -Y face
    (
        [0.0, -1.0, 0.0],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        [0, -1, 0],
    ),
    // +Z face
    (
        [0.0, 0.0, 1.0],
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        [0, 0, 1],
    ),
    // -Z face
    (
        [0.0, 0.0, -1.0],
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [0, 0, -1],
    ),
];

#[inline]
fn grid_index(x: usize, y: usize, z: usize, grid_size: usize) -> usize {
    x + y * grid_size + z * grid_size.pow(2)
}

/// Decodes color from u32 with +1 offset.
/// 0 = empty, 1 = #000000 (black), 0xFFFFFF+1 = #FFFFFF (white)
#[inline]
fn decode_color(packed: u32) -> LinearRgba {
    let val = packed - 1;
    let r = ((val >> 16) & 0xFF) as f32 / 255.0;
    let g = ((val >> 8) & 0xFF) as f32 / 255.0;
    let b = (val & 0xFF) as f32 / 255.0;
    LinearRgba::rgb(r, g, b)
}

#[inline]
fn is_occupied(
    composite: &[u32],
    x: i32,
    y: i32,
    z: i32,
    grid_size: i32,
    grid_size_usize: usize,
) -> bool {
    if !(0..grid_size).contains(&x) || !(0..grid_size).contains(&y) || !(0..grid_size).contains(&z)
    {
        return false;
    }
    composite[grid_index(x as usize, y as usize, z as usize, grid_size_usize)] != 0
}

fn validate_expression(expr_str: &str, dim_mapping: &DimMapping) -> Result<(), String> {
    let trimmed = expr_str.trim();
    if trimmed.is_empty() {
        return Err("Expression cannot be empty".into());
    }

    expr::parse(trimmed, &dim_mapping.into()).map(|_| ())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
async fn wasm_start() {
    let num_threads = web_sys::window()
        .map(|w| w.navigator().hardware_concurrency())
        .map(|c| c as usize)
        .unwrap_or(1)
        .max(1);
    let result = JsFuture::from(wasm_bindgen_rayon::init_thread_pool(num_threads)).await;
    if result.is_ok() {
        set_parallel_available(true);
    } else {
        web_sys::console::log_1(&"init_thread_pool failed, falling back to sequential".into());
    }
    main();
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "HyperVox".into(),
                mode: WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                resolution: WindowResolution::default(),
                present_mode: PresentMode::Fifo,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::WHITE))
        .insert_resource(GridConfig {
            size: 64,
            voxel_size: 1.0,
            voxel_count: 0,
        })
        .insert_resource(DimMapping::default())
        .insert_resource(ExpressionConfig::default())
        .insert_resource(ExpressionStatus::default())
        .insert_resource(RegenerateEveryFrame { enabled: false })
        .insert_resource(ShowAxesPlanes::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                rotate_camera,
                update_ui_scale_from_browser,
                draw_axes_and_planes,
            ),
        )
        .add_systems(EguiPrimaryContextPass, egui_overlays_system)
        .add_systems(EguiPrimaryContextPass, egui_ui_system)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut grid_config: ResMut<GridConfig>,
    mut expr_status: ResMut<ExpressionStatus>,
    expr_config: Res<ExpressionConfig>,
    dim_mapping: Res<DimMapping>,
) {
    let cam_entity = commands
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(
                Vec3::ZERO.x + CAMERA_RADIUS * 0.7,
                Vec3::ZERO.y + CAMERA_HEIGHT,
                Vec3::ZERO.z + CAMERA_RADIUS * 0.7,
            )
            .looking_at(Vec3::ZERO, Vec3::Y),
            Exposure { ev100: 0.0 },
            Tonemapping::None,
            PanOrbitCamera {
                enabled: false,
                focus: Vec3::ZERO,
                target_focus: Vec3::ZERO,
                ..default()
            },
        ))
        .id();

    let total_start = Instant::now();
    let (composite, voxel_count, gen_timings) =
        generate_voxels(&grid_config, &expr_config, &mut expr_status, &dim_mapping);
    grid_config.voxel_count = voxel_count;
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    let mesh_start = Instant::now();

    if parallel_available() {
        build_batched_mesh_with_global_corner_ambient_occlusion_par(
            &mut mesh,
            &composite,
            &grid_config,
            voxel_count,
        );
    } else {
        build_batched_mesh_with_global_corner_ambient_occlusion(
            &mut mesh,
            &composite,
            &grid_config,
            voxel_count,
        );
    }

    let mesh_build_ms = mesh_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    commands.insert_resource(ProfilingData {
        parse_ms: gen_timings.parse_ms,
        sign_grid_ms: gen_timings.sign_grid_ms,
        voxel_fill_ms: gen_timings.voxel_fill_ms,
        mesh_build_ms,
        total_ms,
    });

    let mesh_handle = meshes.add(mesh);

    let material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        perceptual_roughness: 0.0,
        metallic: 0.0,
        reflectance: 0.0,
        ..default()
    });

    let mesh_entity = commands
        .spawn((Mesh3d(mesh_handle), MeshMaterial3d(material_handle)))
        .id();

    commands.insert_resource(SceneEntities {
        camera: cam_entity,
        voxel_mesh: mesh_entity,
    });
    commands.insert_resource(CameraState {
        angle: 0.0,
        speed: 0.5,
        mode: CameraMode::AutoOrbit,
    });
}

fn generate_voxels(
    grid_config: &GridConfig,
    expr_config: &ExpressionConfig,
    expr_status: &mut ExpressionStatus,
    dim_mapping: &DimMapping,
) -> (Vec<u32>, usize, GenerationTimings) {
    let size_usize = grid_config.size as usize;
    let half_extent = (grid_config.size as f64) / 2.0 * grid_config.voxel_size;

    let mut timings = GenerationTimings {
        parse_ms: 0.0,
        sign_grid_ms: 0.0,
        voxel_fill_ms: 0.0,
    };

    expr_status.errors.clear();

    let mut grids = Vec::with_capacity(expr_config.entries.len());

    for (idx, entry) in expr_config.entries.iter().enumerate() {
        if !entry.enabled {
            grids.push(vec![0; size_usize.pow(3)]);
            continue;
        }

        let parse_start = Instant::now();
        if let Err(e) = validate_expression(&entry.expr, dim_mapping) {
            timings.parse_ms += parse_start.elapsed().as_secs_f64() * 1000.0;
            expr_status.is_valid = false;
            expr_status
                .errors
                .push(format!("Expression #{} '{}': {}", idx + 1, entry.expr, e));
            grids.push(vec![0; size_usize.pow(3)]);
            continue;
        }
        timings.parse_ms += parse_start.elapsed().as_secs_f64() * 1000.0;

        let base_color =
            ((entry.color.0 as u32) << 16) | ((entry.color.1 as u32) << 8) | (entry.color.2 as u32);

        let gen_start = Instant::now();
        match math::generate_voxel_grid(
            size_usize,
            &entry.expr,
            base_color,
            half_extent,
            &dim_mapping.into(),
            parallel_available(),
        ) {
            Ok((grid, grid_timings)) => {
                timings.sign_grid_ms += grid_timings.sign_grid_ms;
                timings.voxel_fill_ms += grid_timings.voxel_fill_ms;
                grids.push(grid);
            }
            Err(e) => {
                timings.sign_grid_ms += gen_start.elapsed().as_secs_f64() * 1000.0;
                error!(
                    "Error evaluating expression #{} '{}': {}",
                    idx + 1,
                    entry.expr,
                    e
                );
                expr_status.is_valid = false;
                expr_status.errors.push(format!(
                    "Eval error for #{} '{}': {}",
                    idx + 1,
                    entry.expr,
                    e
                ));
                grids.push(vec![0; size_usize.pow(3)]);
            }
        }
    }

    let total_positions = size_usize.pow(3);
    let mut composite = vec![0u32; total_positions];
    let mut rendered_voxel_count = 0;
    for idx in 0..total_positions {
        for grid in &grids {
            let val = grid[idx];
            if val != 0 {
                composite[idx] = val;
                rendered_voxel_count += 1;
                break;
            }
        }
    }

    // Only mark as valid if no errors occurred AND at least one enabled expression exists
    if expr_status.errors.is_empty()
        && expr_config
            .entries
            .iter()
            .any(|e| e.enabled && !e.expr.trim().is_empty())
    {
        expr_status.is_valid = true;
    }

    (composite, rendered_voxel_count, timings)
}

fn process_z_range_multi(
    z_start: u32,
    z_end: u32,
    composite: &[u32],
    size: u32,
    voxel_count: usize,
) -> MeshData {
    let size_i32 = size as i32;
    let size_usize = size as usize;
    let inv_size = 1.0 / size as f32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(voxel_count * 30);
    let mut indices: Vec<u32> = Vec::with_capacity(voxel_count * 72);

    for z in z_start..z_end {
        for y in 0..size {
            for x in 0..size {
                let idx = grid_index(x as usize, y as usize, z as usize, size_usize);

                let voxel_val = composite[idx];
                if voxel_val == 0 {
                    continue;
                }

                // Skip interior voxels with all 6 faces occluded
                let all_occluded = FACE_DEFS.iter().all(|(_, _, off)| {
                    is_occupied(
                        composite,
                        x as i32 + off[0],
                        y as i32 + off[1],
                        z as i32 + off[2],
                        size_i32,
                        size_usize,
                    )
                });
                if all_occluded {
                    continue;
                }

                let base_linear = decode_color(voxel_val);
                let offset = Vec3::new(
                    x as f32 * inv_size - 0.5,
                    y as f32 * inv_size - 0.5,
                    z as f32 * inv_size - 0.5,
                );

                let mut corner_ambient_occlusion = [1.0f32; 8];
                for cx_off in 0..2usize {
                    for cy_off in 0..2usize {
                        for cz_off in 0..2usize {
                            let corner_idx = cx_off | (cy_off << 1) | (cz_off << 2);
                            let cx = x as i32 + cx_off as i32;
                            let cy = y as i32 + cy_off as i32;
                            let cz = z as i32 + cz_off as i32;

                            let mut occlusion = 0;
                            if is_occupied(composite, cx - 1, cy, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            if is_occupied(composite, cx, cy - 1, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            if is_occupied(composite, cx, cy, cz - 1, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            corner_ambient_occlusion[corner_idx] =
                                AMBIENT_OCCLUSION_FACTORS[occlusion];
                        }
                    }
                }

                for (normal, corners, neighbor_offset) in FACE_DEFS {
                    let nx = x as i32 + neighbor_offset[0];
                    let ny = y as i32 + neighbor_offset[1];
                    let nz = z as i32 + neighbor_offset[2];

                    if is_occupied(composite, nx, ny, nz, size_i32, size_usize) {
                        continue;
                    }

                    let mut ao_vals = [0.0f32; 4];
                    let mut corner_colors = [[0.0; 4]; 4];

                    for (i, &corner) in corners.iter().enumerate() {
                        let [cx, cy, cz] = corner;

                        let cx_off = if cx > 0.0 { 1 } else { 0 };
                        let cy_off = if cy > 0.0 { 1 } else { 0 };
                        let cz_off = if cz > 0.0 { 1 } else { 0 };
                        let corner_idx = cx_off | (cy_off << 1) | (cz_off << 2);

                        let ambient_occlusion = corner_ambient_occlusion[corner_idx];
                        ao_vals[i] = ambient_occlusion;

                        corner_colors[i] = [
                            base_linear.red * ambient_occlusion,
                            base_linear.green * ambient_occlusion,
                            base_linear.blue * ambient_occlusion,
                            base_linear.alpha,
                        ];
                    }

                    let start_idx = positions.len() as u32;

                    for (i, &corner) in corners.iter().enumerate() {
                        let [cx, cy, cz] = corner;
                        positions.push([
                            cx * inv_size + offset.x,
                            cy * inv_size + offset.y,
                            cz * inv_size + offset.z,
                        ]);
                        normals.push(normal);
                        colors.push(corner_colors[i]);
                    }

                    if ao_vals[0] + ao_vals[2] > ao_vals[1] + ao_vals[3] {
                        indices.extend_from_slice(&[
                            start_idx,
                            start_idx + 1,
                            start_idx + 2,
                            start_idx,
                            start_idx + 2,
                            start_idx + 3,
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            start_idx,
                            start_idx + 1,
                            start_idx + 3,
                            start_idx + 1,
                            start_idx + 2,
                            start_idx + 3,
                        ]);
                    }
                }
            }
        }
    }

    (positions, normals, colors, indices)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn build_batched_mesh_with_global_corner_ambient_occlusion(
    mesh: &mut Mesh,
    composite: &[u32],
    grid_config: &GridConfig,
    voxel_count: usize,
) {
    info!("voxel_count: {}", voxel_count);
    let size = grid_config.size;
    let (positions, normals, colors, indices) =
        process_z_range_multi(0, size, composite, size, voxel_count);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
}

#[inline]
fn split_disjoint<'a, T>(slice: &'a mut [T], counts: &[usize]) -> Vec<&'a mut [T]> {
    debug_assert_eq!(
        counts.iter().copied().sum::<usize>(),
        slice.len(),
        "split_disjoint: counts sum ({}) != slice len ({})",
        counts.iter().copied().sum::<usize>(),
        slice.len()
    );
    let mut parts = Vec::with_capacity(counts.len());
    let mut rest = slice;
    for &count in counts {
        let (part, tail) = rest.split_at_mut(count);
        parts.push(part);
        rest = tail;
    }
    parts
}

fn build_batched_mesh_with_global_corner_ambient_occlusion_par(
    mesh: &mut Mesh,
    composite: &[u32],
    grid_config: &GridConfig,
    voxel_count: usize,
) {
    info!("voxel_count: {}", voxel_count);
    let size = grid_config.size;
    let chunk_count = rayon::current_num_threads();
    let chunk_size = (size as usize).div_ceil(chunk_count);
    let voxel_count_per_chunk = voxel_count.div_ceil(chunk_count);
    let results: Vec<_> = (0..size)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|z_range| {
            let z_start = z_range[0];
            let z_end = z_range[z_range.len() - 1] + 1;
            process_z_range_multi(z_start, z_end, composite, size, voxel_count_per_chunk)
        })
        .collect();

    let vertex_counts: Vec<usize> = results.iter().map(|(pos, _, _, _)| pos.len()).collect();
    let index_counts: Vec<usize> = results.iter().map(|(_, _, _, ind)| ind.len()).collect();
    let total_vertices: usize = vertex_counts.iter().sum();
    let total_indices: usize = index_counts.iter().sum();

    let mut vertex_offsets = Vec::with_capacity(results.len());
    {
        let mut v_off = 0u32;
        for &vc in &vertex_counts {
            vertex_offsets.push(v_off);
            v_off += vc as u32;
        }
    }

    let mut positions = Vec::with_capacity(total_vertices);
    let mut normals = Vec::with_capacity(total_vertices);
    let mut colors = Vec::with_capacity(total_vertices);
    let mut indices = Vec::with_capacity(total_indices);

    // SAFETY: every element is overwritten below before any read
    unsafe {
        positions.set_len(total_vertices);
        normals.set_len(total_vertices);
        colors.set_len(total_vertices);
        indices.set_len(total_indices);
    }

    let pos_parts = split_disjoint(&mut positions, &vertex_counts);
    let norm_parts = split_disjoint(&mut normals, &vertex_counts);
    let col_parts = split_disjoint(&mut colors, &vertex_counts);
    let idx_parts = split_disjoint(&mut indices, &index_counts);

    rayon::scope(|s| {
        for (i, (((pos_part, norm_part), col_part), idx_part)) in pos_parts
            .into_iter()
            .zip(norm_parts)
            .zip(col_parts)
            .zip(idx_parts)
            .enumerate()
        {
            let result = &results[i];
            let v_offset = vertex_offsets[i];

            s.spawn(move |_| {
                pos_part.copy_from_slice(&result.0);
                norm_part.copy_from_slice(&result.1);
                col_part.copy_from_slice(&result.2);
                for (dst, &src) in idx_part.iter_mut().zip(result.3.iter()) {
                    *dst = src + v_offset;
                }
            });
        }
    });

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
}

fn egui_overlays_system(
    mut egui_contexts: EguiContexts,
    mut show_axes_planes: ResMut<ShowAxesPlanes>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("Overlays")
        .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
        .show(ctx, |ui| {
            ui.checkbox(&mut show_axes_planes.show_axes, "Show Axes");
            ui.checkbox(&mut show_axes_planes.show_ground_grid, "Show Ground Grid");
            ui.checkbox(&mut show_axes_planes.show_planes, "Show Reference Planes");
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(255, 51, 51), "X — red");
            ui.colored_label(egui::Color32::from_rgb(51, 255, 51), "Y — green");
            ui.colored_label(egui::Color32::from_rgb(51, 51, 255), "Z — blue");
        });
}

fn draw_axes_and_planes(
    mut gizmos: Gizmos,
    grid_config: Res<GridConfig>,
    show: Res<ShowAxesPlanes>,
    camera_query: Query<&Transform, With<Camera3d>>,
) {
    let inv_size = 1.0 / grid_config.size as f32;
    let n = grid_config.size;
    let step = ((n as f32 / 32.0).round() as u32).next_power_of_two();

    if show.show_axes {
        gizmos.line(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Color::srgb(1.0, 0.2, 0.2),
        );
        gizmos.line(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Color::srgb(0.2, 1.0, 0.2),
        );
        gizmos.line(
            Vec3::new(0.0, 0.0, -0.5),
            Vec3::new(0.0, 0.0, 0.5),
            Color::srgb(0.2, 0.2, 1.0),
        );

        let cam_t = camera_query.iter().next().cloned().unwrap_or_default();

        for (pos, label, color) in [
            (Vec3::new(0.55, 0.0, 0.0), "X", Color::srgb(1.0, 0.2, 0.2)),
            (Vec3::new(0.0, 0.55, 0.0), "Y", Color::srgb(0.2, 1.0, 0.2)),
            (Vec3::new(0.0, 0.0, 0.55), "Z", Color::srgb(0.2, 0.2, 1.0)),
        ] {
            let view_dir = (cam_t.translation - pos).normalize();
            let cam_up = cam_t.rotation * Vec3::Y;
            let right = cam_up.cross(view_dir).normalize();
            let up = view_dir.cross(right);
            let rot = Quat::from_mat3(&Mat3::from_cols(right, up, view_dir));
            let isometry = Isometry3d::new(pos, rot);
            gizmos.text(isometry, label, 0.04, Vec2::ZERO, color);
        }
    }

    if show.show_ground_grid {
        let gc = Color::srgba(0.5, 0.5, 0.5, 0.35);

        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 * inv_size - 0.5;
            gizmos.line(Vec3::new(-0.5, -0.5, p), Vec3::new(0.5, -0.5, p), gc);
            gizmos.line(Vec3::new(p, -0.5, -0.5), Vec3::new(p, -0.5, 0.5), gc);
        }
    }

    if show.show_planes {
        // XY plane at z = center.z — red grid
        let cr = Color::srgba(1.0, 0.2, 0.2, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 * inv_size - 0.5;
            gizmos.line(Vec3::new(-0.5, p, 0.0), Vec3::new(0.5, p, 0.0), cr);
            gizmos.line(Vec3::new(p, -0.5, 0.0), Vec3::new(p, 0.5, 0.0), cr);
        }

        // XZ plane at y = center.y — green grid
        let cg = Color::srgba(0.2, 1.0, 0.2, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 * inv_size - 0.5;
            gizmos.line(Vec3::new(-0.5, 0.0, p), Vec3::new(0.5, 0.0, p), cg);
            gizmos.line(Vec3::new(p, 0.0, -0.5), Vec3::new(p, 0.0, 0.5), cg);
        }

        // YZ plane at x = center.x — blue grid
        let cb = Color::srgba(0.2, 0.2, 1.0, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 * inv_size - 0.5;
            gizmos.line(Vec3::new(0.0, -0.5, p), Vec3::new(0.0, 0.5, p), cb);
            gizmos.line(Vec3::new(0.0, p, -0.5), Vec3::new(0.0, p, 0.5), cb);
        }
    }
}

fn rotate_camera(
    time: Res<Time>,
    mut camera: ResMut<CameraState>,
    mut query_camera: Query<&mut Transform, With<Camera3d>>,
) {
    if camera.mode != CameraMode::AutoOrbit {
        return;
    }
    camera.angle += time.delta_secs() * camera.speed;
    for mut t in query_camera.iter_mut() {
        t.translation = Vec3::new(
            Vec3::ZERO.x + CAMERA_RADIUS * camera.angle.cos(),
            Vec3::ZERO.y + CAMERA_HEIGHT,
            Vec3::ZERO.z + CAMERA_RADIUS * camera.angle.sin(),
        );
        t.look_at(Vec3::ZERO, Vec3::Y);
    }
}

#[cfg(target_arch = "wasm32")]
fn update_ui_scale_from_browser(mut egui_contexts: EguiContexts) {
    if let Some(window) = web_sys::window() {
        let scale = window.device_pixel_ratio() as f32;
        if let Ok(ctx) = egui_contexts.ctx_mut() {
            ctx.set_pixels_per_point(scale);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn update_ui_scale_from_browser() {}

#[allow(clippy::too_many_arguments)]
fn egui_ui_system(
    mut commands: Commands,
    mut egui_contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    mut grid_config: ResMut<GridConfig>,
    mut dim_mapping: ResMut<DimMapping>,
    mut camera: ResMut<CameraState>,
    mut query_cam: Query<&mut PanOrbitCamera>,
    mut auto_regen: ResMut<RegenerateEveryFrame>,
    mut regenerate_request: Local<RegenRequest>,
    mut expr_config: ResMut<ExpressionConfig>,
    mut expr_status: ResMut<ExpressionStatus>,
    entities: Res<SceneEntities>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query_cam_transform: Query<&mut Transform>,
    mut profiling: ResMut<ProfilingData>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    let mut viewport_ui = egui::Ui::new(
        ctx.clone(),
        egui::Id::new("viewport_ui"),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    egui::Panel::left("left_panel")
        .resizable(true)
        .show(&mut viewport_ui, |ui| {
            ui.heading("Generation");

            ui.label("Expressions:");

            // Scrollable area for expression list with fixed max height
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let mut remove_idx = None;
                    let mut duplicate_idx = None;
                    let mut move_up_idx = None;
                    let mut move_down_idx = None;
                    let entries_len = expr_config.entries.len();
                    for (idx, entry) in expr_config.entries.iter_mut().enumerate() {
                        egui::CollapsingHeader::new(format!("Function #{}", idx + 1))
                            .id_salt(("func_header", idx))
                            .default_open(true)
                            .show(ui, |ui| {
                                // Trigger regeneration when enabled state changes
                                if ui.checkbox(&mut entry.enabled, "Enabled").changed() {
                                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                                }

                                ui.label("Expression:");
                                if ui.text_edit_singleline(&mut entry.expr).changed() {
                                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                                    expr_status.errors.clear();
                                }

                                ui.label("Color:");
                                ui.horizontal(|ui| {
                                    let mut color_edit = egui::Color32::from_rgb(
                                        entry.color.0,
                                        entry.color.1,
                                        entry.color.2,
                                    );
                                    if ui.color_edit_button_srgba(&mut color_edit).changed() {
                                        entry.color =
                                            (color_edit.r(), color_edit.g(), color_edit.b());
                                        *regenerate_request =
                                            RegenRequest::Debounce(Instant::now());
                                    }
                                });

                                ui.horizontal(|ui| {
                                    if idx > 0 && ui.small_button("⬆").clicked() {
                                        move_up_idx = Some(idx);
                                    }
                                    if idx < entries_len - 1 && ui.small_button("⬇").clicked() {
                                        move_down_idx = Some(idx);
                                    }
                                    if ui.small_button("📋 Duplicate").clicked() {
                                        duplicate_idx = Some(idx);
                                    }
                                    if ui.small_button("❌ Remove").clicked() {
                                        remove_idx = Some(idx);
                                    }
                                });
                            });
                        ui.separator();
                    }

                    // Apply removals after iteration to avoid borrow issues
                    if let Some(idx) = remove_idx {
                        expr_config.entries.remove(idx);
                        *regenerate_request = RegenRequest::Debounce(Instant::now());
                    }

                    if let Some(idx) = duplicate_idx {
                        let entry = expr_config.entries[idx].clone();
                        expr_config.entries.insert(idx + 1, entry);
                        *regenerate_request = RegenRequest::Debounce(Instant::now());
                    }

                    if let Some(idx) = move_up_idx {
                        expr_config.entries.swap(idx, idx - 1);
                        *regenerate_request = RegenRequest::Debounce(Instant::now());
                    }

                    if let Some(idx) = move_down_idx {
                        expr_config.entries.swap(idx, idx + 1);
                        *regenerate_request = RegenRequest::Debounce(Instant::now());
                    }
                });

            ui.separator();

            if ui.button("➕ Add Expression").clicked() {
                expr_config.entries.push(ExpressionEntry {
                    expr: String::new(),
                    color: rand::random(),
                    enabled: true,
                });
                *regenerate_request = RegenRequest::Debounce(Instant::now());
            }

            if !expr_status.errors.is_empty() {
                for error in &expr_status.errors {
                    ui.label(
                        egui::RichText::new(format!("❌ {error}"))
                            .color(egui::Color32::RED)
                            .small(),
                    );
                }
            } else if expr_status.is_valid
                && expr_config
                    .entries
                    .iter()
                    .any(|e| e.enabled && !e.expr.trim().is_empty())
            {
                ui.label(
                    egui::RichText::new("✅ Valid expressions")
                        .color(egui::Color32::GREEN)
                        .small(),
                );
            }

            if dim_mapping.ndim > 3 {
                ui.label(
                    egui::RichText::new(format!(
                        "Tip: ^=power, vars: x,y,z (axes), x0..x{} (dims)",
                        dim_mapping.ndim - 1
                    ))
                    .italics()
                    .small()
                    .color(egui::Color32::from_gray(180)),
                );
            } else {
                ui.label(
                    egui::RichText::new(
                        "Tip: ^ = power (x^2), * = multiply, + - / work as expected",
                    )
                    .italics()
                    .small()
                    .color(egui::Color32::from_gray(180)),
                );
            }

            ui.separator();

            ui.label("Grid Size:");
            ui.horizontal(|ui| {
                if ui.button("-").clicked() && grid_config.size > 2 {
                    grid_config.size = if grid_config.size > 8 {
                        grid_config.size.saturating_sub(8)
                    } else if grid_config.size > 4 {
                        grid_config.size.saturating_sub(4)
                    } else {
                        grid_config.size.saturating_sub(2).max(2)
                    };
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }

                let mut size = grid_config.size as i32;
                if ui
                    .add(
                        egui::Slider::new(&mut size, 2..=256)
                            .logarithmic(false)
                            .custom_formatter(|n, _| format!("{n:.0}")),
                    )
                    .changed()
                {
                    grid_config.size = size as u32;
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }

                if ui.button("+").clicked() && grid_config.size < 256 {
                    grid_config.size = if grid_config.size <= 2 {
                        (grid_config.size + 2).min(256)
                    } else if grid_config.size <= 4 {
                        (grid_config.size + 4).min(256)
                    } else if grid_config.size < 8 {
                        8
                    } else {
                        (grid_config.size + 8).min(256)
                    };
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }
            });

            ui.label("Voxel Size:");
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut grid_config.voxel_size)
                            .speed(0.0)
                            .custom_formatter(|val, _| format_drag_value(val))
                            .range(f64::MIN_POSITIVE..=MAX_VOXEL_SIZE),
                    )
                    .changed()
                {
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }
            });

            ui.separator();

            // --- Dimension Configuration ---
            ui.label("Dimensions (N):");
            ui.horizontal(|ui| {
                let mut ndim = dim_mapping.ndim as i32;
                if ui
                    .add(egui::Slider::new(&mut ndim, 3..=16).logarithmic(false))
                    .changed()
                {
                    dim_mapping.ndim = ndim as usize;
                    let n = dim_mapping.ndim;
                    dim_mapping.fixed.resize(n, 0.0);
                    dim_mapping.x_dim = dim_mapping.x_dim.min(dim_mapping.ndim - 1);
                    dim_mapping.y_dim = dim_mapping.y_dim.min(dim_mapping.ndim - 1);
                    dim_mapping.z_dim = dim_mapping.z_dim.min(dim_mapping.ndim - 1);
                    if dim_mapping.y_dim == dim_mapping.x_dim {
                        dim_mapping.y_dim = (dim_mapping.y_dim + 1) % dim_mapping.ndim;
                    }
                    if dim_mapping.z_dim == dim_mapping.x_dim
                        || dim_mapping.z_dim == dim_mapping.y_dim
                    {
                        for d in 0..dim_mapping.ndim {
                            if d != dim_mapping.x_dim && d != dim_mapping.y_dim {
                                dim_mapping.z_dim = d;
                                break;
                            }
                        }
                    }
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }
            });

            ui.label("Axis Mapping:");
            let ndim = dim_mapping.ndim;
            let mut x_dim = dim_mapping.x_dim;
            let mut y_dim = dim_mapping.y_dim;
            let mut z_dim = dim_mapping.z_dim;

            ui.horizontal(|ui| {
                ui.label("X ←");
                egui::ComboBox::from_id_salt("x_dim_map")
                    .selected_text(format!("x{x_dim}"))
                    .show_ui(ui, |ui| {
                        for d in 0..ndim {
                            let is_taken = d == y_dim || d == z_dim;
                            if !is_taken || d == x_dim {
                                ui.selectable_value(&mut x_dim, d, format!("x{d}"));
                            }
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Y ←");
                egui::ComboBox::from_id_salt("y_dim_map")
                    .selected_text(format!("x{y_dim}"))
                    .show_ui(ui, |ui| {
                        for d in 0..ndim {
                            let is_taken = d == x_dim || d == z_dim;
                            if !is_taken || d == y_dim {
                                ui.selectable_value(&mut y_dim, d, format!("x{d}"));
                            }
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Z ←");
                egui::ComboBox::from_id_salt("z_dim_map")
                    .selected_text(format!("x{z_dim}"))
                    .show_ui(ui, |ui| {
                        for d in 0..ndim {
                            let is_taken = d == x_dim || d == y_dim;
                            if !is_taken || d == z_dim {
                                ui.selectable_value(&mut z_dim, d, format!("x{d}"));
                            }
                        }
                    });
            });

            if x_dim != dim_mapping.x_dim
                || y_dim != dim_mapping.y_dim
                || z_dim != dim_mapping.z_dim
            {
                // Ensure distinct mapping
                if x_dim == y_dim {
                    y_dim = (y_dim + 1) % ndim;
                }
                if x_dim == z_dim || y_dim == z_dim {
                    for d in 0..ndim {
                        if d != x_dim && d != y_dim {
                            z_dim = d;
                            break;
                        }
                    }
                }
                dim_mapping.x_dim = x_dim;
                dim_mapping.y_dim = y_dim;
                dim_mapping.z_dim = z_dim;
                *regenerate_request = RegenRequest::Debounce(Instant::now());
            }

            // Fixed values for non-spatial dims
            let max_offset = first_bad_offset(grid_config.voxel_size)
                - (grid_config.size as f64 / 2.0) * grid_config.voxel_size;
            for d in 0..dim_mapping.ndim {
                if d != dim_mapping.x_dim && d != dim_mapping.y_dim && d != dim_mapping.z_dim {
                    ui.horizontal(|ui| {
                        ui.label(format!("x{d}:"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut dim_mapping.fixed[d])
                                    .speed(0.0)
                                    .custom_formatter(|val, _| format_drag_value(val))
                                    .range(-max_offset..=max_offset),
                            )
                            .changed()
                        {
                            *regenerate_request = RegenRequest::Debounce(Instant::now());
                        }
                    });
                }
            }

            ui.separator();

            if ui.button("Regenerate").clicked() {
                *regenerate_request = RegenRequest::Force;
            }

            ui.checkbox(&mut auto_regen.enabled, "Auto Regenerate");

            ui.separator();

            ui.collapsing("View Offset", |ui| {
                let max_abs_offset = max_offset;

                let mut changed = false;
                let mut ox = dim_mapping.world_offset.0;
                let mut oy = dim_mapping.world_offset.1;
                let mut oz = dim_mapping.world_offset.2;

                ui.horizontal(|ui| {
                    ui.label("X:");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut ox)
                                .speed(0.0)
                                .custom_formatter(|val, _| format_drag_value(val))
                                .range(-max_abs_offset..=max_abs_offset),
                        )
                        .changed();
                    if ui.small_button("↺").clicked() {
                        ox = 0.0;
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Y:");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut oy)
                                .speed(0.0)
                                .custom_formatter(|val, _| format_drag_value(val))
                                .range(-max_abs_offset..=max_abs_offset),
                        )
                        .changed();
                    if ui.small_button("↺").clicked() {
                        oy = 0.0;
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Z:");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut oz)
                                .speed(0.0)
                                .custom_formatter(|val, _| format_drag_value(val))
                                .range(-max_abs_offset..=max_abs_offset),
                        )
                        .changed();
                    if ui.small_button("↺").clicked() {
                        oz = 0.0;
                        changed = true;
                    }
                });
                if changed {
                    dim_mapping.world_offset = (ox, oy, oz);
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }
            });

            let cam_label = match camera.mode {
                CameraMode::AutoOrbit => "Camera: Auto",
                CameraMode::Manual => "Camera: Manual",
            };
            if ui.button(cam_label).clicked() {
                camera.mode = match camera.mode {
                    CameraMode::AutoOrbit => {
                        for mut poc in &mut query_cam {
                            poc.enabled = true;
                            poc.focus = Vec3::ZERO;
                            poc.target_focus = Vec3::ZERO;
                        }
                        CameraMode::Manual
                    }
                    CameraMode::Manual => {
                        for mut poc in &mut query_cam {
                            poc.enabled = false;
                        }
                        CameraMode::AutoOrbit
                    }
                };
            }
        });

    egui::Window::new("Statistics")
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .show(ctx, |ui| {
            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .map(|v| v.round() as u32)
                .unwrap_or(0);
            ui.label(format!("FPS: {fps}"));
            if dim_mapping.ndim > 3 {
                ui.label(format!(
                    "Grid: {}³ (N={}, dims 0..{})",
                    grid_config.size,
                    dim_mapping.ndim,
                    dim_mapping.ndim - 1
                ));
            } else {
                ui.label(format!("Grid: {}³", grid_config.size));
            }
            ui.label(format!("Rendered Voxels: {}", grid_config.voxel_count));
            ui.label(format!(
                "Fill: {:.1}%",
                (grid_config.voxel_count as f32 / (grid_config.size.pow(3) as f32)) * 100.0
            ));

            ui.separator();
            ui.label("Regeneration Timing:");
            let total = profiling.total_ms;
            if total > 0.0 {
                ui.label(format!(
                    "  Parse:     {:.1} ms  ({:.0}%)",
                    profiling.parse_ms,
                    profiling.parse_ms / total * 100.0
                ));
                ui.label(format!(
                    "  Sign grid: {:.1} ms  ({:.0}%)",
                    profiling.sign_grid_ms,
                    profiling.sign_grid_ms / total * 100.0
                ));
                ui.label(format!(
                    "  Voxel fill: {:.1} ms  ({:.0}%)",
                    profiling.voxel_fill_ms,
                    profiling.voxel_fill_ms / total * 100.0
                ));
                ui.label(format!(
                    "  Mesh build: {:.1} ms  ({:.0}%)",
                    profiling.mesh_build_ms,
                    profiling.mesh_build_ms / total * 100.0
                ));
                ui.label("  -----------------");
                ui.label(format!("  Total:     {:.1} ms", total));
            } else {
                ui.label("  (waiting for generation...)");
            }
        });

    let should_regenerate = auto_regen.enabled
        || match *regenerate_request {
            RegenRequest::None => false,
            RegenRequest::Debounce(t) => t.elapsed() >= REGEN_DEBOUNCE,
            RegenRequest::Force => true,
        };
    if should_regenerate {
        *regenerate_request = RegenRequest::None;
        let total_start = Instant::now();
        let (composite, count, gen_timings) =
            generate_voxels(&grid_config, &expr_config, &mut expr_status, &dim_mapping);
        grid_config.voxel_count = count;

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );

        let mesh_start = Instant::now();
        if parallel_available() {
            build_batched_mesh_with_global_corner_ambient_occlusion_par(
                &mut mesh,
                &composite,
                &grid_config,
                count,
            );
        } else {
            build_batched_mesh_with_global_corner_ambient_occlusion(
                &mut mesh,
                &composite,
                &grid_config,
                count,
            );
        }
        let mesh_build_ms = mesh_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        profiling.parse_ms = gen_timings.parse_ms;
        profiling.sign_grid_ms = gen_timings.sign_grid_ms;
        profiling.voxel_fill_ms = gen_timings.voxel_fill_ms;
        profiling.mesh_build_ms = mesh_build_ms;
        profiling.total_ms = total_ms;

        commands
            .entity(entities.voxel_mesh)
            .insert(Mesh3d(meshes.add(mesh)))
            .insert(MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                perceptual_roughness: 0.0,
                metallic: 0.0,
                reflectance: 0.0,
                ..default()
            })));

        match camera.mode {
            CameraMode::AutoOrbit => {
                let angle = camera.angle;
                commands.entity(entities.camera).insert(
                    Transform::from_xyz(
                        Vec3::ZERO.x + CAMERA_RADIUS * angle.cos(),
                        Vec3::ZERO.y + CAMERA_HEIGHT,
                        Vec3::ZERO.z + CAMERA_RADIUS * angle.sin(),
                    )
                    .looking_at(Vec3::ZERO, Vec3::Y),
                );
            }
            CameraMode::Manual => {
                commands.entity(entities.camera).insert(PanOrbitCamera {
                    focus: Vec3::ZERO,
                    target_focus: Vec3::ZERO,
                    ..default()
                });
                if let Ok(mut transform) = query_cam_transform.get_mut(entities.camera) {
                    transform.look_at(Vec3::ZERO, Vec3::Y);
                }
            }
        }
    }
}
