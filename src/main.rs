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

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::math::DimConfig;

mod expr;
mod math;

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

type Vec3Arr = [f32; 3];
type IVec3Arr = [i32; 3];
type CornerArr = [Vec3Arr; 4];

const AMBIENT_OCCLUSION_FACTORS: [f32; 4] = [1.0, 0.75, 0.5, 0.3];

#[derive(Resource)]
struct GridConfig {
    size: u32,
    voxel_size: f64,
    voxel_count: usize,
}

impl GridConfig {
    #[inline]
    fn center(&self) -> Vec3 {
        let f = self.size as f32;
        Vec3::splat(f / 2.0 - 0.5)
    }
    #[inline]
    fn camera_radius(&self) -> f32 {
        self.size as f32 * 1.5
    }
    #[inline]
    fn camera_height(&self) -> f32 {
        self.size as f32 * 0.8
    }
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
struct CameraRotation {
    angle: f32,
    speed: f32,
}

#[derive(Resource, PartialEq)]
enum CameraMode {
    AutoOrbit,
    Manual,
}

#[derive(Resource, Clone, PartialEq)]
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

type Grid = Vec<u32>;
type MeshData = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<u32>);

const FACE_DEFS: [(Vec3Arr, CornerArr, IVec3Arr); 6] = [
    // +X face
    (
        [1.0, 0.0, 0.0],
        [
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
        ],
        [1, 0, 0],
    ),
    // -X face
    (
        [-1.0, 0.0, 0.0],
        [
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        [-1, 0, 0],
    ),
    // +Y face
    (
        [0.0, 1.0, 0.0],
        [
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
        ],
        [0, 1, 0],
    ),
    // -Y face
    (
        [0.0, -1.0, 0.0],
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
        ],
        [0, -1, 0],
    ),
    // +Z face
    (
        [0.0, 0.0, 1.0],
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        [0, 0, 1],
    ),
    // -Z face
    (
        [0.0, 0.0, -1.0],
        [
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
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
    grid: &Grid,
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
    grid[grid_index(x as usize, y as usize, z as usize, grid_size_usize)] != 0
}

fn validate_expression(expr_str: &str, dim_mapping: &DimMapping) -> Result<(), String> {
    let trimmed = expr_str.trim();
    if trimmed.is_empty() {
        return Err("Expression cannot be empty".into());
    }

    expr::parse(trimmed, &dim_mapping.into()).map(|_| ())
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "NDVoxGCalc".into(),
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
    let center = grid_config.center();
    let camera_radius = grid_config.camera_radius();
    let camera_height = grid_config.camera_height();

    let cam_entity = commands
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(
                center.x + camera_radius * 0.7,
                center.y + camera_height,
                center.z + camera_radius * 0.7,
            )
            .looking_at(center, Vec3::Y),
            Exposure { ev100: 0.0 },
            Tonemapping::None,
            PanOrbitCamera {
                enabled: false,
                focus: center,
                target_focus: center,
                ..default()
            },
        ))
        .id();

    let (grids, voxel_count) =
        generate_voxels(&grid_config, &expr_config, &mut expr_status, &dim_mapping);
    grid_config.voxel_count = voxel_count;
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    #[cfg(target_arch = "wasm32")]
    build_batched_mesh_with_global_corner_ambient_occlusion(
        &mut mesh,
        &grids,
        &grid_config,
        voxel_count,
    );

    #[cfg(not(target_arch = "wasm32"))]
    build_batched_mesh_with_global_corner_ambient_occlusion_par(
        &mut mesh,
        &grids,
        &grid_config,
        voxel_count,
    );

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
    commands.insert_resource(CameraRotation {
        angle: 0.0,
        speed: 0.5,
    });
    commands.insert_resource(CameraMode::AutoOrbit);
}

fn generate_voxels(
    grid_config: &GridConfig,
    expr_config: &ExpressionConfig,
    expr_status: &mut ExpressionStatus,
    dim_mapping: &DimMapping,
) -> (Vec<Grid>, usize) {
    let size_usize = grid_config.size as usize;
    let half_extent = (grid_config.size as f64) / 2.0 * grid_config.voxel_size;

    expr_status.errors.clear();

    let mut grids = Vec::with_capacity(expr_config.entries.len());

    for (idx, entry) in expr_config.entries.iter().enumerate() {
        if !entry.enabled {
            grids.push(vec![0; size_usize.pow(3)]);
            continue;
        }

        if let Err(e) = validate_expression(&entry.expr, dim_mapping) {
            expr_status.is_valid = false;
            expr_status
                .errors
                .push(format!("Expression #{} '{}': {}", idx + 1, entry.expr, e));
            grids.push(vec![0; size_usize.pow(3)]);
            continue;
        }

        let base_color =
            ((entry.color.0 as u32) << 16) | ((entry.color.1 as u32) << 8) | (entry.color.2 as u32);

        match math::generate_voxel_grid(
            size_usize,
            &entry.expr,
            base_color,
            half_extent,
            &dim_mapping.into(),
        ) {
            Ok(grid) => {
                grids.push(grid);
            }
            Err(e) => {
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

    let mut rendered_voxel_count = 0;
    let total_positions = size_usize.pow(3);
    for idx in 0..total_positions {
        if grids.iter().any(|grid| grid[idx] != 0) {
            rendered_voxel_count += 1;
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

    (grids, rendered_voxel_count)
}

fn process_x_range_multi(
    x_start: u32,
    x_end: u32,
    grids: &[Grid],
    size: u32,
    voxel_count: usize,
) -> MeshData {
    let size_i32 = size as i32;
    let size_usize = size as usize;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(voxel_count * 30);
    let mut indices: Vec<u32> = Vec::with_capacity(voxel_count * 72);

    for x in x_start..x_end {
        for y in 0..size {
            for z in 0..size {
                let idx = grid_index(x as usize, y as usize, z as usize, size_usize);

                // Find first non-zero voxel value across all grids (priority by order)
                let voxel_val = grids
                    .iter()
                    .map(|grid| grid[idx])
                    .find(|&v| v != 0)
                    .unwrap_or(0);

                if voxel_val == 0 {
                    continue;
                }

                let base_linear = decode_color(voxel_val);
                let offset = Vec3::new(x as f32, y as f32, z as f32);

                // Compute ambient occlusion using composite occupancy
                let mut corner_ambient_occlusion = [1.0f32; 8];
                for cx_off in 0..2 {
                    for cy_off in 0..2 {
                        for cz_off in 0..2 {
                            let corner_idx = cx_off | (cy_off << 1) | (cz_off << 2);
                            let cx = x as i32 + cx_off as i32;
                            let cy = y as i32 + cy_off as i32;
                            let cz = z as i32 + cz_off as i32;

                            let mut occlusion = 0;
                            // Check occupancy across all grids for each axis
                            for grid in grids {
                                if is_occupied(grid, cx - 1, cy, cz, size_i32, size_usize) {
                                    occlusion += 1;
                                    break;
                                }
                            }
                            for grid in grids {
                                if is_occupied(grid, cx, cy - 1, cz, size_i32, size_usize) {
                                    occlusion += 1;
                                    break;
                                }
                            }
                            for grid in grids {
                                if is_occupied(grid, cx, cy, cz - 1, size_i32, size_usize) {
                                    occlusion += 1;
                                    break;
                                }
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

                    // Check if neighbor is occupied by any function
                    let neighbor_occupied = grids
                        .iter()
                        .any(|grid| is_occupied(grid, nx, ny, nz, size_i32, size_usize));

                    if neighbor_occupied {
                        continue;
                    }

                    let mut ambient_occlusion_sum = 0.0;
                    let mut cx_sum = 0.0;
                    let mut cy_sum = 0.0;
                    let mut cz_sum = 0.0;
                    let mut corner_colors = [[0.0; 4]; 4];

                    for (i, &corner) in corners.iter().enumerate() {
                        let [cx, cy, cz] = corner;
                        cx_sum += cx;
                        cy_sum += cy;
                        cz_sum += cz;

                        let cx_off = if cx > 0.0 { 1 } else { 0 };
                        let cy_off = if cy > 0.0 { 1 } else { 0 };
                        let cz_off = if cz > 0.0 { 1 } else { 0 };
                        let corner_idx = cx_off | (cy_off << 1) | (cz_off << 2);

                        let ambient_occlusion = corner_ambient_occlusion[corner_idx];
                        ambient_occlusion_sum += ambient_occlusion;

                        corner_colors[i] = [
                            base_linear.red * ambient_occlusion,
                            base_linear.green * ambient_occlusion,
                            base_linear.blue * ambient_occlusion,
                            base_linear.alpha,
                        ];
                    }

                    let center_ambient_occlusion = ambient_occlusion_sum / 4.0;
                    let center_color = [
                        base_linear.red * center_ambient_occlusion,
                        base_linear.green * center_ambient_occlusion,
                        base_linear.blue * center_ambient_occlusion,
                        base_linear.alpha,
                    ];
                    let center_pos = [
                        cx_sum / 4.0 + offset.x,
                        cy_sum / 4.0 + offset.y,
                        cz_sum / 4.0 + offset.z,
                    ];

                    let start_idx = positions.len() as u32;

                    positions.push(center_pos);
                    normals.push(normal);
                    colors.push(center_color);

                    for (i, &corner) in corners.iter().enumerate() {
                        let [cx, cy, cz] = corner;
                        positions.push([cx + offset.x, cy + offset.y, cz + offset.z]);
                        normals.push(normal);
                        colors.push(corner_colors[i]);
                    }

                    indices.extend_from_slice(&[
                        start_idx,
                        start_idx + 1,
                        start_idx + 2,
                        start_idx,
                        start_idx + 2,
                        start_idx + 3,
                        start_idx,
                        start_idx + 3,
                        start_idx + 4,
                        start_idx,
                        start_idx + 4,
                        start_idx + 1,
                    ]);
                }
            }
        }
    }

    (positions, normals, colors, indices)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn build_batched_mesh_with_global_corner_ambient_occlusion(
    mesh: &mut Mesh,
    grids: &[Grid],
    grid_config: &GridConfig,
    voxel_count: usize,
) {
    info!("voxel_count: {}", voxel_count);
    let size = grid_config.size;
    let (positions, normals, colors, indices) =
        process_x_range_multi(0, size, grids, size, voxel_count);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
}

#[cfg(not(target_arch = "wasm32"))]
fn build_batched_mesh_with_global_corner_ambient_occlusion_par(
    mesh: &mut Mesh,
    grids: &[Grid],
    grid_config: &GridConfig,
    voxel_count: usize,
) {
    info!("voxel_count: {}", voxel_count);
    let size = grid_config.size;
    let chunk_count = num_cpus::get();
    let chunk_size = (size as usize).div_ceil(chunk_count);
    let voxel_count_per_chunk = voxel_count.div_ceil(chunk_count);
    let results: Vec<_> = (0..size)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|x_range| {
            let x_start = x_range[0];
            let x_end = x_range[x_range.len() - 1] + 1;
            process_x_range_multi(x_start, x_end, grids, size, voxel_count_per_chunk)
        })
        .collect();

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(voxel_count * 30);
    let mut indices: Vec<u32> = Vec::with_capacity(voxel_count * 72);

    let mut vertex_offset = 0u32;
    for (pos, norm, col, ind) in results {
        positions.extend(pos);
        normals.extend(norm);
        colors.extend(col);
        indices.extend(ind.into_iter().map(|i| i + vertex_offset));
        vertex_offset = positions.len() as u32;
    }

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
        });
}

fn draw_axes_and_planes(
    mut gizmos: Gizmos,
    grid_config: Res<GridConfig>,
    show: Res<ShowAxesPlanes>,
) {
    let center = grid_config.center();
    let min = -0.5;
    let max = grid_config.size as f32 - 0.5;

    if show.show_axes {
        gizmos.line(
            Vec3::new(min, center.y, center.z),
            Vec3::new(max, center.y, center.z),
            Color::srgb(1.0, 0.2, 0.2),
        );
        gizmos.line(
            Vec3::new(center.x, min, center.z),
            Vec3::new(center.x, max, center.z),
            Color::srgb(0.2, 1.0, 0.2),
        );
        gizmos.line(
            Vec3::new(center.x, center.y, min),
            Vec3::new(center.x, center.y, max),
            Color::srgb(0.2, 0.2, 1.0),
        );
    }

    let n = grid_config.size;
    let step = ((n as f32 / 32.0).round() as u32).next_power_of_two();

    if show.show_ground_grid {
        let y = min;
        let gc = Color::srgba(0.5, 0.5, 0.5, 0.35);

        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 - 0.5;
            gizmos.line(Vec3::new(min, y, p), Vec3::new(max, y, p), gc);
            gizmos.line(Vec3::new(p, y, min), Vec3::new(p, y, max), gc);
        }
    }

    if show.show_planes {
        // XY plane at z = center.z — red grid
        let z = center.z;
        let cr = Color::srgba(1.0, 0.2, 0.2, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 - 0.5;
            gizmos.line(Vec3::new(min, p, z), Vec3::new(max, p, z), cr);
            gizmos.line(Vec3::new(p, min, z), Vec3::new(p, max, z), cr);
        }

        // XZ plane at y = center.y — green grid
        let y = center.y;
        let cg = Color::srgba(0.2, 1.0, 0.2, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 - 0.5;
            gizmos.line(Vec3::new(min, y, p), Vec3::new(max, y, p), cg);
            gizmos.line(Vec3::new(p, y, min), Vec3::new(p, y, max), cg);
        }

        // YZ plane at x = center.x — blue grid
        let x = center.x;
        let cb = Color::srgba(0.2, 0.2, 1.0, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let p = i as f32 - 0.5;
            gizmos.line(Vec3::new(x, min, p), Vec3::new(x, max, p), cb);
            gizmos.line(Vec3::new(x, p, min), Vec3::new(x, p, max), cb);
        }
    }
}

fn rotate_camera(
    time: Res<Time>,
    mut rotation: ResMut<CameraRotation>,
    mut query_camera: Query<&mut Transform, With<Camera3d>>,
    grid_config: Res<GridConfig>,
    camera_mode: Res<CameraMode>,
) {
    if *camera_mode != CameraMode::AutoOrbit {
        return;
    }
    rotation.angle += time.delta_secs() * rotation.speed;
    let center = grid_config.center();
    let radius = grid_config.camera_radius();
    let height = grid_config.camera_height();
    for mut t in query_camera.iter_mut() {
        t.translation = Vec3::new(
            center.x + radius * rotation.angle.cos(),
            center.y + height,
            center.z + radius * rotation.angle.sin(),
        );
        t.look_at(center, Vec3::Y);
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
    mut camera_mode: ResMut<CameraMode>,
    mut query_cam: Query<&mut PanOrbitCamera>,
    mut auto_regen: ResMut<RegenerateEveryFrame>,
    mut regenerate_request: Local<bool>,
    mut expr_config: ResMut<ExpressionConfig>,
    mut expr_status: ResMut<ExpressionStatus>,
    entities: Res<SceneEntities>,
    rotation: Res<CameraRotation>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query_cam_transform: Query<&mut Transform>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    egui::SidePanel::left("left_panel")
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Generation");

            ui.label("Expressions:");

            // Scrollable area for expression list with fixed max height
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let mut remove_idx = None;
                    for (idx, entry) in expr_config.entries.iter_mut().enumerate() {
                        egui::CollapsingHeader::new(format!("Function #{}", idx + 1))
                            .id_salt(("func_header", idx))
                            .default_open(true)
                            .show(ui, |ui| {
                                // Trigger regeneration when enabled state changes
                                if ui.checkbox(&mut entry.enabled, "Enabled").changed() {
                                    *regenerate_request = true;
                                }

                                ui.label("Expression:");
                                if ui.text_edit_singleline(&mut entry.expr).changed() {
                                    *regenerate_request = true;
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
                                        *regenerate_request = true;
                                    }
                                });

                                if ui.small_button("❌ Remove").clicked() {
                                    remove_idx = Some(idx);
                                }
                            });
                        ui.separator();
                    }

                    // Apply removals after iteration to avoid borrow issues
                    if let Some(idx) = remove_idx {
                        expr_config.entries.remove(idx);
                        *regenerate_request = true;
                    }
                });

            ui.separator();

            if ui.button("➕ Add Expression").clicked() {
                expr_config.entries.push(ExpressionEntry {
                    expr: String::new(),
                    color: rand::random(),
                    enabled: true,
                });
                *regenerate_request = true;
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
                    *regenerate_request = true;
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
                    *regenerate_request = true;
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
                    *regenerate_request = true;
                }
            });

            ui.label("Voxel Size:");
            ui.horizontal(|ui| {
                let mut vs = grid_config.voxel_size as f32;
                if ui
                    .add(
                        egui::Slider::new(&mut vs, 0.1..=10.0)
                            .logarithmic(true)
                            .custom_formatter(|n, _| format!("{n:.2}")),
                    )
                    .changed()
                {
                    grid_config.voxel_size = vs as f64;
                    *regenerate_request = true;
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
                    *regenerate_request = true;
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
                *regenerate_request = true;
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
                                    .custom_formatter(|val, _| {
                                        let s = format!("{:.6}", val);
                                        s.trim_end_matches('0').trim_end_matches('.').to_string()
                                    })
                                    .range(-max_offset..=max_offset),
                            )
                            .changed()
                        {
                            *regenerate_request = true;
                        }
                    });
                }
            }

            ui.separator();

            if ui.button("Regenerate").clicked() {
                *regenerate_request = true;
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
                                .custom_formatter(|val, _| {
                                    let s = format!("{:.6}", val);
                                    s.trim_end_matches('0').trim_end_matches('.').to_string()
                                })
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
                                .custom_formatter(|val, _| {
                                    let s = format!("{:.6}", val);
                                    s.trim_end_matches('0').trim_end_matches('.').to_string()
                                })
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
                                .custom_formatter(|val, _| {
                                    let s = format!("{:.6}", val);
                                    s.trim_end_matches('0').trim_end_matches('.').to_string()
                                })
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
                    *regenerate_request = true;
                }
            });

            let cam_label = match *camera_mode {
                CameraMode::AutoOrbit => "Camera: Auto",
                CameraMode::Manual => "Camera: Manual",
            };
            if ui.button(cam_label).clicked() {
                let center = grid_config.center();
                *camera_mode = match *camera_mode {
                    CameraMode::AutoOrbit => {
                        for mut poc in &mut query_cam {
                            poc.enabled = true;
                            poc.focus = center;
                            poc.target_focus = center;
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
        });

    let should_regenerate = *regenerate_request || auto_regen.enabled;
    *regenerate_request = false;

    if should_regenerate {
        let (grids, count) =
            generate_voxels(&grid_config, &expr_config, &mut expr_status, &dim_mapping);
        grid_config.voxel_count = count;

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );

        #[cfg(target_arch = "wasm32")]
        build_batched_mesh_with_global_corner_ambient_occlusion(
            &mut mesh,
            &grids,
            &grid_config,
            count,
        );

        #[cfg(not(target_arch = "wasm32"))]
        build_batched_mesh_with_global_corner_ambient_occlusion_par(
            &mut mesh,
            &grids,
            &grid_config,
            count,
        );

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

        let center = grid_config.center();
        match *camera_mode {
            CameraMode::AutoOrbit => {
                let radius = grid_config.camera_radius();
                let height = grid_config.camera_height();
                let angle = rotation.angle;
                commands.entity(entities.camera).insert(
                    Transform::from_xyz(
                        center.x + radius * angle.cos(),
                        center.y + height,
                        center.z + radius * angle.sin(),
                    )
                    .looking_at(center, Vec3::Y),
                );
            }
            CameraMode::Manual => {
                commands.entity(entities.camera).insert(PanOrbitCamera {
                    focus: center,
                    target_focus: center,
                    ..default()
                });
                if let Ok(mut transform) = query_cam_transform.get_mut(entities.camera) {
                    transform.look_at(center, Vec3::Y);
                }
            }
        }
    }
}
