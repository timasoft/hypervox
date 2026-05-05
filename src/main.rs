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
use rand::RngExt;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

type Vec3Arr = [f32; 3];
type IVec3Arr = [i32; 3];
type CornerArr = [Vec3Arr; 4];

const AMBIENT_OCCLUSION_FACTORS: [f32; 4] = [1.0, 0.75, 0.5, 0.3];

#[derive(Resource)]
struct GridConfig {
    size: u32,
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
struct SceneEntities {
    camera: Entity,
    voxel_mesh: Entity,
}

#[derive(Resource)]
struct CameraRotation {
    angle: f32,
    speed: f32,
}

#[derive(Component)]
struct FpsText;
#[derive(Component)]
struct VoxelRoot;
#[derive(Component)]
struct RegenerateButton;
#[derive(Component)]
struct RegenerateEveryFrameButton;
#[derive(Component)]
struct RegenerateEveryFrameText;
#[derive(Resource)]
struct RegenerateEveryFrame {
    enabled: bool,
}
#[derive(Component)]
struct GridSizeText;
#[derive(Component)]
struct GridMinusButton;
#[derive(Component)]
struct GridPlusButton;

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
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "NDVoxGCalc".into(),
                mode: WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                prevent_default_event_handling: true,
                resolution: WindowResolution::default().with_scale_factor_override(1.0),
                present_mode: PresentMode::Fifo,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::WHITE))
        .insert_resource(GridConfig { size: 64 })
        .insert_resource(RegenerateEveryFrame { enabled: false })
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_systems(Startup, (setup, setup_fps_text, setup_grid_ui))
        .add_systems(
            Update,
            (
                rotate_camera,
                update_fps_text,
                regenerate_voxels,
                update_grid_ui,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    grid_config: Res<GridConfig>,
) {
    let center = grid_config.center();
    let camera_radius = grid_config.camera_radius();
    let camera_height = grid_config.camera_height();

    // 1. Spawn camera and store Entity
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
        ))
        .id();

    // 2. Generate voxels and mesh
    let (grid, voxel_count) = generate_voxels(&grid_config);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    #[cfg(target_arch = "wasm32")]
    build_batched_mesh_with_global_corner_ambient_occlusion(
        &mut mesh,
        &grid,
        &grid_config,
        voxel_count,
    );

    #[cfg(not(target_arch = "wasm32"))]
    build_batched_mesh_with_global_corner_ambient_occlusion_par(
        &mut mesh,
        &grid,
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

    // 3. Spawn mesh and store Entity
    let mesh_entity = commands
        .spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material_handle),
            VoxelRoot,
        ))
        .id();

    // 4. Store references and initialize rotation
    commands.insert_resource(SceneEntities {
        camera: cam_entity,
        voxel_mesh: mesh_entity,
    });
    commands.insert_resource(CameraRotation {
        angle: 0.0,
        speed: 0.5,
    });

    setup_ui(commands);
}

fn generate_voxels(grid_config: &GridConfig) -> (Grid, usize) {
    let mut rng = rand::rng();
    let size = grid_config.size;
    let size_f = grid_config.size as f32;
    let size_usize = grid_config.size as usize;

    // 0 = empty, 1..=0xFFFFFF+1 = voxel with encoded color
    let mut grid = vec![0u32; size_usize.pow(3)];
    let mut voxel_count = 0;

    for x in 0..size {
        for y in 0..size {
            for z in 0..size {
                let chance = ((size - y) as f32 / size_f).powi(4);
                if rng.random::<f32>() < chance {
                    // Generate 24-bit color and add +1 so 0 remains empty
                    let color_val = (rng.random::<u32>() & 0xFFFFFF) + 1;
                    let idx = grid_index(x as usize, y as usize, z as usize, size_usize);
                    grid[idx] = color_val;
                    voxel_count += 1;
                }
            }
        }
    }

    (grid, voxel_count)
}

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(12.0),
                left: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::NONE),
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(160.0),
                        height: Val::Px(40.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(2.0)),
                        margin: UiRect::bottom(Val::Px(8.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.8)),
                    BorderColor::all(Color::srgb(0.5, 0.5, 0.5)),
                    RegenerateButton,
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Text::new("Regenerate"),
                        TextFont {
                            font_size: 18.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.95, 0.95, 0.95)),
                    ));
                });

            parent
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(160.0),
                        height: Val::Px(40.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(2.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.8)),
                    BorderColor::all(Color::srgb(0.5, 0.5, 0.5)),
                    RegenerateEveryFrameButton,
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Text::new("Auto: OFF"),
                        TextFont {
                            font_size: 18.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.95, 0.95, 0.95)),
                        RegenerateEveryFrameText,
                    ));
                });
        });
}

fn setup_fps_text(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                padding: UiRect::all(Val::Px(8.0)),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
            BorderColor::all(Color::srgb(0.3, 0.3, 0.3)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("FPS: --"),
                TextFont {
                    font_size: 20.0,
                    ..default()
                },
                TextColor(Color::srgb(0.95, 0.95, 0.95)),
                FpsText,
            ));
        });
}

fn update_fps_text(diagnostics: Res<DiagnosticsStore>, mut query: Query<&mut Text, With<FpsText>>) {
    let Ok(mut text) = query.single_mut() else {
        warn!("FPS text entity not found or multiple entities with FpsText component");
        return;
    };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .map(|v| v.round() as u32)
        .unwrap_or(0);

    let new_value = format!("FPS: {fps}");
    if **text != new_value {
        text.0 = new_value;
    }
}

fn process_x_range(
    x_start: u32,
    x_end: u32,
    grid: &Grid,
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
                let voxel_val = grid[idx];
                if voxel_val == 0 {
                    continue; // empty voxel
                }

                let base_linear = decode_color(voxel_val);
                let offset = Vec3::new(x as f32, y as f32, z as f32);

                // Calculate AO for 8 voxel corners
                let mut corner_ambient_occlusion = [1.0f32; 8];
                for cx_off in 0..2 {
                    for cy_off in 0..2 {
                        for cz_off in 0..2 {
                            let corner_idx = cx_off | (cy_off << 1) | (cz_off << 2);
                            let cx = x as i32 + cx_off as i32;
                            let cy = y as i32 + cy_off as i32;
                            let cz = z as i32 + cz_off as i32;

                            let mut occlusion = 0;
                            if is_occupied(grid, cx - 1, cy, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            if is_occupied(grid, cx, cy - 1, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            if is_occupied(grid, cx, cy, cz - 1, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            corner_ambient_occlusion[corner_idx] =
                                AMBIENT_OCCLUSION_FACTORS[occlusion];
                        }
                    }
                }

                // Process 6 faces
                for (normal, corners, neighbor_offset) in FACE_DEFS {
                    let nx = x as i32 + neighbor_offset[0];
                    let ny = y as i32 + neighbor_offset[1];
                    let nz = z as i32 + neighbor_offset[2];

                    if is_occupied(grid, nx, ny, nz, size_i32, size_usize) {
                        continue; // face is occluded by neighbor
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

                    // Face center vertex
                    positions.push(center_pos);
                    normals.push(normal);
                    colors.push(center_color);

                    // 4 corner vertices
                    for (i, &corner) in corners.iter().enumerate() {
                        let [cx, cy, cz] = corner;
                        positions.push([cx + offset.x, cy + offset.y, cz + offset.z]);
                        normals.push(normal);
                        colors.push(corner_colors[i]);
                    }

                    // 4 triangles (fan from center)
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
    grid: &Grid,
    grid_config: &GridConfig,
    voxel_count: usize,
) {
    info!("voxel_count: {}", voxel_count);

    let size = grid_config.size;

    let (positions, normals, colors, indices) = process_x_range(0, size, grid, size, voxel_count);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
}

#[cfg(not(target_arch = "wasm32"))]
fn build_batched_mesh_with_global_corner_ambient_occlusion_par(
    mesh: &mut Mesh,
    grid: &Grid,
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
            process_x_range(x_start, x_end, grid, size, voxel_count_per_chunk)
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

fn rotate_camera(
    time: Res<Time>,
    mut rotation: ResMut<CameraRotation>,
    mut query_camera: Query<&mut Transform, With<Camera3d>>,
    grid_config: Res<GridConfig>,
) {
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

fn setup_grid_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                right: Val::Px(12.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(8.0)),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
            BorderColor::all(Color::srgb(0.3, 0.3, 0.3)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Grid Size: "),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(0.95, 0.95, 0.95)),
            ));

            // Minus button
            parent
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(30.0),
                        height: Val::Px(30.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        margin: UiRect::horizontal(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.8)),
                    BorderColor::all(Color::srgb(0.5, 0.5, 0.5)),
                    GridMinusButton,
                ))
                .with_children(|p| {
                    p.spawn((
                        Text::new("-"),
                        TextFont {
                            font_size: 20.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.95, 0.95, 0.95)),
                    ));
                });

            // Size text
            parent.spawn((
                Text::new("64"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(0.95, 0.95, 0.95)),
                GridSizeText,
            ));

            // Plus button
            parent
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(30.0),
                        height: Val::Px(30.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        margin: UiRect::horizontal(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.8)),
                    BorderColor::all(Color::srgb(0.5, 0.5, 0.5)),
                    GridPlusButton,
                ))
                .with_children(|p| {
                    p.spawn((
                        Text::new("+"),
                        TextFont {
                            font_size: 20.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.95, 0.95, 0.95)),
                    ));
                });
        });
}

fn rebuild_scene(
    mut commands: Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    grid_config: &GridConfig,
    entities: &SceneEntities,
    rotation: &CameraRotation,
) {
    let (grid, voxel_count) = generate_voxels(grid_config);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    #[cfg(target_arch = "wasm32")]
    build_batched_mesh_with_global_corner_ambient_occlusion(
        &mut mesh,
        &grid,
        grid_config,
        voxel_count,
    );

    #[cfg(not(target_arch = "wasm32"))]
    build_batched_mesh_with_global_corner_ambient_occlusion_par(
        &mut mesh,
        &grid,
        grid_config,
        voxel_count,
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

#[allow(clippy::too_many_arguments)]
fn update_grid_ui(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query_minus: Query<&Interaction, (Changed<Interaction>, With<GridMinusButton>)>,
    query_plus: Query<&Interaction, (Changed<Interaction>, With<GridPlusButton>)>,
    mut grid_config: ResMut<GridConfig>,
    mut text_query: Query<&mut Text, With<GridSizeText>>,
    entities: Res<SceneEntities>,
    rotation: Res<CameraRotation>,
) {
    let mut changed = false;

    if let Ok(interaction) = query_minus.single()
        && *interaction == Interaction::Pressed
    {
        if grid_config.size > 8 {
            grid_config.size = grid_config.size.saturating_sub(8);
            changed = true;
        } else if grid_config.size > 4 {
            grid_config.size = grid_config.size.saturating_sub(4);
            changed = true;
        } else if grid_config.size > 2 {
            grid_config.size = grid_config.size.saturating_sub(2).max(2);
            changed = true;
        }
    }

    if let Ok(interaction) = query_plus.single()
        && *interaction == Interaction::Pressed
        && grid_config.size < 256
    {
        if grid_config.size <= 2 {
            grid_config.size = (grid_config.size + 2).min(256);
        } else if grid_config.size <= 4 {
            grid_config.size = (grid_config.size + 4).min(256);
        } else if grid_config.size < 8 {
            grid_config.size = 8;
        } else {
            grid_config.size = (grid_config.size + 8).min(256);
        }
        changed = true;
    }

    if changed {
        if let Ok(mut text) = text_query.single_mut() {
            text.0 = grid_config.size.to_string();
        } else {
            warn!("GridSizeText entity not found or duplicated, UI may be out of sync");
        }
        rebuild_scene(
            commands.reborrow(),
            &mut meshes,
            &mut materials,
            &grid_config,
            &entities,
            &rotation,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn regenerate_voxels(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query_button: Query<&Interaction, (Changed<Interaction>, With<RegenerateButton>)>,
    query_auto_button: Query<
        &Interaction,
        (Changed<Interaction>, With<RegenerateEveryFrameButton>),
    >,
    mut auto_regen: ResMut<RegenerateEveryFrame>,
    grid_config: Res<GridConfig>,
    entities: Res<SceneEntities>,
    rotation: Res<CameraRotation>,
    mut query_auto_text: Query<&mut Text, With<RegenerateEveryFrameText>>,
) {
    if let Ok(interaction) = query_auto_button.single()
        && *interaction == Interaction::Pressed
    {
        auto_regen.enabled = !auto_regen.enabled;
        if let Ok(mut text) = query_auto_text.single_mut() {
            text.0 = if auto_regen.enabled {
                "Auto: ON"
            } else {
                "Auto: OFF"
            }
            .into();
        }
    }

    let should_regenerate = query_button
        .single()
        .is_ok_and(|i| *i == Interaction::Pressed)
        || auto_regen.enabled;

    if !should_regenerate {
        return;
    }

    rebuild_scene(
        commands.reborrow(),
        &mut meshes,
        &mut materials,
        &grid_config,
        &entities,
        &rotation,
    );
}
