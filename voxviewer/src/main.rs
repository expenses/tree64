use bevy::{
    core_pipeline::core_3d::{graph::Core3d, CORE_3D_DEPTH_FORMAT},
    ecs::{
        query::QueryItem,
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    math::*,
    pbr::{MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey, SetMeshViewBindGroup},
    prelude::*,
    render::primitives::Aabb,
    render::render_graph::{NodeRunError, RenderGraphContext, ViewNodeRunner},
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{self, RenderGraphApp, RenderLabel},
        render_phase::*,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice},
        sync_world::MainEntity,
        view::*,
        *,
    },
};
use std::borrow::Cow;
use std::sync::atomic::AtomicU32;

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_plugins(DefaultPlugins)
        .add_plugins(bevy_panorbit_camera::PanOrbitCameraPlugin)
        .add_plugins(VoxelRendererPlugin)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Visibility::default(),
        Transform::default(),
        // This `Aabb` is necessary for the visibility checks to work.
        Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(0.5),
        },
        VoxelModel,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 1000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 2.0, 0.0)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 4.)),
    ));

    commands.spawn((
        Transform::from_xyz(-20.0, 20.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy_panorbit_camera::PanOrbitCamera::default(),
    ));
}

fn prepare_voxel_render_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<RenderingPipeline>>,
    pipeline: Res<RenderingPipeline>,
    views: Query<(Entity, &ExtractedView, &Msaa)>,
) {
    for (entity, view, msaa) in &views {
        let view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, view_key);

        commands
            .entity(entity)
            .insert(VoxelRenderPipeline(pipeline_id));
    }
}

#[derive(Component)]
struct VoxelRenderPipeline(CachedRenderPipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VoxelRendererLabel;

struct VoxelRendererPlugin;

impl Plugin for VoxelRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<VoxelModel>::default())
            .add_systems(
                PostUpdate,
                view::check_visibility::<With<VoxelModel>>
                    .in_set(VisibilitySystems::CheckVisibility),
            );
        //app.add_systems(Startup, prepare_bind_groups)

        /*
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(ExtractResourcePlugin::<GameOfLifeImages>::default());
        */
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<DrawFunctions<VoxelBinnedPhaseItem>>()
            .init_resource::<ViewBinnedRenderPhases<VoxelBinnedPhaseItem>>()
            .init_resource::<SpecializedRenderPipelines<RenderingPipeline>>()
            .add_render_command::<VoxelBinnedPhaseItem, DrawVoxelCubesCommands>()
            .add_systems(
                Render,
                (
                    queue_custom_phase_item.in_set(RenderSet::Queue),
                    prepare_voxel_render_pipelines.in_set(RenderSet::Prepare),
                ),
            );
        /*render_app.add_systems(
            Render,
            prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
        );*/

        render_app
            .add_render_graph_node::<ViewNodeRunner<VoxelRendererNode>>(Core3d, VoxelRendererLabel);
        render_app.add_render_graph_edges(
            Core3d,
            (
                VoxelRendererLabel,
                bevy::core_pipeline::core_3d::graph::Node3d::MainOpaquePass,
            ),
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<VoxelRendererPipeline>();
        render_app.init_resource::<RenderingPipeline>();
    }
}

#[derive(Copy, Clone, ShaderType, Default, Debug)]
struct Cube {
    pos: UVec3,
    size: u32,
    value: u32,
}

#[derive(Copy, Clone, ShaderType, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Uniforms {
    half_size: u32,
}

#[derive(Copy, Clone, ShaderType, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct NodeData {
    reserved_indices: u32,
}

#[derive(Copy, Clone, ShaderType, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct WorkItem {
    pos: UVec3,
    index: u32,
}

struct WorkItems {
    buffer: Buffer,
    dispatch: Buffer,
}

#[derive(Resource)]
struct RenderingPipeline {
    shader_handle: Handle<Shader>,
    draw_bgl: BindGroupLayout,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for RenderingPipeline {
    fn from_world(world: &mut World) -> Self {
        let vrp = world.resource::<VoxelRendererPipeline>();
        // Load the shader
        let shader_handle: Handle<Shader> = world.resource::<AssetServer>().load("render.wgsl");
        Self {
            shader_handle,
            draw_bgl: vrp.draw_bgl.clone(),
            mesh_pipeline: MeshPipeline::from_world(world),
        }
    }
}

impl bevy::render::render_resource::SpecializedRenderPipeline for RenderingPipeline {
    type Key = bevy::pbr::MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("VoxelRenderer::render pipeline".into()),
            layout: vec![
                self.mesh_pipeline
                    .get_view_layout(MeshPipelineViewLayoutKey::from(key))
                    .clone(),
                /*mesh_pipeline_view_layout.bind_group_layout.clone(), */
                self.draw_bgl.clone(),
            ],
            vertex: VertexState {
                shader: self.shader_handle.clone(),
                entry_point: "vertex".into(),
                buffers: Default::default(),
                shader_defs: Default::default(),
            },
            fragment: Some(FragmentState {
                shader: self.shader_handle.clone(),
                shader_defs: vec![],
                // Make sure this matches the entry point of your shader.
                // It can be anything as long as it matches here and in the shader.
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                //topology: bevy::render::render_resource::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: 4,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(Resource)]
struct VoxelRendererPipeline {
    base_bgl: BindGroupLayout,
    uniform_bgl: BindGroupLayout,
    pipeline: CachedComputePipelineId,
    bind_group: BindGroup,
    nodes: Buffer,
    uniform_bind_groups: Vec<BindGroup>,
    first_work_item: Buffer,
    flip_flop_bind_groups: [BindGroup; 2],
    work_items: [WorkItems; 2],
    u32_1_buffer: Buffer,
    cubes: Buffer,
    draw_indirect: Buffer,
    draw_bind_group: BindGroup,
    draw_bgl: BindGroupLayout,
}

impl FromWorld for VoxelRendererPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        //let mesh_pipeline_view_layout = world.resource::<bevy::pbr::MeshPipelineViewLayout>();
        let base_bgl = render_device.create_bind_group_layout(
            "VoxelRenderer::base_bgl",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, None),
                    uniform_buffer::<NodeData>(false),
                    storage_buffer_sized(false, None),
                    storage_buffer::<AtomicU32>(false),
                ),
            ),
        );
        let draw_bgl = render_device.create_bind_group_layout(
            "VoxelRenderer::draw_bgl",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (storage_buffer_read_only_sized(false, None),),
            ),
        );

        let workitem_bgl = render_device.create_bind_group_layout(
            "VoxelRenderer::workitem_bgl",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer::<AtomicU32>(false),
                ),
            ),
        );
        let uniform_bgl = render_device.create_bind_group_layout(
            "VoxelRenderer::uniform_bgl",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<Uniforms>(false),),
            ),
        );
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("VoxelRenderer::compute pipeline".into()),
            layout: vec![base_bgl.clone(), workitem_bgl.clone(), uniform_bgl.clone()],
            push_constant_ranges: Vec::new(),
            shader: world.load_asset("shader.wgsl"),
            shader_defs: vec![],
            entry_point: Cow::from("expand_voxels"),
            zero_initialize_workgroup_memory: false,
        });

        let render_device = world.resource::<RenderDevice>();
        let mut array = [1; 8 * 8 * 8];
        for z in 0..8 {
            if z % 2 == 0 {
                for i in 0..8 * 8 {
                    array[i + z * 8 * 8] = 0;
                }
            }
        }
        array[0] = 2;
        let dag = svo_dag::SvoDag::new(&array, 8, 8, 8, 256);

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("VoxelRenderer::nodes"),
            contents: dag.node_bytes(),
            usage: bevy::render::render_resource::BufferUsages::STORAGE,
        });

        let cubes = render_device.create_buffer(&BufferDescriptor {
            label: Some("VoxelRenderer::cubes"),
            size: 16 + (1_000_000 * std::mem::size_of::<Cube>()) as u64,
            usage: bevy::render::render_resource::BufferUsages::STORAGE
                | bevy::render::render_resource::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let node_data = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("VoxelRenderer::node_data"),
            contents: bytemuck::bytes_of(&NodeData {
                reserved_indices: 256,
            }),
            usage: bevy::render::render_resource::BufferUsages::UNIFORM,
        });

        let uniform_bind_groups = [1, 2, 4]
            .into_iter()
            .map(|i| {
                let uniforms = render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some(&format!("VoxelRenderer::uniforms {}", i)),
                    contents: bytemuck::bytes_of(&Uniforms { half_size: i }),
                    usage: bevy::render::render_resource::BufferUsages::UNIFORM,
                });

                render_device.create_bind_group(
                    None,
                    &uniform_bgl,
                    &BindGroupEntries::sequential((uniforms.as_entire_buffer_binding(),)),
                )
            })
            .collect();

        let draw_indirect = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("VoxelRenderer::draw_indirect_args"),
            contents: bevy::render::render_resource::DrawIndirectArgs {
                vertex_count: 0,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }
            .as_bytes(),
            usage: bevy::render::render_resource::BufferUsages::COPY_DST
                | bevy::render::render_resource::BufferUsages::STORAGE
                | bevy::render::render_resource::BufferUsages::INDIRECT,
        });

        let bind_group_0 = render_device.create_bind_group(
            None,
            &base_bgl,
            &BindGroupEntries::sequential((
                buffer.as_entire_buffer_binding(),
                node_data.as_entire_buffer_binding(),
                cubes.as_entire_buffer_binding(),
                bevy::render::render_resource::BufferBinding {
                    buffer: &draw_indirect,
                    offset: 0,
                    size: Some(std::num::NonZero::new(4).unwrap()),
                },
            )),
        );

        let draw_bind_group = render_device.create_bind_group(
            None,
            &draw_bgl,
            &BindGroupEntries::sequential((cubes.as_entire_buffer_binding(),)),
        );

        let first_work_item = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("VoxelRenderer::first_work_item"),
            contents: bytemuck::cast_slice(&[WorkItem {
                pos: UVec3::ZERO,
                index: ((dag.num_nodes() as u32) - 1),
            }]),
            usage: bevy::render::render_resource::BufferUsages::COPY_SRC,
        });

        let work_items = [
            (
                "VoxelRenderer::work_items_0",
                "VoxelRenderer::work_items_0_dispatch",
            ),
            (
                "VoxelRenderer::work_items_1",
                "VoxelRenderer::work_items_1_dispatch",
            ),
        ]
        .map(|(label, dispatch_label)| WorkItems {
            buffer: render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: 16 + (std::mem::size_of::<WorkItem>() * 100_000) as u64,
                usage: bevy::render::render_resource::BufferUsages::COPY_DST
                    | bevy::render::render_resource::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            dispatch: render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(dispatch_label),
                contents: bevy::render::render_resource::DispatchIndirectArgs { x: 1, y: 1, z: 1 }
                    .as_bytes(),
                usage: bevy::render::render_resource::BufferUsages::COPY_DST
                    | bevy::render::render_resource::BufferUsages::STORAGE
                    | bevy::render::render_resource::BufferUsages::INDIRECT,
            }),
        });

        let workitems_fn = |input: &WorkItems, output: &WorkItems| {
            render_device.create_bind_group(
                None,
                &workitem_bgl,
                &BindGroupEntries::sequential((
                    input.buffer.as_entire_buffer_binding(),
                    output.buffer.as_entire_buffer_binding(),
                    bevy::render::render_resource::BufferBinding {
                        buffer: &output.dispatch,
                        offset: 0,
                        size: Some(std::num::NonZero::new(4).unwrap()),
                    },
                )),
            )
        };

        Self {
            pipeline,
            base_bgl,
            uniform_bgl,
            nodes: buffer,
            bind_group: bind_group_0,
            flip_flop_bind_groups: [
                workitems_fn(&work_items[0], &work_items[1]),
                workitems_fn(&work_items[1], &work_items[0]),
            ],
            cubes,
            uniform_bind_groups,
            first_work_item,
            work_items,
            draw_bind_group,
            u32_1_buffer: render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("VoxelRenderer::u32_1_buffer"),
                contents: bytemuck::cast_slice(&[1_u32]),
                usage: bevy::render::render_resource::BufferUsages::COPY_SRC,
            }),
            draw_indirect,
            draw_bgl,
        }
    }
}

/// A render-world system that enqueues the entity with custom rendering into
/// the opaque render phases of each view.
fn queue_custom_phase_item(
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<VoxelBinnedPhaseItem>>,
    opaque_draw_functions: Res<DrawFunctions<VoxelBinnedPhaseItem>>,
    views: Query<(Entity, &RenderVisibleEntities)>,
) {
    let draw_custom_phase_item = opaque_draw_functions.read().id::<DrawVoxelCubesCommands>();

    // Render phases are per-view, so we need to iterate over all views so that
    // the entity appears in them. (In this example, we have only one view, but
    // it's good practice to loop over all views anyway.)
    for (view_entity, view_visible_entities) in &views {
        let opaque_phase = opaque_render_phases.entry(view_entity).or_default();
        opaque_phase.clear();

        // Find all the custom rendered entities that are visible from this
        // view.
        for &entity in view_visible_entities.get::<With<VoxelModel>>().iter() {
            // Add the custom render item. We use the
            // [`BinnedRenderPhaseType::NonMesh`] type to skip the special
            // handling that Bevy has for meshes (preprocessing, indirect
            // draws, etc.)
            //
            // The asset ID is arbitrary; we simply use [`AssetId::invalid`],
            // but you can use anything you like. Note that the asset ID need
            // not be the ID of a [`Mesh`].
            opaque_phase.add(
                VoxelBinnedPhaseItemBinKey {
                    draw_function: draw_custom_phase_item,
                },
                entity,
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

type DrawVoxelCubesCommands = (SetMeshViewBindGroup<0>, DrawVoxelCubes);

#[derive(Clone, Component, ExtractComponent)]
struct VoxelModel;

struct DrawVoxelCubes;

impl<P> RenderCommand<P> for DrawVoxelCubes
where
    P: PhaseItem,
{
    type Param = SRes<VoxelRendererPipeline>;

    type ViewQuery = ();

    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, Self::ItemQuery>>,
        custom_phase_item_buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let pipeline = custom_phase_item_buffers.into_inner();
        pass.draw_indirect(&pipeline.draw_indirect, 0);

        RenderCommandResult::Success
    }
}

use std::ops::Range;

#[derive(Default)]
struct VoxelRendererNode {
    ready: bool,
}

pub struct VoxelBinnedPhaseItem {
    pub draw_function: DrawFunctionId,
    // An entity from which data will be fetched, including the mesh if
    // applicable.
    pub representative_entity: (Entity, MainEntity),
    // The ranges of instances.
    pub batch_range: Range<u32>,
    // An extra index, which is either a dynamic offset or an index in the
    // indirect parameters list.
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for VoxelBinnedPhaseItem {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    #[inline]
    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for VoxelBinnedPhaseItem {
    type BinKey = VoxelBinnedPhaseItemBinKey;

    #[inline]
    fn new(
        key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            draw_function: key.draw_function,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

// Data that must be identical in order to batch phase items together.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoxelBinnedPhaseItemBinKey {
    pub draw_function: DrawFunctionId,
}

impl render_graph::ViewNode for VoxelRendererNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<VoxelRendererPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if !self.ready {
            match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
                CachedPipelineState::Ok(_) => {
                    self.ready = true;
                }
                #[cfg(target_arch = "wasm32")]
                CachedPipelineState::Err(
                    err @ bevy::render::render_resource::PipelineCacheError::ShaderNotLoaded(_),
                ) => {
                    log::debug!("Caught {err}. Optimistically assuming that the shader is still being downloaded.");
                }
                CachedPipelineState::Err(err) => {
                    panic!("{err}");
                }
                _ => {}
            }
        }
    }

    type ViewQuery = (
        Entity,
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static VoxelRenderPipeline,
        //Option<&'static SkyboxBindGroup>,
        //&'static ViewUniformOffset,
    );

    // Required method
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext<'_>,
        render_context: &mut RenderContext<'w>,
        (
            view,
            camera,
            target,
            depth,
            render_pipeline,
            //skybox_pipeline,
            //skybox_bind_group,
            //view_uniform_offset,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if !self.ready {
            return Ok(());
        }

        //let opaque_draw_functions = world.resource::<DrawFunctions<Opaque3d>>();
        //let draw_custom_phase_item = opaque_draw_functions.read().get::<DrawVoxelCubesCommands>();

        let voxel_phases = world.resource::<ViewBinnedRenderPhases<VoxelBinnedPhaseItem>>();
        let voxel_phase = voxel_phases.get(&view).unwrap();

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<VoxelRendererPipeline>();

        let color_attachments = [Some(target.get_color_attachment())];
        let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

        let view_entity = graph.view_entity();
        render_context.add_command_buffer_generation_task(move |render_device| {
            // Command encoder setup
            let mut command_encoder =
                render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("voxel_command_encoder"),
                });

            let render_pipeline = match pipeline_cache.get_render_pipeline(render_pipeline.0) {
                Some(render_pipeline) => render_pipeline,
                None => return command_encoder.finish(),
            };

            command_encoder.clear_buffer(&pipeline.cubes, 0, Some(4));
            command_encoder.clear_buffer(&pipeline.draw_indirect, 0, Some(4));

            command_encoder.copy_buffer_to_buffer(
                &pipeline.u32_1_buffer,
                0,
                &pipeline.work_items[0].buffer,
                0,
                4,
            );

            command_encoder.copy_buffer_to_buffer(
                &pipeline.first_work_item,
                0,
                &pipeline.work_items[0].buffer,
                // There's padding after the atomic in the buffer as it's not tightly packed.
                16,
                std::mem::size_of::<WorkItem>() as u64,
            );

            command_encoder.copy_buffer_to_buffer(
                &pipeline.u32_1_buffer,
                0,
                &pipeline.work_items[0].dispatch,
                0,
                std::mem::size_of::<u32>() as u64,
            );

            let init_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.pipeline)
                .unwrap();

            for (i, uniform_bind_group) in pipeline.uniform_bind_groups.iter().rev().enumerate() {
                command_encoder.clear_buffer(&pipeline.work_items[(i + 1) % 2].buffer, 0, Some(4));
                command_encoder.clear_buffer(
                    &pipeline.work_items[(i + 1) % 2].dispatch,
                    0,
                    Some(std::mem::size_of::<u32>() as u64),
                );

                let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("voxel_command_encoder_compute_pass"),
                    ..Default::default()
                });

                pass.set_pipeline(init_pipeline);
                pass.set_bind_group(0, &pipeline.bind_group, &[]);
                pass.set_bind_group(1, &pipeline.flip_flop_bind_groups[i % 2], &[]);
                pass.set_bind_group(2, uniform_bind_group, &[]);
                pass.dispatch_workgroups_indirect(&pipeline.work_items[i % 2].dispatch, 0);
            }

            // Render pass setup
            let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("voxel_command_encoder_render_pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);
            //let pass_span = diagnostics.pass_span(&mut render_pass, "main_opaque_pass_3d");

            if let Some(viewport) = camera.viewport.as_ref() {
                render_pass.set_camera_viewport(viewport);
            }

            render_pass.set_render_pipeline(render_pipeline);
            render_pass.set_bind_group(1, &pipeline.draw_bind_group, &[]);

            if let Err(err) = voxel_phase.render(&mut render_pass, world, view_entity) {
                error!("Error encountered while rendering the opaque phase {err:?}");
            }

            //pass_span.end(&mut render_pass);
            drop(render_pass);

            command_encoder.finish()
        });

        Ok(())
    }
}
