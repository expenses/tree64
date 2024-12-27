use bevy::{
    core_pipeline::core_3d::{Opaque3d, Opaque3dBinKey, CORE_3D_DEPTH_FORMAT},
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    math::*,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        primitives::*,
        render_asset::{RenderAsset, RenderAssets},
        render_graph::{self, RenderGraph, RenderLabel},
        render_phase::*,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice},
        storage::*,
        view::{self, ExtractedView, RenderVisibleEntities, VisibilitySystems},
        *,
    },
};
use std::sync::atomic::AtomicU32;
use std::{borrow::Cow, error::Error};

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_plugins(DefaultPlugins)
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
        CustomRenderedEntity,
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VoxelRendererLabel;

struct VoxelRendererPlugin;

impl Plugin for VoxelRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<CustomRenderedEntity>::default())
            .add_systems(
                PostUpdate,
                view::check_visibility::<With<CustomRenderedEntity>>
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
            .add_render_command::<Opaque3d, DrawCustomPhaseItemCommands>()
            .add_systems(Render, queue_custom_phase_item.in_set(RenderSet::Queue));
        /*render_app.add_systems(
            Render,
            prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
        );*/

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(VoxelRendererLabel, VoxelRendererNode::default());
        render_graph.add_node_edge(VoxelRendererLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<VoxelRendererPipeline>();
        //render_app.init_resource::<VoxelRendererBaseResources>();
    }
}

#[derive(Copy, Clone, ShaderType, Default, Debug)]
struct Cube {
    pos: UVec3,
    size: u32,
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
    render_pipeline: CachedRenderPipelineId,
    draw_bind_group: BindGroup,
}

impl FromWorld for VoxelRendererPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        //let mesh_pipeline_view_layout = world.resource::<bevy::pbr::MeshPipelineViewLayout>();
        let base_bgl = render_device.create_bind_group_layout(
            "VoxelRendererBase",
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
            "VoxelRendererDraw",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (storage_buffer_read_only_sized(false, None),),
            ),
        );

        let workitem_bgl = render_device.create_bind_group_layout(
            "VoxelRendererWorkItems",
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
            "VoxelRendererUniform",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<Uniforms>(false),),
            ),
        );
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("compute pipeline".into()),
            layout: vec![base_bgl.clone(), workitem_bgl.clone(), uniform_bgl.clone()],
            push_constant_ranges: Vec::new(),
            shader: world.load_asset("shader.wgsl"),
            shader_defs: vec![],
            entry_point: Cow::from("expand_voxels"),
            zero_initialize_workgroup_memory: false,
        });

        let render_shader = world.load_asset("render.wgsl");

        let render_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("render pipeline".into()),
            layout: vec![/*mesh_pipeline_view_layout.bind_group_layout.clone(), */draw_bgl.clone()],
            vertex: VertexState {
                shader: render_shader.clone(),
                entry_point: "vertex".into(),
                buffers: Default::default(),
                shader_defs: Default::default(),
            },
            fragment: Some(FragmentState {
                shader: render_shader,
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
                topology: bevy::render::render_resource::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: false,
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
        });

        let render_device = world.resource::<RenderDevice>();
        let mut array = [1; 8 * 8 * 8];
        array[0] = 0;
        let dag = svo_dag::SvoDag::new(&array, 8, 8, 8, 256);

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("nodes"),
            contents: dag.node_bytes(),
            usage: bevy::render::render_resource::BufferUsages::STORAGE,
        });

        let cubes = render_device.create_buffer(&BufferDescriptor {
            label: Some("cubes"),
            size: 16 + (1_000_000 * std::mem::size_of::<Cube>()) as u64,
            usage: bevy::render::render_resource::BufferUsages::STORAGE
                | bevy::render::render_resource::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atomic_u32 = |label| {
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[0_u32]),
                usage: bevy::render::render_resource::BufferUsages::STORAGE
                    | bevy::render::render_resource::BufferUsages::COPY_DST,
            })
        };

        let node_data = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("NodeData"),
            contents: bytemuck::bytes_of(&NodeData {
                reserved_indices: 256,
            }),
            usage: bevy::render::render_resource::BufferUsages::UNIFORM,
        });

        let uniform_bind_groups = [1, 2, 4]
            .into_iter()
            .map(|i| {
                let uniforms = render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some(&format!("uniforms {}", i)),
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
            label: Some("draw_indirect_args"),
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
            label: Some("first_work_item"),
            contents: bytemuck::cast_slice(&[WorkItem {
                pos: UVec3::ZERO,
                index: ((dag.num_nodes() as u32) - 1),
            }]),
            usage: bevy::render::render_resource::BufferUsages::COPY_SRC,
        });

        let work_items = [
            ("work_items_0", "work_items_0_dispatch"),
            ("work_items_1", "work_items_1_dispatch"),
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
                label: Some(&dispatch_label),
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
                label: Some("u32_1_buffer"),
                contents: bytemuck::cast_slice(&[1_u32]),
                usage: bevy::render::render_resource::BufferUsages::COPY_SRC,
            }),
            draw_indirect,
            render_pipeline,
        }
    }
}

/// A render-world system that enqueues the entity with custom rendering into
/// the opaque render phases of each view.
fn queue_custom_phase_item(
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    pipeline: Res<VoxelRendererPipeline>,
    views: Query<(Entity, &RenderVisibleEntities, &Msaa), With<ExtractedView>>,
) {
    let draw_custom_phase_item = opaque_draw_functions
        .read()
        .id::<DrawCustomPhaseItemCommands>();

    // Render phases are per-view, so we need to iterate over all views so that
    // the entity appears in them. (In this example, we have only one view, but
    // it's good practice to loop over all views anyway.)
    for (view_entity, view_visible_entities, msaa) in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view_entity) else {
            continue;
        };

        // Find all the custom rendered entities that are visible from this
        // view.
        for &entity in view_visible_entities
            .get::<With<CustomRenderedEntity>>()
            .iter()
        {
            // Add the custom render item. We use the
            // [`BinnedRenderPhaseType::NonMesh`] type to skip the special
            // handling that Bevy has for meshes (preprocessing, indirect
            // draws, etc.)
            //
            // The asset ID is arbitrary; we simply use [`AssetId::invalid`],
            // but you can use anything you like. Note that the asset ID need
            // not be the ID of a [`Mesh`].
            opaque_phase.add(
                Opaque3dBinKey {
                    draw_function: draw_custom_phase_item,
                    pipeline: pipeline.render_pipeline,
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                    material_bind_group_id: None,
                    lightmap_image: None,
                },
                entity,
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

type DrawCustomPhaseItemCommands = (SetItemPipeline, DrawCustomPhaseItem);

#[derive(Clone, Component, ExtractComponent)]
struct CustomRenderedEntity;

struct DrawCustomPhaseItem;

impl<P> RenderCommand<P> for DrawCustomPhaseItem
where
    P: PhaseItem,
{
    type Param = (SRes<VoxelRendererPipeline>);

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

        pass.set_bind_group(0, &pipeline.draw_bind_group, &[]);
        pass.draw_indirect(&pipeline.draw_indirect, 0);

        RenderCommandResult::Success
    }
}

#[derive(Default)]
struct VoxelRendererNode {
    ready: bool,
}

impl render_graph::Node for VoxelRendererNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<VoxelRendererPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if !self.ready {
            match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
                CachedPipelineState::Ok(_) => {
                    self.ready = true;
                }
                CachedPipelineState::Err(err) => {
                    panic!("{err}");
                }
                _ => {}
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !self.ready {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<VoxelRendererPipeline>();

        let mut command_encoder = render_context.command_encoder();

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

            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_bind_group(0, &pipeline.bind_group, &[]);
            pass.set_bind_group(1, &pipeline.flip_flop_bind_groups[i % 2], &[]);
            pass.set_bind_group(2, uniform_bind_group, &[]);
            pass.set_pipeline(init_pipeline);
            pass.dispatch_workgroups_indirect(&pipeline.work_items[i % 2].dispatch, 0);
        }

        Ok(())
    }
}
