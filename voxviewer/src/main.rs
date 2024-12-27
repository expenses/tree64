use bevy::{
    prelude::*,
    render::{
        render_asset::{RenderAsset, RenderAssets},
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice},
        storage::*,
        Render, RenderApp, RenderSet,
    },
};
use std::sync::atomic::AtomicU32;
use std::{borrow::Cow, error::Error};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelRendererPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    //commands.init_resource::<Pipeline>();
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VoxelRendererLabel;

struct VoxelRendererPlugin;

impl Plugin for VoxelRendererPlugin {
    fn build(&self, app: &mut App) {
        //app.add_systems(Startup, prepare_bind_groups)

        /*
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(ExtractResourcePlugin::<GameOfLifeImages>::default());
        */
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
        );

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

fn prepare_bind_groups(
    render_device: Res<RenderDevice>,
    pipeline: Res<VoxelRendererPipeline>,
    mut commands: Commands,
    mut buffers: ResMut<RenderAssets<GpuShaderStorageBuffer>>,
) {
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
    workitems: [BindGroup; 2],
    work_items: [WorkItems; 2],
    u32_1_buffer: Buffer,
    cubes: Buffer,
}

impl FromWorld for VoxelRendererPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let base_bgl = render_device.create_bind_group_layout(
            "VoxelRendererBase",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, None),
                    uniform_buffer::<NodeData>(false),
                    storage_buffer_sized(false, None),
                ),
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
        let shader = world.load_asset("shader.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![base_bgl.clone(), workitem_bgl.clone(), uniform_bgl.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("expand_voxels"),
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
            size: (1_000_000 * std::mem::size_of::<Cube>()) as u64,
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

        let bind_group_0 = render_device.create_bind_group(
            None,
            &base_bgl,
            &BindGroupEntries::sequential((
                buffer.as_entire_buffer_binding(),
                node_data.as_entire_buffer_binding(),
                cubes.as_entire_buffer_binding(),
            )),
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
            (
                "work_items_0",
                "work_items_0_count",
                "work_items_0_dispatch",
            ),
            (
                "work_items_1",
                "work_items_1_count",
                "work_items_1_dispatch",
            ),
        ]
        .map(|(label, atomic_label, dispatch_label)| WorkItems {
            buffer: render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (std::mem::size_of::<WorkItem>() * 100_000) as u64 + 4,
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
            workitems: [
                workitems_fn(&work_items[0], &work_items[1]),
                workitems_fn(&work_items[1], &work_items[0]),
            ],
            cubes,
            uniform_bind_groups,
            first_work_item,
            work_items,
            u32_1_buffer: render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("u32_1_buffer"),
                contents: bytemuck::cast_slice(&[1_u32]),
                usage: bevy::render::render_resource::BufferUsages::COPY_SRC,
            }),
        }
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
            4,
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

        for i in (0..3) {
            let bind_group = &pipeline.workitems[i % 2];

            command_encoder.clear_buffer(&pipeline.work_items[1 - (i % 2)].buffer, 0, Some(4));
            command_encoder.clear_buffer(
                &pipeline.work_items[1 - (i % 2)].dispatch,
                0,
                Some(std::mem::size_of::<u32>() as u64),
            );

            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_bind_group(0, &pipeline.bind_group, &[]);
            pass.set_bind_group(1, bind_group, &[]);
            pass.set_bind_group(2, &pipeline.uniform_bind_groups[2 - i], &[]);
            pass.set_pipeline(init_pipeline);
            pass.dispatch_workgroups_indirect(&pipeline.work_items[i % 2].dispatch, 0);
        }

        Ok(())
    }
}
