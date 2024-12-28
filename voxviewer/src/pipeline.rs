use bevy::pbr::{MeshPipeline, MeshPipelineViewLayoutKey};
use bevy::prelude::*;
use bevy::render::{
    render_resource::{binding_types::*, *},
    renderer::RenderDevice,
};
use std::borrow::Cow;
use std::sync::atomic::AtomicU32;

use bevy::core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;

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
pub struct WorkItem {
    pos: UVec3,
    index: u32,
}

pub struct WorkItems {
    pub buffer: Buffer,
    pub dispatch: Buffer,
}

#[derive(Resource)]
pub struct RenderingPipeline {
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
pub struct VoxelRendererPipeline {
    pub base_bgl: BindGroupLayout,
    pub uniform_bgl: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
    pub bind_group: BindGroup,
    pub nodes: Buffer,
    pub uniform_bind_groups: Vec<BindGroup>,
    pub first_work_item: Buffer,
    pub flip_flop_bind_groups: [BindGroup; 2],
    pub work_items: [WorkItems; 2],
    pub u32_1_buffer: Buffer,
    pub cubes: Buffer,
    pub draw_indirect: Buffer,
    pub draw_bind_group: BindGroup,
    pub draw_bgl: BindGroupLayout,
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
