struct UniformBuffer {
    half_size_log2: u32,
};

struct NodeData {
    reserved_indices: u32,
};

struct WorkItem {
    pos: vec3<u32>,
    index: u32,
}

struct Cube {
    pos: vec3<u32>,
    size_log2: u32,
    value: u32
}

struct Cubes {
    len: atomic<u32>,
    data: array<Cube>
}

struct InputWorkItems {
    len: u32,
    data: array<WorkItem>
}

struct OutputWorkItems {
    len: atomic<u32>,
    data: array<WorkItem>
}

@group(0) @binding(0) var<storage, read> nodes: array<array<u32, 8>>;
@group(0) @binding(1) var<uniform> node_data: NodeData;

@group(1) @binding(0) var<storage, read_write> cubes: Cubes;
@group(1) @binding(1) var<storage, read_write> draw_indirect_vertex_count: atomic<u32>;

@group(2) @binding(0) var<storage, read> input_work_items: InputWorkItems;
@group(2) @binding(1) var<storage, read_write> output_work_items: OutputWorkItems;
@group(2) @binding(2) var<storage, read_write> output_invocations_count: atomic<u32>;

@group(3) @binding(0) var<uniform> uniforms: UniformBuffer;

fn cull(pos: vec3<u32>) -> bool {
    return false;
}

@compute @workgroup_size(8, 8, 1)
fn expand_voxels(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let work_item_index = invocation_id.x;
    let octant = invocation_id.y;

    if (work_item_index >= input_work_items.len) {
        return;
    }

    let work_item = input_work_items.data[work_item_index];
    let value = nodes[work_item.index][octant];

    if (value == 0) {
        return;
    }

    let half_size = u32(1) << uniforms.half_size_log2;

    let pos = work_item.pos + vec3(
            (octant % 2),
            ((octant/2) % 2),
            ((octant/4) % 2)
        ) * half_size;

    if (cull(pos)) {
        return;
    }

    if (value < node_data.reserved_indices) {
        let index = atomicAdd(&cubes.len, u32(1));
        cubes.data[index].pos=pos;
        cubes.data[index].size_log2=uniforms.half_size_log2;
        cubes.data[index].value = value;
        atomicAdd(&draw_indirect_vertex_count, u32(18));
    } else {
        let index = atomicAdd(&output_work_items.len, u32(1));
        atomicMax(&output_invocations_count, (index / 8) + 1);
        output_work_items.data[index].pos = pos;
        output_work_items.data[index].index = value - node_data.reserved_indices;
    }
}
