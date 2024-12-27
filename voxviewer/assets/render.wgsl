#import bevy_pbr::{
    view_transformations::position_world_to_clip,
    mesh_view_bindings as view_bindings
}

struct Cube {
    pos: vec3<u32>,
    size: u32
}

struct Cubes {
    len: u32,
    data: array<Cube>
}

@group(1) @binding(0) var<storage, read> cubes: Cubes;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var cube = cubes.data[vertex_id / 18];

    let center = vec3<f32>(cube.pos + cube.size / 2);
    let direction = view_bindings::view.world_position.xyz - center;

    let x = u32(direction.x > 0.0);
    let y = u32(direction.y > 0.0);
    let z = u32(direction.z > 0.0);

    let triangles = array<vec3<u32>, 18>(
        // x
        vec3(x,0,0),
        vec3(x,1,0),
        vec3(x,0,1),
        vec3(x,1,0),
        vec3(x,0,1),
        vec3(x,1,1),
        // y
        vec3(0,y,0),
        vec3(1,y,0),
        vec3(0,y,1),
        vec3(1,y,0),
        vec3(0,y,1),
        vec3(1,y,1),
        // z
        vec3(0,0,z),
        vec3(1,0,z),
        vec3(0,1,z),
        vec3(1,0,z),
        vec3(0,1,z),
        vec3(1,1,z),
    );

    let normals = array<vec3<f32>, 3>(
        vec3(select(-1.0, 1.0, direction.x > 0.0), 0.0, 0.0),
        vec3(0.0, select(-1.0, 1.0, direction.y > 0.0), 0.0),
        vec3(0.0, 0.0, select(-1.0, 1.0, direction.z > 0.0)),
    );

    var out:VertexOutput;
    out.clip_position = position_world_to_clip(vec3<f32>(cube.pos + triangles[vertex_id % 18] * cube.size));
    out.normal = normals[(vertex_id/6)%3];

    return out;
}

@fragment
fn fragment(@location(0) normal: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4(normal, 1.0);
}
