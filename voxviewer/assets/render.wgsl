/*#import bevy_pbr::{
    view_transformations::position_world_to_clip
}*/


struct Cube {
    pos: vec3<u32>,
    size: u32
}

struct Cubes {
    len: u32,
    data: array<Cube>
}

@group(0) @binding(0) var<storage, read> cubes: Cubes;

@vertex
fn vertex(@builtin(vertex_index) vertex_id: u32) -> @builtin(position) vec4<f32> {
    let cube = cubes.data[vertex_id];
    return vec4<f32>(vec3<f32>(cube.pos), 1.0);
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4(1.0, 0.0, 0.0, 1.0);
}
