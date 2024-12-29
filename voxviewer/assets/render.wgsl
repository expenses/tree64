#import bevy_pbr::{
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions,
    pbr_functions::alpha_discard
}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::FragmentOutput,
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::FragmentOutput,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif


struct Cube {
    pos: vec3<u32>,
    size: u32,
    value: u32
}

struct Cubes {
    len: u32,
    data: array<Cube>
}

@group(1) @binding(0) var<storage, read> cubes: Cubes;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) colour: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};


const TRIANGLE_UVS = array<vec2<u32>, 6>(
    vec2(0,0),
    vec2(1,0),
    vec2(0,1),
    vec2(1,0),
    vec2(0,1),
    vec2(1,1)
);

@vertex
fn vertex(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var cube = cubes.data[vertex_id / 18];

    let center = vec3<f32>(cube.pos + cube.size / 2);
    let direction = view.world_position.xyz - center;

    let uv = TRIANGLE_UVS[vertex_id % 6];
    let index_in_cube = vertex_id % 18;
    var triangle = vec3<u32>(0);

    if index_in_cube < 6 {
        let x = u32(direction.x > 0.0);
        triangle = vec3(x, uv);
    } else if index_in_cube < 12 {
        let y = u32(direction.y > 0.0);
        triangle = vec3(uv.x, y, uv.y);
    } else {
        let z = u32(direction.z > 0.0);
        triangle = vec3(uv, z);
    }

    let normals = array<vec3<f32>, 3>(
        vec3(select(-1.0, 1.0, direction.x > 0.0), 0.0, 0.0),
        vec3(0.0, select(-1.0, 1.0, direction.y > 0.0), 0.0),
        vec3(0.0, 0.0, select(-1.0, 1.0, direction.z > 0.0)),
    );

    let world_pos = vec3<f32>(cube.pos + triangle * cube.size);

    var out:VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    out.normal = normals[index_in_cube / 6];
    out.colour = select(vec3(1.0), vec3(1.0, 0.0, 0.0), cube.value == 2);
    out.world_position = world_pos;

    return out;
}

@fragment
fn fragment(input: VertexOutput) -> FragmentOutput {
    var pbr_input: PbrInput = pbr_input_new();
    pbr_input.material.base_color = vec4(input.colour, 1.0);
    pbr_input.material.perceptual_roughness = 1.0;
    pbr_input.world_position = vec4(input.world_position, 1.0);
    pbr_input.frag_coord = input.clip_position;
    pbr_input.world_normal = input.normal;
    pbr_input.N = pbr_input.world_normal;
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.V = pbr_functions::calculate_view(pbr_input.world_position, pbr_input.is_orthographic);

    // alpha discard
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    // in deferred mode we can't modify anything after that, as lighting is run in a separate fullscreen shader.
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // apply lighting
    out.color = apply_pbr_lighting(pbr_input);

    // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
    // note this does not include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    // we can optionally modify the final result here
    out.color = out.color * 2.0;
#endif

    return out;
}
