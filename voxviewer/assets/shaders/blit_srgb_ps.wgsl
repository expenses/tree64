@binding(0) @group(0) var entryPointParams_hdr_0 : texture_2d<f32>;

@binding(1) @group(0) var entryPointParams_sampler_0 : sampler;

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct V2P_0
{
    @builtin(position) Pos_0 : vec4<f32>,
    @location(0) Uv_0 : vec2<f32>,
};

@fragment
fn PSMain( psIn_0 : V2P_0) -> pixelOutput_0
{
    var _S1 : pixelOutput_0 = pixelOutput_0( vec4<f32>((textureSample((entryPointParams_hdr_0), (entryPointParams_sampler_0), (psIn_0.Uv_0)).xyz), 1.0f) );
    return _S1;
}
