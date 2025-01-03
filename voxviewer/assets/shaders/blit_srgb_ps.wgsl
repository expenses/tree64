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
    var _S1 : pixelOutput_0 = pixelOutput_0( vec4<f32>(psIn_0.Uv_0, 1.0f, 1.0f) );
    return _S1;
}

