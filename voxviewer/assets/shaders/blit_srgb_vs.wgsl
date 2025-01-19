struct V2P_0
{
    @builtin(position) Pos_0 : vec4<f32>,
    @location(0) Uv_0 : vec2<f32>,
};

@vertex
fn VSMain(@builtin(vertex_index) vId_0 : u32) -> V2P_0
{
    var _S1 : f32 = f32((((vId_0 << (u32(1)))) & (u32(2))));
    var _S2 : f32 = f32((vId_0 & (u32(2))));
    var uv_0 : vec2<f32> = vec2<f32>(_S1, _S2);
    var vsOut_0 : V2P_0;
    vsOut_0.Uv_0 = vec2<f32>(_S1, 1.0f - _S2);
    vsOut_0.Pos_0 = vec4<f32>(vec2<f32>(2.0f) * uv_0 - vec2<f32>(1.0f), 0.0f, 1.0f);
    return vsOut_0;
}

