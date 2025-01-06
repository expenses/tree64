
struct V2P
{
    float4 Pos : SV_Position;
    float2 Uv : COLOR0;
};

uniform uint accumulated_frame_index;

[shader("vertex")]
V2P VSMain(uint vId : SV_VertexID)
{
    float2 uv = float2((vId << 1) & 2, vId & 2);
    V2P vsOut;
    vsOut.Uv = float2(uv.x, 1.0 - uv.y);
    vsOut.Pos = float4((2.0 * uv) - 1.0, 0.0, 1.0);
    return vsOut;
}

[shader("pixel")]
float4 PSMain(
V2P psIn,
    Texture2D<float3> hdr,
    SamplerState sampler,
) : SV_Target0
{
    return float4(hdr.Sample(sampler, psIn.Uv) / float(accumulated_frame_index+1), 1.0);
}
