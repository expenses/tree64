#include "inverse.slang"

static uint dagData[48] = uint[48](
1u,1u,1u,0u,2u,0u,0u,0u,
1u,1u,1u,256u,1u,256u,256u,0u,
1u,1u,1u,257u,1u,257u,257u,0u,
1u,1u,1u,258u,1u,258u,258u,0u,
1u,1u,1u,259u,1u,259u,259u,0u,
1u,1u,1u,260u,1u,260u,260u,0u
);

uint GET_NODE_FN(uint node, uint childId)
{
	//return dagData[node * 8 + childId];
	return uint(dagData[(node - MaterialCount) * 8u + childId]);
}

#include "../../cubiquity/src/application/commands/view/glsl/pathtracing.h"

uniform struct {
    float4x4 VInv;
    float4x4 PInv;
    float3 cameraPos;
    float3 sun_direction;
    uint2 resolution;
} uniforms;

float3 compute_shading(RayVolumeIntersection intersection) {
    float3[] palette = {float3(1.), float3(1.,0.,0.)};

    float3 base_colour = palette[intersection.material - 1u];

    float lighting = max(dot(uniforms.sun_direction, intersection.normal), 0.);

    return base_colour * lighting;
}

float3 trace(Ray3f ray) {
    SubDAG subDAG;
	subDAG.lowerBound[0] = 0;
	subDAG.lowerBound[1] = 0;
	subDAG.lowerBound[2] = 0;
	subDAG.nodeHeight = 6;
	subDAG.nodeIndex = MaterialCount + 5u;

	RayVolumeIntersection intersection = intersect_subdag(ray, subDAG, true, gMaxFootprint);

    if (intersection.hit) {
        return compute_shading(intersection);
    }

    return float3(abs(sin(ray.mDir.y*10)));
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void main(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    [[vk::image_format("rgba16f")]]
    WTexture2D<float3> image,
) {
    var TexCoords = (float2(dispatch_thread_id.xy) + .5) /  uniforms.resolution;
    TexCoords.y = 1.0 - TexCoords.y;

	Ray3f ray;
	ray.mOrigin = uniforms.cameraPos;
	ray.mDir = createRay(TexCoords, uniforms.PInv, uniforms.VInv);

    image.Store(dispatch_thread_id.xy, trace(ray));
}
