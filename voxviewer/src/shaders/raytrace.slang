#include "inverse.slang"

static uint dagData[48] = uint[48](
1u,1u,1u,0u,2u,0u,0u,0u,
1u,1u,1u,256u,1u,256u,256u,0u,
1u,1u,1u,257u,1u,257u,257u,0u,
1u,1u,1u,258u,1u,258u,258u,0u,
1u,1u,1u,259u,1u,259u,259u,0u,
1u,1u,1u,260u,1u,260u,260u,0u
);

[ForceInline]
uint GET_NODE_FN(uint node, uint childId)
{
	//return dagData[node * 8 + childId];
	return uint(dagData[(node - MaterialCount) * 8u + childId]);
}

#include "../cubiquity/src/application/commands/view/glsl/pathtracing.h"

[ForceInline]
float3 compute_shading(RayVolumeIntersection intersection) {
    float3[] palette = {float3(1.), float3(1.,0.,0.)};

    float3 base_colour = palette[intersection.material - 1u];

    float3 sun = normalize(float3(1.,2.,.5));
    float lighting = max(dot(sun, intersection.normal), 0.);

    return base_colour * lighting;
}

[ForceInline]
float3 trace(float4x4 PInv, float4x4 VInv, float3 cameraPos, uint2 resolution, uint2 thread_id) {
    let _VInv = inverse(float4x4(
            -0.767033, -0.486404, 0.418415, 0,
            0.641608, -0.581489, 0.500209, 0,
            -2.16103e-08, 0.652136, 0.758102, 0,
            20.065, 33.2061, -147.342, 1
    ));

    let _cameraPos = float3(93.1922, 80.1369, 90.0454);

    let TexCoords = (float2(thread_id) + .5) /  resolution;

    //VInv = inverse(VInv);

	Ray3f ray;
	ray.mOrigin = cameraPos;
	ray.mDir = createRay(TexCoords, PInv, VInv);

    //return abs(ray.mDir);

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

    return float3(0.);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void main(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    [[vk::image_format("rgba16f")]]
    RWTexture2D<float3> image,
    uniform float4x4 VInv,
    uniform float4x4 PInv,
    uniform float3 cameraPos,
    uniform uint2 resolution
) {
    image[dispatch_thread_id.xy] = trace(VInv, PInv, cameraPos, resolution, dispatch_thread_id.xy);
}
