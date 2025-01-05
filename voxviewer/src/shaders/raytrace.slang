import Rendering.Materials.StandardMaterialInstance;
import Rendering.Materials.PBRT.PBRTDiffuseMaterialInstance;
import Rendering.Materials.BSDFs.StandardBSDF;
import Rendering.Materials.BSDFs.SimpleBTDF;
import Rendering.Materials.BSDFs.DisneyDiffuseBRDF;
import Scene.Material.MaterialData;
import Utils.Sampling.TinyUniformSampleGenerator;

import Utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

StructuredBuffer<uint[8]> svo_dag;

uint GET_NODE_FN(uint node, uint childId)
{
	//return dagData[node * 8 + childId];
	return svo_dag[node - MaterialCount][childId];
}

#include "../../cubiquity/src/application/commands/view/glsl/pathtracing.h"

uniform float4x4 VInv;
uniform float4x4 PInv;
uniform float3 cameraPos;
uniform float3 sun_direction;
uniform uint2 resolution;
uniform Settings settings;
uniform uint frame_index;
uniform float cos_sun_aparent_size;
uniform uint accumulated_frame_index;

[Flags]
enum Settings {
    EnableShadows,
    AccumulateSamples
};

[ForceInline]
SubDAG initial_subdag() {
    SubDAG subDAG;
    subDAG.lowerBound = uint3(0);
    subDAG.nodeHeight = 6;
    subDAG.nodeIndex = MaterialCount + 5u;
    return subDAG;
}

ShadingData create_shading_data_from_intersection(RayVolumeIntersection intersection, Ray3f ray) {
    ShadingData shading_data;
    shading_data.frame = ShadingFrame.createIdentity();
    shading_data.frame.N = intersection.normal;
    shading_data.posW = intersection.position;
    shading_data.faceN = shading_data.frame.N;
    shading_data.V = -ray.mDir;
    shading_data.frontFacing = true;
    shading_data.IoR = 1.;
    return shading_data;
}

float3x3 create_rotation_matrix(float3 dir) {
    // Choose an arbitrary vector for T
    var T = float3(1.0, 0.0, 0.0);
    if (abs(dir.x) > 0.99) { // Avoid parallel case
        T = float3(0.0, 1.0, 0.0);
    }

    // Compute U and V
    let U = normalize(cross(T, dir));
    let V = cross(dir, U);

    // Construct the rotation matrix
    return float3x3(U, V, dir);
}

float3 sample_light<S: ISampleGenerator>(inout S sampler) {
    float3 dir = sample_cone(sampleNext2D(sampler), cos_sun_aparent_size);
    return mul(transpose(create_rotation_matrix(sun_direction)), dir);
}

float3 compute_shading<S: ISampleGenerator>(RayVolumeIntersection intersection, Ray3f ray, inout S sampler) {
    SubDAG subDAG = initial_subdag();

    float3[] palette = {float3(1.), float3(1.,0.,0.)};

    float3 base_colour = palette[intersection.material - 1u];

    float lighting = 3.0;

    if (settings & Settings::EnableShadows) {
        Ray3f shadow_ray;
        shadow_ray.mOrigin = intersection.position + intersection.normal * 0.0001;
        shadow_ray.mDir = normalize(sample_light(sampler));
        let shadow_intersection = intersect_subdag(shadow_ray, initial_subdag(), false, gMaxFootprint);
        lighting *= float(!shadow_intersection.hit);
    }

    let shading_data = create_shading_data_from_intersection(intersection, ray);

    StandardBSDFData data;
    data.diffuse = base_colour;
    data.specular = float3(1.0);
    data.roughness = 1.0;
    data.metallic = 0.0;
    data.eta = 1.0;
    data.transmission = 0.;
    data.diffuseTransmission = 0.0;
    data.specularTransmission = 0.0;
    data.volumeScattering = 0.0;
    data.volumeAnsiotropy = 0.0;
    data.hasEntryPointVolumeProperties = false;

    let material = PBRTDiffuseMaterialInstance(
        shading_data.frame,
        {base_colour},
        //float3(.01)
    );

    return material.eval(shading_data, sun_direction, sampler) * lighting;// + material.emission;
}

float3 trace<S: ISampleGenerator>(Ray3f ray, inout S sampler) {
	RayVolumeIntersection intersection = intersect_subdag(ray, initial_subdag(), true, gMaxFootprint);

    if (intersection.hit) {
        return compute_shading(intersection, ray, sampler);
    }

    return float3(abs(sin(ray.mDir.y*10)));
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void main(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    [[vk::image_format("rgba16f")]]
    WTexture2D<float3> current,
    [[vk::image_format("rgba16f")]]
    Texture2D<float3> previous,
    SamplerState sampler,
) {
    var TexCoords = (float2(dispatch_thread_id.xy) + .5) /  resolution;
    var original_tex_coords = TexCoords;
    TexCoords.y = 1.0 - TexCoords.y;

    var rng = TinyUniformSampleGenerator(dispatch_thread_id.xy, frame_index);

	Ray3f ray;
	ray.mOrigin = cameraPos;
	ray.mDir = createRay(TexCoords, PInv, VInv);

    var sample = trace(ray, rng);

    if ((settings & Settings::AccumulateSamples) && (accumulated_frame_index > 0)) {
        float3 accumulated_samples = previous.SampleLevel(sampler, original_tex_coords, 0);
        float3 combined_value = accumulated_samples * (float(accumulated_frame_index) / float(accumulated_frame_index+1)) + sample * (1. / float(accumulated_frame_index+1));
        sample = combined_value;
    }

    current.Store(dispatch_thread_id.xy, sample);
}
