import Rendering.Materials.StandardMaterialInstance;
import Rendering.Materials.PBRT.PBRTDiffuseMaterialInstance;
import Rendering.Materials.BSDFs.StandardBSDF;
import Rendering.Materials.BSDFs.SimpleBTDF;
import Rendering.Materials.BSDFs.DisneyDiffuseBRDF;
import Scene.Material.MaterialData;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Sampling.UniformSampleGenerator;

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
uniform float3 sun_colour;
uniform float3 background_colour;
uniform uint2 resolution;
uniform Settings settings;
uniform uint frame_index;
uniform float cos_sun_aparent_size;
uniform uint accumulated_frame_index;
uniform uint max_bounces;
uniform uint node_height;
uniform uint root_node_index;

[Flags]
enum Settings {
    EnableShadows,
    AccumulateSamples,
    ShowHeatmap
};

SubDAG initial_subdag() {
    SubDAG subDAG;
    subDAG.lowerBound = uint3(0);
    subDAG.nodeHeight = node_height;
    subDAG.nodeIndex = MaterialCount + root_node_index;
    return subDAG;
}

ShadingData create_shading_data_from_intersection(RayVolumeIntersection intersection, Ray3f ray) {
    ShadingData shading_data;
    shading_data.frame.N = intersection.normal;
    shading_data.frame.T = shading_data.frame.N.yzx;
    shading_data.frame.B = shading_data.frame.N.zxy;
    shading_data.posW = intersection.position;
    shading_data.faceN = shading_data.frame.N;
    shading_data.V = -ray.mDir;
    shading_data.frontFacing = true;
    shading_data.IoR = 1.;
    return shading_data;
}

struct Material {
    float3 base_colour;
    float emission_factor;
};

StructuredBuffer<Material> materials;

struct MaterialAndShadingData {
    PBRTDiffuseMaterialInstance material = {};
    ShadingData shading_data;
    float3 emission = 0;

    float3 get_direct_lighting<S: ISampleGenerator>(inout S sampler) {
        float lighting = float(!shoot_shadow_ray(Ray3f(shading_data.computeRayOrigin(), sample_light(sampler))));
        return material.eval(shading_data, sun_direction, sampler) * lighting * sun_colour + emission;
    }
};

MaterialAndShadingData create_material_from_intersection(RayVolumeIntersection intersection, Ray3f ray) {
    MaterialAndShadingData output;
    output.shading_data = create_shading_data_from_intersection(intersection, ray);

    let material = materials[intersection.material - 1u];

    output.material = PBRTDiffuseMaterialInstance(
        output.shading_data.frame,
        {material.base_colour},
    );
    output.emission = material.base_colour * material.emission_factor;

    return output;
}

float3x3 create_rotation_matrix(dynamic_uniform float3 dir) {
    // Choose an arbitrary vector for T
    var T = float3(1.0, 0.0, 0.0);
    if (abs(dir.x) > 0.99) { // Avoid parallel case
        T = float3(0.0, 1.0, 0.0);
    }

    // Compute U and V
    let U = normalize(cross(T, dir));
    let V = cross(dir, U);

    // Construct the rotation matrix
    return transpose(float3x3(U, V, dir));
}

float3 sample_light<S: ISampleGenerator>(inout S sampler) {
    float3 dir = sample_cone(sampleNext2D(sampler), cos_sun_aparent_size);
    return mul(create_rotation_matrix(sun_direction), dir);
}

bool shoot_shadow_ray(Ray3f ray) {
    if (settings & Settings::EnableShadows) {
        return intersect_subdag(ray, initial_subdag(), false, gMaxFootprint).hit;
    }

    return false;
}

float3 compute_shading<S: ISampleGenerator>(RayVolumeIntersection intersection, Ray3f ray, inout S sampler) {
    var material = create_material_from_intersection(intersection, ray);
    var radiance = material.get_direct_lighting(sampler);
    var throughput = float3(1.0);

    for (uint i = 0; i < max_bounces; i++) {
        BSDFSample sample_result;
        if (!material.material.sample(material.shading_data, sampler, sample_result)) {
            break;
        }

        throughput *= sample_result.weight;

        if (all(throughput < 0.01)) {
            break;
        }

        let ray = Ray3f(material.shading_data.computeRayOrigin(), sample_result.wo);
        intersection = intersect_subdag(ray, initial_subdag(), true, gMaxFootprint);
        if (intersection.hit) {
            material = create_material_from_intersection(intersection, ray);
            radiance += material.get_direct_lighting(sampler) * throughput;
        } else {
            radiance += background_colour * throughput;
            break;
        }
    }

    return radiance;
}

float3 trace<S: ISampleGenerator>(Ray3f ray, inout S sampler) {
    RayVolumeIntersection intersection = intersect_subdag(ray, initial_subdag(), true, gMaxFootprint);
    if (intersection.hit) {
        return compute_shading(intersection, ray, sampler);
    }

    return background_colour;
}

#include "gradients.slang"

uint get_clock_portable() {
    __target_switch
    {
        case spirv:
            return getRealtimeClockLow();
        default:
            return 0;
    }
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void main(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    [[vk::image_format("rgba32f")]]
    WTexture2D<float3> current,
    [[vk::image_format("rgba32f")]]
    Texture2D<float3> previous,
    SamplerState sampler,
) {
    let start = get_clock_portable();

    if (any(dispatch_thread_id.xy >= resolution)) {
        return;
    }

    let original_tex_coords = (float2(dispatch_thread_id.xy) + .5) /  resolution;

    var rng = TinyUniformSampleGenerator(dispatch_thread_id.xy, frame_index);

    var thread_offset = float2(.5);

    if (settings & Settings::AccumulateSamples) {
        thread_offset = sampleNext2D(rng);
    }

    var TexCoords = (float2(dispatch_thread_id.xy) + thread_offset) /  resolution;
    TexCoords.y = 1.0 - TexCoords.y;


    Ray3f ray;
    ray.mOrigin = cameraPos;
    ray.mDir = createRay(TexCoords, PInv, VInv);

    var sample = trace(ray, rng);


    if ((settings & Settings::AccumulateSamples) && (accumulated_frame_index > 0)) {
        float3 past_samples = previous.SampleLevel(sampler, original_tex_coords, 0);
        sample += past_samples;
    }

    current.Store(dispatch_thread_id.xy, sample);

    if (settings & Settings::ShowHeatmap) {
        let end = get_clock_portable();

        current.Store(dispatch_thread_id.xy, viridis(float(end.x - start.x) / 50000.0));
    }
}
