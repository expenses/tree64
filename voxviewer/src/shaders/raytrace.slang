import Rendering.Materials.BSDFs.DisneyDiffuseBRDF;
import Rendering.Materials.BSDFs.SpecularMicrofacet;
import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.Fresnel;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Sampling.UniformSampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;
import Scene.ShadingData;

#include "materials.slang"
#include "gradients.slang"
#include "../../VoxelRT/src/VoxelRT/Shaders/Backends/Tree64/Map.slang"

struct Ray {
    float3 pos;
    float3 dir;
}

Ray GetPrimaryRay<S: ISampleGenerator>(int2 screenPos, inout S rng) {
    var thread_offset = float2(.5);

    if (settings & Settings::AccumulateSamples) {
        thread_offset = sampleNext2D(rng);
    }

    // nit: UV re-scaling and anti-alias jitter can be pre-baked in the matrix.
    float2 uv = (screenPos + thread_offset) / resolution;
    uv = uv * 2 - 1;
    uv.y = -uv.y;
    float4 far = mul(camera.p_inv, float4(uv, 1, 1));

    float3 rayDirection = normalize(far.xyz / far.w);

    Ray ray;
    ray.pos = camera.pos % 1;
    ray.dir = rayDirection;

    return ray;
}


StructuredBuffer<backend::tree64::Node> tree_nodes;
StructuredBuffer<uint> leaf_data;

HitInfo ray_cast(Ray ray, bool coarse) {
    backend::tree64::VoxelMap voxel_map;
    voxel_map.TreeNodes = tree_nodes;
    voxel_map.LeafData = leaf_data;
    voxel_map.TreeScale = tree.scale;
    voxel_map.RootNodeIndex = tree.root_node_index;

    return voxel_map.RayCast(tree.offset + int3(camera.pos), ray.pos, ray.dir, coarse);
}

struct Camera {
    float4x4 p_inv;
    float4x4 view;
    float4x4 view_inv;
    float3 pos;
    float3 forward;
    float3 up;
    float3 right;

    // https://blog.demofox.org/2018/07/04/pathtraced-depth-of-field-bokeh/
    float4 sensor_plane() {
        float4 sensorPlane;
        sensorPlane.xyz = forward;
        sensorPlane.w = -dot(forward, (pos - forward));
        return sensorPlane;
    }

    float3 calculate_sensor_pos(float3 ray_direction) {
        float4 sensorPlane = sensor_plane();
        float t = -(dot(pos, sensorPlane.xyz) + sensorPlane.w) / dot(ray_direction, sensorPlane.xyz);
        float3 sensorPos = pos + ray_direction * t;

        // convert the sensorPos from world space to camera space
        float3 cameraSpaceSensorPos = mul(float4(sensorPos, 1.0f), view).xyz;

        float DOFFocalLength = 50.0;

        // elongate z by the focal length
        cameraSpaceSensorPos.z *= DOFFocalLength;

        // convert back into world space
        return mul(float4(cameraSpaceSensorPos, 1.0f), view_inv).xyz;
    }


};

uniform Camera camera;

uniform struct {
    float3 direction;
    float3 emission;
    float cosine_apparent_size;
} sun;

uniform struct {
    int3 offset;
    uint scale;
    uint root_node_index;
} tree;

uniform float3 background_colour;
uniform uint2 resolution;
uniform Settings settings;
uniform uint frame_index;
uniform uint accumulated_frame_index;
uniform uint max_bounces;

[Flags]
enum Settings {
    EnableShadows,
    AccumulateSamples,
    ShowHeatmap
};

ShadingData create_shading_data_from_hit(HitInfo hit, float3 ray_dir) {
    ShadingData shading_data;
    shading_data.frame.N = hit.Normal;
    shading_data.frame.T = shading_data.frame.N.yzx;
    shading_data.frame.B = shading_data.frame.N.zxy;
    shading_data.posW = hit.Pos;
    shading_data.faceN = shading_data.frame.N;
    shading_data.V = -ray_dir;
    shading_data.frontFacing = true;
    shading_data.IoR = 1.5;
    return shading_data;
}

struct MaterialInstance {
    float3 emission;
    DisneyDiffuseBRDF diffuse_brdf;
    SpecularMicrofacetBRDF specular_brdf;

    float fresnel;

    __init(Material material, ShadingData sd) {
        emission = material.base_colour * material.emission_factor;

        float3 wiLocal = sd.frame.toLocal(sd.V);
        fresnel = luminance(evalFresnelSchlick(material.f0(), material.f90(), wiLocal.z));

        // Initialize BRDFs
        diffuse_brdf = DisneyDiffuseBRDF(material.diffuse_colour(), material.linear_roughness);
        specular_brdf.albedo = material.base_colour;
        specular_brdf.alpha = material.alpha_roughness();
        specular_brdf.activeLobes = (uint)LobeType::SpecularReflection;
    }

    bool is_emissive() {
        return any(emission > 0.0);
    }

    float3 eval<S: ISampleGenerator>(const ShadingData sd, const float3 wo, float3 light_intensity, inout S sampler) {
        if (is_emissive()) {
            return emission;
        }

        if (all(light_intensity == 0.0)) {
            return 0.0;
        }

        float3 wiLocal = sd.frame.toLocal(sd.V);
        float3 woLocal = sd.frame.toLocal(wo);

        float3 result = 0.0;

        // Blend the BRDF evaluations
        result += (1.0 - fresnel) * diffuse_brdf.eval(wiLocal, woLocal, sampler, BSDFContext());
        result += fresnel * specular_brdf.eval(wiLocal, woLocal, sampler, BSDFContext());

        return result * light_intensity;
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result) {
        result = {};

        if (is_emissive()) {
            return false;
        }

        float3 wiLocal = sd.frame.toLocal(sd.V);
        float3 woLocal = {};

        // Sample selection based on probabilities
        float selection = sampleNext1D(sg);

        bool valid = false;

        if (selection > fresnel) {
            valid = diffuse_brdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg, BSDFContext());
        } else {
            valid = specular_brdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg, BSDFContext());
        }

        result.wo = sd.frame.fromLocal(woLocal);
        return valid;
    }
}

StructuredBuffer<Material> materials;

struct MaterialAndShadingData {
    MaterialInstance material = {};
    ShadingData shading_data;

    float3 get_direct_lighting<S: ISampleGenerator>(inout S sampler) {
        float lighting = float(!shoot_shadow_ray(Ray(shading_data.computeRayOrigin(), sample_light(sampler))));
        return material.eval(shading_data, sun.direction, lighting * sun.emission, sampler);
    }
};

MaterialAndShadingData create_material_from_hit(HitInfo hit, float3 ray_dir) {
    MaterialAndShadingData output;
    output.shading_data = create_shading_data_from_hit(hit, ray_dir);

    let material = materials[hit.MaterialId - 1u];

    output.material = MaterialInstance(
        material, output.shading_data
    );

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
    float3 dir = sample_cone(sampleNext2D(sampler), sun.cosine_apparent_size);
    return mul(create_rotation_matrix(sun.direction), dir);
}

bool shoot_shadow_ray(Ray ray) {
    if (settings & Settings::EnableShadows) {
        return !ray_cast(ray, true).Miss;
    }

    return false;
}

float3 compute_shading<S: ISampleGenerator>(HitInfo hit, float3 ray_dir, inout S sampler) {
    var material = create_material_from_hit(hit, ray_dir);
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

        let ray = Ray(material.shading_data.computeRayOrigin(), sample_result.wo);
        hit = ray_cast(ray, true);
        if (!hit.Miss) {
            material = create_material_from_hit(hit, ray.dir);
            radiance += material.get_direct_lighting(sampler) * throughput;
        } else {
            radiance += background_colour * throughput;
            break;
        }
    }

    return radiance;
}

float3 trace<S: ISampleGenerator>(Ray ray, inout S sampler) {
    HitInfo hit = ray_cast(ray, false);
    if (!hit.Miss) {
        return compute_shading(hit, ray.dir, sampler);
    }

    return background_colour;
}

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

    var sample = trace(GetPrimaryRay(dispatch_thread_id.xy, rng), rng);

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
