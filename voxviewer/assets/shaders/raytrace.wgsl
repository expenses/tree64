struct backend_tree64_Node_std430_0
{
    @align(4) PackedData_0 : array<u32, i32(3)>,
};

@binding(1) @group(0) var<storage, read> tree_nodes_0 : array<backend_tree64_Node_std430_0>;

@binding(2) @group(0) var<storage, read> leaf_data_0 : array<u32>;

struct Material_std430_0
{
    @align(16) base_colour_0 : vec3<f32>,
    @align(4) emission_factor_0 : f32,
    @align(16) linear_roughness_0 : f32,
    @align(4) metallic_0 : f32,
};

@binding(3) @group(0) var<storage, read> materials_0 : array<Material_std430_0>;

@binding(4) @group(0) var entryPointParams_current_0 : texture_storage_2d<rgba32float, write>;

@binding(5) @group(0) var entryPointParams_previous_0 : texture_2d<f32>;

@binding(6) @group(0) var entryPointParams_sampler_0 : sampler;

struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct GlobalParams_std140_0
{
    @align(16) PInv_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) cameraPos_0 : vec3<f32>,
    @align(16) sun_direction_0 : vec3<f32>,
    @align(16) sun_colour_0 : vec3<f32>,
    @align(16) background_colour_0 : vec3<f32>,
    @align(16) root_offset_0 : vec3<i32>,
    @align(16) resolution_0 : vec2<u32>,
    @align(8) settings_0 : i32,
    @align(4) frame_index_0 : u32,
    @align(16) cos_sun_aparent_size_0 : f32,
    @align(4) accumulated_frame_index_0 : u32,
    @align(8) max_bounces_0 : u32,
    @align(4) tree_scale_0 : u32,
    @align(16) root_node_index_0 : u32,
};

@binding(0) @group(0) var<uniform> globalParams_0 : GlobalParams_std140_0;
fn unpackStorage_0( _S1 : _MatrixStorage_float4x4_ColMajorstd140_0) -> mat4x4<f32>
{
    return mat4x4<f32>(_S1.data_0[i32(0)][i32(0)], _S1.data_0[i32(1)][i32(0)], _S1.data_0[i32(2)][i32(0)], _S1.data_0[i32(3)][i32(0)], _S1.data_0[i32(0)][i32(1)], _S1.data_0[i32(1)][i32(1)], _S1.data_0[i32(2)][i32(1)], _S1.data_0[i32(3)][i32(1)], _S1.data_0[i32(0)][i32(2)], _S1.data_0[i32(1)][i32(2)], _S1.data_0[i32(2)][i32(2)], _S1.data_0[i32(3)][i32(2)], _S1.data_0[i32(0)][i32(3)], _S1.data_0[i32(1)][i32(3)], _S1.data_0[i32(2)][i32(3)], _S1.data_0[i32(3)][i32(3)]);
}

struct LCG_0
{
    @align(4) state_0 : u32,
};

struct TinyUniformSampleGenerator_0
{
    @align(4) rng_0 : LCG_0,
};

fn interleave_32bit_0( v_0 : vec2<u32>) -> u32
{
    var x_0 : u32 = ((v_0.x) & (u32(65535)));
    var x_1 : u32 = (((x_0 | (((x_0 << (u32(8))))))) & (u32(16711935)));
    var x_2 : u32 = (((x_1 | (((x_1 << (u32(4))))))) & (u32(252645135)));
    var x_3 : u32 = (((x_2 | (((x_2 << (u32(2))))))) & (u32(858993459)));
    var y_0 : u32 = ((v_0.y) & (u32(65535)));
    var y_1 : u32 = (((y_0 | (((y_0 << (u32(8))))))) & (u32(16711935)));
    var y_2 : u32 = (((y_1 | (((y_1 << (u32(4))))))) & (u32(252645135)));
    var y_3 : u32 = (((y_2 | (((y_2 << (u32(2))))))) & (u32(858993459)));
    return (((((x_3 | (((x_3 << (u32(1))))))) & (u32(1431655765)))) | (((((((y_3 | (((y_3 << (u32(1))))))) & (u32(1431655765)))) << (u32(1))))));
}

fn blockCipherTEA_0( v0_0 : u32,  v1_0 : u32,  iterations_0 : u32) -> vec2<u32>
{
    var i_0 : u32 = u32(0);
    var sum_0 : u32 = u32(0);
    var _S2 : u32 = v1_0;
    var _S3 : u32 = v0_0;
    for(;;)
    {
        if(i_0 < iterations_0)
        {
        }
        else
        {
            break;
        }
        var sum_1 : u32 = sum_0 + u32(2654435769);
        var _S4 : u32 = _S3 + (((((((_S2 << (u32(4)))) + u32(2738958700)) ^ ((_S2 + sum_1)))) ^ ((((_S2 >> (u32(5)))) + u32(3355524772)))));
        var _S5 : u32 = _S2 + (((((((_S4 << (u32(4)))) + u32(2911926141)) ^ ((_S4 + sum_1)))) ^ ((((_S4 >> (u32(5)))) + u32(2123724318)))));
        i_0 = i_0 + u32(1);
        sum_0 = sum_1;
        _S2 = _S5;
        _S3 = _S4;
    }
    return vec2<u32>(_S3, _S2);
}

fn createLCG_0( s0_0 : u32) -> LCG_0
{
    var rng_1 : LCG_0;
    rng_1.state_0 = s0_0;
    return rng_1;
}

fn TinyUniformSampleGenerator_x24init_0( pixel_0 : vec2<u32>,  sampleNumber_0 : u32) -> TinyUniformSampleGenerator_0
{
    var _S6 : TinyUniformSampleGenerator_0;
    _S6.rng_0 = createLCG_0(blockCipherTEA_0(interleave_32bit_0(pixel_0), sampleNumber_0, u32(16)).x);
    return _S6;
}

fn nextRandom_0( rng_2 : ptr<function, LCG_0>) -> u32
{
    var _S7 : u32 = u32(1664525) * (*rng_2).state_0 + u32(1013904223);
    (*rng_2).state_0 = _S7;
    return _S7;
}

fn TinyUniformSampleGenerator_next_0( this_0 : ptr<function, TinyUniformSampleGenerator_0>) -> u32
{
    var _S8 : LCG_0 = (*this_0).rng_0;
    var _S9 : u32 = nextRandom_0(&(_S8));
    (*this_0).rng_0 = _S8;
    return _S9;
}

fn sampleNext1D_0( sg_0 : ptr<function, TinyUniformSampleGenerator_0>) -> f32
{
    var bits_0 : u32 = TinyUniformSampleGenerator_next_0(&((*sg_0)));
    return f32((bits_0 >> (u32(8)))) * 5.9604644775390625e-08f;
}

fn sampleNext2D_0( sg_1 : ptr<function, TinyUniformSampleGenerator_0>) -> vec2<f32>
{
    var sample_0 : vec2<f32>;
    var _S10 : f32 = sampleNext1D_0(&((*sg_1)));
    sample_0[i32(0)] = _S10;
    var _S11 : f32 = sampleNext1D_0(&((*sg_1)));
    sample_0[i32(1)] = _S11;
    return sample_0;
}

fn GetPrimaryRay_0( screenPos_0 : vec2<i32>,  rng_3 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S12 : vec2<f32> = vec2<f32>(0.5f);
    var thread_offset_0 : vec2<f32>;
    if((((globalParams_0.settings_0) & (i32(2)))) != i32(0))
    {
        var _S13 : vec2<f32> = sampleNext2D_0(&((*rng_3)));
        thread_offset_0 = _S13;
    }
    else
    {
        thread_offset_0 = _S12;
    }
    var _S14 : vec2<f32> = (vec2<f32>(screenPos_0) + thread_offset_0) / vec2<f32>(globalParams_0.resolution_0) * vec2<f32>(2.0f) - vec2<f32>(1.0f);
    var uv_0 : vec2<f32> = _S14;
    uv_0[i32(1)] = - _S14.y;
    var far_0 : vec4<f32> = (((vec4<f32>(uv_0, 1.0f, 1.0f)) * (unpackStorage_0(globalParams_0.PInv_0))));
    return normalize(far_0.xyz / vec3<f32>(far_0.w));
}

struct SpecularMicrofacetBRDF_0
{
     albedo_0 : vec3<f32>,
     alpha_0 : f32,
     activeLobes_0 : u32,
};

fn SpecularMicrofacetBRDF_hasLobe_0( this_1 : SpecularMicrofacetBRDF_0,  lobeType_0 : i32) -> bool
{
    return (((this_1.activeLobes_0) & (u32(lobeType_0)))) != u32(0);
}

fn evalFresnelSchlick_0( f0_0 : vec3<f32>,  f90_0 : vec3<f32>,  cosTheta_0 : f32) -> vec3<f32>
{
    return f0_0 + (f90_0 - f0_0) * vec3<f32>(pow(max(1.0f - cosTheta_0, 0.0f), 5.0f));
}

fn SampleVndf_Hemisphere_0( u_0 : vec2<f32>,  Vh_0 : vec3<f32>) -> vec3<f32>
{
    var phi_0 : f32 = 6.28318548202514648f * u_0.x;
    var _S15 : f32 = Vh_0.z;
    var z_0 : f32 = fma(1.0f - u_0.y, 1.0f + _S15, - _S15);
    var sinTheta_0 : f32 = sqrt(saturate(1.0f - z_0 * z_0));
    return vec3<f32>(sinTheta_0 * cos(phi_0), sinTheta_0 * sin(phi_0), z_0) + Vh_0;
}

fn evalG1GGX_0( alphaSqr_0 : f32,  cosTheta_1 : f32) -> f32
{
    if(cosTheta_1 <= 0.0f)
    {
        return 0.0f;
    }
    var cosThetaSqr_0 : f32 = cosTheta_1 * cosTheta_1;
    return 2.0f / (1.0f + sqrt(1.0f + alphaSqr_0 * (max(1.0f - cosThetaSqr_0, 0.0f) / cosThetaSqr_0)));
}

fn evalNdfGGX_0( alpha_1 : f32,  cosTheta_2 : f32) -> f32
{
    var a2_0 : f32 = alpha_1 * alpha_1;
    var d_0 : f32 = (cosTheta_2 * a2_0 - cosTheta_2) * cosTheta_2 + 1.0f;
    return a2_0 / (d_0 * d_0 * 3.14159274101257324f);
}

fn evalPdfGGX_VNDF_0( alpha_2 : f32,  wi_0 : vec3<f32>,  h_0 : vec3<f32>) -> f32
{
    var _S16 : f32 = wi_0.z;
    return evalG1GGX_0(alpha_2 * alpha_2, _S16) * evalNdfGGX_0(alpha_2, h_0.z) * max(0.0f, dot(wi_0, h_0)) / _S16;
}

fn sampleGGX_VNDF_0( alpha_3 : f32,  wi_1 : vec3<f32>,  u_1 : vec2<f32>,  pdf_0 : ptr<function, f32>) -> vec3<f32>
{
    var Nh_0 : vec3<f32> = SampleVndf_Hemisphere_0(u_1, normalize(vec3<f32>(alpha_3 * wi_1.x, alpha_3 * wi_1.y, wi_1.z)));
    var h_1 : vec3<f32> = normalize(vec3<f32>(alpha_3 * Nh_0.x, alpha_3 * Nh_0.y, Nh_0.z));
    (*pdf_0) = evalPdfGGX_VNDF_0(alpha_3, wi_1, h_1);
    return h_1;
}

fn evalLambdaGGX_0( alphaSqr_1 : f32,  cosTheta_3 : f32) -> f32
{
    if(cosTheta_3 <= 0.0f)
    {
        return 0.0f;
    }
    var cosThetaSqr_1 : f32 = cosTheta_3 * cosTheta_3;
    return 0.5f * (-1.0f + sqrt(1.0f + alphaSqr_1 * (max(1.0f - cosThetaSqr_1, 0.0f) / cosThetaSqr_1)));
}

fn evalMaskingSmithGGXCorrelated_0( alpha_4 : f32,  cosThetaI_0 : f32,  cosThetaO_0 : f32) -> f32
{
    var alphaSqr_2 : f32 = alpha_4 * alpha_4;
    return 1.0f / (1.0f + evalLambdaGGX_0(alphaSqr_2, cosThetaI_0) + evalLambdaGGX_0(alphaSqr_2, cosThetaO_0));
}

struct BSDFContext_0
{
     iorI_0 : f32,
     iorT_0 : f32,
     inited_0 : bool,
};

fn SpecularMicrofacetBRDF_sample_0( this_2 : SpecularMicrofacetBRDF_0,  wi_2 : vec3<f32>,  wo_0 : ptr<function, vec3<f32>>,  pdf_1 : ptr<function, f32>,  weight_0 : ptr<function, vec3<f32>>,  lobeType_1 : ptr<function, u32>,  sg_2 : ptr<function, TinyUniformSampleGenerator_0>,  bc_0 : BSDFContext_0) -> bool
{
    const _S17 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    (*wo_0) = _S17;
    (*weight_0) = _S17;
    (*pdf_1) = 0.0f;
    (*lobeType_1) = u32(2);
    var _S18 : f32 = wi_2.z;
    if(_S18 < 9.99999997475242708e-07f)
    {
        return false;
    }
    if((this_2.alpha_0) == 0.0f)
    {
        if(!SpecularMicrofacetBRDF_hasLobe_0(this_2, i32(4)))
        {
            return false;
        }
        (*wo_0) = vec3<f32>(- wi_2.x, - wi_2.y, _S18);
        (*pdf_1) = 0.0f;
        (*weight_0) = evalFresnelSchlick_0(this_2.albedo_0, vec3<f32>(1.0f), _S18);
        (*lobeType_1) = u32(4);
        return true;
    }
    if(!SpecularMicrofacetBRDF_hasLobe_0(this_2, i32(2)))
    {
        return false;
    }
    var _S19 : vec2<f32> = sampleNext2D_0(&((*sg_2)));
    var h_2 : vec3<f32> = sampleGGX_VNDF_0(this_2.alpha_0, wi_2, _S19, &((*pdf_1)));
    var wiDotH_0 : f32 = dot(wi_2, h_2);
    var _S20 : vec3<f32> = vec3<f32>((2.0f * wiDotH_0)) * h_2 - wi_2;
    (*wo_0) = _S20;
    if((_S20.z) < 9.99999997475242708e-07f)
    {
        return false;
    }
    var GOverG1wo_0 : f32 = evalMaskingSmithGGXCorrelated_0(this_2.alpha_0, _S18, (*wo_0).z) * (1.0f + evalLambdaGGX_0(this_2.alpha_0 * this_2.alpha_0, _S18));
    var F_0 : vec3<f32> = evalFresnelSchlick_0(this_2.albedo_0, vec3<f32>(1.0f), wiDotH_0);
    (*pdf_1) = (*pdf_1) / (4.0f * wiDotH_0);
    (*weight_0) = F_0 * vec3<f32>(GOverG1wo_0);
    (*lobeType_1) = u32(2);
    return true;
}

fn sample_disk_concentric_0( u_2 : vec2<f32>) -> vec2<f32>
{
    var _S21 : vec2<f32> = vec2<f32>(2.0f) * u_2 - vec2<f32>(1.0f);
    var _S22 : f32 = _S21.x;
    var _S23 : bool;
    if(_S22 == 0.0f)
    {
        _S23 = (_S21.y) == 0.0f;
    }
    else
    {
        _S23 = false;
    }
    if(_S23)
    {
        return _S21;
    }
    var _S24 : f32 = _S21.y;
    var r_0 : f32;
    var phi_1 : f32;
    if((abs(_S22)) > (abs(_S24)))
    {
        var _S25 : f32 = _S24 / _S22 * 0.78539818525314331f;
        r_0 = _S22;
        phi_1 = _S25;
    }
    else
    {
        var _S26 : f32 = 1.57079637050628662f - _S22 / _S24 * 0.78539818525314331f;
        r_0 = _S24;
        phi_1 = _S26;
    }
    return vec2<f32>(r_0) * vec2<f32>(cos(phi_1), sin(phi_1));
}

fn sample_cosine_hemisphere_concentric_0( u_3 : vec2<f32>,  pdf_2 : ptr<function, f32>) -> vec3<f32>
{
    var d_1 : vec2<f32> = sample_disk_concentric_0(u_3);
    var z_1 : f32 = sqrt(max(0.0f, 1.0f - dot(d_1, d_1)));
    (*pdf_2) = z_1 * 0.31830987334251404f;
    return vec3<f32>(d_1, z_1);
}

fn evalFresnelSchlick_1( f0_1 : f32,  f90_1 : f32,  cosTheta_4 : f32) -> f32
{
    return f0_1 + (f90_1 - f0_1) * pow(max(1.0f - cosTheta_4, 0.0f), 5.0f);
}

struct DisneyDiffuseBRDF_0
{
     albedo_1 : vec3<f32>,
     roughness_0 : f32,
};

fn DisneyDiffuseBRDF_evalWeight_0( this_3 : DisneyDiffuseBRDF_0,  wi_3 : vec3<f32>,  wo_1 : vec3<f32>) -> vec3<f32>
{
    var woDotH_0 : f32 = dot(wo_1, normalize(wi_3 + wo_1));
    var fd90_0 : f32 = 0.5f + 2.0f * woDotH_0 * woDotH_0 * this_3.roughness_0;
    return this_3.albedo_1 * vec3<f32>(evalFresnelSchlick_1(1.0f, fd90_0, wi_3.z)) * vec3<f32>(evalFresnelSchlick_1(1.0f, fd90_0, wo_1.z));
}

fn DisneyDiffuseBRDF_sample_0( this_4 : DisneyDiffuseBRDF_0,  wi_4 : vec3<f32>,  wo_2 : ptr<function, vec3<f32>>,  pdf_3 : ptr<function, f32>,  weight_1 : ptr<function, vec3<f32>>,  lobeType_2 : ptr<function, u32>,  sg_3 : ptr<function, TinyUniformSampleGenerator_0>,  bc_1 : BSDFContext_0) -> bool
{
    var _S27 : vec2<f32> = sampleNext2D_0(&((*sg_3)));
    var _S28 : vec3<f32> = sample_cosine_hemisphere_concentric_0(_S27, &((*pdf_3)));
    (*wo_2) = _S28;
    (*lobeType_2) = u32(1);
    if((min(wi_4.z, (*wo_2).z)) < 9.99999997475242708e-07f)
    {
        (*weight_1) = vec3<f32>(0.0f, 0.0f, 0.0f);
        return false;
    }
    (*weight_1) = DisneyDiffuseBRDF_evalWeight_0(this_4, wi_4, (*wo_2));
    return true;
}

struct MaterialInstance_0
{
     emission_0 : vec3<f32>,
     diffuse_brdf_0 : DisneyDiffuseBRDF_0,
     specular_brdf_0 : SpecularMicrofacetBRDF_0,
     fresnel_0 : f32,
};

fn MaterialInstance_is_emissive_0( this_5 : MaterialInstance_0) -> bool
{
    return any((this_5.emission_0) > vec3<f32>(0.0f));
}

struct ShadingFrame_0
{
     T_0 : vec3<f32>,
     B_0 : vec3<f32>,
     N_0 : vec3<f32>,
};

fn ShadingFrame_toLocal_0( this_6 : ShadingFrame_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(v_1, this_6.T_0), dot(v_1, this_6.B_0), dot(v_1, this_6.N_0));
}

fn BSDFContext_x24init_0() -> BSDFContext_0
{
    var _S29 : BSDFContext_0;
    _S29.iorI_0 = 1.0f;
    _S29.iorT_0 = 1.0f;
    _S29.inited_0 = false;
    return _S29;
}

fn ShadingFrame_fromLocal_0( this_7 : ShadingFrame_0,  v_2 : vec3<f32>) -> vec3<f32>
{
    return this_7.T_0 * vec3<f32>(v_2.x) + this_7.B_0 * vec3<f32>(v_2.y) + this_7.N_0 * vec3<f32>(v_2.z);
}

struct MaterialHeader_0
{
     packedData_0 : vec4<u32>,
};

struct ShadingData_0
{
     posW_0 : vec3<f32>,
     V_0 : vec3<f32>,
     uv_1 : vec2<f32>,
     frame_0 : ShadingFrame_0,
     faceN_0 : vec3<f32>,
     tangentW_0 : vec4<f32>,
     frontFacing_0 : bool,
     curveRadius_0 : f32,
     mtl_0 : MaterialHeader_0,
     materialID_0 : u32,
     IoR_0 : f32,
     materialGradOffset_0 : u32,
     geometryGradOffset_0 : u32,
     threadID_0 : u32,
};

struct BSDFSample_0
{
     wo_3 : vec3<f32>,
     pdf_4 : f32,
     weight_2 : vec3<f32>,
     lobeType_3 : u32,
};

fn MaterialInstance_sample_0( this_8 : MaterialInstance_0,  sd_0 : ShadingData_0,  sg_4 : ptr<function, TinyUniformSampleGenerator_0>,  result_0 : ptr<function, BSDFSample_0>) -> bool
{
    const _S30 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    (*result_0).wo_3 = _S30;
    (*result_0).pdf_4 = 0.0f;
    (*result_0).weight_2 = _S30;
    (*result_0).lobeType_3 = u32(0);
    if(MaterialInstance_is_emissive_0(this_8))
    {
        return false;
    }
    var wiLocal_0 : vec3<f32> = ShadingFrame_toLocal_0(sd_0.frame_0, sd_0.V_0);
    var woLocal_0 : vec3<f32> = _S30;
    var selection_0 : f32 = sampleNext1D_0(&((*sg_4)));
    var valid_0 : bool;
    if(selection_0 > (this_8.fresnel_0))
    {
        var _S31 : BSDFContext_0 = BSDFContext_x24init_0();
        var _S32 : f32 = (*result_0).pdf_4;
        var _S33 : vec3<f32> = (*result_0).weight_2;
        var _S34 : u32 = (*result_0).lobeType_3;
        var _S35 : bool = DisneyDiffuseBRDF_sample_0(this_8.diffuse_brdf_0, wiLocal_0, &(woLocal_0), &(_S32), &(_S33), &(_S34), &((*sg_4)), _S31);
        (*result_0).pdf_4 = _S32;
        (*result_0).weight_2 = _S33;
        (*result_0).lobeType_3 = _S34;
        valid_0 = _S35;
    }
    else
    {
        var _S36 : BSDFContext_0 = BSDFContext_x24init_0();
        var _S37 : f32 = (*result_0).pdf_4;
        var _S38 : vec3<f32> = (*result_0).weight_2;
        var _S39 : u32 = (*result_0).lobeType_3;
        var _S40 : bool = SpecularMicrofacetBRDF_sample_0(this_8.specular_brdf_0, wiLocal_0, &(woLocal_0), &(_S37), &(_S38), &(_S39), &((*sg_4)), _S36);
        (*result_0).pdf_4 = _S37;
        (*result_0).weight_2 = _S38;
        (*result_0).lobeType_3 = _S39;
        valid_0 = _S40;
    }
    (*result_0).wo_3 = ShadingFrame_fromLocal_0(sd_0.frame_0, woLocal_0);
    return valid_0;
}

fn SpecularMicrofacetBRDF_eval_0( this_9 : SpecularMicrofacetBRDF_0,  wi_5 : vec3<f32>,  wo_4 : vec3<f32>,  sg_5 : ptr<function, TinyUniformSampleGenerator_0>,  bc_2 : BSDFContext_0) -> vec3<f32>
{
    var _S41 : f32 = wi_5.z;
    var _S42 : f32 = wo_4.z;
    if((min(_S41, _S42)) < 9.99999997475242708e-07f)
    {
        return vec3<f32>(0.0f);
    }
    if((this_9.alpha_0) == 0.0f)
    {
        return vec3<f32>(0.0f);
    }
    if(!SpecularMicrofacetBRDF_hasLobe_0(this_9, i32(2)))
    {
        return vec3<f32>(0.0f);
    }
    var h_3 : vec3<f32> = normalize(wi_5 + wo_4);
    return evalFresnelSchlick_0(this_9.albedo_0, vec3<f32>(1.0f), dot(wi_5, h_3)) * vec3<f32>(evalNdfGGX_0(this_9.alpha_0, h_3.z)) * vec3<f32>(evalMaskingSmithGGXCorrelated_0(this_9.alpha_0, _S41, _S42)) * vec3<f32>(0.25f) / vec3<f32>(_S41);
}

fn DisneyDiffuseBRDF_eval_0( this_10 : DisneyDiffuseBRDF_0,  wi_6 : vec3<f32>,  wo_5 : vec3<f32>,  sg_6 : ptr<function, TinyUniformSampleGenerator_0>,  bc_3 : BSDFContext_0) -> vec3<f32>
{
    var _S43 : f32 = wo_5.z;
    if((min(wi_6.z, _S43)) < 9.99999997475242708e-07f)
    {
        return vec3<f32>(0.0f);
    }
    return DisneyDiffuseBRDF_evalWeight_0(this_10, wi_6, wo_5) * vec3<f32>(0.31830987334251404f) * vec3<f32>(_S43);
}

fn MaterialInstance_eval_0( this_11 : MaterialInstance_0,  sd_1 : ShadingData_0,  wo_6 : vec3<f32>,  light_intensity_0 : vec3<f32>,  sampler_0 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    if(MaterialInstance_is_emissive_0(this_11))
    {
        return this_11.emission_0;
    }
    if(all(light_intensity_0 == vec3<f32>(0.0f)))
    {
        return vec3<f32>(0.0f);
    }
    var wiLocal_1 : vec3<f32> = ShadingFrame_toLocal_0(sd_1.frame_0, sd_1.V_0);
    var woLocal_1 : vec3<f32> = ShadingFrame_toLocal_0(sd_1.frame_0, wo_6);
    var _S44 : f32 = 1.0f - this_11.fresnel_0;
    var _S45 : BSDFContext_0 = BSDFContext_x24init_0();
    var _S46 : vec3<f32> = DisneyDiffuseBRDF_eval_0(this_11.diffuse_brdf_0, wiLocal_1, woLocal_1, &((*sampler_0)), _S45);
    var _S47 : vec3<f32> = vec3<f32>(_S44) * _S46;
    var _S48 : vec3<f32> = SpecularMicrofacetBRDF_eval_0(this_11.specular_brdf_0, wiLocal_1, woLocal_1, &((*sampler_0)), _S45);
    return (_S47 + vec3<f32>(this_11.fresnel_0) * _S48) * light_intensity_0;
}

fn sample_cone_0( u_4 : vec2<f32>,  cosTheta_5 : f32) -> vec3<f32>
{
    var z_2 : f32 = u_4.x * (1.0f - cosTheta_5) + cosTheta_5;
    var r_1 : f32 = sqrt(1.0f - z_2 * z_2);
    var phi_2 : f32 = 6.28318548202514648f * u_4.y;
    return vec3<f32>(r_1 * cos(phi_2), r_1 * sin(phi_2), z_2);
}

fn create_rotation_matrix_0( dir_0 : vec3<f32>) -> mat3x3<f32>
{
    const _S49 : vec3<f32> = vec3<f32>(1.0f, 0.0f, 0.0f);
    var T_1 : vec3<f32>;
    if((abs(dir_0.x)) > 0.99000000953674316f)
    {
        T_1 = vec3<f32>(0.0f, 1.0f, 0.0f);
    }
    else
    {
        T_1 = _S49;
    }
    var _S50 : vec3<f32> = normalize(cross(T_1, dir_0));
    return transpose(mat3x3<f32>(_S50, cross(dir_0, _S50), dir_0));
}

fn sample_light_0( sampler_1 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S51 : vec2<f32> = sampleNext2D_0(&((*sampler_1)));
    return (((sample_cone_0(_S51, globalParams_0.cos_sun_aparent_size_0)) * (create_rotation_matrix_0(globalParams_0.sun_direction_0))));
}

fn computeRayOrigin_0( pos_0 : vec3<f32>,  normal_0 : vec3<f32>) -> vec3<f32>
{
    var iOff_0 : vec3<i32> = vec3<i32>(normal_0 * vec3<f32>(768.0f));
    return select((bitcast<vec3<f32>>(((bitcast<vec3<i32>>((pos_0))) + select(iOff_0, - iOff_0, pos_0 < vec3<f32>(0.0f))))), pos_0 + normal_0 * vec3<f32>(0.0000457763671875f), (abs(pos_0)) < vec3<f32>(0.0625f));
}

fn ShadingData_computeRayOrigin_0( this_12 : ShadingData_0,  viewside_0 : bool) -> vec3<f32>
{
    var _S52 : vec3<f32>;
    if((this_12.frontFacing_0) == viewside_0)
    {
        _S52 = this_12.faceN_0;
    }
    else
    {
        _S52 = - this_12.faceN_0;
    }
    return computeRayOrigin_0(this_12.posW_0, _S52);
}

struct backend_tree64_Node_0
{
     PackedData_0 : array<u32, i32(3)>,
};

fn unpackStorage_1( _S53 : backend_tree64_Node_std430_0) -> backend_tree64_Node_0
{
    var _S54 : backend_tree64_Node_0 = backend_tree64_Node_0( _S53.PackedData_0 );
    return _S54;
}

fn backend_tree64_VoxelMap_GetMirroredPos_0( pos_1 : vec3<f32>,  dir_1 : vec3<f32>,  rangeCheck_0 : bool) -> vec3<f32>
{
    var _S55 : vec3<f32> = (bitcast<vec3<f32>>((((bitcast<vec3<u32>>((pos_1))) ^ (vec3<u32>(u32(8388607)))))));
    var _S56 : bool;
    if(rangeCheck_0)
    {
        _S56 = any(((pos_1 < vec3<f32>(1.0f)) | ((pos_1 >= vec3<f32>(2.0f)))));
    }
    else
    {
        _S56 = false;
    }
    var mirrored_0 : vec3<f32>;
    if(_S56)
    {
        mirrored_0 = vec3<f32>(3.0f) - pos_1;
    }
    else
    {
        mirrored_0 = _S55;
    }
    return select(pos_1, mirrored_0, dir_1 > vec3<f32>(0.0f));
}

fn backend_tree64_Node_IsLeaf_get_0( this_13 : backend_tree64_Node_0) -> bool
{
    return (((this_13.PackedData_0[i32(0)]) & (u32(1)))) != u32(0);
}

fn backend_tree64_VoxelMap_GetNodeCellIndex_0( pos_2 : vec3<f32>,  scale_exp_0 : i32) -> i32
{
    var cellPos_0 : vec3<u32> = ((((bitcast<vec3<u32>>((pos_2))) >> (vec3<u32>(u32(scale_exp_0))))) & (vec3<u32>(u32(3))));
    return i32(cellPos_0.x + cellPos_0.z * u32(4) + cellPos_0.y * u32(16));
}

fn backend_tree64_Node_pop_mask_shifted_0( this_14 : backend_tree64_Node_0,  shift_0 : u32) -> u32
{
    if(shift_0 >= u32(32))
    {
        return ((this_14.PackedData_0[i32(2)]) >> ((shift_0 - u32(32))));
    }
    return ((this_14.PackedData_0[i32(1)]) >> (shift_0));
}

fn backend_tree64_Node_ChildPtr_get_0( this_15 : backend_tree64_Node_0) -> u32
{
    return ((this_15.PackedData_0[i32(0)]) >> (u32(1)));
}

fn backend_tree64_Node_PopMask_get_0( this_16 : backend_tree64_Node_0) -> vec2<u32>
{
    return vec2<u32>(this_16.PackedData_0[i32(2)], this_16.PackedData_0[i32(1)]);
}

fn popcnt_var64_0( mask_0 : vec2<u32>,  width_0 : u32) -> u32
{
    var himask_0 : u32;
    var count_0 : u32;
    if(width_0 >= u32(32))
    {
        var _S57 : u32 = countOneBits(mask_0[i32(1)]);
        himask_0 = mask_0[i32(0)];
        count_0 = _S57;
    }
    else
    {
        himask_0 = mask_0[i32(1)];
        count_0 = u32(0);
    }
    return count_0 + countOneBits((himask_0 & ((((u32(1) << (((width_0 & (u32(31))))))) - u32(1)))));
}

fn backend_tree64_VoxelMap_FloorScale_0( pos_3 : vec3<f32>,  scale_exp_1 : i32) -> vec3<f32>
{
    return (bitcast<vec3<f32>>((((bitcast<vec3<u32>>((pos_3))) & (vec3<u32>(((u32(4294967295) << (u32(scale_exp_1))))))))));
}

struct HitInfo_0
{
     Dist_0 : f32,
     Pos_0 : vec3<f32>,
     Normal_0 : vec3<f32>,
     FaceUV_0 : vec2<f32>,
     MaterialId_0 : u32,
};

struct Ray3f_0
{
     mOrigin_0 : vec3<f32>,
     mDir_0 : vec3<f32>,
};

struct backend_tree64_VoxelMap_0
{
     TreeScale_0 : u32,
     RootNodeIndex_0 : u32,
};

fn backend_tree64_VoxelMap_load_leaf_data_0( _S58 : backend_tree64_VoxelMap_0,  _S59 : u32) -> u32
{
    return unpack4xU8(u32(leaf_data_0[_S59 / u32(4)]))[_S59 % u32(4)];
}

fn backend_tree64_VoxelMap_Traversal_OctMirror_0( _S60 : backend_tree64_VoxelMap_0,  _S61 : vec3<f32>,  _S62 : vec3<f32>,  _S63 : bool) -> HitInfo_0
{
    var _S64 : bool;
    var stack_0 : array<u32, i32(11)>;
    var _S65 : backend_tree64_Node_0 = unpackStorage_1(tree_nodes_0[_S60.RootNodeIndex_0]);
    var _S66 : vec3<f32> = abs(_S62);
    var _S67 : vec3<f32> = vec3<f32>(1.0f) / - _S66;
    var mirrorMask_0 : u32;
    if((_S62.x) > 0.0f)
    {
        mirrorMask_0 = u32(3);
    }
    else
    {
        mirrorMask_0 = u32(0);
    }
    if((_S62.y) > 0.0f)
    {
        mirrorMask_0 = (mirrorMask_0 | (u32(48)));
    }
    if((_S62.z) > 0.0f)
    {
        mirrorMask_0 = (mirrorMask_0 | (u32(12)));
    }
    var _S68 : bool;
    var _S69 : vec3<f32> = backend_tree64_VoxelMap_GetMirroredPos_0(_S61, _S62, true);
    var _S70 : vec3<f32> = clamp(_S69, vec3<f32>(1.0f), vec3<f32>(1.99999988079071045f));
    var _S71 : vec3<i32> = vec3<i32>(i32(-1));
    var sideDist_0 : vec3<f32>;
    var childIdx_0 : i32;
    var node_0 : backend_tree64_Node_0 = _S65;
    var pos_4 : vec3<f32> = _S70;
    var nodeIdx_0 : u32 = _S60.RootNodeIndex_0;
    var i_1 : i32 = i32(0);
    var scaleExp_0 : i32 = i32(21);
    for(;;)
    {
        if(i_1 < i32(256))
        {
        }
        else
        {
            break;
        }
        if(_S63)
        {
            _S68 = i_1 > i32(20);
        }
        else
        {
            _S68 = false;
        }
        var _S72 : bool;
        if(_S68)
        {
            _S72 = backend_tree64_Node_IsLeaf_get_0(node_0);
        }
        else
        {
            _S72 = false;
        }
        if(_S72)
        {
            break;
        }
        var _S73 : i32 = i32((u32(backend_tree64_VoxelMap_GetNodeCellIndex_0(pos_4, scaleExp_0)) ^ (mirrorMask_0)));
        var node_1 : backend_tree64_Node_0 = node_0;
        var childIdx_1 : i32 = _S73;
        var nodeIdx_1 : u32 = nodeIdx_0;
        var scaleExp_1 : i32 = scaleExp_0;
        for(;;)
        {
            var _S74 : u32 = u32(childIdx_1);
            var _S75 : bool = (((backend_tree64_Node_pop_mask_shifted_0(node_1, _S74)) & (u32(1)))) != u32(0);
            _S64 = _S75;
            var _S76 : bool;
            if(_S75)
            {
                _S76 = !backend_tree64_Node_IsLeaf_get_0(node_1);
            }
            else
            {
                _S76 = false;
            }
            if(_S76)
            {
            }
            else
            {
                break;
            }
            stack_0[(scaleExp_1 >> (u32(1)))] = nodeIdx_1;
            var nodeIdx_2 : u32 = backend_tree64_Node_ChildPtr_get_0(node_1) + popcnt_var64_0(backend_tree64_Node_PopMask_get_0(node_1), _S74);
            var scaleExp_2 : i32 = scaleExp_1 - i32(2);
            var _S77 : i32 = i32((u32(backend_tree64_VoxelMap_GetNodeCellIndex_0(pos_4, scaleExp_2)) ^ (mirrorMask_0)));
            node_1 = unpackStorage_1(tree_nodes_0[nodeIdx_2]);
            childIdx_1 = _S77;
            nodeIdx_1 = nodeIdx_2;
            scaleExp_1 = scaleExp_2;
        }
        var _S78 : bool;
        if(_S64)
        {
            _S78 = backend_tree64_Node_IsLeaf_get_0(node_1);
        }
        else
        {
            _S78 = false;
        }
        if(_S78)
        {
            node_0 = node_1;
            childIdx_0 = childIdx_1;
            scaleExp_0 = scaleExp_1;
            break;
        }
        var advScaleExp_0 : i32;
        if((((backend_tree64_Node_pop_mask_shifted_0(node_1, u32((childIdx_1 & (i32(42)))))) & (u32(3342387)))) == u32(0))
        {
            advScaleExp_0 = scaleExp_1 + i32(1);
        }
        else
        {
            advScaleExp_0 = scaleExp_1;
        }
        var edgePos_0 : vec3<f32> = backend_tree64_VoxelMap_FloorScale_0(pos_4, advScaleExp_0);
        var sideDist_1 : vec3<f32> = (edgePos_0 - _S69) * _S67;
        var _S79 : vec3<f32> = vec3<f32>(min(min(sideDist_1.x, sideDist_1.y), sideDist_1.z));
        var pos_5 : vec3<f32> = min(_S69 - _S66 * _S79, (bitcast<vec3<f32>>((vec3<i32>(bitcast<i32>(edgePos_0[u32(0)]), bitcast<i32>(edgePos_0[u32(1)]), bitcast<i32>(edgePos_0[u32(2)])) + select(vec3<i32>((((i32(1) << (u32(advScaleExp_0)))) - i32(1))), _S71, sideDist_1 == _S79)))));
        var diffPos_0 : vec3<u32> = ((bitcast<vec3<u32>>((pos_5))) ^ ((bitcast<vec3<u32>>((edgePos_0)))));
        var diffExp_0 : i32 = i32(firstLeadingBit(((((diffPos_0.x) | ((diffPos_0.y)))) | ((diffPos_0.z)))));
        var diffExp_1 : i32;
        if((diffExp_0 % i32(2)) == i32(0))
        {
            diffExp_1 = diffExp_0 - i32(1);
        }
        else
        {
            diffExp_1 = diffExp_0;
        }
        if(diffExp_1 > scaleExp_1)
        {
            if(diffExp_1 > i32(21))
            {
                node_0 = node_1;
                pos_4 = pos_5;
                childIdx_0 = childIdx_1;
                sideDist_0 = sideDist_1;
                scaleExp_0 = diffExp_1;
                break;
            }
            node_0 = unpackStorage_1(tree_nodes_0[stack_0[(diffExp_1 >> (u32(1)))]]);
            nodeIdx_0 = stack_0[(diffExp_1 >> (u32(1)))];
            scaleExp_0 = diffExp_1;
        }
        else
        {
            node_0 = node_1;
            nodeIdx_0 = nodeIdx_1;
            scaleExp_0 = scaleExp_1;
        }
        var _S80 : i32 = i_1 + i32(1);
        pos_4 = pos_5;
        childIdx_0 = childIdx_1;
        sideDist_0 = sideDist_1;
        i_1 = _S80;
    }
    var hit_0 : HitInfo_0;
    hit_0.MaterialId_0 = u32(0);
    if(backend_tree64_Node_IsLeaf_get_0(node_0))
    {
        _S68 = scaleExp_0 <= i32(21);
    }
    else
    {
        _S68 = false;
    }
    if(_S68)
    {
        var pos_6 : vec3<f32> = backend_tree64_VoxelMap_GetMirroredPos_0(pos_4, _S62, false);
        hit_0.MaterialId_0 = backend_tree64_VoxelMap_load_leaf_data_0(_S60, backend_tree64_Node_ChildPtr_get_0(node_0) + popcnt_var64_0(backend_tree64_Node_PopMask_get_0(node_0), u32(childIdx_0)));
        hit_0.Pos_0 = pos_6;
        hit_0.Normal_0 = select(vec3<f32>(0.0f), vec3<f32>(- sign(_S62)), vec3<f32>((min(min(sideDist_0.x, sideDist_0.y), sideDist_0.z))) >= sideDist_0);
    }
    return hit_0;
}

fn backend_tree64_VoxelMap_RayCast_0( _S81 : backend_tree64_VoxelMap_0,  _S82 : vec3<i32>,  _S83 : vec3<f32>,  _S84 : vec3<f32>,  _S85 : bool) -> HitInfo_0
{
    var _S86 : f32 = f32((i32(1) << ((_S81.TreeScale_0))));
    var _S87 : vec3<f32> = vec3<f32>((1.0f / _S86));
    var _S88 : vec3<f32> = vec3<f32>(_S82) * _S87;
    var _S89 : vec3<f32> = vec3<f32>(1.0f);
    var hit_1 : HitInfo_0 = backend_tree64_VoxelMap_Traversal_OctMirror_0(_S81, _S88 + _S83 * _S87 + _S89, _S84, _S85);
    if((hit_1.MaterialId_0) != u32(0))
    {
        hit_1.Pos_0 = (hit_1.Pos_0 - _S89 - _S88) * vec3<f32>(_S86);
    }
    return hit_1;
}

fn ray_cast_0( ray_0 : Ray3f_0,  coarse_0 : bool) -> HitInfo_0
{
    var voxel_map_0 : backend_tree64_VoxelMap_0;
    voxel_map_0.TreeScale_0 = globalParams_0.tree_scale_0;
    voxel_map_0.RootNodeIndex_0 = globalParams_0.root_node_index_0;
    return backend_tree64_VoxelMap_RayCast_0(voxel_map_0, globalParams_0.root_offset_0 + vec3<i32>(globalParams_0.cameraPos_0), ray_0.mOrigin_0, ray_0.mDir_0, coarse_0);
}

fn HitInfo_Miss_get_0( this_17 : HitInfo_0) -> bool
{
    return (this_17.MaterialId_0) == u32(0);
}

fn shoot_shadow_ray_0( ray_1 : Ray3f_0) -> bool
{
    if((((globalParams_0.settings_0) & (i32(1)))) != i32(0))
    {
        return !HitInfo_Miss_get_0(ray_cast_0(ray_1, true));
    }
    return false;
}

struct MaterialAndShadingData_0
{
     material_0 : MaterialInstance_0,
     shading_data_0 : ShadingData_0,
};

fn MaterialAndShadingData_get_direct_lighting_0( this_18 : MaterialAndShadingData_0,  sampler_2 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S90 : vec3<f32> = ShadingData_computeRayOrigin_0(this_18.shading_data_0, true);
    var _S91 : vec3<f32> = sample_light_0(&((*sampler_2)));
    var _S92 : Ray3f_0 = Ray3f_0( _S90, _S91 );
    var _S93 : vec3<f32> = MaterialInstance_eval_0(this_18.material_0, this_18.shading_data_0, globalParams_0.sun_direction_0, vec3<f32>(f32(!shoot_shadow_ray_0(_S92))) * globalParams_0.sun_colour_0, &((*sampler_2)));
    return _S93;
}

fn MaterialHeader_x24init_0() -> MaterialHeader_0
{
    var _S94 : MaterialHeader_0;
    _S94.packedData_0 = vec4<u32>(u32(0), u32(0), u32(0), u32(0));
    return _S94;
}

fn ShadingData_x24init_0() -> ShadingData_0
{
    var _S95 : ShadingData_0;
    _S95.mtl_0 = MaterialHeader_x24init_0();
    return _S95;
}

fn MaterialAndShadingData_x24init_0() -> MaterialAndShadingData_0
{
    var _S96 : MaterialAndShadingData_0;
    const _S97 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var _S98 : DisneyDiffuseBRDF_0 = DisneyDiffuseBRDF_0( _S97, 0.0f );
    var _S99 : SpecularMicrofacetBRDF_0 = SpecularMicrofacetBRDF_0( _S97, 0.0f, u32(0) );
    _S96.material_0.emission_0 = _S97;
    _S96.material_0.diffuse_brdf_0 = _S98;
    _S96.material_0.specular_brdf_0 = _S99;
    _S96.material_0.fresnel_0 = 0.0f;
    _S96.shading_data_0 = ShadingData_x24init_0();
    return _S96;
}

fn create_shading_data_from_hit_0( hit_2 : HitInfo_0,  ray_dir_0 : vec3<f32>) -> ShadingData_0
{
    var shading_data_1 : ShadingData_0 = ShadingData_x24init_0();
    shading_data_1.frame_0.N_0 = hit_2.Normal_0;
    shading_data_1.frame_0.T_0 = hit_2.Normal_0.yzx;
    shading_data_1.frame_0.B_0 = shading_data_1.frame_0.N_0.zxy;
    shading_data_1.posW_0 = hit_2.Pos_0;
    shading_data_1.faceN_0 = shading_data_1.frame_0.N_0;
    shading_data_1.V_0 = - ray_dir_0;
    shading_data_1.frontFacing_0 = true;
    shading_data_1.IoR_0 = 1.5f;
    return shading_data_1;
}

struct Material_0
{
     base_colour_0 : vec3<f32>,
     emission_factor_0 : f32,
     linear_roughness_0 : f32,
     metallic_0 : f32,
};

fn unpackStorage_2( _S100 : Material_std430_0) -> Material_0
{
    var _S101 : Material_0 = Material_0( _S100.base_colour_0, _S100.emission_factor_0, _S100.linear_roughness_0, _S100.metallic_0 );
    return _S101;
}

fn Material_ior_get_0( this_19 : Material_0) -> f32
{
    return 1.5f;
}

fn Material_f0_from_ior_0( this_20 : Material_0) -> f32
{
    var _S102 : f32 = Material_ior_get_0(this_20);
    var _S103 : f32 = (1.0f - _S102) / (1.0f + _S102);
    return _S103 * _S103;
}

fn Material_specular_colour_get_0( this_21 : Material_0) -> vec3<f32>
{
    return vec3<f32>(1.0f);
}

fn Material_specular_factor_get_0( this_22 : Material_0) -> f32
{
    return 1.0f;
}

fn Material_dielectic_f0_0( this_23 : Material_0) -> vec3<f32>
{
    return vec3<f32>(Material_f0_from_ior_0(this_23)) * Material_specular_colour_get_0(this_23) * vec3<f32>(Material_specular_factor_get_0(this_23));
}

fn Material_f0_0( this_24 : Material_0) -> vec3<f32>
{
    return mix(Material_dielectic_f0_0(this_24), this_24.base_colour_0, vec3<f32>(this_24.metallic_0));
}

fn Material_f90_0( this_25 : Material_0) -> f32
{
    return mix(Material_specular_factor_get_0(this_25), 1.0f, this_25.metallic_0);
}

fn luminance_0( rgb_0 : vec3<f32>) -> f32
{
    return dot(rgb_0, vec3<f32>(0.2125999927520752f, 0.71520000696182251f, 0.07220000028610229f));
}

fn Material_diffuse_colour_0( this_26 : Material_0) -> vec3<f32>
{
    return mix(this_26.base_colour_0, vec3<f32>(0.0f), vec3<f32>(this_26.metallic_0));
}

fn Material_alpha_roughness_0( this_27 : Material_0) -> f32
{
    var _S104 : f32 = this_27.linear_roughness_0;
    return max(_S104 * _S104, 9.99999997475242708e-07f);
}

fn MaterialInstance_x24init_0( material_1 : Material_0,  sd_2 : ShadingData_0) -> MaterialInstance_0
{
    var _S105 : MaterialInstance_0;
    _S105.emission_0 = material_1.base_colour_0 * vec3<f32>(material_1.emission_factor_0);
    _S105.fresnel_0 = luminance_0(evalFresnelSchlick_0(Material_f0_0(material_1), vec3<f32>(Material_f90_0(material_1)), ShadingFrame_toLocal_0(sd_2.frame_0, sd_2.V_0).z));
    _S105.diffuse_brdf_0.albedo_1 = Material_diffuse_colour_0(material_1);
    _S105.diffuse_brdf_0.roughness_0 = material_1.linear_roughness_0;
    _S105.specular_brdf_0.albedo_0 = material_1.base_colour_0;
    _S105.specular_brdf_0.alpha_0 = Material_alpha_roughness_0(material_1);
    _S105.specular_brdf_0.activeLobes_0 = u32(2);
    return _S105;
}

fn create_material_from_hit_0( hit_3 : HitInfo_0,  ray_dir_1 : vec3<f32>) -> MaterialAndShadingData_0
{
    var output_0 : MaterialAndShadingData_0 = MaterialAndShadingData_x24init_0();
    var _S106 : ShadingData_0 = create_shading_data_from_hit_0(hit_3, ray_dir_1);
    output_0.shading_data_0 = _S106;
    output_0.material_0 = MaterialInstance_x24init_0(unpackStorage_2(materials_0[hit_3.MaterialId_0 - u32(1)]), _S106);
    return output_0;
}

fn compute_shading_0( hit_4 : HitInfo_0,  ray_dir_2 : vec3<f32>,  sampler_3 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var material_2 : MaterialAndShadingData_0 = create_material_from_hit_0(hit_4, ray_dir_2);
    var _S107 : vec3<f32> = MaterialAndShadingData_get_direct_lighting_0(material_2, &((*sampler_3)));
    var _S108 : vec3<f32> = vec3<f32>(1.0f);
    var material_3 : MaterialAndShadingData_0 = material_2;
    var i_2 : u32 = u32(0);
    var throughput_0 : vec3<f32> = _S108;
    var radiance_0 : vec3<f32> = _S107;
    var _S109 : vec3<f32> = vec3<f32>(0.00999999977648258f);
    for(;;)
    {
        if(i_2 < (globalParams_0.max_bounces_0))
        {
        }
        else
        {
            break;
        }
        var _S110 : MaterialAndShadingData_0 = material_3;
        var sample_result_0 : BSDFSample_0;
        var _S111 : bool = MaterialInstance_sample_0(material_3.material_0, material_3.shading_data_0, &((*sampler_3)), &(sample_result_0));
        if(!_S111)
        {
            break;
        }
        var throughput_1 : vec3<f32> = throughput_0 * sample_result_0.weight_2;
        if(all(throughput_1 < _S109))
        {
            break;
        }
        var _S112 : vec3<f32>;
        var _S113 : MaterialAndShadingData_0;
        var _S114 : vec3<f32> = sample_result_0.wo_3;
        var _S115 : Ray3f_0 = Ray3f_0( ShadingData_computeRayOrigin_0(_S110.shading_data_0, true), sample_result_0.wo_3 );
        var _S116 : HitInfo_0 = ray_cast_0(_S115, true);
        if(!HitInfo_Miss_get_0(_S116))
        {
            var material_4 : MaterialAndShadingData_0 = create_material_from_hit_0(_S116, _S114);
            _S113 = material_4;
            var _S117 : vec3<f32> = MaterialAndShadingData_get_direct_lighting_0(material_4, &((*sampler_3)));
            _S112 = radiance_0 + _S117 * throughput_1;
        }
        else
        {
            radiance_0 = radiance_0 + globalParams_0.background_colour_0 * throughput_1;
            break;
        }
        var _S118 : u32 = i_2 + u32(1);
        material_3 = _S113;
        i_2 = _S118;
        throughput_0 = throughput_1;
        radiance_0 = _S112;
    }
    return radiance_0;
}

fn trace_0( direction_0 : vec3<f32>,  sampler_4 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S119 : Ray3f_0 = Ray3f_0( globalParams_0.cameraPos_0 % vec3<f32>(1.0f), direction_0 );
    var hit_5 : HitInfo_0 = ray_cast_0(_S119, false);
    if(!HitInfo_Miss_get_0(hit_5))
    {
        var _S120 : vec3<f32> = compute_shading_0(hit_5, direction_0, &((*sampler_4)));
        return _S120;
    }
    return globalParams_0.background_colour_0;
}

fn viridis_0( t_0 : f32) -> vec3<f32>
{
    var _S121 : vec3<f32> = vec3<f32>(t_0);
    return vec3<f32>(0.27772733569145203f, 0.00540734454989433f, 0.33409979939460754f) + _S121 * (vec3<f32>(0.10509303957223892f, 1.40461349487304688f, 1.38459014892578125f) + _S121 * (vec3<f32>(-0.33086183667182922f, 0.21484756469726562f, 0.09509516507387161f) + _S121 * (vec3<f32>(-4.63423061370849609f, -5.79910087585449219f, -19.33244132995605469f) + _S121 * (vec3<f32>(6.22827005386352539f, 14.17993354797363281f, 56.6905517578125f) + _S121 * (vec3<f32>(4.77638483047485352f, -13.74514579772949219f, -65.35303497314453125f) + _S121 * vec3<f32>(-5.4354557991027832f, 4.64585256576538086f, 26.31243515014648438f))))));
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatch_thread_id_0 : vec3<u32>)
{
    var _S122 : vec2<u32> = dispatch_thread_id_0.xy;
    if(any(_S122 >= (globalParams_0.resolution_0)))
    {
        return;
    }
    var _S123 : vec2<f32> = (vec2<f32>(_S122) + vec2<f32>(0.5f)) / vec2<f32>(globalParams_0.resolution_0);
    var rng_4 : TinyUniformSampleGenerator_0 = TinyUniformSampleGenerator_x24init_0(_S122, globalParams_0.frame_index_0);
    if((((globalParams_0.settings_0) & (i32(2)))) != i32(0))
    {
        var _S124 : vec2<f32> = sampleNext2D_0(&(rng_4));
    }
    var _S125 : vec3<f32> = GetPrimaryRay_0(vec2<i32>(_S122), &(rng_4));
    var sample_1 : vec3<f32> = trace_0(_S125, &(rng_4));
    var sample_2 : vec3<f32>;
    if(((((globalParams_0.settings_0) & (i32(2)))) != i32(0)) && ((globalParams_0.accumulated_frame_index_0) > u32(0)))
    {
        sample_2 = sample_1 + (textureSampleLevel((entryPointParams_previous_0), (entryPointParams_sampler_0), (_S123), (0.0f)).xyz);
    }
    else
    {
        sample_2 = sample_1;
    }
    textureStore((entryPointParams_current_0), (_S122), vec4<f32>((sample_2), 1));
    if((((globalParams_0.settings_0) & (i32(4)))) != i32(0))
    {
        textureStore((entryPointParams_current_0), (_S122), vec4<f32>((viridis_0(0.0f)), 1));
    }
    return;
}

struct Xoshiro128StarStar_0
{
    @align(4) state_1 : array<u32, i32(4)>,
};

struct UniformSampleGenerator_0
{
    @align(4) rng_5 : Xoshiro128StarStar_0,
};

