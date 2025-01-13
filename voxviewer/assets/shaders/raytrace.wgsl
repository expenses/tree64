@binding(1) @group(0) var<storage, read> svo_dag_0 : array<array<u32, i32(8)>>;

struct Material_std430_0
{
    @align(16) base_colour_0 : vec3<f32>,
    @align(4) emission_factor_0 : f32,
    @align(16) linear_roughness_0 : f32,
    @align(4) metallic_0 : f32,
};

@binding(2) @group(0) var<storage, read> materials_0 : array<Material_std430_0>;

@binding(3) @group(0) var entryPointParams_current_0 : texture_storage_2d<rgba32float, write>;

@binding(4) @group(0) var entryPointParams_previous_0 : texture_2d<f32>;

@binding(5) @group(0) var entryPointParams_sampler_0 : sampler;

struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct GlobalParams_std140_0
{
    @align(16) VInv_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) PInv_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) cameraPos_0 : vec3<f32>,
    @align(16) sun_direction_0 : vec3<f32>,
    @align(16) sun_colour_0 : vec3<f32>,
    @align(16) background_colour_0 : vec3<f32>,
    @align(16) resolution_0 : vec2<u32>,
    @align(8) settings_0 : i32,
    @align(4) frame_index_0 : u32,
    @align(16) cos_sun_aparent_size_0 : f32,
    @align(4) accumulated_frame_index_0 : u32,
    @align(8) max_bounces_0 : u32,
    @align(4) node_height_0 : u32,
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
    var x_0 : u32 = (v_0.x & (u32(65535)));
    var x_1 : u32 = (((x_0 | ((x_0 << (u32(8)))))) & (u32(16711935)));
    var x_2 : u32 = (((x_1 | ((x_1 << (u32(4)))))) & (u32(252645135)));
    var x_3 : u32 = (((x_2 | ((x_2 << (u32(2)))))) & (u32(858993459)));
    var y_0 : u32 = (v_0.y & (u32(65535)));
    var y_1 : u32 = (((y_0 | ((y_0 << (u32(8)))))) & (u32(16711935)));
    var y_2 : u32 = (((y_1 | ((y_1 << (u32(4)))))) & (u32(252645135)));
    var y_3 : u32 = (((y_2 | ((y_2 << (u32(2)))))) & (u32(858993459)));
    return (((((x_3 | ((x_3 << (u32(1)))))) & (u32(1431655765)))) | ((((((y_3 | ((y_3 << (u32(1)))))) & (u32(1431655765)))) << (u32(1)))));
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
        var _S4 : u32 = _S3 + ((((((_S2 << (u32(4)))) + u32(2738958700)) ^ ((_S2 + sum_1))) ^ ((((_S2 >> (u32(5)))) + u32(3355524772)))));
        var _S5 : u32 = _S2 + ((((((_S4 << (u32(4)))) + u32(2911926141)) ^ ((_S4 + sum_1))) ^ ((((_S4 >> (u32(5)))) + u32(2123724318)))));
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

fn createRay_0( px_0 : vec2<f32>,  PInv_1 : mat4x4<f32>,  VInv_1 : mat4x4<f32>) -> vec3<f32>
{
    var dirEye_0 : vec4<f32> = (((vec4<f32>(vec3<f32>(px_0 * vec2<f32>(2.0f) - vec2<f32>(1.0f), -1.0f), 1.0f)) * (PInv_1)));
    dirEye_0[i32(3)] = 0.0f;
    return normalize((((dirEye_0) * (VInv_1))).xyz);
}

struct SpecularMicrofacetBRDF_0
{
     albedo_0 : vec3<f32>,
     alpha_0 : f32,
     activeLobes_0 : u32,
};

fn SpecularMicrofacetBRDF_hasLobe_0( this_1 : SpecularMicrofacetBRDF_0,  lobeType_0 : i32) -> bool
{
    return ((this_1.activeLobes_0 & (u32(lobeType_0)))) != u32(0);
}

fn evalFresnelSchlick_0( f0_0 : vec3<f32>,  f90_0 : vec3<f32>,  cosTheta_0 : f32) -> vec3<f32>
{
    return f0_0 + (f90_0 - f0_0) * vec3<f32>(pow(max(1.0f - cosTheta_0, 0.0f), 5.0f));
}

fn SampleVndf_Hemisphere_0( u_0 : vec2<f32>,  Vh_0 : vec3<f32>) -> vec3<f32>
{
    var phi_0 : f32 = 6.28318548202514648f * u_0.x;
    var _S12 : f32 = Vh_0.z;
    var z_0 : f32 = fma(1.0f - u_0.y, 1.0f + _S12, - _S12);
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
    var _S13 : f32 = wi_0.z;
    return evalG1GGX_0(alpha_2 * alpha_2, _S13) * evalNdfGGX_0(alpha_2, h_0.z) * max(0.0f, dot(wi_0, h_0)) / _S13;
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
    const _S14 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    (*wo_0) = _S14;
    (*weight_0) = _S14;
    (*pdf_1) = 0.0f;
    (*lobeType_1) = u32(2);
    var _S15 : f32 = wi_2.z;
    if(_S15 < 9.99999997475242708e-07f)
    {
        return false;
    }
    if(this_2.alpha_0 == 0.0f)
    {
        if(!SpecularMicrofacetBRDF_hasLobe_0(this_2, i32(4)))
        {
            return false;
        }
        (*wo_0) = vec3<f32>(- wi_2.x, - wi_2.y, _S15);
        (*pdf_1) = 0.0f;
        (*weight_0) = evalFresnelSchlick_0(this_2.albedo_0, vec3<f32>(1.0f), _S15);
        (*lobeType_1) = u32(4);
        return true;
    }
    if(!SpecularMicrofacetBRDF_hasLobe_0(this_2, i32(2)))
    {
        return false;
    }
    var _S16 : vec2<f32> = sampleNext2D_0(&((*sg_2)));
    var h_2 : vec3<f32> = sampleGGX_VNDF_0(this_2.alpha_0, wi_2, _S16, &((*pdf_1)));
    var wiDotH_0 : f32 = dot(wi_2, h_2);
    var _S17 : vec3<f32> = vec3<f32>((2.0f * wiDotH_0)) * h_2 - wi_2;
    (*wo_0) = _S17;
    if(_S17.z < 9.99999997475242708e-07f)
    {
        return false;
    }
    var GOverG1wo_0 : f32 = evalMaskingSmithGGXCorrelated_0(this_2.alpha_0, _S15, (*wo_0).z) * (1.0f + evalLambdaGGX_0(this_2.alpha_0 * this_2.alpha_0, _S15));
    var F_0 : vec3<f32> = evalFresnelSchlick_0(this_2.albedo_0, vec3<f32>(1.0f), wiDotH_0);
    (*pdf_1) = (*pdf_1) / (4.0f * wiDotH_0);
    (*weight_0) = F_0 * vec3<f32>(GOverG1wo_0);
    (*lobeType_1) = u32(2);
    return true;
}

fn sample_disk_concentric_0( u_2 : vec2<f32>) -> vec2<f32>
{
    var _S18 : vec2<f32> = vec2<f32>(2.0f) * u_2 - vec2<f32>(1.0f);
    var _S19 : f32 = _S18.x;
    var _S20 : bool;
    if(_S19 == 0.0f)
    {
        _S20 = _S18.y == 0.0f;
    }
    else
    {
        _S20 = false;
    }
    if(_S20)
    {
        return _S18;
    }
    var _S21 : f32 = _S18.y;
    var r_0 : f32;
    var phi_1 : f32;
    if(abs(_S19) > abs(_S21))
    {
        var _S22 : f32 = _S21 / _S19 * 0.78539818525314331f;
        r_0 = _S19;
        phi_1 = _S22;
    }
    else
    {
        var _S23 : f32 = 1.57079637050628662f - _S19 / _S21 * 0.78539818525314331f;
        r_0 = _S21;
        phi_1 = _S23;
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
    var _S24 : vec2<f32> = sampleNext2D_0(&((*sg_3)));
    var _S25 : vec3<f32> = sample_cosine_hemisphere_concentric_0(_S24, &((*pdf_3)));
    (*wo_2) = _S25;
    (*lobeType_2) = u32(1);
    if(min(wi_4.z, (*wo_2).z) < 9.99999997475242708e-07f)
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
    return any(this_5.emission_0 > vec3<f32>(0.0f));
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
    var _S26 : BSDFContext_0;
    _S26.iorI_0 = 1.0f;
    _S26.iorT_0 = 1.0f;
    _S26.inited_0 = false;
    return _S26;
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
     uv_0 : vec2<f32>,
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
    const _S27 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    (*result_0).wo_3 = _S27;
    (*result_0).pdf_4 = 0.0f;
    (*result_0).weight_2 = _S27;
    (*result_0).lobeType_3 = u32(0);
    if(MaterialInstance_is_emissive_0(this_8))
    {
        return false;
    }
    var wiLocal_0 : vec3<f32> = ShadingFrame_toLocal_0(sd_0.frame_0, sd_0.V_0);
    var woLocal_0 : vec3<f32> = _S27;
    var selection_0 : f32 = sampleNext1D_0(&((*sg_4)));
    var valid_0 : bool;
    if(selection_0 > this_8.fresnel_0)
    {
        var _S28 : BSDFContext_0 = BSDFContext_x24init_0();
        var _S29 : f32 = (*result_0).pdf_4;
        var _S30 : vec3<f32> = (*result_0).weight_2;
        var _S31 : u32 = (*result_0).lobeType_3;
        var _S32 : bool = DisneyDiffuseBRDF_sample_0(this_8.diffuse_brdf_0, wiLocal_0, &(woLocal_0), &(_S29), &(_S30), &(_S31), &((*sg_4)), _S28);
        (*result_0).pdf_4 = _S29;
        (*result_0).weight_2 = _S30;
        (*result_0).lobeType_3 = _S31;
        valid_0 = _S32;
    }
    else
    {
        var _S33 : BSDFContext_0 = BSDFContext_x24init_0();
        var _S34 : f32 = (*result_0).pdf_4;
        var _S35 : vec3<f32> = (*result_0).weight_2;
        var _S36 : u32 = (*result_0).lobeType_3;
        var _S37 : bool = SpecularMicrofacetBRDF_sample_0(this_8.specular_brdf_0, wiLocal_0, &(woLocal_0), &(_S34), &(_S35), &(_S36), &((*sg_4)), _S33);
        (*result_0).pdf_4 = _S34;
        (*result_0).weight_2 = _S35;
        (*result_0).lobeType_3 = _S36;
        valid_0 = _S37;
    }
    (*result_0).wo_3 = ShadingFrame_fromLocal_0(sd_0.frame_0, woLocal_0);
    return valid_0;
}

fn SpecularMicrofacetBRDF_eval_0( this_9 : SpecularMicrofacetBRDF_0,  wi_5 : vec3<f32>,  wo_4 : vec3<f32>,  sg_5 : ptr<function, TinyUniformSampleGenerator_0>,  bc_2 : BSDFContext_0) -> vec3<f32>
{
    var _S38 : f32 = wi_5.z;
    var _S39 : f32 = wo_4.z;
    if(min(_S38, _S39) < 9.99999997475242708e-07f)
    {
        return vec3<f32>(0.0f);
    }
    if(this_9.alpha_0 == 0.0f)
    {
        return vec3<f32>(0.0f);
    }
    if(!SpecularMicrofacetBRDF_hasLobe_0(this_9, i32(2)))
    {
        return vec3<f32>(0.0f);
    }
    var h_3 : vec3<f32> = normalize(wi_5 + wo_4);
    return evalFresnelSchlick_0(this_9.albedo_0, vec3<f32>(1.0f), dot(wi_5, h_3)) * vec3<f32>(evalNdfGGX_0(this_9.alpha_0, h_3.z)) * vec3<f32>(evalMaskingSmithGGXCorrelated_0(this_9.alpha_0, _S38, _S39)) * vec3<f32>(0.25f) / vec3<f32>(_S38);
}

fn DisneyDiffuseBRDF_eval_0( this_10 : DisneyDiffuseBRDF_0,  wi_6 : vec3<f32>,  wo_5 : vec3<f32>,  sg_6 : ptr<function, TinyUniformSampleGenerator_0>,  bc_3 : BSDFContext_0) -> vec3<f32>
{
    var _S40 : f32 = wo_5.z;
    if(min(wi_6.z, _S40) < 9.99999997475242708e-07f)
    {
        return vec3<f32>(0.0f);
    }
    return DisneyDiffuseBRDF_evalWeight_0(this_10, wi_6, wo_5) * vec3<f32>(0.31830987334251404f) * vec3<f32>(_S40);
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
    var _S41 : f32 = 1.0f - this_11.fresnel_0;
    var _S42 : BSDFContext_0 = BSDFContext_x24init_0();
    var _S43 : vec3<f32> = DisneyDiffuseBRDF_eval_0(this_11.diffuse_brdf_0, wiLocal_1, woLocal_1, &((*sampler_0)), _S42);
    var _S44 : vec3<f32> = vec3<f32>(_S41) * _S43;
    var _S45 : vec3<f32> = SpecularMicrofacetBRDF_eval_0(this_11.specular_brdf_0, wiLocal_1, woLocal_1, &((*sampler_0)), _S42);
    return (_S44 + vec3<f32>(this_11.fresnel_0) * _S45) * light_intensity_0;
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
    const _S46 : vec3<f32> = vec3<f32>(1.0f, 0.0f, 0.0f);
    var T_1 : vec3<f32>;
    if(abs(dir_0.x) > 0.99000000953674316f)
    {
        T_1 = vec3<f32>(0.0f, 1.0f, 0.0f);
    }
    else
    {
        T_1 = _S46;
    }
    var _S47 : vec3<f32> = normalize(cross(T_1, dir_0));
    return transpose(mat3x3<f32>(_S47, cross(dir_0, _S47), dir_0));
}

fn sample_light_0( sampler_1 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S48 : vec2<f32> = sampleNext2D_0(&((*sampler_1)));
    return (((sample_cone_0(_S48, globalParams_0.cos_sun_aparent_size_0)) * (create_rotation_matrix_0(globalParams_0.sun_direction_0))));
}

fn computeRayOrigin_0( pos_0 : vec3<f32>,  normal_0 : vec3<f32>) -> vec3<f32>
{
    var iOff_0 : vec3<i32> = vec3<i32>(normal_0 * vec3<f32>(768.0f));
    return select((bitcast<vec3<f32>>(((bitcast<vec3<i32>>((pos_0))) + select(iOff_0, - iOff_0, pos_0 < vec3<f32>(0.0f))))), pos_0 + normal_0 * vec3<f32>(0.0000457763671875f), abs(pos_0) < vec3<f32>(0.0625f));
}

fn ShadingData_computeRayOrigin_0( this_12 : ShadingData_0,  viewside_0 : bool) -> vec3<f32>
{
    var _S49 : vec3<f32>;
    if(this_12.frontFacing_0 == viewside_0)
    {
        _S49 = this_12.faceN_0;
    }
    else
    {
        _S49 = - this_12.faceN_0;
    }
    return computeRayOrigin_0(this_12.posW_0, _S49);
}

struct SubDAG_0
{
     lowerBound_0 : vec3<i32>,
     nodeHeight_0 : u32,
     nodeIndex_0 : u32,
};

fn initial_subdag_0() -> SubDAG_0
{
    var subDAG_0 : SubDAG_0;
    subDAG_0.lowerBound_0 = vec3<i32>(vec3<u32>(u32(0)));
    subDAG_0.nodeHeight_0 = globalParams_0.node_height_0;
    subDAG_0.nodeIndex_0 = u32(256) + globalParams_0.root_node_index_0;
    return subDAG_0;
}

fn bitand_noninline_0( a_0 : vec3<bool>,  b_0 : vec3<bool>) -> vec3<bool>
{
    return (a_0 & (b_0));
}

fn max3_0( vec_0 : vec3<f32>) -> f32
{
    return max(max(vec_0.x, vec_0.y), vec_0.z);
}

fn min3_0( vec_1 : vec3<f32>) -> f32
{
    return min(min(vec_1.x, vec_1.y), vec_1.z);
}

fn findFirstChild_0( nodeEntry_0 : f32,  rayOrigin_0 : vec3<f32>,  invRayDir_0 : vec3<f32>,  nodeCentre_0 : vec3<i32>) -> vec3<i32>
{
    var _S50 : vec3<f32> = vec3<f32>(nodeCentre_0);
    var childId_0 : vec3<i32> = vec3<i32>((_S50 - rayOrigin_0) * invRayDir_0 < vec3<f32>(nodeEntry_0, nodeEntry_0, nodeEntry_0));
    var childId_1 : vec3<i32>;
    if(nodeEntry_0 <= 0.0f)
    {
        childId_1 = (childId_0 | (vec3<i32>(rayOrigin_0 >= _S50)));
    }
    else
    {
        childId_1 = childId_0;
    }
    return childId_1;
}

fn GET_NODE_FN_0( node_0 : u32,  childId_2 : u32) -> u32
{
    return svo_dag_0[node_0 - u32(256)][childId_2];
}

fn findNearestMaterial_0( nodeIndex_1 : u32,  rayDirSignBits_0 : u32) -> u32
{
    var _S51 : u32 = (u32(4275817112) ^ ((rayDirSignBits_0 * u32(286331153))));
    var _S52 : u32 = nodeIndex_1;
    for(;;)
    {
        if(_S52 >= u32(256))
        {
        }
        else
        {
            break;
        }
        var childIds_0 : u32 = _S51;
        for(;;)
        {
            if(childIds_0 != u32(0))
            {
            }
            else
            {
                break;
            }
            var childNodeIndex_0 : u32 = GET_NODE_FN_0(_S52, (childIds_0 & (u32(7))));
            if(childNodeIndex_0 > u32(0))
            {
                _S52 = childNodeIndex_0;
                break;
            }
            childIds_0 = (childIds_0 >> (u32(4)));
        }
    }
    return _S52;
}

struct RayVolumeIntersection_0
{
     hit_0 : bool,
     distance_0 : f32,
     material_0 : u32,
     position_0 : vec3<f32>,
     normal_1 : vec3<f32>,
};

struct Ray3f_0
{
     mOrigin_0 : vec3<f32>,
     mDir_0 : vec3<f32>,
};

fn intersectRayNodeESVO_0( nodeIndex_2 : u32,  nodePos_0 : vec3<i32>,  nodeHeight_1 : i32,  ray_0 : Ray3f_0,  rayDirSign_0 : vec3<f32>,  rayDirSignBits_1 : u32,  computeSurfaceProperties_0 : bool,  maxFootprint_0 : f32) -> RayVolumeIntersection_0
{
    const _S53 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var intersection_0 : RayVolumeIntersection_0;
    intersection_0.hit_0 = false;
    intersection_0.distance_0 = 0.0f;
    intersection_0.material_0 = u32(0);
    intersection_0.position_0 = _S53;
    intersection_0.normal_1 = _S53;
    var nodeSize_0 : u32 = (u32(1) << (u32(nodeHeight_1)));
    var invRayDir_1 : vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f) / ray_0.mDir_0;
    var _S54 : vec3<f32> = vec3<f32>(nodePos_0);
    var nodeEntry_1 : f32 = max3_0((_S54 - ray_0.mOrigin_0) * invRayDir_1);
    var nodeExit_0 : f32 = min3_0((_S54 + vec3<f32>(f32(nodeSize_0)) - ray_0.mOrigin_0) * invRayDir_1);
    if(nodeEntry_1 < nodeExit_0)
    {
        var childNodeSize_0 : i32 = i32(nodeSize_0 / u32(2));
        var _S55 : vec3<i32> = vec3<i32>(childNodeSize_0);
        var childId_3 : vec3<i32> = findFirstChild_0(nodeEntry_1, ray_0.mOrigin_0, invRayDir_1, nodePos_0 + _S55);
        var _S56 : vec3<i32> = nodePos_0 + childId_3 * _S55;
        var nodeStack_0 : array<u32, i32(33)>;
        var _S57 : vec3<f32> = - rayDirSign_0;
        var _S58 : u32 = nodeIndex_2;
        var lastExit_0 : f32 = nodeExit_0;
        var childPos_0 : vec3<i32> = _S56;
        var childNodeSize_1 : i32 = childNodeSize_0;
        var childId_4 : vec3<i32> = childId_3;
        var _S59 : i32 = nodeHeight_1;
        var _S60 : vec3<i32> = vec3<i32>(i32(1));
        for(;;)
        {
            var childT1_0 : vec3<f32> = (vec3<f32>(childPos_0 + vec3<i32>(childNodeSize_1)) - ray_0.mOrigin_0) * invRayDir_1;
            var tChildExit_0 : f32 = min3_0(childT1_0);
            var childNodeIndex_1 : u32 = GET_NODE_FN_0(_S58, (u32(((((childId_4[i32(0)] & (i32(1)))) | ((((childId_4[i32(1)] & (i32(1)))) << (u32(1))))) | ((((childId_4[i32(2)] & (i32(1)))) << (u32(2)))))) ^ (rayDirSignBits_1)));
            var _S61 : u32;
            var lastExit_1 : f32;
            var childPos_1 : vec3<i32>;
            var childId_5 : vec3<i32>;
            var _S62 : i32;
            var childNodeSize_2 : i32;
            var _S63 : bool;
            if(childNodeIndex_1 > u32(0))
            {
                var childT0_0 : vec3<f32> = (vec3<f32>(childPos_0) - ray_0.mOrigin_0) * invRayDir_1;
                var tChildEntry_0 : f32 = max3_0(childT0_0);
                var hasLargeFootprint_0 : bool = f32(childNodeSize_1) / tChildExit_0 > maxFootprint_0;
                if(childNodeIndex_1 >= u32(256))
                {
                    _S63 = hasLargeFootprint_0;
                }
                else
                {
                    _S63 = false;
                }
                if(_S63)
                {
                    if(tChildExit_0 < lastExit_0)
                    {
                        nodeStack_0[_S59] = _S58;
                    }
                    var _S64 : i32 = _S59 - i32(1);
                    var childNodeSize_3 : i32 = childNodeSize_1 / i32(2);
                    var _S65 : vec3<i32> = vec3<i32>(childNodeSize_3);
                    var childId_6 : vec3<i32> = findFirstChild_0(tChildEntry_0, ray_0.mOrigin_0, invRayDir_1, childPos_0 + _S65);
                    var _S66 : vec3<i32> = childPos_0 + childId_6 * _S65;
                    _S61 = childNodeIndex_1;
                    lastExit_1 = tChildExit_0;
                    _S62 = _S64;
                    childPos_1 = _S66;
                    childNodeSize_2 = childNodeSize_3;
                    childId_5 = childId_6;
                }
                else
                {
                    intersection_0.hit_0 = true;
                    intersection_0.distance_0 = tChildEntry_0;
                    if(computeSurfaceProperties_0)
                    {
                        intersection_0.material_0 = findNearestMaterial_0(childNodeIndex_1, rayDirSignBits_1);
                        intersection_0.normal_1 = vec3<f32>(vec3<f32>(tChildEntry_0, tChildEntry_0, tChildEntry_0) == childT0_0) * _S57;
                    }
                    _S61 = _S58;
                    lastExit_1 = lastExit_0;
                    _S62 = _S59;
                    childPos_1 = childPos_0;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_4;
                }
                _S58 = _S61;
                _S59 = _S62;
            }
            else
            {
                var nextChildFlipsVec_0 : vec3<i32> = vec3<i32>(childT1_0 <= vec3<f32>(tChildExit_0, tChildExit_0, tChildExit_0));
                var childId_7 : vec3<i32> = (childId_4 ^ (nextChildFlipsVec_0));
                var childPos_2 : vec3<i32> = childPos_0 + nextChildFlipsVec_0 * vec3<i32>(childNodeSize_1);
                if(!all(((childId_7 & (nextChildFlipsVec_0))) == nextChildFlipsVec_0))
                {
                    var differingBits_0 : vec3<i32> = (childPos_0 ^ (childPos_2));
                    var msb_0 : u32 = firstLeadingBit(u32(((differingBits_0[i32(0)] | (differingBits_0[i32(1)])) | (differingBits_0[i32(2)]))));
                    var _S67 : i32 = i32(msb_0 + u32(1));
                    var childNodeSize_4 : i32 = i32((u32(1) << (msb_0)));
                    var childId_8 : vec3<i32> = ((childPos_2 >> (vec3<u32>(msb_0))) & (_S60));
                    var _S68 : vec3<u32> = vec3<u32>(u32(_S67));
                    var childPos_3 : vec3<i32> = (((childPos_2 >> (_S68)) << (_S68))) + childId_8 * vec3<i32>(childNodeSize_4);
                    _S61 = nodeStack_0[_S67];
                    lastExit_1 = 0.0f;
                    _S62 = _S67;
                    childPos_1 = childPos_3;
                    childNodeSize_2 = childNodeSize_4;
                    childId_5 = childId_8;
                }
                else
                {
                    _S61 = _S58;
                    lastExit_1 = lastExit_0;
                    _S62 = _S59;
                    childPos_1 = childPos_2;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_7;
                }
                _S58 = _S61;
                _S59 = _S62;
            }
            if(intersection_0.hit_0 == false)
            {
                _S63 = _S59 <= nodeHeight_1;
            }
            else
            {
                _S63 = false;
            }
            if(!_S63)
            {
                break;
            }
            lastExit_0 = lastExit_1;
            childPos_0 = childPos_1;
            childNodeSize_1 = childNodeSize_2;
            childId_4 = childId_5;
        }
    }
    return intersection_0;
}

fn intersect_subdag_0( ray_1 : Ray3f_0,  subDAG_1 : SubDAG_0,  computeSurfaceProperties_1 : bool,  maxFootprint_1 : f32) -> RayVolumeIntersection_0
{
    const _S69 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var intersection_1 : RayVolumeIntersection_0;
    intersection_1.hit_0 = false;
    intersection_1.distance_0 = 0.0f;
    intersection_1.material_0 = u32(0);
    intersection_1.position_0 = _S69;
    intersection_1.normal_1 = _S69;
    var _S70 : vec3<f32> = vec3<f32>(0.0f);
    var _S71 : vec3<bool> = ray_1.mDir_0 < _S70;
    if(any(bitand_noninline_0(_S71, ray_1.mOrigin_0 < _S70)))
    {
        return intersection_1;
    }
    var _S72 : u32 = (u32(1) << (subDAG_1.nodeHeight_0));
    if(any(bitand_noninline_0(ray_1.mDir_0 >= _S70, ray_1.mOrigin_0 >= vec3<f32>((f32(_S72) - 0.5f)))))
    {
        return intersection_1;
    }
    var rayDirSignBits_2 : u32 = u32(((i32(_S71[i32(0)]) | ((i32(_S71[i32(1)]) << (u32(1))))) | ((i32(_S71[i32(2)]) << (u32(2))))));
    var _S73 : vec3<i32> = vec3<i32>(_S71);
    var rayDirSign_1 : vec3<f32> = vec3<f32>(_S73 * vec3<i32>(i32(-2)) + vec3<i32>(i32(1)));
    var reflectedRay_0 : Ray3f_0 = ray_1;
    reflectedRay_0.mOrigin_0 = (reflectedRay_0.mOrigin_0 + vec3<f32>(0.5f)) * rayDirSign_1;
    reflectedRay_0.mDir_0 = abs(reflectedRay_0.mDir_0);
    var nodeSize_1 : i32 = i32(_S72);
    intersection_1 = intersectRayNodeESVO_0(subDAG_1.nodeIndex_0, subDAG_1.lowerBound_0 * vec3<i32>(rayDirSign_1) - _S73 * vec3<i32>(nodeSize_1, nodeSize_1, nodeSize_1), i32(subDAG_1.nodeHeight_0), reflectedRay_0, rayDirSign_1, rayDirSignBits_2, computeSurfaceProperties_1, maxFootprint_1);
    if(intersection_1.hit_0)
    {
        intersection_1.position_0 = ray_1.mOrigin_0 + ray_1.mDir_0 * vec3<f32>(intersection_1.distance_0);
        return intersection_1;
    }
    return intersection_1;
}

fn shoot_shadow_ray_0( ray_2 : Ray3f_0) -> bool
{
    if(((globalParams_0.settings_0 & (i32(1)))) != i32(0))
    {
        return intersect_subdag_0(ray_2, initial_subdag_0(), false, 0.00350000010803342f).hit_0;
    }
    return false;
}

struct MaterialAndShadingData_0
{
     material_1 : MaterialInstance_0,
     shading_data_0 : ShadingData_0,
};

fn MaterialAndShadingData_get_direct_lighting_0( this_13 : MaterialAndShadingData_0,  sampler_2 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S74 : vec3<f32> = ShadingData_computeRayOrigin_0(this_13.shading_data_0, true);
    var _S75 : vec3<f32> = sample_light_0(&((*sampler_2)));
    var _S76 : Ray3f_0 = Ray3f_0( _S74, _S75 );
    var _S77 : vec3<f32> = MaterialInstance_eval_0(this_13.material_1, this_13.shading_data_0, globalParams_0.sun_direction_0, vec3<f32>(f32(!shoot_shadow_ray_0(_S76))) * globalParams_0.sun_colour_0, &((*sampler_2)));
    return _S77;
}

fn MaterialHeader_x24init_0() -> MaterialHeader_0
{
    var _S78 : MaterialHeader_0;
    _S78.packedData_0 = vec4<u32>(u32(0), u32(0), u32(0), u32(0));
    return _S78;
}

fn ShadingData_x24init_0() -> ShadingData_0
{
    var _S79 : ShadingData_0;
    _S79.mtl_0 = MaterialHeader_x24init_0();
    return _S79;
}

fn MaterialAndShadingData_x24init_0() -> MaterialAndShadingData_0
{
    var _S80 : MaterialAndShadingData_0;
    const _S81 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var _S82 : DisneyDiffuseBRDF_0 = DisneyDiffuseBRDF_0( _S81, 0.0f );
    var _S83 : SpecularMicrofacetBRDF_0 = SpecularMicrofacetBRDF_0( _S81, 0.0f, u32(0) );
    _S80.material_1.emission_0 = _S81;
    _S80.material_1.diffuse_brdf_0 = _S82;
    _S80.material_1.specular_brdf_0 = _S83;
    _S80.material_1.fresnel_0 = 0.0f;
    _S80.shading_data_0 = ShadingData_x24init_0();
    return _S80;
}

fn create_shading_data_from_intersection_0( intersection_2 : RayVolumeIntersection_0,  ray_3 : Ray3f_0) -> ShadingData_0
{
    var shading_data_1 : ShadingData_0 = ShadingData_x24init_0();
    shading_data_1.frame_0.N_0 = intersection_2.normal_1;
    shading_data_1.frame_0.T_0 = intersection_2.normal_1.yzx;
    shading_data_1.frame_0.B_0 = shading_data_1.frame_0.N_0.zxy;
    shading_data_1.posW_0 = intersection_2.position_0;
    shading_data_1.faceN_0 = shading_data_1.frame_0.N_0;
    shading_data_1.V_0 = - ray_3.mDir_0;
    shading_data_1.frontFacing_0 = true;
    shading_data_1.IoR_0 = 1.0f;
    return shading_data_1;
}

struct Material_0
{
     base_colour_0 : vec3<f32>,
     emission_factor_0 : f32,
     linear_roughness_0 : f32,
     metallic_0 : f32,
};

fn unpackStorage_1( _S84 : Material_std430_0) -> Material_0
{
    var _S85 : Material_0 = Material_0( _S84.base_colour_0, _S84.emission_factor_0, _S84.linear_roughness_0, _S84.metallic_0 );
    return _S85;
}

fn Material_ior_get_0( this_14 : Material_0) -> f32
{
    return 1.5f;
}

fn Material_f0_from_ior_0( this_15 : Material_0) -> f32
{
    var _S86 : f32 = Material_ior_get_0(this_15);
    var _S87 : f32 = (1.0f - _S86) / (1.0f + _S86);
    return _S87 * _S87;
}

fn Material_specular_colour_get_0( this_16 : Material_0) -> vec3<f32>
{
    return vec3<f32>(1.0f);
}

fn Material_specular_factor_get_0( this_17 : Material_0) -> f32
{
    return 1.0f;
}

fn Material_dielectic_f0_0( this_18 : Material_0) -> vec3<f32>
{
    return vec3<f32>(Material_f0_from_ior_0(this_18)) * Material_specular_colour_get_0(this_18) * vec3<f32>(Material_specular_factor_get_0(this_18));
}

fn Material_f0_0( this_19 : Material_0) -> vec3<f32>
{
    return mix(Material_dielectic_f0_0(this_19), this_19.base_colour_0, vec3<f32>(this_19.metallic_0));
}

fn Material_f90_0( this_20 : Material_0) -> f32
{
    return mix(Material_specular_factor_get_0(this_20), 1.0f, this_20.metallic_0);
}

fn luminance_0( rgb_0 : vec3<f32>) -> f32
{
    return dot(rgb_0, vec3<f32>(0.2125999927520752f, 0.71520000696182251f, 0.07220000028610229f));
}

fn Material_diffuse_colour_0( this_21 : Material_0) -> vec3<f32>
{
    return mix(this_21.base_colour_0, vec3<f32>(0.0f), vec3<f32>(this_21.metallic_0));
}

fn Material_alpha_roughness_0( this_22 : Material_0) -> f32
{
    var _S88 : f32 = this_22.linear_roughness_0;
    return max(_S88 * _S88, 9.99999997475242708e-07f);
}

fn MaterialInstance_x24init_0( material_2 : Material_0,  sd_2 : ShadingData_0) -> MaterialInstance_0
{
    var _S89 : MaterialInstance_0;
    _S89.emission_0 = material_2.base_colour_0 * vec3<f32>(material_2.emission_factor_0);
    _S89.fresnel_0 = luminance_0(evalFresnelSchlick_0(Material_f0_0(material_2), vec3<f32>(Material_f90_0(material_2)), ShadingFrame_toLocal_0(sd_2.frame_0, sd_2.V_0).z));
    _S89.diffuse_brdf_0.albedo_1 = Material_diffuse_colour_0(material_2);
    _S89.diffuse_brdf_0.roughness_0 = material_2.linear_roughness_0;
    _S89.specular_brdf_0.albedo_0 = material_2.base_colour_0;
    _S89.specular_brdf_0.alpha_0 = Material_alpha_roughness_0(material_2);
    _S89.specular_brdf_0.activeLobes_0 = u32(2);
    return _S89;
}

fn create_material_from_intersection_0( intersection_3 : RayVolumeIntersection_0,  ray_4 : Ray3f_0) -> MaterialAndShadingData_0
{
    var output_0 : MaterialAndShadingData_0 = MaterialAndShadingData_x24init_0();
    var _S90 : ShadingData_0 = create_shading_data_from_intersection_0(intersection_3, ray_4);
    output_0.shading_data_0 = _S90;
    output_0.material_1 = MaterialInstance_x24init_0(unpackStorage_1(materials_0[intersection_3.material_0 - u32(1)]), _S90);
    return output_0;
}

fn compute_shading_0( intersection_4 : RayVolumeIntersection_0,  ray_5 : Ray3f_0,  sampler_3 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var material_3 : MaterialAndShadingData_0 = create_material_from_intersection_0(intersection_4, ray_5);
    var _S91 : vec3<f32> = MaterialAndShadingData_get_direct_lighting_0(material_3, &((*sampler_3)));
    var _S92 : vec3<f32> = vec3<f32>(1.0f);
    var _S93 : SubDAG_0 = initial_subdag_0();
    var material_4 : MaterialAndShadingData_0 = material_3;
    var i_1 : u32 = u32(0);
    var throughput_0 : vec3<f32> = _S92;
    var radiance_0 : vec3<f32> = _S91;
    var _S94 : vec3<f32> = vec3<f32>(0.00999999977648258f);
    for(;;)
    {
        if(i_1 < globalParams_0.max_bounces_0)
        {
        }
        else
        {
            break;
        }
        var _S95 : MaterialAndShadingData_0 = material_4;
        var sample_result_0 : BSDFSample_0;
        var _S96 : bool = MaterialInstance_sample_0(material_4.material_1, material_4.shading_data_0, &((*sampler_3)), &(sample_result_0));
        if(!_S96)
        {
            break;
        }
        var throughput_1 : vec3<f32> = throughput_0 * sample_result_0.weight_2;
        if(all(throughput_1 < _S94))
        {
            break;
        }
        var _S97 : vec3<f32>;
        var _S98 : MaterialAndShadingData_0;
        var _S99 : Ray3f_0 = Ray3f_0( ShadingData_computeRayOrigin_0(_S95.shading_data_0, true), sample_result_0.wo_3 );
        var _S100 : RayVolumeIntersection_0 = intersect_subdag_0(_S99, _S93, true, 0.00350000010803342f);
        if(_S100.hit_0)
        {
            var material_5 : MaterialAndShadingData_0 = create_material_from_intersection_0(_S100, _S99);
            _S98 = material_5;
            var _S101 : vec3<f32> = MaterialAndShadingData_get_direct_lighting_0(material_5, &((*sampler_3)));
            _S97 = radiance_0 + _S101 * throughput_1;
        }
        else
        {
            radiance_0 = radiance_0 + globalParams_0.background_colour_0 * throughput_1;
            break;
        }
        var _S102 : u32 = i_1 + u32(1);
        material_4 = _S98;
        i_1 = _S102;
        throughput_0 = throughput_1;
        radiance_0 = _S97;
    }
    return radiance_0;
}

fn trace_0( ray_6 : Ray3f_0,  sampler_4 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var intersection_5 : RayVolumeIntersection_0 = intersect_subdag_0(ray_6, initial_subdag_0(), true, 0.00350000010803342f);
    if(intersection_5.hit_0)
    {
        var _S103 : vec3<f32> = compute_shading_0(intersection_5, ray_6, &((*sampler_4)));
        return _S103;
    }
    return globalParams_0.background_colour_0;
}

fn viridis_0( t_0 : f32) -> vec3<f32>
{
    var _S104 : vec3<f32> = vec3<f32>(t_0);
    return vec3<f32>(0.27772733569145203f, 0.00540734454989433f, 0.33409979939460754f) + _S104 * (vec3<f32>(0.10509303957223892f, 1.40461349487304688f, 1.38459014892578125f) + _S104 * (vec3<f32>(-0.33086183667182922f, 0.21484756469726562f, 0.09509516507387161f) + _S104 * (vec3<f32>(-4.63423061370849609f, -5.79910087585449219f, -19.33244132995605469f) + _S104 * (vec3<f32>(6.22827005386352539f, 14.17993354797363281f, 56.6905517578125f) + _S104 * (vec3<f32>(4.77638483047485352f, -13.74514579772949219f, -65.35303497314453125f) + _S104 * vec3<f32>(-5.4354557991027832f, 4.64585256576538086f, 26.31243515014648438f))))));
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatch_thread_id_0 : vec3<u32>)
{
    var _S105 : vec2<u32> = dispatch_thread_id_0.xy;
    if(any(_S105 >= globalParams_0.resolution_0))
    {
        return;
    }
    var _S106 : vec2<f32> = vec2<f32>(_S105);
    var _S107 : vec2<f32> = (_S106 + vec2<f32>(0.5f)) / vec2<f32>(globalParams_0.resolution_0);
    var rng_3 : TinyUniformSampleGenerator_0 = TinyUniformSampleGenerator_x24init_0(_S105, globalParams_0.frame_index_0);
    var _S108 : vec2<f32> = vec2<f32>(0.5f);
    var thread_offset_0 : vec2<f32>;
    if(((globalParams_0.settings_0 & (i32(2)))) != i32(0))
    {
        var _S109 : vec2<f32> = sampleNext2D_0(&(rng_3));
        thread_offset_0 = _S109;
    }
    else
    {
        thread_offset_0 = _S108;
    }
    var _S110 : vec2<f32> = (_S106 + thread_offset_0) / vec2<f32>(globalParams_0.resolution_0);
    var TexCoords_0 : vec2<f32> = _S110;
    TexCoords_0[i32(1)] = 1.0f - _S110.y;
    var ray_7 : Ray3f_0;
    ray_7.mOrigin_0 = globalParams_0.cameraPos_0;
    ray_7.mDir_0 = createRay_0(TexCoords_0, unpackStorage_0(globalParams_0.PInv_0), unpackStorage_0(globalParams_0.VInv_0));
    var sample_1 : vec3<f32> = trace_0(ray_7, &(rng_3));
    var sample_2 : vec3<f32>;
    if(((globalParams_0.settings_0 & (i32(2)))) != i32(0) && globalParams_0.accumulated_frame_index_0 > u32(0))
    {
        sample_2 = sample_1 + (textureSampleLevel((entryPointParams_previous_0), (entryPointParams_sampler_0), (_S107), (0.0f)).xyz);
    }
    else
    {
        sample_2 = sample_1;
    }
    textureStore((entryPointParams_current_0), (_S105), vec4<f32>((sample_2), 1));
    if(((globalParams_0.settings_0 & (i32(4)))) != i32(0))
    {
        textureStore((entryPointParams_current_0), (_S105), vec4<f32>((viridis_0(0.0f)), 1));
    }
    return;
}

struct Xoshiro128StarStar_0
{
    @align(4) state_1 : array<u32, i32(4)>,
};

struct UniformSampleGenerator_0
{
    @align(4) rng_4 : Xoshiro128StarStar_0,
};

