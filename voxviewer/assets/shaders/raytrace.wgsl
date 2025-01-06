@binding(1) @group(0) var<storage, read> svo_dag_0 : array<array<u32, i32(8)>>;

struct Material_std430_0
{
    @align(16) base_colour_0 : vec3<f32>,
    @align(4) emission_factor_0 : f32,
};

@binding(2) @group(0) var<storage, read> materials_0 : array<Material_std430_0>;

@binding(3) @group(0) var entryPointParams_current_0 : texture_storage_2d<rgba16float, write>;

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
    @align(8) num_bounces_0 : u32,
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

fn add_noninline_0( a_0 : u32,  b_0 : u32) -> u32
{
    return a_0 + b_0;
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
        var _S4 : u32 = _S3 + (((add_noninline_0((_S2 << (u32(4))), u32(2738958700)) ^ (_S2 + sum_1)) ^ (add_noninline_0((_S2 >> (u32(5))), u32(3355524772)))));
        var _S5 : u32 = _S2 + (((add_noninline_0((_S4 << (u32(4))), u32(2911926141)) ^ (_S4 + sum_1)) ^ (add_noninline_0((_S4 >> (u32(5))), u32(2123724318)))));
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

fn sample_disk_concentric_0( u_0 : vec2<f32>) -> vec2<f32>
{
    var _S12 : vec2<f32> = vec2<f32>(2.0f) * u_0 - vec2<f32>(1.0f);
    var _S13 : f32 = _S12.x;
    var _S14 : bool;
    if(_S13 == 0.0f)
    {
        _S14 = _S12.y == 0.0f;
    }
    else
    {
        _S14 = false;
    }
    if(_S14)
    {
        return _S12;
    }
    var _S15 : f32 = _S12.y;
    var r_0 : f32;
    var phi_0 : f32;
    if(abs(_S13) > abs(_S15))
    {
        var _S16 : f32 = _S15 / _S13 * 0.78539818525314331f;
        r_0 = _S13;
        phi_0 = _S16;
    }
    else
    {
        var _S17 : f32 = 1.57079637050628662f - _S13 / _S15 * 0.78539818525314331f;
        r_0 = _S15;
        phi_0 = _S17;
    }
    return vec2<f32>(r_0) * vec2<f32>(cos(phi_0), sin(phi_0));
}

fn sample_cosine_hemisphere_concentric_0( u_1 : vec2<f32>,  pdf_0 : ptr<function, f32>) -> vec3<f32>
{
    var d_0 : vec2<f32> = sample_disk_concentric_0(u_1);
    var z_0 : f32 = sqrt(max(0.0f, 1.0f - dot(d_0, d_0)));
    (*pdf_0) = z_0 * 0.31830987334251404f;
    return vec3<f32>(d_0, z_0);
}

struct PBRTDiffuseBSDF_0
{
     albedo_0 : vec3<f32>,
};

struct BSDFContext_0
{
     iorI_0 : f32,
     iorT_0 : f32,
     inited_0 : bool,
};

fn PBRTDiffuseBSDF_sample_0( this_1 : PBRTDiffuseBSDF_0,  wiLocal_0 : vec3<f32>,  wo_0 : ptr<function, vec3<f32>>,  pdf_1 : ptr<function, f32>,  weight_0 : ptr<function, vec3<f32>>,  lobeType_0 : ptr<function, u32>,  sg_2 : ptr<function, TinyUniformSampleGenerator_0>,  bc_0 : BSDFContext_0) -> bool
{
    var _S18 : vec2<f32> = sampleNext2D_0(&((*sg_2)));
    var _S19 : vec3<f32> = sample_cosine_hemisphere_concentric_0(_S18, &((*pdf_1)));
    (*wo_0) = _S19;
    (*weight_0) = this_1.albedo_0;
    (*lobeType_0) = u32(1);
    return true;
}

struct ShadingFrame_0
{
     T_0 : vec3<f32>,
     B_0 : vec3<f32>,
     N_0 : vec3<f32>,
};

fn ShadingFrame_toLocal_0( this_2 : ShadingFrame_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(v_1, this_2.T_0), dot(v_1, this_2.B_0), dot(v_1, this_2.N_0));
}

fn BSDFContext_x24init_0() -> BSDFContext_0
{
    var _S20 : BSDFContext_0;
    _S20.iorI_0 = 1.0f;
    _S20.iorT_0 = 1.0f;
    _S20.inited_0 = false;
    return _S20;
}

fn ShadingFrame_fromLocal_0( this_3 : ShadingFrame_0,  v_2 : vec3<f32>) -> vec3<f32>
{
    return this_3.T_0 * vec3<f32>(v_2.x) + this_3.B_0 * vec3<f32>(v_2.y) + this_3.N_0 * vec3<f32>(v_2.z);
}

fn ne_noninline_0( a_1 : bool,  b_1 : bool) -> bool
{
    return a_1 != b_1;
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

fn isValidHemisphereReflection_0( sd_0 : ShadingData_0,  sf_0 : ShadingFrame_0,  wiLocal_1 : vec3<f32>,  woLocal_0 : vec3<f32>,  wo_1 : vec3<f32>) -> bool
{
    if(min(wiLocal_1.z, woLocal_0.z) < 9.99999997475242708e-07f)
    {
        return false;
    }
    if(ne_noninline_0(sd_0.frontFacing_0, dot(wo_1, sd_0.faceN_0) >= 0.0f))
    {
        return false;
    }
    if(ne_noninline_0(sd_0.frontFacing_0, dot(sf_0.N_0, sd_0.faceN_0) >= 0.0f))
    {
        return false;
    }
    return true;
}

struct PBRTDiffuseMaterialInstance_0
{
     sf_1 : ShadingFrame_0,
     bsdf_0 : PBRTDiffuseBSDF_0,
};

struct BSDFSample_0
{
     wo_2 : vec3<f32>,
     pdf_2 : f32,
     weight_1 : vec3<f32>,
     lobeType_1 : u32,
};

fn PBRTDiffuseMaterialInstance_sample_0( this_4 : PBRTDiffuseMaterialInstance_0,  sd_1 : ShadingData_0,  sg_3 : ptr<function, TinyUniformSampleGenerator_0>,  result_0 : ptr<function, BSDFSample_0>,  useImportanceSampling_0 : bool) -> bool
{
    var wiLocal_2 : vec3<f32> = ShadingFrame_toLocal_0(this_4.sf_1, sd_1.V_0);
    var woLocal_1 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var _S21 : BSDFContext_0 = BSDFContext_x24init_0();
    var _S22 : f32 = (*result_0).pdf_2;
    var _S23 : vec3<f32> = (*result_0).weight_1;
    var _S24 : u32 = (*result_0).lobeType_1;
    var valid_0 : bool = PBRTDiffuseBSDF_sample_0(this_4.bsdf_0, wiLocal_2, &(woLocal_1), &(_S22), &(_S23), &(_S24), &((*sg_3)), _S21);
    (*result_0).pdf_2 = _S22;
    (*result_0).weight_1 = _S23;
    (*result_0).lobeType_1 = _S24;
    var _S25 : vec3<f32> = ShadingFrame_fromLocal_0(this_4.sf_1, woLocal_1);
    (*result_0).wo_2 = _S25;
    var _S26 : bool;
    if(!isValidHemisphereReflection_0(sd_1, this_4.sf_1, wiLocal_2, woLocal_1, _S25))
    {
        _S26 = true;
    }
    else
    {
        _S26 = (*result_0).pdf_2 == 0.0f;
    }
    if(_S26)
    {
        return false;
    }
    return valid_0;
}

fn PBRTDiffuseBSDF_eval_0( this_5 : PBRTDiffuseBSDF_0,  wiLocal_3 : vec3<f32>,  woLocal_2 : vec3<f32>,  sg_4 : ptr<function, TinyUniformSampleGenerator_0>,  bc_1 : BSDFContext_0) -> vec3<f32>
{
    return this_5.albedo_0 * vec3<f32>(max(woLocal_2.z, 0.0f)) * vec3<f32>(0.31830987334251404f);
}

fn PBRTDiffuseMaterialInstance_eval_0( this_6 : PBRTDiffuseMaterialInstance_0,  sd_2 : ShadingData_0,  wo_3 : vec3<f32>,  sg_5 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var wiLocal_4 : vec3<f32> = ShadingFrame_toLocal_0(this_6.sf_1, sd_2.V_0);
    var woLocal_3 : vec3<f32> = ShadingFrame_toLocal_0(this_6.sf_1, wo_3);
    if(!isValidHemisphereReflection_0(sd_2, this_6.sf_1, wiLocal_4, woLocal_3, wo_3))
    {
        return vec3<f32>(0.0f);
    }
    var _S27 : vec3<f32> = PBRTDiffuseBSDF_eval_0(this_6.bsdf_0, wiLocal_4, woLocal_3, &((*sg_5)), BSDFContext_x24init_0());
    return _S27;
}

fn sample_cone_0( u_2 : vec2<f32>,  cosTheta_0 : f32) -> vec3<f32>
{
    var z_1 : f32 = u_2.x * (1.0f - cosTheta_0) + cosTheta_0;
    var r_1 : f32 = sqrt(1.0f - z_1 * z_1);
    var phi_1 : f32 = 6.28318548202514648f * u_2.y;
    return vec3<f32>(r_1 * cos(phi_1), r_1 * sin(phi_1), z_1);
}

fn create_rotation_matrix_0( dir_0 : vec3<f32>) -> mat3x3<f32>
{
    const _S28 : vec3<f32> = vec3<f32>(1.0f, 0.0f, 0.0f);
    var T_1 : vec3<f32>;
    if(abs(dir_0.x) > 0.99000000953674316f)
    {
        T_1 = vec3<f32>(0.0f, 1.0f, 0.0f);
    }
    else
    {
        T_1 = _S28;
    }
    var _S29 : vec3<f32> = normalize(cross(T_1, dir_0));
    return transpose(mat3x3<f32>(_S29, cross(dir_0, _S29), dir_0));
}

fn sample_light_0( sampler_0 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S30 : vec2<f32> = sampleNext2D_0(&((*sampler_0)));
    return (((sample_cone_0(_S30, globalParams_0.cos_sun_aparent_size_0)) * (create_rotation_matrix_0(globalParams_0.sun_direction_0))));
}

fn computeRayOrigin_0( pos_0 : vec3<f32>,  normal_0 : vec3<f32>) -> vec3<f32>
{
    var iOff_0 : vec3<i32> = vec3<i32>(normal_0 * vec3<f32>(768.0f));
    return select((bitcast<vec3<f32>>(((bitcast<vec3<i32>>((pos_0))) + select(iOff_0, - iOff_0, pos_0 < vec3<f32>(0.0f))))), pos_0 + normal_0 * vec3<f32>(0.0000457763671875f), abs(pos_0) < vec3<f32>(0.0625f));
}

fn ShadingData_computeRayOrigin_0( this_7 : ShadingData_0,  viewside_0 : bool) -> vec3<f32>
{
    var _S31 : vec3<f32>;
    if(this_7.frontFacing_0 == viewside_0)
    {
        _S31 = this_7.faceN_0;
    }
    else
    {
        _S31 = - this_7.faceN_0;
    }
    return computeRayOrigin_0(this_7.posW_0, _S31);
}

struct SubDAG_0
{
     lowerBound_0 : vec3<i32>,
     nodeHeight_0 : i32,
     padding0_0 : u32,
     nodeIndex_0 : u32,
     padding1_0 : u32,
     padding2_0 : u32,
};

fn initial_subdag_0() -> SubDAG_0
{
    var subDAG_0 : SubDAG_0;
    subDAG_0.lowerBound_0 = vec3<i32>(vec3<u32>(u32(0)));
    subDAG_0.nodeHeight_0 = i32(6);
    subDAG_0.nodeIndex_0 = u32(261);
    return subDAG_0;
}

fn bool3_and_0( a_2 : vec3<bool>,  b_2 : vec3<bool>) -> vec3<bool>
{
    return (a_2 & (b_2));
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
    var _S32 : vec3<f32> = vec3<f32>(nodeCentre_0);
    var childId_0 : vec3<i32> = vec3<i32>((_S32 - rayOrigin_0) * invRayDir_0 < vec3<f32>(nodeEntry_0, nodeEntry_0, nodeEntry_0));
    var childId_1 : vec3<i32>;
    if(nodeEntry_0 <= 0.0f)
    {
        childId_1 = (childId_0 | (vec3<i32>(rayOrigin_0 >= _S32)));
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
    var _S33 : u32 = (u32(4275817112) ^ (rayDirSignBits_0 * u32(286331153)));
    var _S34 : u32 = nodeIndex_1;
    for(;;)
    {
        if(_S34 >= u32(256))
        {
        }
        else
        {
            break;
        }
        var childIds_0 : u32 = _S33;
        for(;;)
        {
            if(childIds_0 != u32(0))
            {
            }
            else
            {
                break;
            }
            var childNodeIndex_0 : u32 = GET_NODE_FN_0(_S34, (childIds_0 & (u32(7))));
            if(childNodeIndex_0 > u32(0))
            {
                _S34 = childNodeIndex_0;
                break;
            }
            childIds_0 = (childIds_0 >> (u32(4)));
        }
    }
    return _S34;
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
    const _S35 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var intersection_0 : RayVolumeIntersection_0;
    intersection_0.hit_0 = false;
    intersection_0.distance_0 = 0.0f;
    intersection_0.material_0 = u32(0);
    intersection_0.position_0 = _S35;
    intersection_0.normal_1 = _S35;
    var nodeSize_0 : u32 = (u32(1) << (u32(nodeHeight_1)));
    var invRayDir_1 : vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f) / ray_0.mDir_0;
    var _S36 : vec3<f32> = vec3<f32>(nodePos_0);
    var nodeEntry_1 : f32 = max3_0((_S36 - ray_0.mOrigin_0) * invRayDir_1);
    var nodeExit_0 : f32 = min3_0((_S36 + vec3<f32>(f32(nodeSize_0)) - ray_0.mOrigin_0) * invRayDir_1);
    if(nodeEntry_1 < nodeExit_0)
    {
        var childNodeSize_0 : i32 = i32(nodeSize_0 / u32(2));
        var _S37 : vec3<i32> = vec3<i32>(childNodeSize_0);
        var childId_3 : vec3<i32> = findFirstChild_0(nodeEntry_1, ray_0.mOrigin_0, invRayDir_1, nodePos_0 + _S37);
        var _S38 : vec3<i32> = nodePos_0 + childId_3 * _S37;
        var nodeStack_0 : array<u32, i32(33)>;
        var _S39 : vec3<f32> = - rayDirSign_0;
        var _S40 : u32 = nodeIndex_2;
        var lastExit_0 : f32 = nodeExit_0;
        var childPos_0 : vec3<i32> = _S38;
        var childNodeSize_1 : i32 = childNodeSize_0;
        var childId_4 : vec3<i32> = childId_3;
        var _S41 : i32 = nodeHeight_1;
        var _S42 : vec3<i32> = vec3<i32>(i32(1));
        for(;;)
        {
            var childT1_0 : vec3<f32> = (vec3<f32>(childPos_0 + vec3<i32>(childNodeSize_1)) - ray_0.mOrigin_0) * invRayDir_1;
            var tChildExit_0 : f32 = min3_0(childT1_0);
            var childNodeIndex_1 : u32 = GET_NODE_FN_0(_S40, (u32(((((childId_4[i32(0)] & (i32(1)))) | ((((childId_4[i32(1)] & (i32(1)))) << bitcast<u32>(i32(1))))) | ((((childId_4[i32(2)] & (i32(1)))) << bitcast<u32>(i32(2)))))) ^ (rayDirSignBits_1)));
            var _S43 : u32;
            var lastExit_1 : f32;
            var childPos_1 : vec3<i32>;
            var childId_5 : vec3<i32>;
            var _S44 : i32;
            var childNodeSize_2 : i32;
            var _S45 : bool;
            if(childNodeIndex_1 > u32(0))
            {
                var childT0_0 : vec3<f32> = (vec3<f32>(childPos_0) - ray_0.mOrigin_0) * invRayDir_1;
                var tChildEntry_0 : f32 = max3_0(childT0_0);
                var hasLargeFootprint_0 : bool = f32(childNodeSize_1) / tChildExit_0 > maxFootprint_0;
                if(childNodeIndex_1 >= u32(256))
                {
                    _S45 = hasLargeFootprint_0;
                }
                else
                {
                    _S45 = false;
                }
                if(_S45)
                {
                    if(tChildExit_0 < lastExit_0)
                    {
                        nodeStack_0[_S41] = _S40;
                    }
                    var _S46 : i32 = _S41 - i32(1);
                    var childNodeSize_3 : i32 = childNodeSize_1 / i32(2);
                    var _S47 : vec3<i32> = vec3<i32>(childNodeSize_3);
                    var childId_6 : vec3<i32> = findFirstChild_0(tChildEntry_0, ray_0.mOrigin_0, invRayDir_1, childPos_0 + _S47);
                    var _S48 : vec3<i32> = childPos_0 + childId_6 * _S47;
                    _S43 = childNodeIndex_1;
                    lastExit_1 = tChildExit_0;
                    _S44 = _S46;
                    childPos_1 = _S48;
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
                        intersection_0.normal_1 = vec3<f32>(vec3<f32>(tChildEntry_0, tChildEntry_0, tChildEntry_0) == childT0_0) * _S39;
                    }
                    _S43 = _S40;
                    lastExit_1 = lastExit_0;
                    _S44 = _S41;
                    childPos_1 = childPos_0;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_4;
                }
                _S40 = _S43;
                _S41 = _S44;
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
                    var _S49 : i32 = i32(msb_0 + u32(1));
                    var childNodeSize_4 : i32 = (i32(1) << bitcast<u32>(i32(msb_0)));
                    var childId_8 : vec3<i32> = ((childPos_2 >> (vec3<u32>(msb_0))) & (_S42));
                    var _S50 : vec3<u32> = vec3<u32>(u32(_S49));
                    var childPos_3 : vec3<i32> = (((childPos_2 >> (_S50)) << (_S50))) + childId_8 * vec3<i32>(childNodeSize_4);
                    _S43 = nodeStack_0[_S49];
                    lastExit_1 = 0.0f;
                    _S44 = _S49;
                    childPos_1 = childPos_3;
                    childNodeSize_2 = childNodeSize_4;
                    childId_5 = childId_8;
                }
                else
                {
                    _S43 = _S40;
                    lastExit_1 = lastExit_0;
                    _S44 = _S41;
                    childPos_1 = childPos_2;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_7;
                }
                _S40 = _S43;
                _S41 = _S44;
            }
            if(intersection_0.hit_0 == false)
            {
                _S45 = _S41 <= nodeHeight_1;
            }
            else
            {
                _S45 = false;
            }
            if(!_S45)
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
    var _S51 : vec3<f32> = vec3<f32>(0.0f);
    var intersection_1 : RayVolumeIntersection_0;
    intersection_1.hit_0 = false;
    intersection_1.distance_0 = 0.0f;
    intersection_1.material_0 = u32(0);
    intersection_1.position_0 = _S51;
    intersection_1.normal_1 = _S51;
    if(any(bool3_and_0(ray_1.mDir_0 < _S51, ray_1.mOrigin_0 < _S51)))
    {
        return intersection_1;
    }
    var _S52 : u32 = (u32(1) << (u32(subDAG_1.nodeHeight_0)));
    if(any(bool3_and_0(ray_1.mDir_0 >= _S51, ray_1.mOrigin_0 >= vec3<f32>(f32(_S52)) - vec3<f32>(0.5f))))
    {
        return intersection_1;
    }
    var rayDirSignBitsAsVec_0 : vec3<i32> = vec3<i32>(ray_1.mDir_0 < vec3<f32>(0.0f, 0.0f, 0.0f));
    var rayDirSignBits_2 : u32 = u32(((rayDirSignBitsAsVec_0[i32(0)] | ((rayDirSignBitsAsVec_0[i32(1)] << bitcast<u32>(i32(1))))) | ((rayDirSignBitsAsVec_0[i32(2)] << bitcast<u32>(i32(2))))));
    var rayDirSign_1 : vec3<f32> = vec3<f32>(rayDirSignBitsAsVec_0 * vec3<i32>(i32(-2)) + vec3<i32>(i32(1)));
    var reflectedRay_0 : Ray3f_0 = ray_1;
    reflectedRay_0.mOrigin_0 = (reflectedRay_0.mOrigin_0 + vec3<f32>(0.5f, 0.5f, 0.5f)) * rayDirSign_1;
    reflectedRay_0.mDir_0 = abs(reflectedRay_0.mDir_0);
    var nodeSize_1 : i32 = i32(_S52);
    intersection_1 = intersectRayNodeESVO_0(subDAG_1.nodeIndex_0, subDAG_1.lowerBound_0 * vec3<i32>(rayDirSign_1) - rayDirSignBitsAsVec_0 * vec3<i32>(nodeSize_1, nodeSize_1, nodeSize_1), subDAG_1.nodeHeight_0, reflectedRay_0, rayDirSign_1, rayDirSignBits_2, computeSurfaceProperties_1, maxFootprint_1);
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
     material_1 : PBRTDiffuseMaterialInstance_0,
     shading_data_0 : ShadingData_0,
     emission_0 : vec3<f32>,
};

fn MaterialAndShadingData_get_direct_lighting_0( this_8 : MaterialAndShadingData_0,  sampler_1 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S53 : vec3<f32> = ShadingData_computeRayOrigin_0(this_8.shading_data_0, true);
    var _S54 : vec3<f32> = sample_light_0(&((*sampler_1)));
    var _S55 : Ray3f_0 = Ray3f_0( _S53, _S54 );
    var lighting_0 : f32 = f32(!shoot_shadow_ray_0(_S55));
    var _S56 : vec3<f32> = PBRTDiffuseMaterialInstance_eval_0(this_8.material_1, this_8.shading_data_0, globalParams_0.sun_direction_0, &((*sampler_1)));
    return _S56 * vec3<f32>(lighting_0) * globalParams_0.sun_colour_0 + this_8.emission_0;
}

fn MaterialHeader_x24init_0() -> MaterialHeader_0
{
    var _S57 : MaterialHeader_0;
    _S57.packedData_0 = vec4<u32>(u32(0), u32(0), u32(0), u32(0));
    return _S57;
}

fn ShadingData_x24init_0() -> ShadingData_0
{
    var _S58 : ShadingData_0;
    _S58.mtl_0 = MaterialHeader_x24init_0();
    return _S58;
}

fn MaterialAndShadingData_x24init_0() -> MaterialAndShadingData_0
{
    var _S59 : MaterialAndShadingData_0;
    _S59.shading_data_0 = ShadingData_x24init_0();
    return _S59;
}

fn create_shading_data_from_intersection_0( intersection_2 : RayVolumeIntersection_0,  ray_3 : Ray3f_0) -> ShadingData_0
{
    var shading_data_1 : ShadingData_0 = ShadingData_x24init_0();
    shading_data_1.frame_0.N_0 = intersection_2.normal_1;
    shading_data_1.frame_0.T_0 = intersection_2.normal_1.yzx;
    shading_data_1.frame_0.B_0 = intersection_2.normal_1.zxy;
    shading_data_1.posW_0 = intersection_2.position_0;
    shading_data_1.faceN_0 = shading_data_1.frame_0.N_0;
    shading_data_1.V_0 = - ray_3.mDir_0;
    shading_data_1.frontFacing_0 = true;
    shading_data_1.IoR_0 = 1.0f;
    return shading_data_1;
}

fn create_material_from_intersection_0( intersection_3 : RayVolumeIntersection_0,  ray_4 : Ray3f_0) -> MaterialAndShadingData_0
{
    var output_0 : MaterialAndShadingData_0 = MaterialAndShadingData_x24init_0();
    output_0.shading_data_0 = create_shading_data_from_intersection_0(intersection_3, ray_4);
    var _S60 : vec3<f32> = materials_0[intersection_3.material_0 - u32(1)].base_colour_0;
    var _S61 : PBRTDiffuseBSDF_0 = PBRTDiffuseBSDF_0( materials_0[intersection_3.material_0 - u32(1)].base_colour_0 );
    output_0.material_1.sf_1 = output_0.shading_data_0.frame_0;
    output_0.material_1.bsdf_0 = _S61;
    output_0.emission_0 = _S60 * vec3<f32>(materials_0[intersection_3.material_0 - u32(1)].emission_factor_0);
    return output_0;
}

fn compute_shading_0( intersection_4 : RayVolumeIntersection_0,  ray_5 : Ray3f_0,  sampler_2 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var material_2 : MaterialAndShadingData_0 = create_material_from_intersection_0(intersection_4, ray_5);
    var _S62 : vec3<f32> = MaterialAndShadingData_get_direct_lighting_0(material_2, &((*sampler_2)));
    var _S63 : vec3<f32> = vec3<f32>(1.0f);
    var _S64 : SubDAG_0 = initial_subdag_0();
    var material_3 : MaterialAndShadingData_0 = material_2;
    var i_1 : u32 = u32(0);
    var throughput_0 : vec3<f32> = _S63;
    var radiance_0 : vec3<f32> = _S62;
    for(;;)
    {
        if(i_1 < globalParams_0.num_bounces_0)
        {
        }
        else
        {
            break;
        }
        var _S65 : MaterialAndShadingData_0 = material_3;
        var sample_result_0 : BSDFSample_0;
        var _S66 : bool = PBRTDiffuseMaterialInstance_sample_0(material_3.material_1, material_3.shading_data_0, &((*sampler_2)), &(sample_result_0), true);
        if(!_S66)
        {
            break;
        }
        var _S67 : vec3<f32>;
        var _S68 : MaterialAndShadingData_0;
        var throughput_1 : vec3<f32> = throughput_0 * sample_result_0.weight_1;
        var _S69 : Ray3f_0 = Ray3f_0( ShadingData_computeRayOrigin_0(_S65.shading_data_0, true), sample_result_0.wo_2 );
        var _S70 : RayVolumeIntersection_0 = intersect_subdag_0(_S69, _S64, true, 0.00350000010803342f);
        if(_S70.hit_0)
        {
            var material_4 : MaterialAndShadingData_0 = create_material_from_intersection_0(_S70, _S69);
            _S68 = material_4;
            var _S71 : vec3<f32> = MaterialAndShadingData_get_direct_lighting_0(material_4, &((*sampler_2)));
            _S67 = radiance_0 + _S71 * throughput_1;
        }
        else
        {
            radiance_0 = radiance_0 + globalParams_0.background_colour_0 * throughput_1;
            break;
        }
        var _S72 : u32 = i_1 + u32(1);
        material_3 = _S68;
        i_1 = _S72;
        throughput_0 = throughput_1;
        radiance_0 = _S67;
    }
    return radiance_0;
}

fn trace_0( ray_6 : Ray3f_0,  sampler_3 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var intersection_5 : RayVolumeIntersection_0 = intersect_subdag_0(ray_6, initial_subdag_0(), true, 0.00350000010803342f);
    if(intersection_5.hit_0)
    {
        var _S73 : vec3<f32> = compute_shading_0(intersection_5, ray_6, &((*sampler_3)));
        return _S73;
    }
    return globalParams_0.background_colour_0;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatch_thread_id_0 : vec3<u32>)
{
    var _S74 : vec2<u32> = dispatch_thread_id_0.xy;
    var _S75 : vec2<f32> = vec2<f32>(_S74);
    var _S76 : vec2<f32> = (_S75 + vec2<f32>(0.5f)) / vec2<f32>(globalParams_0.resolution_0);
    var rng_3 : TinyUniformSampleGenerator_0 = TinyUniformSampleGenerator_x24init_0(_S74, globalParams_0.frame_index_0);
    var _S77 : vec2<f32> = vec2<f32>(0.5f);
    var thread_offset_0 : vec2<f32>;
    if(((globalParams_0.settings_0 & (i32(2)))) != i32(0))
    {
        var _S78 : vec2<f32> = sampleNext2D_0(&(rng_3));
        thread_offset_0 = _S78;
    }
    else
    {
        thread_offset_0 = _S77;
    }
    var _S79 : vec2<f32> = (_S75 + thread_offset_0) / vec2<f32>(globalParams_0.resolution_0);
    var TexCoords_0 : vec2<f32> = _S79;
    TexCoords_0[i32(1)] = 1.0f - _S79.y;
    var ray_7 : Ray3f_0;
    ray_7.mOrigin_0 = globalParams_0.cameraPos_0;
    ray_7.mDir_0 = createRay_0(TexCoords_0, unpackStorage_0(globalParams_0.PInv_0), unpackStorage_0(globalParams_0.VInv_0));
    var sample_1 : vec3<f32> = trace_0(ray_7, &(rng_3));
    var sample_2 : vec3<f32>;
    if(((globalParams_0.settings_0 & (i32(2)))) != i32(0) && globalParams_0.accumulated_frame_index_0 > u32(0))
    {
        sample_2 = sample_1 + (textureSampleLevel((entryPointParams_previous_0), (entryPointParams_sampler_0), (_S76), (0.0f)).xyz);
    }
    else
    {
        sample_2 = sample_1;
    }
    textureStore((entryPointParams_current_0), (_S74), vec4<f32>((sample_2), 1));
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

