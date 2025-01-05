@binding(1) @group(0) var<storage, read> svo_dag_0 : array<array<u32, i32(8)>>;

@binding(2) @group(0) var entryPointParams_current_0 : texture_storage_2d<rgba16float, write>;

@binding(3) @group(0) var entryPointParams_previous_0 : texture_2d<f32>;

@binding(4) @group(0) var entryPointParams_sampler_0 : sampler;

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
    @align(16) resolution_0 : vec2<u32>,
    @align(8) settings_0 : i32,
    @align(4) frame_index_0 : u32,
    @align(16) cos_sun_aparent_size_0 : f32,
    @align(4) accumulated_frame_index_0 : u32,
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

fn PBRTDiffuseBSDF_eval_0( this_0 : PBRTDiffuseBSDF_0,  wiLocal_0 : vec3<f32>,  woLocal_0 : vec3<f32>,  sg_0 : ptr<function, TinyUniformSampleGenerator_0>,  bc_0 : BSDFContext_0) -> vec3<f32>
{
    return this_0.albedo_0 * vec3<f32>(max(woLocal_0.z, 0.0f)) * vec3<f32>(0.31830987334251404f);
}

struct ShadingFrame_0
{
     T_0 : vec3<f32>,
     B_0 : vec3<f32>,
     N_0 : vec3<f32>,
};

fn ShadingFrame_toLocal_0( this_1 : ShadingFrame_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(v_1, this_1.T_0), dot(v_1, this_1.B_0), dot(v_1, this_1.N_0));
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

fn isValidHemisphereReflection_0( sd_0 : ShadingData_0,  sf_0 : ShadingFrame_0,  wiLocal_1 : vec3<f32>,  woLocal_1 : vec3<f32>,  wo_0 : vec3<f32>) -> bool
{
    if(min(wiLocal_1.z, woLocal_1.z) < 9.99999997475242708e-07f)
    {
        return false;
    }
    if(ne_noninline_0(sd_0.frontFacing_0, dot(wo_0, sd_0.faceN_0) >= 0.0f))
    {
        return false;
    }
    if(ne_noninline_0(sd_0.frontFacing_0, dot(sf_0.N_0, sd_0.faceN_0) >= 0.0f))
    {
        return false;
    }
    return true;
}

fn BSDFContext_x24init_0() -> BSDFContext_0
{
    var _S7 : BSDFContext_0;
    _S7.iorI_0 = 1.0f;
    _S7.iorT_0 = 1.0f;
    _S7.inited_0 = false;
    return _S7;
}

struct PBRTDiffuseMaterialInstance_0
{
     sf_1 : ShadingFrame_0,
     bsdf_0 : PBRTDiffuseBSDF_0,
};

fn PBRTDiffuseMaterialInstance_eval_0( this_2 : PBRTDiffuseMaterialInstance_0,  sd_1 : ShadingData_0,  wo_1 : vec3<f32>,  sg_1 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var wiLocal_2 : vec3<f32> = ShadingFrame_toLocal_0(this_2.sf_1, sd_1.V_0);
    var woLocal_2 : vec3<f32> = ShadingFrame_toLocal_0(this_2.sf_1, wo_1);
    if(!isValidHemisphereReflection_0(sd_1, this_2.sf_1, wiLocal_2, woLocal_2, wo_1))
    {
        return vec3<f32>(0.0f);
    }
    var _S8 : vec3<f32> = PBRTDiffuseBSDF_eval_0(this_2.bsdf_0, wiLocal_2, woLocal_2, &((*sg_1)), BSDFContext_x24init_0());
    return _S8;
}

fn nextRandom_0( rng_2 : ptr<function, LCG_0>) -> u32
{
    var _S9 : u32 = u32(1664525) * (*rng_2).state_0 + u32(1013904223);
    (*rng_2).state_0 = _S9;
    return _S9;
}

fn TinyUniformSampleGenerator_next_0( this_3 : ptr<function, TinyUniformSampleGenerator_0>) -> u32
{
    var _S10 : LCG_0 = (*this_3).rng_0;
    var _S11 : u32 = nextRandom_0(&(_S10));
    (*this_3).rng_0 = _S10;
    return _S11;
}

fn sampleNext1D_0( sg_2 : ptr<function, TinyUniformSampleGenerator_0>) -> f32
{
    var bits_0 : u32 = TinyUniformSampleGenerator_next_0(&((*sg_2)));
    return f32((bits_0 >> (u32(8)))) * 5.9604644775390625e-08f;
}

fn sampleNext2D_0( sg_3 : ptr<function, TinyUniformSampleGenerator_0>) -> vec2<f32>
{
    var sample_0 : vec2<f32>;
    var _S12 : f32 = sampleNext1D_0(&((*sg_3)));
    sample_0[i32(0)] = _S12;
    var _S13 : f32 = sampleNext1D_0(&((*sg_3)));
    sample_0[i32(1)] = _S13;
    return sample_0;
}

fn sample_cone_0( u_0 : vec2<f32>,  cosTheta_0 : f32) -> vec3<f32>
{
    var z_0 : f32 = u_0.x * (1.0f - cosTheta_0) + cosTheta_0;
    var r_0 : f32 = sqrt(1.0f - z_0 * z_0);
    var phi_0 : f32 = 6.28318548202514648f * u_0.y;
    return vec3<f32>(r_0 * cos(phi_0), r_0 * sin(phi_0), z_0);
}

fn create_rotation_matrix_0( dir_0 : vec3<f32>) -> mat3x3<f32>
{
    const _S14 : vec3<f32> = vec3<f32>(1.0f, 0.0f, 0.0f);
    var T_1 : vec3<f32>;
    if(abs(dir_0.x) > 0.99000000953674316f)
    {
        T_1 = vec3<f32>(0.0f, 1.0f, 0.0f);
    }
    else
    {
        T_1 = _S14;
    }
    var _S15 : vec3<f32> = normalize(cross(T_1, dir_0));
    return mat3x3<f32>(_S15, cross(dir_0, _S15), dir_0);
}

fn sample_light_0( sampler_0 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S16 : vec2<f32> = sampleNext2D_0(&((*sampler_0)));
    return (((sample_cone_0(_S16, globalParams_0.cos_sun_aparent_size_0)) * (transpose(create_rotation_matrix_0(globalParams_0.sun_direction_0)))));
}

fn bool3_and_0( a_2 : vec3<bool>,  b_2 : vec3<bool>) -> vec3<bool>
{
    return (a_2 & (b_2));
}

fn findFirstChild_0( nodeEntry_0 : f32,  rayOrigin_0 : vec3<f32>,  invRayDir_0 : vec3<f32>,  nodeCentre_0 : vec3<i32>) -> vec3<i32>
{
    var _S17 : vec3<f32> = vec3<f32>(nodeCentre_0);
    var childId_0 : vec3<i32> = vec3<i32>((_S17 - rayOrigin_0) * invRayDir_0 < vec3<f32>(nodeEntry_0, nodeEntry_0, nodeEntry_0));
    var childId_1 : vec3<i32>;
    if(nodeEntry_0 <= 0.0f)
    {
        childId_1 = (childId_0 | (vec3<i32>(rayOrigin_0 >= _S17)));
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

fn findNearestMaterial_0( nodeIndex_0 : u32,  rayDirSignBits_0 : u32) -> u32
{
    var _S18 : u32 = (u32(4275817112) ^ (rayDirSignBits_0 * u32(286331153)));
    var _S19 : u32 = nodeIndex_0;
    for(;;)
    {
        if(_S19 >= u32(256))
        {
        }
        else
        {
            break;
        }
        var childIds_0 : u32 = _S18;
        for(;;)
        {
            if(childIds_0 != u32(0))
            {
            }
            else
            {
                break;
            }
            var childNodeIndex_0 : u32 = GET_NODE_FN_0(_S19, (childIds_0 & (u32(7))));
            if(childNodeIndex_0 > u32(0))
            {
                _S19 = childNodeIndex_0;
                break;
            }
            childIds_0 = (childIds_0 >> (u32(4)));
        }
    }
    return _S19;
}

struct RayVolumeIntersection_0
{
     hit_0 : bool,
     distance_0 : f32,
     material_0 : u32,
     position_0 : vec3<f32>,
     normal_0 : vec3<f32>,
};

struct Ray3f_0
{
     mOrigin_0 : vec3<f32>,
     mDir_0 : vec3<f32>,
};

fn intersectRayNodeESVO_0( nodeIndex_1 : u32,  nodePos_0 : vec3<i32>,  nodeHeight_0 : i32,  ray_0 : Ray3f_0,  rayDirSign_0 : vec3<f32>,  rayDirSignBits_1 : u32,  computeSurfaceProperties_0 : bool,  maxFootprint_0 : f32) -> RayVolumeIntersection_0
{
    const _S20 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var intersection_0 : RayVolumeIntersection_0;
    intersection_0.hit_0 = false;
    intersection_0.distance_0 = 0.0f;
    intersection_0.material_0 = u32(0);
    intersection_0.position_0 = _S20;
    intersection_0.normal_0 = _S20;
    var nodeSize_0 : u32 = (u32(1) << (u32(nodeHeight_0)));
    var invRayDir_1 : vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f) / ray_0.mDir_0;
    var _S21 : vec3<f32> = vec3<f32>(nodePos_0);
    var nodeT0_0 : vec3<f32> = (_S21 - ray_0.mOrigin_0) * invRayDir_1;
    var nodeT1_0 : vec3<f32> = (_S21 + vec3<f32>(f32(nodeSize_0)) - ray_0.mOrigin_0) * invRayDir_1;
    var _S22 : f32 = max(max(nodeT0_0.x, nodeT0_0.y), nodeT0_0.z);
    var _S23 : f32 = min(min(nodeT1_0.x, nodeT1_0.y), nodeT1_0.z);
    if(_S22 < _S23)
    {
        var childNodeSize_0 : i32 = i32(nodeSize_0 / u32(2));
        var _S24 : vec3<i32> = vec3<i32>(childNodeSize_0);
        var childId_3 : vec3<i32> = findFirstChild_0(_S22, ray_0.mOrigin_0, invRayDir_1, nodePos_0 + _S24);
        var _S25 : vec3<i32> = nodePos_0 + childId_3 * _S24;
        var nodeStack_0 : array<u32, i32(33)>;
        var _S26 : vec3<f32> = - rayDirSign_0;
        var _S27 : u32 = nodeIndex_1;
        var lastExit_0 : f32 = _S23;
        var childPos_0 : vec3<i32> = _S25;
        var childNodeSize_1 : i32 = childNodeSize_0;
        var childId_4 : vec3<i32> = childId_3;
        var _S28 : i32 = nodeHeight_0;
        var _S29 : vec3<i32> = vec3<i32>(i32(1));
        for(;;)
        {
            var childT1_0 : vec3<f32> = (vec3<f32>(childPos_0 + vec3<i32>(childNodeSize_1)) - ray_0.mOrigin_0) * invRayDir_1;
            var _S30 : f32 = min(min(childT1_0.x, childT1_0.y), childT1_0.z);
            var childNodeIndex_1 : u32 = GET_NODE_FN_0(_S27, (u32(((((childId_4[i32(0)] & (i32(1)))) | ((((childId_4[i32(1)] & (i32(1)))) << bitcast<u32>(i32(1))))) | ((((childId_4[i32(2)] & (i32(1)))) << bitcast<u32>(i32(2)))))) ^ (rayDirSignBits_1)));
            var _S31 : u32;
            var lastExit_1 : f32;
            var childPos_1 : vec3<i32>;
            var childId_5 : vec3<i32>;
            var _S32 : i32;
            var childNodeSize_2 : i32;
            var _S33 : bool;
            if(childNodeIndex_1 > u32(0))
            {
                var childT0_0 : vec3<f32> = (vec3<f32>(childPos_0) - ray_0.mOrigin_0) * invRayDir_1;
                var _S34 : f32 = max(max(childT0_0.x, childT0_0.y), childT0_0.z);
                var hasLargeFootprint_0 : bool = f32(childNodeSize_1) / _S30 > maxFootprint_0;
                if(childNodeIndex_1 >= u32(256))
                {
                    _S33 = hasLargeFootprint_0;
                }
                else
                {
                    _S33 = false;
                }
                if(_S33)
                {
                    if(_S30 < lastExit_0)
                    {
                        nodeStack_0[_S28] = _S27;
                    }
                    var _S35 : i32 = _S28 - i32(1);
                    var childNodeSize_3 : i32 = childNodeSize_1 / i32(2);
                    var _S36 : vec3<i32> = vec3<i32>(childNodeSize_3);
                    var childId_6 : vec3<i32> = findFirstChild_0(_S34, ray_0.mOrigin_0, invRayDir_1, childPos_0 + _S36);
                    var _S37 : vec3<i32> = childPos_0 + childId_6 * _S36;
                    _S31 = childNodeIndex_1;
                    lastExit_1 = _S30;
                    _S32 = _S35;
                    childPos_1 = _S37;
                    childNodeSize_2 = childNodeSize_3;
                    childId_5 = childId_6;
                }
                else
                {
                    intersection_0.hit_0 = true;
                    intersection_0.distance_0 = _S34;
                    if(computeSurfaceProperties_0)
                    {
                        intersection_0.material_0 = findNearestMaterial_0(childNodeIndex_1, rayDirSignBits_1);
                        intersection_0.normal_0 = vec3<f32>(vec3<f32>(_S34, _S34, _S34) == childT0_0) * _S26;
                    }
                    _S31 = _S27;
                    lastExit_1 = lastExit_0;
                    _S32 = _S28;
                    childPos_1 = childPos_0;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_4;
                }
                _S27 = _S31;
                _S28 = _S32;
            }
            else
            {
                var nextChildFlipsVec_0 : vec3<i32> = vec3<i32>(childT1_0 <= vec3<f32>(_S30, _S30, _S30));
                var childId_7 : vec3<i32> = (childId_4 ^ (nextChildFlipsVec_0));
                var childPos_2 : vec3<i32> = childPos_0 + nextChildFlipsVec_0 * vec3<i32>(childNodeSize_1);
                if(!all(((childId_7 & (nextChildFlipsVec_0))) == nextChildFlipsVec_0))
                {
                    var differingBits_0 : vec3<i32> = (childPos_0 ^ (childPos_2));
                    var msb_0 : u32 = firstLeadingBit(u32(((differingBits_0[i32(0)] | (differingBits_0[i32(1)])) | (differingBits_0[i32(2)]))));
                    var _S38 : i32 = i32(msb_0 + u32(1));
                    var childNodeSize_4 : i32 = (i32(1) << bitcast<u32>(i32(msb_0)));
                    var childId_8 : vec3<i32> = ((childPos_2 >> (vec3<u32>(msb_0))) & (_S29));
                    var _S39 : vec3<u32> = vec3<u32>(u32(_S38));
                    var childPos_3 : vec3<i32> = (((childPos_2 >> (_S39)) << (_S39))) + childId_8 * vec3<i32>(childNodeSize_4);
                    _S31 = nodeStack_0[_S38];
                    lastExit_1 = 0.0f;
                    _S32 = _S38;
                    childPos_1 = childPos_3;
                    childNodeSize_2 = childNodeSize_4;
                    childId_5 = childId_8;
                }
                else
                {
                    _S31 = _S27;
                    lastExit_1 = lastExit_0;
                    _S32 = _S28;
                    childPos_1 = childPos_2;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_7;
                }
                _S27 = _S31;
                _S28 = _S32;
            }
            if(intersection_0.hit_0 == false)
            {
                _S33 = _S28 <= nodeHeight_0;
            }
            else
            {
                _S33 = false;
            }
            if(!_S33)
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

struct SubDAG_0
{
     lowerBound_0 : vec3<i32>,
     nodeHeight_1 : i32,
     padding0_0 : u32,
     nodeIndex_2 : u32,
     padding1_0 : u32,
     padding2_0 : u32,
};

fn intersect_subdag_0( ray_1 : Ray3f_0,  subDAG_0 : SubDAG_0,  computeSurfaceProperties_1 : bool,  maxFootprint_1 : f32) -> RayVolumeIntersection_0
{
    var _S40 : vec3<f32> = vec3<f32>(0.0f);
    var intersection_1 : RayVolumeIntersection_0;
    intersection_1.hit_0 = false;
    intersection_1.distance_0 = 0.0f;
    intersection_1.material_0 = u32(0);
    intersection_1.position_0 = _S40;
    intersection_1.normal_0 = _S40;
    if(any(bool3_and_0(ray_1.mDir_0 < _S40, ray_1.mOrigin_0 < _S40)))
    {
        return intersection_1;
    }
    var _S41 : u32 = (u32(1) << (u32(subDAG_0.nodeHeight_1)));
    if(any(bool3_and_0(ray_1.mDir_0 >= _S40, ray_1.mOrigin_0 >= vec3<f32>(f32(_S41)) - vec3<f32>(0.5f))))
    {
        return intersection_1;
    }
    var rayDirSignBitsAsVec_0 : vec3<i32> = vec3<i32>(ray_1.mDir_0 < vec3<f32>(0.0f, 0.0f, 0.0f));
    var rayDirSignBits_2 : u32 = u32(((rayDirSignBitsAsVec_0[i32(0)] | ((rayDirSignBitsAsVec_0[i32(1)] << bitcast<u32>(i32(1))))) | ((rayDirSignBitsAsVec_0[i32(2)] << bitcast<u32>(i32(2))))));
    var rayDirSign_1 : vec3<f32> = vec3<f32>(rayDirSignBitsAsVec_0 * vec3<i32>(i32(-2)) + vec3<i32>(i32(1)));
    var reflectedRay_0 : Ray3f_0 = ray_1;
    reflectedRay_0.mOrigin_0 = (reflectedRay_0.mOrigin_0 + vec3<f32>(0.5f, 0.5f, 0.5f)) * rayDirSign_1;
    reflectedRay_0.mDir_0 = abs(reflectedRay_0.mDir_0);
    var nodeSize_1 : i32 = i32(_S41);
    intersection_1 = intersectRayNodeESVO_0(subDAG_0.nodeIndex_2, subDAG_0.lowerBound_0 * vec3<i32>(rayDirSign_1) - rayDirSignBitsAsVec_0 * vec3<i32>(nodeSize_1, nodeSize_1, nodeSize_1), subDAG_0.nodeHeight_1, reflectedRay_0, rayDirSign_1, rayDirSignBits_2, computeSurfaceProperties_1, maxFootprint_1);
    if(intersection_1.hit_0)
    {
        intersection_1.position_0 = ray_1.mOrigin_0 + ray_1.mDir_0 * vec3<f32>(intersection_1.distance_0);
        return intersection_1;
    }
    return intersection_1;
}

fn MaterialHeader_x24init_0() -> MaterialHeader_0
{
    var _S42 : MaterialHeader_0;
    _S42.packedData_0 = vec4<u32>(u32(0), u32(0), u32(0), u32(0));
    return _S42;
}

fn ShadingData_x24init_0() -> ShadingData_0
{
    var _S43 : ShadingData_0;
    _S43.mtl_0 = MaterialHeader_x24init_0();
    return _S43;
}

fn ShadingFrame_createIdentity_0() -> ShadingFrame_0
{
    var sf_2 : ShadingFrame_0;
    sf_2.T_0 = vec3<f32>(1.0f, 0.0f, 0.0f);
    sf_2.B_0 = vec3<f32>(0.0f, 1.0f, 0.0f);
    sf_2.N_0 = vec3<f32>(0.0f, 0.0f, 1.0f);
    return sf_2;
}

fn create_shading_data_from_intersection_0( intersection_2 : RayVolumeIntersection_0,  ray_2 : Ray3f_0) -> ShadingData_0
{
    var shading_data_0 : ShadingData_0 = ShadingData_x24init_0();
    shading_data_0.frame_0 = ShadingFrame_createIdentity_0();
    shading_data_0.frame_0.N_0 = intersection_2.normal_0;
    shading_data_0.posW_0 = intersection_2.position_0;
    shading_data_0.faceN_0 = shading_data_0.frame_0.N_0;
    shading_data_0.V_0 = - ray_2.mDir_0;
    shading_data_0.frontFacing_0 = true;
    shading_data_0.IoR_0 = 1.0f;
    return shading_data_0;
}

struct StandardBSDFData_0
{
     diffuse_0 : vec3<f32>,
     specular_0 : vec3<f32>,
     roughness_0 : f32,
     metallic_0 : f32,
     eta_0 : f32,
     transmission_0 : vec3<f32>,
     diffuseTransmission_0 : f32,
     specularTransmission_0 : f32,
     volumeScattering_0 : vec3<f32>,
     volumeAnsiotropy_0 : f32,
     hasEntryPointVolumeProperties_0 : bool,
     hasSigmaSGreaterZero_0 : bool,
};

fn compute_shading_0( intersection_3 : RayVolumeIntersection_0,  ray_3 : Ray3f_0,  sampler_1 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var _S44 : vec3<f32> = vec3<f32>(1.0f);
    var palette_0 : array<vec3<f32>, i32(2)> = array<vec3<f32>, i32(2)>( _S44, vec3<f32>(1.0f, 0.0f, 0.0f) );
    var _S45 : u32 = intersection_3.material_0 - u32(1);
    var lighting_0 : f32;
    if(((globalParams_0.settings_0 & (i32(1)))) != i32(0))
    {
        var shadow_ray_0 : Ray3f_0;
        shadow_ray_0.mOrigin_0 = intersection_3.position_0 + intersection_3.normal_0 * vec3<f32>(0.00009999999747379f);
        var _S46 : vec3<f32> = sample_light_0(&((*sampler_1)));
        shadow_ray_0.mDir_0 = normalize(_S46);
        var subDAG_1 : SubDAG_0;
        subDAG_1.lowerBound_0 = vec3<i32>(vec3<u32>(u32(0)));
        subDAG_1.nodeHeight_1 = i32(6);
        subDAG_1.nodeIndex_2 = u32(261);
        lighting_0 = 3.0f * f32(!intersect_subdag_0(shadow_ray_0, subDAG_1, false, 0.00350000010803342f).hit_0);
    }
    else
    {
        lighting_0 = 3.0f;
    }
    var _S47 : ShadingData_0 = create_shading_data_from_intersection_0(intersection_3, ray_3);
    var data_1 : StandardBSDFData_0;
    data_1.diffuse_0 = palette_0[_S45];
    data_1.specular_0 = _S44;
    data_1.roughness_0 = 1.0f;
    data_1.metallic_0 = 0.0f;
    data_1.eta_0 = 1.0f;
    var _S48 : vec3<f32> = vec3<f32>(0.0f);
    data_1.transmission_0 = _S48;
    data_1.diffuseTransmission_0 = 0.0f;
    data_1.specularTransmission_0 = 0.0f;
    data_1.volumeScattering_0 = _S48;
    data_1.volumeAnsiotropy_0 = 0.0f;
    data_1.hasEntryPointVolumeProperties_0 = false;
    var _S49 : PBRTDiffuseBSDF_0 = PBRTDiffuseBSDF_0( palette_0[_S45] );
    var _S50 : PBRTDiffuseMaterialInstance_0 = PBRTDiffuseMaterialInstance_0( _S47.frame_0, _S49 );
    var _S51 : vec3<f32> = PBRTDiffuseMaterialInstance_eval_0(_S50, _S47, globalParams_0.sun_direction_0, &((*sampler_1)));
    return _S51 * vec3<f32>(lighting_0);
}

fn trace_0( ray_4 : Ray3f_0,  sampler_2 : ptr<function, TinyUniformSampleGenerator_0>) -> vec3<f32>
{
    var subDAG_2 : SubDAG_0;
    subDAG_2.lowerBound_0 = vec3<i32>(vec3<u32>(u32(0)));
    subDAG_2.nodeHeight_1 = i32(6);
    subDAG_2.nodeIndex_2 = u32(261);
    var intersection_4 : RayVolumeIntersection_0 = intersect_subdag_0(ray_4, subDAG_2, true, 0.00350000010803342f);
    if(intersection_4.hit_0)
    {
        var _S52 : vec3<f32> = compute_shading_0(intersection_4, ray_4, &((*sampler_2)));
        return _S52;
    }
    return vec3<f32>(abs(sin(ray_4.mDir_0.y * 10.0f)));
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatch_thread_id_0 : vec3<u32>)
{
    var _S53 : vec2<u32> = dispatch_thread_id_0.xy;
    var _S54 : vec2<f32> = (vec2<f32>(_S53) + vec2<f32>(0.5f)) / vec2<f32>(globalParams_0.resolution_0);
    var TexCoords_0 : vec2<f32> = _S54;
    TexCoords_0[i32(1)] = 1.0f - _S54.y;
    var rng_3 : TinyUniformSampleGenerator_0 = TinyUniformSampleGenerator_x24init_0(_S53, globalParams_0.frame_index_0);
    var ray_5 : Ray3f_0;
    ray_5.mOrigin_0 = globalParams_0.cameraPos_0;
    var _S55 : mat4x4<f32> = unpackStorage_0(globalParams_0.VInv_0);
    var dirEye_0 : vec4<f32> = (((vec4<f32>(vec3<f32>(TexCoords_0 * vec2<f32>(2.0f) - vec2<f32>(1.0f), -1.0f), 1.0f)) * (unpackStorage_0(globalParams_0.PInv_0))));
    dirEye_0[i32(3)] = 0.0f;
    ray_5.mDir_0 = normalize((((dirEye_0) * (_S55))).xyz);
    var sample_1 : vec3<f32> = trace_0(ray_5, &(rng_3));
    var sample_2 : vec3<f32>;
    if(((globalParams_0.settings_0 & (i32(2)))) != i32(0) && globalParams_0.accumulated_frame_index_0 > u32(0))
    {
        sample_2 = (textureSampleLevel((entryPointParams_previous_0), (entryPointParams_sampler_0), (_S54), (0.0f)).xyz) * vec3<f32>((f32(globalParams_0.accumulated_frame_index_0) / f32(globalParams_0.accumulated_frame_index_0 + u32(1)))) + sample_1 * vec3<f32>((1.0f / f32(globalParams_0.accumulated_frame_index_0 + u32(1))));
    }
    else
    {
        sample_2 = sample_1;
    }
    textureStore((entryPointParams_current_0), (_S53), vec4<f32>((sample_2), 1));
    return;
}

