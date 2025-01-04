@binding(1) @group(0) var entryPointParams_image_0 : texture_storage_2d<rgba16float, write>;

struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct SLANG_anonymous_0_std140_0
{
    @align(16) VInv_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) PInv_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) cameraPos_0 : vec3<f32>,
    @align(16) sun_direction_0 : vec3<f32>,
    @align(16) resolution_0 : vec2<u32>,
};

struct GlobalParams_std140_0
{
    @align(16) uniforms_0 : SLANG_anonymous_0_std140_0,
};

@binding(0) @group(0) var<uniform> globalParams_0 : GlobalParams_std140_0;
fn unpackStorage_0( _S1 : _MatrixStorage_float4x4_ColMajorstd140_0) -> mat4x4<f32>
{
    return mat4x4<f32>(_S1.data_0[i32(0)][i32(0)], _S1.data_0[i32(1)][i32(0)], _S1.data_0[i32(2)][i32(0)], _S1.data_0[i32(3)][i32(0)], _S1.data_0[i32(0)][i32(1)], _S1.data_0[i32(1)][i32(1)], _S1.data_0[i32(2)][i32(1)], _S1.data_0[i32(3)][i32(1)], _S1.data_0[i32(0)][i32(2)], _S1.data_0[i32(1)][i32(2)], _S1.data_0[i32(2)][i32(2)], _S1.data_0[i32(3)][i32(2)], _S1.data_0[i32(0)][i32(3)], _S1.data_0[i32(1)][i32(3)], _S1.data_0[i32(2)][i32(3)], _S1.data_0[i32(3)][i32(3)]);
}

struct SLANG_anonymous_0_0
{
     VInv_0 : mat4x4<f32>,
     PInv_0 : mat4x4<f32>,
     cameraPos_0 : vec3<f32>,
     sun_direction_0 : vec3<f32>,
     resolution_0 : vec2<u32>,
};

fn unpackStorage_1( _S2 : SLANG_anonymous_0_std140_0) -> SLANG_anonymous_0_0
{
    var _S3 : SLANG_anonymous_0_0 = SLANG_anonymous_0_0( unpackStorage_0(_S2.VInv_0), unpackStorage_0(_S2.PInv_0), _S2.cameraPos_0, _S2.sun_direction_0, _S2.resolution_0 );
    return _S3;
}

fn findFirstChild_0( nodeEntry_0 : f32,  rayOrigin_0 : vec3<f32>,  invRayDir_0 : vec3<f32>,  nodeCentre_0 : vec3<i32>) -> vec3<i32>
{
    var _S4 : vec3<f32> = vec3<f32>(nodeCentre_0);
    var childId_0 : vec3<i32> = vec3<i32>((_S4 - rayOrigin_0) * invRayDir_0 < vec3<f32>(nodeEntry_0, nodeEntry_0, nodeEntry_0));
    var childId_1 : vec3<i32>;
    if(nodeEntry_0 <= 0.0f)
    {
        childId_1 = (childId_0 | (vec3<i32>(rayOrigin_0 >= _S4)));
    }
    else
    {
        childId_1 = childId_0;
    }
    return childId_1;
}

var<private> dagData_0 : array<u32, i32(48)> = array<u32, i32(48)>( u32(1), u32(1), u32(1), u32(0), u32(2), u32(0), u32(0), u32(0), u32(1), u32(1), u32(1), u32(256), u32(1), u32(256), u32(256), u32(0), u32(1), u32(1), u32(1), u32(257), u32(1), u32(257), u32(257), u32(0), u32(1), u32(1), u32(1), u32(258), u32(1), u32(258), u32(258), u32(0), u32(1), u32(1), u32(1), u32(259), u32(1), u32(259), u32(259), u32(0), u32(1), u32(1), u32(1), u32(260), u32(1), u32(260), u32(260), u32(0) );

fn GET_NODE_FN_0( node_0 : u32,  childId_2 : u32) -> u32
{
    return dagData_0[(node_0 - u32(256)) * u32(8) + childId_2];
}

fn findNearestMaterial_0( nodeIndex_0 : u32,  rayDirSignBits_0 : u32) -> u32
{
    var _S5 : u32 = (u32(4275817112) ^ (rayDirSignBits_0 * u32(286331153)));
    var _S6 : u32 = nodeIndex_0;
    for(;;)
    {
        if(_S6 >= u32(256))
        {
        }
        else
        {
            break;
        }
        var childIds_0 : u32 = _S5;
        for(;;)
        {
            if(childIds_0 != u32(0))
            {
            }
            else
            {
                break;
            }
            var childNodeIndex_0 : u32 = GET_NODE_FN_0(_S6, (childIds_0 & (u32(7))));
            if(childNodeIndex_0 > u32(0))
            {
                _S6 = childNodeIndex_0;
                break;
            }
            childIds_0 = (childIds_0 >> (u32(4)));
        }
    }
    return _S6;
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
    const _S7 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var intersection_0 : RayVolumeIntersection_0;
    intersection_0.hit_0 = false;
    intersection_0.distance_0 = 0.0f;
    intersection_0.material_0 = u32(0);
    intersection_0.position_0 = _S7;
    intersection_0.normal_0 = _S7;
    var nodeSize_0 : u32 = (u32(1) << (u32(nodeHeight_0)));
    var invRayDir_1 : vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f) / ray_0.mDir_0;
    var _S8 : vec3<f32> = vec3<f32>(nodePos_0);
    var nodeT0_0 : vec3<f32> = (_S8 - ray_0.mOrigin_0) * invRayDir_1;
    var nodeT1_0 : vec3<f32> = (_S8 + vec3<f32>(f32(nodeSize_0)) - ray_0.mOrigin_0) * invRayDir_1;
    var _S9 : f32 = max(max(nodeT0_0.x, nodeT0_0.y), nodeT0_0.z);
    var _S10 : f32 = min(min(nodeT1_0.x, nodeT1_0.y), nodeT1_0.z);
    if(_S9 < _S10)
    {
        var childNodeSize_0 : i32 = i32(nodeSize_0 / u32(2));
        var _S11 : vec3<i32> = vec3<i32>(childNodeSize_0);
        var childId_3 : vec3<i32> = findFirstChild_0(_S9, ray_0.mOrigin_0, invRayDir_1, nodePos_0 + _S11);
        var _S12 : vec3<i32> = nodePos_0 + childId_3 * _S11;
        var nodeStack_0 : array<u32, i32(33)>;
        var _S13 : vec3<f32> = - rayDirSign_0;
        var _S14 : u32 = nodeIndex_1;
        var lastExit_0 : f32 = _S10;
        var childPos_0 : vec3<i32> = _S12;
        var childNodeSize_1 : i32 = childNodeSize_0;
        var childId_4 : vec3<i32> = childId_3;
        var _S15 : i32 = nodeHeight_0;
        var _S16 : vec3<i32> = vec3<i32>(i32(1));
        for(;;)
        {
            var childT1_0 : vec3<f32> = (vec3<f32>(childPos_0 + vec3<i32>(childNodeSize_1)) - ray_0.mOrigin_0) * invRayDir_1;
            var _S17 : f32 = min(min(childT1_0.x, childT1_0.y), childT1_0.z);
            var childNodeIndex_1 : u32 = GET_NODE_FN_0(_S14, (u32(((((childId_4[i32(0)] & (i32(1)))) | ((((childId_4[i32(1)] & (i32(1)))) << bitcast<u32>(i32(1))))) | ((((childId_4[i32(2)] & (i32(1)))) << bitcast<u32>(i32(2)))))) ^ (rayDirSignBits_1)));
            var _S18 : u32;
            var lastExit_1 : f32;
            var childPos_1 : vec3<i32>;
            var childId_5 : vec3<i32>;
            var _S19 : i32;
            var childNodeSize_2 : i32;
            var _S20 : bool;
            if(childNodeIndex_1 > u32(0))
            {
                var childT0_0 : vec3<f32> = (vec3<f32>(childPos_0) - ray_0.mOrigin_0) * invRayDir_1;
                var _S21 : f32 = max(max(childT0_0.x, childT0_0.y), childT0_0.z);
                var hasLargeFootprint_0 : bool = f32(childNodeSize_1) / _S17 > maxFootprint_0;
                if(childNodeIndex_1 >= u32(256))
                {
                    _S20 = hasLargeFootprint_0;
                }
                else
                {
                    _S20 = false;
                }
                if(_S20)
                {
                    if(_S17 < lastExit_0)
                    {
                        nodeStack_0[_S15] = _S14;
                    }
                    var _S22 : i32 = _S15 - i32(1);
                    var childNodeSize_3 : i32 = childNodeSize_1 / i32(2);
                    var _S23 : vec3<i32> = vec3<i32>(childNodeSize_3);
                    var childId_6 : vec3<i32> = findFirstChild_0(_S21, ray_0.mOrigin_0, invRayDir_1, childPos_0 + _S23);
                    var _S24 : vec3<i32> = childPos_0 + childId_6 * _S23;
                    _S18 = childNodeIndex_1;
                    lastExit_1 = _S17;
                    _S19 = _S22;
                    childPos_1 = _S24;
                    childNodeSize_2 = childNodeSize_3;
                    childId_5 = childId_6;
                }
                else
                {
                    intersection_0.hit_0 = true;
                    intersection_0.distance_0 = _S21;
                    if(computeSurfaceProperties_0)
                    {
                        intersection_0.material_0 = findNearestMaterial_0(childNodeIndex_1, rayDirSignBits_1);
                        intersection_0.normal_0 = vec3<f32>(vec3<f32>(_S21, _S21, _S21) == childT0_0) * _S13;
                    }
                    _S18 = _S14;
                    lastExit_1 = lastExit_0;
                    _S19 = _S15;
                    childPos_1 = childPos_0;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_4;
                }
                _S14 = _S18;
                _S15 = _S19;
            }
            else
            {
                var nextChildFlipsVec_0 : vec3<i32> = vec3<i32>(childT1_0 <= vec3<f32>(_S17, _S17, _S17));
                var childId_7 : vec3<i32> = (childId_4 ^ (nextChildFlipsVec_0));
                var childPos_2 : vec3<i32> = childPos_0 + nextChildFlipsVec_0 * vec3<i32>(childNodeSize_1);
                if(!all(((childId_7 & (nextChildFlipsVec_0))) == nextChildFlipsVec_0))
                {
                    var differingBits_0 : vec3<i32> = (childPos_0 ^ (childPos_2));
                    var msb_0 : u32 = firstLeadingBit(u32(((differingBits_0[i32(0)] | (differingBits_0[i32(1)])) | (differingBits_0[i32(2)]))));
                    var _S25 : i32 = i32(msb_0 + u32(1));
                    var childNodeSize_4 : i32 = (i32(1) << bitcast<u32>(i32(msb_0)));
                    var childId_8 : vec3<i32> = ((childPos_2 >> (vec3<u32>(msb_0))) & (_S16));
                    var _S26 : vec3<u32> = vec3<u32>(u32(_S25));
                    var childPos_3 : vec3<i32> = (((childPos_2 >> (_S26)) << (_S26))) + childId_8 * vec3<i32>(childNodeSize_4);
                    _S18 = nodeStack_0[_S25];
                    lastExit_1 = 0.0f;
                    _S19 = _S25;
                    childPos_1 = childPos_3;
                    childNodeSize_2 = childNodeSize_4;
                    childId_5 = childId_8;
                }
                else
                {
                    _S18 = _S14;
                    lastExit_1 = lastExit_0;
                    _S19 = _S15;
                    childPos_1 = childPos_2;
                    childNodeSize_2 = childNodeSize_1;
                    childId_5 = childId_7;
                }
                _S14 = _S18;
                _S15 = _S19;
            }
            if(intersection_0.hit_0 == false)
            {
                _S20 = _S15 <= nodeHeight_0;
            }
            else
            {
                _S20 = false;
            }
            if(!_S20)
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
    var rayDirSignBitsAsVec_0 : vec3<i32> = vec3<i32>(ray_1.mDir_0 < vec3<f32>(0.0f, 0.0f, 0.0f));
    var rayDirSignBits_2 : u32 = u32(((rayDirSignBitsAsVec_0[i32(0)] | ((rayDirSignBitsAsVec_0[i32(1)] << bitcast<u32>(i32(1))))) | ((rayDirSignBitsAsVec_0[i32(2)] << bitcast<u32>(i32(2))))));
    var rayDirSign_1 : vec3<f32> = vec3<f32>(rayDirSignBitsAsVec_0 * vec3<i32>(i32(-2)) + vec3<i32>(i32(1)));
    var reflectedRay_0 : Ray3f_0 = ray_1;
    reflectedRay_0.mOrigin_0 = (reflectedRay_0.mOrigin_0 + vec3<f32>(0.5f, 0.5f, 0.5f)) * rayDirSign_1;
    reflectedRay_0.mDir_0 = abs(reflectedRay_0.mDir_0);
    var nodeSize_1 : i32 = i32((u32(1) << (u32(subDAG_0.nodeHeight_1))));
    var intersection_1 : RayVolumeIntersection_0 = intersectRayNodeESVO_0(subDAG_0.nodeIndex_2, subDAG_0.lowerBound_0 * vec3<i32>(rayDirSign_1) - rayDirSignBitsAsVec_0 * vec3<i32>(nodeSize_1, nodeSize_1, nodeSize_1), subDAG_0.nodeHeight_1, reflectedRay_0, rayDirSign_1, rayDirSignBits_2, computeSurfaceProperties_1, maxFootprint_1);
    if(intersection_1.hit_0)
    {
        intersection_1.position_0 = ray_1.mOrigin_0 + ray_1.mDir_0 * vec3<f32>(intersection_1.distance_0);
        return intersection_1;
    }
    return intersection_1;
}

fn compute_shading_0( intersection_2 : RayVolumeIntersection_0) -> vec3<f32>
{
    var palette_0 : array<vec3<f32>, i32(2)> = array<vec3<f32>, i32(2)>( vec3<f32>(1.0f), vec3<f32>(1.0f, 0.0f, 0.0f) );
    return palette_0[intersection_2.material_0 - u32(1)] * vec3<f32>(max(dot(unpackStorage_1(globalParams_0.uniforms_0).sun_direction_0, intersection_2.normal_0), 0.0f));
}

fn trace_0( ray_2 : Ray3f_0) -> vec3<f32>
{
    var subDAG_1 : SubDAG_0;
    subDAG_1.lowerBound_0[i32(0)] = i32(0);
    subDAG_1.lowerBound_0[i32(1)] = i32(0);
    subDAG_1.lowerBound_0[i32(2)] = i32(0);
    subDAG_1.nodeHeight_1 = i32(6);
    subDAG_1.nodeIndex_2 = u32(261);
    var intersection_3 : RayVolumeIntersection_0 = intersect_subdag_0(ray_2, subDAG_1, true, 0.00350000010803342f);
    if(intersection_3.hit_0)
    {
        return compute_shading_0(intersection_3);
    }
    return vec3<f32>(abs(sin(ray_2.mDir_0.y * 10.0f)));
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) dispatch_thread_id_0 : vec3<u32>)
{
    var _S27 : vec2<u32> = dispatch_thread_id_0.xy;
    var _S28 : vec2<f32> = (vec2<f32>(_S27) + vec2<f32>(0.5f)) / vec2<f32>(unpackStorage_1(globalParams_0.uniforms_0).resolution_0);
    var TexCoords_0 : vec2<f32> = _S28;
    TexCoords_0[i32(1)] = 1.0f - _S28.y;
    var ray_3 : Ray3f_0;
    ray_3.mOrigin_0 = unpackStorage_1(globalParams_0.uniforms_0).cameraPos_0;
    var _S29 : SLANG_anonymous_0_0 = unpackStorage_1(globalParams_0.uniforms_0);
    var dirEye_0 : vec4<f32> = (((vec4<f32>(vec3<f32>(TexCoords_0 * vec2<f32>(2.0f) - vec2<f32>(1.0f), -1.0f), 1.0f)) * (unpackStorage_1(globalParams_0.uniforms_0).PInv_0)));
    dirEye_0[i32(3)] = 0.0f;
    ray_3.mDir_0 = normalize((((dirEye_0) * (_S29.VInv_0))).xyz);
    textureStore((entryPointParams_image_0), (_S27), vec4<f32>((trace_0(ray_3)), 1));
    return;
}

