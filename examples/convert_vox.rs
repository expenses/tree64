fn swizzle(v: glam::IVec3, indices: glam::IVec3) -> glam::IVec3 {
    let indices = indices.abs() - 1;
    glam::IVec3::new(
        v[indices.x as usize],
        v[indices.y as usize],
        v[indices.z as usize],
    )
}

fn merge_vox_models(vox: dot_vox::DotVoxData) -> (Vec<u8>, glam::UVec3) {
    dbg!(vox.models.len());

    let mut stack = vec![(0, glam::IVec3::ZERO, dot_vox::Rotation::IDENTITY)];
    let mut models_and_positions = Vec::new();

    while let Some((index, translation, rotation)) = stack.pop() {
        match &vox.scenes[index as usize] {
            dot_vox::SceneNode::Transform { child, frames, .. } => {
                // In case of a Transform node, the potential translation and rotation is added
                // to the global transform to all of the nodes children nodes
                let translation = if let Some(t) = frames[0].attributes.get("_t") {
                    let translation_delta = t
                        .split(" ")
                        .map(|x| x.parse().expect("Not an integer!"))
                        .collect::<Vec<i32>>();
                    debug_assert_eq!(translation_delta.len(), 3);
                    translation
                        + glam::IVec3::new(
                            translation_delta[0],
                            translation_delta[1],
                            translation_delta[2],
                        )
                } else {
                    translation
                };
                let rotation = if let Some(r) = frames[0].attributes.get("_r") {
                    rotation
                        * dot_vox::Rotation::from_byte(
                            r.parse()
                                .expect("Expected valid u8 byte to parse rotation matrix"),
                        )
                } else {
                    rotation
                };

                stack.push((*child, translation, rotation));
            }
            dot_vox::SceneNode::Group { children, .. } => {
                for &child in children {
                    stack.push((child, translation, rotation));
                }
            }
            dot_vox::SceneNode::Shape { models, .. } => {
                let model = models[0].model_id;
                dbg!(models[0].model_id, translation);
                let model = &vox.models[model as usize];

                let channel_reordering =
                    (glam::Mat3::from_cols_array_2d(&rotation.to_cols_array_2d())
                        * glam::Vec3::new(1.0, 2.0, 3.0))
                    .as_ivec3();

                let size = swizzle(
                    glam::UVec3::new(model.size.x, model.size.y, model.size.z).as_ivec3(),
                    channel_reordering,
                )
                .as_uvec3();

                models_and_positions.push((translation, size, rotation, model));
            }
        }
    }

    let mut min = glam::IVec3::splat(i32::MAX);
    let mut max = glam::IVec3::splat(i32::MIN);

    for &(pos, model_size, ..) in &models_and_positions {
        min = min.min(pos - (model_size / 2).as_ivec3());
        max = max.max(pos + (model_size / 2).as_ivec3());
    }

    let size = max - min + 1;

    let mut array = vec![0; size.x as usize * size.y as usize * size.z as usize];

    for (pos, model_size, rotation, model) in models_and_positions {
        let offset = pos - min - (model_size / 2).as_ivec3();

        let channel_reordering = (glam::Mat3::from_cols_array_2d(&rotation.to_cols_array_2d())
            * glam::Vec3::new(1.0, 2.0, 3.0))
        .as_ivec3();
        let model_size_swizzled = swizzle(model_size.as_ivec3(), channel_reordering.abs());

        for voxel in &model.voxels {
            let voxel_pos = glam::IVec3::new(voxel.x as i32, voxel.y as i32, voxel.z as i32);

            let mut voxel_pos_swizzled = swizzle(voxel_pos, channel_reordering);

            for i in 0..3 {
                if channel_reordering[i] < 0 {
                    voxel_pos_swizzled[i] = model_size_swizzled[i] - 1 - voxel_pos_swizzled[i];
                }
            }

            let voxel_pos = offset + voxel_pos_swizzled;
            array[voxel_pos.x as usize
                + voxel_pos.y as usize * size.x as usize
                + voxel_pos.z as usize * size.x as usize * size.y as usize] = voxel.i;
        }
    }

    (array, size.as_uvec3())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let vox_filename = args.next().unwrap();
    let tree_filename = args.next().unwrap();

    let (array, size) = merge_vox_models(dot_vox::load(&vox_filename).unwrap());

    let tree = tree64::Tree64::new((&array[..], size.into()));

    tree.serialize(std::fs::File::create(&tree_filename).unwrap())
        .unwrap();

    #[cfg(feature = "caching")]
    dbg!(tree.stats);
    dbg!(tree.data.len());
    dbg!(tree.nodes.len());
}
