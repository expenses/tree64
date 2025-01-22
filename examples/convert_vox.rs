fn merge_vox_models(vox: dot_vox::DotVoxData) -> (Vec<u8>, glam::UVec3) {
    dbg!(vox.models.len());

    let root_transform = &vox.scenes[0];
    let root_group = &vox.scenes[match root_transform {
        dot_vox::SceneNode::Transform { child, .. } => *child as usize,
        _ => panic!(),
    }];
    let children = match root_group {
        dot_vox::SceneNode::Group { children, .. } => children,
        _ => panic!(),
    };

    let mut models_and_positions: Vec<(glam::IVec3, glam::UVec3, &dot_vox::Model)> = Vec::new();

    for child in children {
        let (child, frames) = match &vox.scenes[*child as usize] {
            dot_vox::SceneNode::Transform { frames, child, .. } => (child, frames),
            _ => panic!(),
        };

        let translation = &frames[0].attributes["_t"];

        let mut splits = translation.split(' ');

        let x = splits.next().unwrap().parse::<i32>().unwrap();
        let y = splits.next().unwrap().parse::<i32>().unwrap();
        let z = splits.next().unwrap().parse::<i32>().unwrap();

        let model = match &vox.scenes[*child as usize] {
            dot_vox::SceneNode::Shape { models, .. } => models[0].model_id,
            _ => panic!(),
        };

        let model = &vox.models[model as usize];

        let size = glam::UVec3::new(model.size.x, model.size.y, model.size.z);

        models_and_positions.push(((x, y, z).into(), size, model));
    }

    let mut min = glam::IVec3::splat(i32::MAX);
    let mut max = glam::IVec3::splat(i32::MIN);

    for &(pos, model_size, _) in &models_and_positions {
        min = min.min(pos - (model_size / 2).as_ivec3());
        max = max.max(pos + (model_size / 2).as_ivec3());
    }

    let size = max - min + 1;

    let mut array = vec![0; size.x as usize * size.y as usize * size.z as usize];

    for (pos, model_size, model) in models_and_positions {
        let offset = pos - min - (model_size / 2).as_ivec3();

        for voxel in &model.voxels {
            let voxel_pos =
                offset + glam::IVec3::new(voxel.x as i32, voxel.y as i32, voxel.z as i32);
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

    let vox = dot_vox::load(&vox_filename).unwrap();

    let (array, size) = merge_vox_models(vox);

    let tree = tree64::Tree64::new((&array[..], size.into()));

    tree.serialize(std::fs::File::create(&tree_filename).unwrap())
        .unwrap();

    dbg!(tree.stats);
}
