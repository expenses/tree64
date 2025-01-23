use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn benchmark(c: &mut Criterion) {
    let (array, size) = merge_vox_models(dot_vox::load("sponza.vox").unwrap());

    c.bench_function("new_iterative", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new_iterative((&array[..], size.into()));
            black_box(tree);
        })
    });
    c.bench_function("new_recursive", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new((&array[..], size.into()));
            black_box(tree);
        })
    });
    c.bench_function("new_iterative_empty", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new_iterative(EmptyVoxModel([1024; 3], false));
            black_box(tree);
        })
    });
    c.bench_function("new_recursive_empty", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new(EmptyVoxModel([1024; 3], false));
            black_box(tree);
        })
    });
    c.bench_function("new_iterative_solid", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new_iterative(EmptyVoxModel([1024; 3], true));
            black_box(tree);
        })
    });
    c.bench_function("new_recursive_sold", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new(EmptyVoxModel([1024; 3], true));
            black_box(tree);
        })
    });
    c.bench_function("billion_voxel_deletion", |b| {
        b.iter(|| {
            let empty_slice: &[u8] = &[];
            let mut tree = tree64::Tree64::new((empty_slice, [0; 3]));
            tree.modify_nodes_in_box([0; 3], [1024; 3], Some(1));
            let range = tree.modify_nodes_in_box([1; 3], [1024 - 1; 3], None);
            black_box(range)
        })
    });

    let (array, size) = merge_vox_models(dot_vox::load("church.vox").unwrap());

    c.bench_function("church_iterative", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new_iterative((&array[..], size.into()));
            black_box(tree);
        })
    });
    c.bench_function("church_recursive", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new((&array[..], size.into()));
            black_box(tree);
        })
    });
}

struct EmptyVoxModel([u32; 3], bool);

impl tree64::VoxelModel<()> for EmptyVoxModel {
    fn dimensions(&self) -> [u32; 3] {
        self.0
    }

    fn access(&self, _coord: [usize; 3]) -> Option<()> {
        if self.1 {
            Some(())
        } else {
            None
        }
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
