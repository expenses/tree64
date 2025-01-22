use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark(c: &mut Criterion) {
    let vox_filename = "sponza.vox";

    let vox = dot_vox::load(&vox_filename).unwrap();

    //assert_eq!(vox.models.len(), 1, "Expected 1 model");

    let model = &vox.models[0];

    let mut array = vec![0; model.size.x as usize * model.size.y as usize * model.size.z as usize];

    for voxel in &model.voxels {
        array[voxel.x as usize
            + voxel.y as usize * model.size.x as usize
            + voxel.z as usize * model.size.x as usize * model.size.y as usize] = voxel.i;
    }

    c.bench_function("new_iterative", |b| {
        b.iter(|| {
            let tree =
                tree64::Tree64::new((&array[..], [model.size.x, model.size.y, model.size.z]));
            black_box(tree);
        })
    });
    c.bench_function("new_recursive", |b| {
        b.iter(|| {
            let tree = tree64::Tree64::new_recursive((
                &array[..],
                [model.size.x, model.size.y, model.size.z],
            ));
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
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
