fn main() {
    let mut args = std::env::args().skip(1);
    let vox_filename = args.next().unwrap();
    let tree_filename = args.next().unwrap();

    let vox = dot_vox::load(&vox_filename).unwrap();

    //assert_eq!(vox.models.len(), 1, "Expected 1 model");

    let model = &vox.models[0];

    let mut array = vec![0; model.size.x as usize * model.size.y as usize * model.size.z as usize];

    for voxel in &model.voxels {
        array[voxel.x as usize
            + voxel.y as usize * model.size.x as usize
            + voxel.z as usize * model.size.x as usize * model.size.y as usize] = voxel.i;
    }

    let tree = tree64::Tree64::new((
        &array[..],
        [model.size.x as _, model.size.y as _, model.size.z as _],
    ));

    tree.serialize(std::fs::File::create(&tree_filename).unwrap())
        .unwrap();

    dbg!(tree.stats);
}
