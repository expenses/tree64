#![no_main]

use libfuzzer_sys::fuzz_target;

#[derive(arbitrary::Arbitrary, Debug)]
pub struct Array<'a> {
    pub size: [u32; 3],
    modifications: Vec<([i32; 3], [i32; 3], u8)>,
    array: &'a [u8],
}

fuzz_target!(|data: Array| {
    let size = glam::UVec3::from(data.size);

    if size.cmpgt(glam::UVec3::splat(256)).any() {
        return;
    }

    let mut tree = tree64::Tree64::new((&data.array[..], size.into()));

    for (min, max, value) in data.modifications {
        let min = glam::IVec3::from(min);
        let max = glam::IVec3::from(max);

        let check = |val: glam::IVec3| {
            val.cmple(glam::IVec3::splat(i32::MIN / 2)).any()
                || val.cmpgt(glam::IVec3::splat(i32::MAX / 2)).any()
        };

        if check(min) || check(max) {
            continue;
        }

        if (max - min).cmpgt(glam::IVec3::splat(500)).any() {
            continue;
        }

        let ranges = tree.modify_nodes_in_box(min.into(), max.into(), value);
    }
});
