use std::fmt::Debug;
use std::hash::BuildHasher;
use std::io::{self, Write};
use std::ops::{Add, Sub};

pub trait NodeValue:
    PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + TryInto<usize, Error: Debug>
    + TryFrom<usize, Error: Debug>
    + From<u8>
    + Copy
    + std::hash::Hash
    + Eq
    + bytemuck::Pod
    + bytemuck::Zeroable
    + Debug
{
}

impl<
        T: PartialOrd
            + Add<Output = Self>
            + Sub<Output = T>
            + TryInto<usize, Error: Debug>
            + TryFrom<usize, Error: Debug>
            + From<u8>
            + Copy
            + std::hash::Hash
            + Eq
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Debug,
    > NodeValue for T
{
}

pub struct EditableSvoDag<T> {
    inner: SvoDag<T>,
    indices: hashbrown::HashTable<T>,
}

impl<T: NodeValue> EditableSvoDag<T> {
    pub fn into_read_only(self) -> SvoDag<T> {
        self.inner
    }

    pub fn new(
        slice: &[u8],
        width: usize,
        height: usize,
        depth: usize,
        reserved_indices: T,
    ) -> Self {
        let size = width.max(height).max(depth).next_power_of_two();

        let access = |x, y, z| {
            T::from(
                slice
                    .get(x + y * width + z * width * height)
                    .copied()
                    .unwrap_or(0),
            )
        };

        let mut cached_indices: hashbrown::HashTable<T> = hashbrown::HashTable::new();

        let hasher = fnv::FnvBuildHasher::default();

        let mut layer: Vec<T> = Vec::with_capacity((size * size * size) / 8);

        let mut nodes: Vec<Node<T>> = Vec::new();

        let mut insert_node = |node: Node<T>, level_size: usize| match node.uniform_value() {
            Some(value) if value < reserved_indices && level_size > 2 => value,
            _ => {
                *cached_indices
                    .entry(
                        hasher.hash_one(&node),
                        |&other| nodes[other.try_into().unwrap()] == node,
                        |&index| hasher.hash_one(&nodes[index.try_into().unwrap()]),
                    )
                    .or_insert_with(|| {
                        let next_index = T::try_from(nodes.len()).unwrap();
                        nodes.push(node);
                        next_index
                    })
                    .get()
                    + reserved_indices
            }
        };

        for z in (0..size).step_by(2) {
            for y in (0..size).step_by(2) {
                for x in (0..size).step_by(2) {
                    layer.push(insert_node(
                        Node([
                            access(x, y, z),
                            access(x + 1, y, z),
                            access(x, y + 1, z),
                            access(x + 1, y + 1, z),
                            access(x, y, z + 1),
                            access(x + 1, y, z + 1),
                            access(x, y + 1, z + 1),
                            access(x + 1, y + 1, z + 1),
                        ]),
                        size,
                    ));
                }
            }
        }
        debug_assert_eq!(layer.capacity(), (size * size * size) / 8);

        let mut level_size = size / 2;

        let mut prev_layer = layer;
        let mut current_layer = Vec::with_capacity((level_size * level_size * level_size) / 8);

        while level_size > 1 {
            let access = |x, y, z| prev_layer[x + y * level_size + z * level_size * level_size];

            for z in (0..level_size).step_by(2) {
                for y in (0..level_size).step_by(2) {
                    for x in (0..level_size).step_by(2) {
                        current_layer.push(insert_node(
                            Node([
                                access(x, y, z),
                                access(x + 1, y, z),
                                access(x, y + 1, z),
                                access(x + 1, y + 1, z),
                                access(x, y, z + 1),
                                access(x + 1, y, z + 1),
                                access(x, y + 1, z + 1),
                                access(x + 1, y + 1, z + 1),
                            ]),
                            level_size,
                        ));
                    }
                }
            }

            level_size /= 2;
            std::mem::swap(&mut prev_layer, &mut current_layer);
            current_layer.clear();
        }

        Self {
            inner: SvoDag {
                nodes,
                size: size as _,
                reserved_indices,
            },
            indices: cached_indices,
        }
    }
}

pub struct SvoDag<T> {
    size: u32,
    nodes: Vec<Node<T>>,
    reserved_indices: T,
}

impl<T: NodeValue> SvoDag<T> {
    pub fn new(
        slice: &[u8],
        width: usize,
        height: usize,
        depth: usize,
        reserved_indices: T,
    ) -> Self {
        EditableSvoDag::new(slice, width, height, depth, reserved_indices).into_read_only()
    }

    // todo: modifications.
    /*
    pub fn touch(&mut self, mut x: u32, mut y: u32, mut z: u32) {
        let old_root = self.nodes.last().copied().unwrap();
        self.nodes.add_node(old_root);

    }
    */

    fn root(&self) -> Node<T> {
        *self.nodes.last().unwrap()
    }

    #[allow(dead_code)]
    fn add_node(&mut self, node: Node<T>) -> T {
        let next_index = T::try_from(self.nodes.len()).unwrap();
        self.nodes.push(node);
        next_index + self.reserved_indices
    }

    pub fn index(&self, mut x: u32, mut y: u32, mut z: u32) -> T {
        let mut node = self.root();
        let mut node_size = self.size;

        loop {
            let child_index = get_child_index(node_size, x, y, z);
            let child = node.0[child_index];

            if child < self.reserved_indices {
                return child;
            }

            node_size /= 2;
            node = self.nodes[(child - self.reserved_indices).try_into().unwrap()];
            x %= node_size;
            y %= node_size;
            z %= node_size;
        }
    }

    pub fn serialize<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(b"DAG")?;
        writer.write_all(&(std::mem::size_of::<T>() as u8).to_le_bytes())?;
        writer.write_all(&(self.reserved_indices.try_into().unwrap() as u32).to_le_bytes())?;
        writer.write_all(&self.size.to_le_bytes())?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        writer.write_all(bytemuck::cast_slice(&self.nodes))?;
        Ok(())
    }

    pub fn cubes(&self) -> (Vec<[f32; 3]>, Vec<u32>, Vec<T>) {
        let mut stack = vec![(
            T::try_from(self.nodes.len() - 1).unwrap() + self.reserved_indices,
            self.size,
            0,
        )];

        let mut positions = Vec::with_capacity(self.nodes.len() / 2);
        let mut sizes = Vec::with_capacity(self.nodes.len() / 2);
        let mut values = Vec::with_capacity(self.nodes.len() / 2);

        while let Some((value, size, index)) = stack.pop() {
            if value == T::from(0) {
                continue;
            }

            let x = index % self.size;
            let y = (index / self.size) % self.size;
            let z = index / self.size / self.size;

            if value < self.reserved_indices {
                values.push(value);
                sizes.push(size);
                positions.push([x as _, y as _, z as _]);
            } else {
                let size = size / 2;
                let node = self.nodes[(value - self.reserved_indices).try_into().unwrap()];
                for (i, value) in node.0.iter().copied().enumerate() {
                    let x = x + (i % 2 == 1) as u32 * size;
                    let y = y + (i % 4 > 1) as u32 * size;
                    let z = z + (i > 3) as u32 * size;
                    let index = z * self.size * self.size + y * self.size + x;
                    stack.push((value, size, index));
                }
            }
        }

        (positions, sizes, values)
    }

    pub fn node_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.nodes)
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_levels(&self) -> u8 {
        self.size.ilog2() as u8
    }
}

fn get_child_index(size: u32, x: u32, y: u32, z: u32) -> usize {
    let half_size = size / 2;
    (z >= half_size) as usize * 4 + (y >= half_size) as usize * 2 + (x >= half_size) as usize
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
struct Node<T>([T; 8]);

impl<T: NodeValue> Node<T> {
    fn uniform_value(self) -> Option<T> {
        if self.0[0] == self.0[1]
            && self.0[0] == self.0[2]
            && self.0[0] == self.0[3]
            && self.0[0] == self.0[4]
            && self.0[0] == self.0[5]
            && self.0[0] == self.0[6]
            && self.0[0] == self.0[7]
        {
            Some(self.0[0])
        } else {
            None
        }
    }
}

#[test]
fn uniform_cubes() {
    assert_eq!(
        SvoDag::new(&[1; 2 * 2 * 2], 2, 2, 2, 256).nodes,
        vec![Node([1; 8])]
    );
    assert_eq!(
        SvoDag::new(&[1; 4 * 4 * 4], 4, 4, 4, 256).nodes,
        vec![Node([1; 8])]
    );
    assert_eq!(
        SvoDag::new(&[1; 8 * 8 * 8], 8, 8, 8, 256).nodes,
        vec![Node([1; 8])]
    );
    assert_eq!(
        SvoDag::new(&[1; 16 * 16 * 16], 16, 16, 16, 256).nodes,
        vec![Node([1; 8])]
    );
    assert_eq!(
        SvoDag::new(&[2; 16 * 16 * 16], 16, 16, 16, 256).nodes,
        vec![Node([2; 8])]
    );
}

#[test]
fn non_power_of_2() {
    assert_eq!(
        SvoDag::new(&[0; 9 * 9 * 9], 9, 9, 9, 256).nodes,
        vec![Node([0; 8])]
    );
    assert_eq!(
        SvoDag::new(&[1; 8 * 8 * 7], 8, 8, 7, 256).nodes,
        vec![
            Node([1, 1, 1, 1, 0, 0, 0, 0]),
            Node([1, 1, 1, 1, 256, 256, 256, 256]),
            Node([1, 1, 1, 1, 257, 257, 257, 257])
        ]
    );
}

#[test]
fn different_corners() {
    {
        let mut array = [1; 2 * 2 * 2];
        array[0] = 0;
        assert_eq!(
            SvoDag::new(&array, 2, 2, 2, 256).nodes,
            vec![Node(array.map(|v| v as u32))]
        );
    }
    {
        let mut array = [1; 4 * 4 * 4];
        array[0] = 0;
        assert_eq!(
            SvoDag::new(&array, 4, 4, 4, 256).nodes,
            vec![
                Node([0, 1, 1, 1, 1, 1, 1, 1]),
                Node([256, 1, 1, 1, 1, 1, 1, 1])
            ]
        );
    }
    {
        let mut array = [1; 8 * 8 * 8];
        array[0] = 0;
        assert_eq!(
            SvoDag::new(&array, 8, 8, 8, 256).nodes,
            vec![
                Node([0, 1, 1, 1, 1, 1, 1, 1]),
                Node([256, 1, 1, 1, 1, 1, 1, 1]),
                Node([257, 1, 1, 1, 1, 1, 1, 1])
            ]
        );
    }
}

#[test]
fn cubes() {
    let mut array = [1; 4 * 4 * 4];
    array[0] = 0;
    let svo_dag = SvoDag::new(&array, 4, 4, 4, 256);
    dbg!(&svo_dag.nodes);
    assert_eq!(svo_dag.cubes(), (vec![], vec![], vec![]));
}

#[test]
fn no_join_uniform_pointer_nodes() {
    let mut array = [0; 8 * 8 * 8];
    for z in (0..8).step_by(2) {
        dbg!(z);
        array[z * 8 * 8..(z + 1) * 8 * 8].copy_from_slice(&[1; 8 * 8]);
    }

    assert_eq!(
        SvoDag::new(&array, 8, 8, 8, 256).nodes,
        [
            Node([1, 1, 1, 1, 0, 0, 0, 0]),
            Node([256, 256, 256, 256, 256, 256, 256, 256]),
            Node([257, 257, 257, 257, 257, 257, 257, 257])
        ]
    );
}

#[test]
fn paper_example() {
    #[rustfmt::skip]
    let flat_array = [
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,
        1,1,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,
        1,1,1,1,0,0,0,0,
        1,1,1,1,1,0,0,0,
    ];
    let mut full_array = [0; 8 * 8 * 8];
    for chunk in full_array.chunks_exact_mut(8 * 8) {
        chunk.copy_from_slice(&flat_array);
    }

    let dag = SvoDag::new(&full_array, 8, 8, 1, 256);
    assert_eq!(
        dag.nodes,
        [
            Node([0, 0, 1, 0, 0, 0, 1, 0]),
            Node([0, 0, 256, 0, 0, 0, 256, 0]),
            Node([1, 256, 1, 1, 1, 256, 1, 1]),
            Node([257, 0, 258, 257, 257, 0, 258, 257])
        ]
    );
}

#[test]
fn basic_indexing() {
    let mut array = [1; 8 * 8 * 8];
    array[0] = 0;
    let dag = SvoDag::new(&array, 8, 8, 8, 256);

    for z in 0..8 {
        for y in 0..8 {
            for x in 0..y {
                assert_eq!(
                    dag.index(x, y, z),
                    if (x, y, z) == (0, 0, 0) { 0 } else { 1 }
                );
            }
        }
    }
}

#[test]
fn advanced_indexing() {
    {
        let mut array = [1; 8 * 8 * 8];
        array[(8 * 8 * 8) - 1] = 0;
        let dag = SvoDag::new(&array, 8, 8, 8, 256);
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(
                        dag.index(x, y, z),
                        if (x, y, z) == (7, 7, 7) { 0 } else { 1 }
                    );
                }
            }
        }
    }

    {
        let mut array = [1; 64 * 64 * 64];
        array[4 + 5 * 64 + 6 * 64 * 64] = 0;
        let dag = SvoDag::new(&array, 64, 64, 64, 256);
        for z in 0..64 {
            for y in 0..64 {
                for x in 0..64 {
                    assert_eq!(
                        dag.index(x, y, z),
                        if (x, y, z) == (4, 5, 6) { 0 } else { 1 }
                    );
                }
            }
        }
    }
}
