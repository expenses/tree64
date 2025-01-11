use std::fmt::Debug;
use std::hash::BuildHasher;
use std::io::{self, Read, Write};

pub struct EditableSvoDag {
    inner: SvoDag,
    #[allow(dead_code)]
    indices: hashbrown::HashTable<u32>,
}

impl EditableSvoDag {
    pub fn into_read_only(self) -> SvoDag {
        self.inner
    }

    pub fn new(
        slice: &[u8],
        width: usize,
        height: usize,
        depth: usize,
        reserved_indices: u32,
    ) -> Self {
        let size = width.max(height).max(depth).next_power_of_two();

        let access = |x, y, z| match slice.get(x + y * width + z * width * height) {
            Some(&value) if x < width && y < height => value as u32,
            _ => 0,
        };

        let mut cached_indices: hashbrown::HashTable<u32> = hashbrown::HashTable::new();

        let hasher = fnv::FnvBuildHasher::default();

        let mut layer: Vec<u32> = Vec::with_capacity((size * size * size) / 8);

        let mut nodes: Vec<Node<u32>> = Vec::new();

        let mut insert_node = |node: Node<u32>, level_size: usize| match node.uniform_value() {
            Some(value) if value < reserved_indices && level_size > 2 => value,
            _ => {
                *cached_indices
                    .entry(
                        hasher.hash_one(node),
                        |&other| nodes[other as usize] == node,
                        |&index| hasher.hash_one(nodes[index as usize]),
                    )
                    .or_insert_with(|| {
                        let next_index = nodes.len() as u32;
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

pub struct SvoDag {
    size: u32,
    nodes: Vec<Node<u32>>,
    reserved_indices: u32,
}

impl SvoDag {
    pub fn new(
        slice: &[u8],
        width: usize,
        height: usize,
        depth: usize,
        reserved_indices: u32,
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

    pub fn as_array(&self) -> Vec<u8> {
        let size = self.size as usize;
        let mut array = vec![0; size * size * size];
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    array[x + y * size + z * size * size] =
                        self.index(x as _, y as _, z as _) as u8;
                }
            }
        }

        array
    }

    fn root(&self) -> Node<u32> {
        *self.nodes.last().unwrap()
    }

    #[allow(dead_code)]
    fn add_node(&mut self, node: Node<u32>) -> u32 {
        let next_index = self.nodes.len() as u32;
        self.nodes.push(node);
        next_index + self.reserved_indices
    }

    pub fn index(&self, mut x: u32, mut y: u32, mut z: u32) -> u32 {
        let mut node = self.root();
        let mut node_size = self.size;

        loop {
            let child_index = get_child_index(node_size, x, y, z);
            let child = node.0[child_index];

            if child < self.reserved_indices {
                return child;
            }

            node_size /= 2;
            node = self.nodes[(child - self.reserved_indices) as usize];
            x %= node_size;
            y %= node_size;
            z %= node_size;
        }
    }

    pub fn serialize<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(b"DAG")?;
        writer.write_all(&(std::mem::size_of::<u32>() as u8).to_le_bytes())?;
        writer.write_all(&(self.reserved_indices).to_le_bytes())?;
        writer.write_all(&self.size.to_le_bytes())?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        writer.write_all(bytemuck::cast_slice(&self.nodes))?;
        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut buffer = [0; 3];
        reader.read_exact(&mut buffer)?;
        assert_eq!(&buffer, b"DAG");
        let mut size = 0_u8;
        reader.read_exact(bytemuck::bytes_of_mut(&mut size))?;
        assert_eq!(size as usize, std::mem::size_of::<u32>());
        let mut reserved_indices = 0_u32;
        let mut size = 0_u32;
        let mut num_nodes = 0_u32;
        reader.read_exact(bytemuck::bytes_of_mut(&mut reserved_indices))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut size))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_nodes))?;
        let mut nodes = vec![Node([0; 8]); num_nodes as usize];
        reader.read_exact(bytemuck::cast_slice_mut(&mut nodes))?;
        Ok(Self {
            size,
            nodes,
            reserved_indices,
        })
    }

    pub fn cubes(&self) -> (Vec<[f32; 3]>, Vec<u32>, Vec<u32>) {
        let mut stack = vec![(
            self.nodes.len() as u32 - 1 + self.reserved_indices,
            self.size,
            0,
        )];

        let mut positions = Vec::with_capacity(self.nodes.len() / 2);
        let mut sizes = Vec::with_capacity(self.nodes.len() / 2);
        let mut values = Vec::with_capacity(self.nodes.len() / 2);

        while let Some((value, size, index)) = stack.pop() {
            if value == 0 {
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
                let node = self.nodes[(value - self.reserved_indices) as usize];
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

    pub fn nodes(&self) -> &[Node<u32>] {
        &self.nodes
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_levels(&self) -> u8 {
        self.size.ilog2() as u8
    }

    pub fn size(&self) -> u32 {
        self.size
    }
}

fn get_child_index(size: u32, x: u32, y: u32, z: u32) -> usize {
    let half_size = size / 2;
    (z >= half_size) as usize * 4 + (y >= half_size) as usize * 2 + (x >= half_size) as usize
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Node<T>([T; 8]);

impl<T: PartialEq + Copy> Node<T> {
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
    assert_eq!(svo_dag.cubes().0.len(), 7 + 7);
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

#[derive(Default, Debug)]
pub struct Tree64Stats {
    value_cache_hits: usize,
    value_cache_bytes_saved: usize,
    value_search_hits: usize,
    value_search_bytes_saved: usize,
    value_overlapping_saved: usize,
    node_search_hits: usize,
    node_search_bytes_saved: usize,
    nodes_overlapping_saved: usize,
}

#[repr(C, packed)]
#[derive(Default, Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Tree64Node {
    pub is_leaf_and_ptr: u32,
    pub pop_mask: u64,
}

#[derive(Debug)]
pub struct Tree64 {
    pub nodes: Vec<Tree64Node>,
    pub data: Vec<u8>,
    pub scale: u8,
    stats: Tree64Stats,
    indices: hashbrown::HashTable<std::ops::Range<u32>>,
}

#[test]
fn test_tree_node_size() {
    assert_eq!(std::mem::size_of::<Tree64Node>(), 4 + 8);
}

impl Tree64 {
    fn insert_values(&mut self, values: &[u8]) -> u32 {
        if values.is_empty() {
            return 0;
        }

        let hasher = fnv::FnvBuildHasher::default();

        let mut hit_cache = true;

        let index = self
            .indices
            .entry(
                hasher.hash_one(values),
                |range| &self.data[range.start as usize..range.end as usize] == values,
                |range| hasher.hash_one(&self.data[range.start as usize..range.end as usize]),
            )
            .or_insert_with(|| {
                hit_cache = false;

                let start = /*if let Some(index) = memchr::memmem::find(&self.data, values) {
                    self.stats.value_search_hits += 1;
                    self.stats.value_search_bytes_saved += values.len();
                    index
                } else*/ {
                    let old_index = self.data.len();
                    let index = extend_overlapping(&mut self.data, values);

                    self.stats.value_overlapping_saved +=
                        values.len() - (self.data.len() - old_index);

                    index
                } as u32;

                start..start + values.len() as u32
            })
            .get()
            .start;

        if hit_cache {
            self.stats.value_cache_hits += 1;
            self.stats.value_cache_bytes_saved += values.len();
        }

        index
    }

    fn insert_nodes(&mut self, nodes: &[Tree64Node]) -> u32 {
        if nodes.is_empty() {
            return 0;
        }

        /*if let Some(index) = memchr::memmem::find(
            bytemuck::cast_slice(&self.nodes),
            bytemuck::cast_slice(nodes),
        ) {
            self.stats.node_search_hits += 1;
            self.stats.node_search_bytes_saved += nodes.len() * std::mem::size_of::<Tree64Node>();

            //dbg!(index, index % std::mem::size_of::<Tree64Node>());
            if index % std::mem::size_of::<Tree64Node>() == 0 {
                return (index / std::mem::size_of::<Tree64Node>()) as u32;
            }
            panic!();
            //return index as u32;
        }*/

        let old_index = self.nodes.len();
        let index = extend_overlapping(&mut self.nodes, nodes) as u32;

        self.stats.nodes_overlapping_saved += nodes.len() - (self.nodes.len() - old_index);
        index
    }

    pub fn new(array: &[u8], dims: [usize; 3]) -> Self {
        let mut scale = dims[0].max(dims[1]).max(dims[2]).next_power_of_two() as u32;
        if scale.ilog2() % 2 == 1 {
            scale *= 2;
        }
        let mut this = Self {
            scale: dbg!(scale.ilog2() as _),
            nodes: Default::default(),
            data: Default::default(),
            stats: Default::default(),
            indices: Default::default(),
        };
        this.scale = scale as _;
        let root = this.insert(array, dims, [0; 3], scale);
        this.nodes.insert(0, root);
        dbg!(&this.stats);
        println!("{:?}", this.stats);
        this
    }

    fn insert(
        &mut self,
        array: &[u8],
        dims: [usize; 3],
        offset: [usize; 3],
        scale: u32,
    ) -> Tree64Node {
        let access = |mut x, mut y, mut z| {
            x += offset[0];
            y += offset[1];
            z += offset[2];
            match array.get(x + y * dims[0] + z * dims[0] * dims[1]) {
                Some(&value) if x < dims[0] && y < dims[1] => value,
                _ => 0,
            }
        };

        if scale == 4 {
            let mut bitmask = 0;
            let mut vec = arrayvec::ArrayVec::<_, 64>::new();
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        let value = access(x, y, z);
                        if value != 0 {
                            vec.push(value);
                            bitmask |= 1 << (z * 16 + y * 4 + x) as u64;
                        }
                    }
                }
            }

            let pointer = self.insert_values(&vec);

            Tree64Node {
                is_leaf_and_ptr: (pointer << 1) | 1,
                pop_mask: bitmask,
            }
        } else {
            let new_scale = scale / 4;
            let mut nodes = arrayvec::ArrayVec::<_, 64>::new();
            let mut bitmask = 0;
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        let value = self.insert(
                            array,
                            dims,
                            [
                                offset[0] + x * new_scale as usize,
                                offset[1] + y * new_scale as usize,
                                offset[2] + z * new_scale as usize,
                            ],
                            new_scale,
                        );
                        if value.pop_mask != 0 {
                            nodes.push(value);
                            bitmask |= 1 << (z * 16 + y * 4 + x) as u64;
                        }
                    }
                }
            }

            let pointer = self.insert_nodes(&nodes) + 1;

            Tree64Node {
                is_leaf_and_ptr: (pointer << 1),
                pop_mask: bitmask,
            }
        }
    }

    pub fn serialize<W: io::Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(&[self.scale])?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.data.len() as u32).to_le_bytes())?;
        writer.write_all(bytemuck::cast_slice(&self.nodes))?;
        writer.write_all(bytemuck::cast_slice(&self.data))?;
        Ok(())
    }

    pub fn deserialize<R: io::Read>(mut reader: R) -> io::Result<Self> {
        let mut scale = 0;
        let mut num_nodes = 0_u32;
        let mut num_data = 0_u32;
        reader.read_exact(bytemuck::bytes_of_mut(&mut scale))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_nodes))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_data))?;
        let mut this = Self {
            scale,
            nodes: vec![Tree64Node::default(); num_nodes as usize],
            data: vec![0; num_data as usize],
            stats: Default::default(),
            indices: Default::default(),
        };
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.nodes))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.data))?;
        Ok(this)
    }
}

fn extend_overlapping<T: PartialEq + Clone>(vec: &mut Vec<T>, data: &[T]) -> usize {
    for i in (1..data.len()).rev() {
        let slice_to_match = &data[..i];

        if slice_to_match.len() >= vec.len() {
            continue;
        }

        let pointer = vec.len() - slice_to_match.len();
        if slice_to_match == &vec[pointer..] {
            vec.extend_from_slice(&data[i..]);
            return pointer;
        }
    }

    let pointer = vec.len();
    vec.extend_from_slice(data);
    pointer
}

#[test]
fn test_tree() {
    {
        let tree = Tree64::new(&[1, 1, 1, 1], [2, 2, 1]);
        assert_eq!(tree.data, &[1; 4]);
        assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
        assert_eq!({ tree.nodes[0].pop_mask }, 0b00110011);
    }

    {
        let tree = Tree64::new(&[1, 1, 1, 1, 1, 0, 1, 1], [2, 2, 2]);
        assert_eq!(tree.data, &[1; 7]);
        assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
        assert_eq!(
            { tree.nodes[0].pop_mask },
            0b0000000000000000000000000000000000000000001100010000000000110011,
            "{:064b}",
            { tree.nodes[0].pop_mask }
        );
    }

    {
        let mut values = [1; 64];
        values[63] = 0;

        let tree = Tree64::new(&values, [4, 4, 4]);
        assert_eq!(tree.data, &[1; 63]);
        assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
        assert_eq!({ tree.nodes[0].pop_mask }, (!0 & !(1 << 63)));
    }

    {
        let mut values = [1; 64];
        values[63] = 0;
        let tree = Tree64::new(&values, [4, 4, 4]);

        let mut data = Vec::new();
        tree.serialize(&mut data).unwrap();
        let tree2 = Tree64::deserialize(io::Cursor::new(&data)).unwrap();
        assert_eq!(tree.data, tree2.data);
        assert_eq!(tree.nodes, tree2.nodes);
        assert_eq!(tree.scale, tree2.scale);

        std::fs::write("out.tree64", &data).unwrap();
    }

    {
        let mut values = [1; 5 * 5 * 5];
        values[5 * 5 * 5 - 1] = 0;
        let mut tree = Tree64::new(&values, [5, 5, 5]);
        dbg!(tree.data.len());
        tree.serialize(std::fs::File::create("out.tree64").unwrap())
            .unwrap();
    }
}
