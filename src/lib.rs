use std::fmt::Debug;
use std::io;

#[repr(C, packed)]
#[derive(
    Default, Debug, Clone, Copy, std::hash::Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Node {
    pub is_leaf_and_ptr: u32,
    pub pop_mask: u64,
}

impl Node {
    fn empty(is_leaf: bool) -> Self {
        Self::new(is_leaf, 0, 0)
    }

    fn new(is_leaf: bool, ptr: u32, pop_mask: u64) -> Self {
        Self {
            is_leaf_and_ptr: (ptr << 1) | (is_leaf as u32),
            pop_mask,
        }
    }

    fn is_leaf(&self) -> bool {
        (self.is_leaf_and_ptr & 1) == 1
    }

    fn ptr(&self) -> u32 {
        self.is_leaf_and_ptr >> 1
    }

    fn is_occupied(&self, index: u32) -> bool {
        self.pop_mask >> index & 1 == 1
    }

    fn get_index_for_child(&self, child: u32) -> Option<u32> {
        Some(self.ptr() + self.get_index_in_children(child)?)
    }

    fn get_index_in_children(&self, index: u32) -> Option<u32> {
        if !self.is_occupied(index) {
            return None;
        }

        Some(count_ones_variable(self.pop_mask, index))
    }

    fn range(&self) -> std::ops::Range<usize> {
        self.ptr() as usize..self.ptr() as usize + self.pop_mask.count_ones() as usize
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct RootState {
    pub index: u32,
    pub num_levels: u8,
    pub offset: glam::IVec3,
}

#[derive(Debug, Default, PartialEq)]
pub struct Edits {
    root_states: Vec<RootState>,
    index: u32,
}

impl Edits {
    fn push(&mut self, state: RootState) {
        if !self.root_states.is_empty() {
            if state == self.current() {
                return;
            }

            // Ensure that history is linear.
            while self.can_redo() {
                self.root_states.pop();
            }
        }

        self.index = self.root_states.len() as u32;
        self.root_states.push(state);
    }

    fn current(&self) -> RootState {
        self.root_states[self.index as usize]
    }

    pub fn can_undo(&self) -> bool {
        self.index > 0
    }

    pub fn can_redo(&self) -> bool {
        self.index < (self.root_states.len() as u32 - 1)
    }

    pub fn undo(&mut self) {
        if self.can_undo() {
            self.index -= 1;
        }
    }

    pub fn redo(&mut self) {
        if self.can_redo() {
            self.index += 1;
        }
    }
}

#[derive(Debug)]
pub struct Tree64 {
    pub nodes: VecWithCaching<Node>,
    pub data: VecWithCaching<u8>,
    pub stats: Stats,
    pub edits: Edits,
}

#[test]
fn test_tree_node_size() {
    assert_eq!(std::mem::size_of::<Node>(), 4 + 8);
}

impl Tree64 {
    fn insert_values(&mut self, values: &[u8]) -> u32 {
        if values.is_empty() {
            return 0;
        }

        let mut hit_cache = true;
        let old_length = self.data.len();

        let index = self.data.insert(values, |_data| {
            hit_cache = false;

            // use memmem to find where the values aready exist in the array
            // but spread across multiple leaf nodes. very slow.
            //if let Some(index) = memchr::memmem::find(_data, values) {
            //    self.stats.value.search_hits += 1;
            //    self.stats.value.search_bytes_saved += values.len();
            //    return Some(index as u32);
            //}

            None
        });

        if hit_cache {
            self.stats.value.cache_hits += 1;
            self.stats.value.cache_bytes_saved += values.len();
        }

        let new_length = self.data.len();

        if old_length != new_length {
            let added = new_length - old_length;
            self.stats.value.overlapping_saved += values.len() - added;
        }

        index
    }

    fn insert_nodes(&mut self, nodes: &[Node]) -> u32 {
        if nodes.is_empty() {
            return 0;
        }

        let mut hit_cache = true;
        let old_length = self.nodes.len();

        let index = self.nodes.insert(nodes, |_data| {
            hit_cache = false;

            /*
            if let Some(index) =
                memchr::memmem::find(bytemuck::cast_slice(_data), bytemuck::cast_slice(nodes))
            {
                if index % std::mem::size_of::<Tree64Node>() != 0 {
                    panic!()
                }
                let node_index = index / std::mem::size_of::<Tree64Node>();

                self.stats.nodes.search_hits += 1;
                self.stats.nodes.search_bytes_saved += nodes.len() * std::mem::size_of::<Tree64Node>();
                return Some(node_index as u32);
            }
            */

            None
        });

        if hit_cache {
            self.stats.nodes.cache_hits += 1;
            self.stats.nodes.cache_bytes_saved += nodes.len();
        }

        let new_length = self.nodes.len();

        if old_length != new_length {
            let added = new_length - old_length;
            self.stats.nodes.overlapping_saved += nodes.len() - added;
        }

        index
    }

    pub fn new(array: &[u8], dims: [usize; 3]) -> Self {
        let mut scale = dims[0].max(dims[1]).max(dims[2]).next_power_of_two() as u32;
        scale = scale.max(4);
        if scale.ilog2() % 2 == 1 {
            scale *= 2;
        }
        let num_levels = scale.ilog(4) as _;
        let mut this = Self {
            nodes: Default::default(),
            data: Default::default(),
            stats: Default::default(),
            edits: Default::default(),
        };
        let root = this.insert(array, dims, [0; 3], scale);
        let root_index = this.insert_nodes(&[root]);
        this.edits.push(RootState {
            index: root_index,
            num_levels,
            offset: glam::IVec3::ZERO,
        });
        this
    }

    pub fn root_state(&self) -> RootState {
        self.edits.current()
    }

    fn push_new_root_node(&mut self, node: Node, num_levels: u8, offset: glam::IVec3) -> u32 {
        let index = self.insert_nodes(&[node]);
        self.edits.push(RootState {
            index,
            num_levels,
            offset,
        });
        index
    }

    pub fn expand(&mut self) -> std::ops::Range<usize> {
        let root_state = self.root_state();
        let current_len = self.nodes.len();

        let new_nodes = [self.nodes[root_state.index as usize]; 64];
        let children = self.insert_nodes(&new_nodes);
        self.push_new_root_node(
            Node::new(false, children, !0),
            root_state.num_levels + 1,
            root_state.offset,
        );

        current_len..self.nodes.len()
    }

    pub fn modify(&mut self, at: [i32; 3], value: u8) -> UpdatedRanges {
        self.modify_nodes_in_box(at, [at[0] + 1, at[1] + 1, at[2] + 1], value)
    }

    pub fn modify_nodes_in_box(
        &mut self,
        min: [i32; 3],
        max: [i32; 3],
        value: u8,
    ) -> UpdatedRanges {
        let num_data = self.data.len();
        let num_nodes = self.nodes.len();

        let mut root_state = self.root_state();

        let mut min = glam::IVec3::from(min) + root_state.offset;
        let mut max = glam::IVec3::from(max) + root_state.offset;

        if min.cmpge(max).any() {
            return UpdatedRanges {
                data: num_data..self.data.len(),
                nodes: num_nodes..self.nodes.len(),
            };
        }

        let mut bbox = BoundingBox {
            min: min.as_uvec3(),
            max: max.as_uvec3(),
        };

        while bbox
            .get_intersection(&BoundingBox::from_levels(root_state.num_levels))
            .map(|intersection| intersection != bbox)
            .unwrap_or(true)
            && value != 0
        {
            let index_of_existing_root = 1 + 4 + 16;
            root_state.index = self.insert_nodes(&[Node::new(
                false,
                root_state.index,
                1 << index_of_existing_root,
            )]);

            let offset = glam::IVec3::splat(4_i32.pow(root_state.num_levels as u32));

            min += offset;
            max += offset;

            bbox = BoundingBox {
                min: min.as_uvec3(),
                max: max.as_uvec3(),
            };

            root_state.num_levels += 1;
            root_state.offset += offset;
        }

        let modify_leaf_node = |this: &mut Self, node: Node, mut intersection: BoundingBox| {
            let new_min = intersection.min % 4;
            intersection.max = new_min + (intersection.max - intersection.min);
            intersection.min = new_min;

            let mut pop_masked_data = PopMaskedData::new(&this.data[node.range()], node.pop_mask);
            for x in intersection.min.x..intersection.max.x {
                for y in intersection.min.y..intersection.max.y {
                    for z in intersection.min.z..intersection.max.z {
                        let index = x + y * 4 + z * 16;
                        pop_masked_data.set(index, Some(value).filter(|&v| v != 0));
                    }
                }
            }

            Node::new(
                true,
                this.insert_values(&pop_masked_data.as_compact()),
                pop_masked_data.pop_mask,
            )
        };

        let root = self.nodes[root_state.index as usize];

        if root.is_leaf() {
            let node_bbox = BoundingBox {
                min: glam::UVec3::splat(0),
                max: glam::UVec3::splat(4),
            };
            if let Some(intersection) = bbox.get_intersection(&node_bbox) {
                let new_root = modify_leaf_node(self, root, intersection);
                self.push_new_root_node(new_root, root_state.num_levels, root_state.offset);
            }

            return UpdatedRanges {
                data: num_data..self.data.len(),
                nodes: num_nodes..self.nodes.len(),
            };
        }

        #[derive(Clone)]
        struct StackItem {
            node_min_pos: glam::UVec3,
            node_level: u8,
            corresponding_node_index: Option<u32>,
            children: PopMaskedData<Node>,
            parent_and_index: Option<(u32, u8)>,
        }

        let mut stack: Vec<StackItem> = vec![StackItem {
            node_min_pos: glam::UVec3::ZERO,
            node_level: root_state.num_levels,
            corresponding_node_index: Some(root_state.index),
            children: PopMaskedData::new(&self.nodes[root.range()], root.pop_mask),
            parent_and_index: None,
        }];

        let mut last_parent = None;

        let mut new_root = Node::default();

        while let Some(mut item) = stack.last().cloned() {
            let index = stack.len() as u32 - 1;
            let child_size = 4_u32.pow(item.node_level as u32 - 1);
            let children_are_leaves = child_size == 4;

            let going_up = last_parent == Some(index);
            let mut any_pushed = false;

            last_parent = item.parent_and_index.map(|(parent, _)| parent);

            if !going_up {
                for child_index in 0..64 {
                    let pos = glam::UVec3::new(child_index, child_index / 4, child_index / 16) % 4;
                    let node_bbox = BoundingBox {
                        min: item.node_min_pos + child_size * pos,
                        max: item.node_min_pos + child_size * (pos + 1),
                    };

                    let intersection = match node_bbox.get_intersection(&bbox) {
                        Some(intersection) => intersection,
                        None => continue,
                    };

                    if intersection == node_bbox {
                        item.children.set(
                            child_index,
                            if value > 0 {
                                // leaf node for the specific value.
                                let mut child_node =
                                    Node::new(true, self.insert_values(&[value; 64]), !0);

                                // build up a tree for the child if the children for this node are not leaves.
                                for _ in 0..item.node_level.saturating_sub(2) {
                                    child_node =
                                        Node::new(false, self.insert_nodes(&[child_node; 64]), !0);
                                }

                                Some(child_node)
                            } else {
                                None
                            },
                        );
                    } else {
                        let corresponding_child_index =
                            item.corresponding_node_index.and_then(|node_index| {
                                self.nodes[node_index as usize].get_index_for_child(child_index)
                            });
                        let child = if let Some(child_index) = corresponding_child_index {
                            self.nodes[child_index as usize]
                        } else {
                            Node::empty(children_are_leaves)
                        };

                        if !children_are_leaves {
                            any_pushed = true;
                            stack.push(StackItem {
                                node_min_pos: node_bbox.min,
                                node_level: item.node_level - 1,
                                corresponding_node_index: corresponding_child_index,
                                children: PopMaskedData::new(
                                    &self.nodes[child.range()],
                                    child.pop_mask,
                                ),
                                parent_and_index: Some((index, child_index as u8)),
                            });
                        } else {
                            item.children.set(
                                child_index,
                                Some(modify_leaf_node(self, child, intersection))
                                    .filter(|node| node.pop_mask != 0),
                            );
                        }
                    }
                }
            }

            if !any_pushed {
                let node = Node::new(
                    false,
                    self.insert_nodes(&item.children.as_compact()),
                    item.children.pop_mask,
                );

                if let Some((parent, index)) = item.parent_and_index {
                    stack[parent as usize]
                        .children
                        .set(index as u32, Some(node).filter(|node| node.pop_mask != 0));
                } else {
                    new_root = node;
                }

                stack.pop();
            } else {
                // if we're decending into the node's children, then we need to make sure any updates are retained.
                stack[index as usize].children = item.children;
            }
        }

        self.push_new_root_node(new_root, root_state.num_levels, root_state.offset);

        UpdatedRanges {
            data: num_data..self.data.len(),
            nodes: num_nodes..self.nodes.len(),
        }
    }

    fn insert(&mut self, array: &[u8], dims: [usize; 3], offset: [usize; 3], scale: u32) -> Node {
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

            Node::new(true, self.insert_values(&vec), bitmask)
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

            Node::new(false, self.insert_nodes(&nodes), bitmask)
        }
    }

    pub fn serialize<W: io::Write>(&self, mut writer: W) -> io::Result<()> {
        let root_state = self.root_state();
        writer.write_all(&[root_state.num_levels])?;
        writer.write_all(&root_state.index.to_le_bytes())?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.data.len() as u32).to_le_bytes())?;
        writer.write_all(bytemuck::cast_slice(&self.nodes))?;
        writer.write_all(bytemuck::cast_slice(&self.data))?;
        Ok(())
    }

    pub fn deserialize<R: io::Read>(mut reader: R) -> io::Result<Self> {
        let mut num_levels = 0;
        let mut num_nodes = 0_u32;
        let mut num_data = 0_u32;
        let mut root_node_index = 0_u32;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_levels))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut root_node_index))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_nodes))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_data))?;
        let mut this = Self {
            nodes: VecWithCaching::from_vec(vec![Node::default(); num_nodes as usize]),
            data: VecWithCaching::from_vec(vec![0; num_data as usize]),
            stats: Default::default(),
            edits: Edits {
                root_states: vec![RootState {
                    num_levels,
                    offset: glam::IVec3::ZERO,
                    index: root_node_index,
                }],
                index: 0,
            },
        };
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.nodes.inner))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.data.inner))?;
        Ok(this)
    }

    pub fn get_value_at(&self, pos: [u32; 3]) -> u8 {
        let pos = glam::UVec3::from(pos);
        let state = self.root_state();
        let mut node = self.nodes[state.index as usize];
        let mut child_size = 4_u32.pow(state.num_levels as u32 - 1);

        while !node.is_leaf() {
            let child_index = ((pos / child_size) % 4).dot(glam::UVec3::new(1, 4, 16));
            child_size /= 4;
            if let Some(child_node_index) = node.get_index_for_child(child_index) {
                node = self.nodes[child_node_index as usize];
            } else {
                return 0;
            }
        }

        debug_assert_eq!(child_size, 1);

        let child_index = (pos % 4).dot(glam::UVec3::new(1, 4, 16));
        node.get_index_for_child(child_index)
            .map(|index| self.data[index as usize])
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, Eq, PartialEq)]
struct CompactRange {
    start: u32,
    length: u8,
}

impl CompactRange {
    fn as_range(&self) -> std::ops::Range<usize> {
        self.start as usize..self.start as usize + self.length as usize
    }
}

#[derive(Default, Debug)]
pub struct VecWithCaching<T, Hasher = fnv::FnvBuildHasher> {
    pub inner: Vec<T>,
    cache: hashbrown::HashTable<CompactRange>,
    _phantom: std::marker::PhantomData<Hasher>,
}

impl<T: std::hash::Hash + Clone + PartialEq + Debug, Hasher: std::hash::BuildHasher + Default>
    VecWithCaching<T, Hasher>
{
    fn from_vec(vec: Vec<T>) -> Self {
        Self {
            inner: vec,
            cache: Default::default(),
            _phantom: Default::default(),
        }
    }

    fn insert<F: FnMut(&[T]) -> Option<u32>>(&mut self, slice: &[T], mut try_find: F) -> u32 {
        let hasher = Hasher::default();

        self.cache
            .entry(
                hasher.hash_one(slice),
                |compact_range| &self.inner[compact_range.as_range()] == slice,
                |compact_range| hasher.hash_one(&self.inner[compact_range.as_range()]),
            )
            .or_insert_with(|| CompactRange {
                start: try_find(&self.inner)
                    .unwrap_or_else(|| extend_overlapping(&mut self.inner, slice) as u32),
                length: slice.len() as u8,
            })
            .get()
            .start
    }
}

impl<T, Hasher> std::ops::Deref for VecWithCaching<T, Hasher> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Default, Debug)]
pub struct VecStats {
    cache_hits: usize,
    cache_bytes_saved: usize,
    //search_hits: usize,
    //search_bytes_saved: usize,
    overlapping_saved: usize,
}

#[derive(Default, Debug)]
pub struct Stats {
    value: VecStats,
    nodes: VecStats,
}

fn extend_overlapping<T: PartialEq + Clone + Debug>(vec: &mut Vec<T>, data: &[T]) -> usize {
    let pointer = vec.len();

    for i in (1..=data.len()).rev() {
        let slice_to_match = &data[..i];

        if vec.ends_with(slice_to_match) {
            vec.extend_from_slice(&data[i..]);
            return pointer - slice_to_match.len();
        }
    }

    vec.extend_from_slice(data);
    pointer
}

#[derive(Default, Debug, PartialEq)]
pub struct UpdatedRanges {
    pub nodes: std::ops::Range<usize>,
    pub data: std::ops::Range<usize>,
}

#[test]
fn test_pm() {
    let mut x = PopMaskedData::<u8>::default();
    for i in (0..64) {
        x.set(i, Some(1));
    }
    assert_eq!(&x.as_compact()[..], &[1; 64]);
    for i in (0..64).rev() {
        x.set(i, Some(2));
    }
    assert_eq!(&x.as_compact()[..], &[2; 64]);
    let mut x = PopMaskedData::<u8>::default();
    for i in (0..64).rev() {
        x.set(i, Some(3));
    }
    assert_eq!(&x.as_compact()[..], &[3; 64]);
    for i in (0..64).rev() {
        x.set(i, None);
    }
    assert_eq!(&x.as_compact()[..], &[]);
    let mut x = PopMaskedData::<u8>::default();
    for i in (0..64) {
        x.set(i, Some(1));
    }
    x.set(1, None);
    assert_eq!(&x.as_compact()[..], &[1; 63]);

    let mut x = PopMaskedData::<u8>::new(&[1_u8; 4], 0b11110);
    x.set(0, Some(255));
    assert_eq!(&x.as_compact()[..], &[255, 1, 1, 1, 1]);
}

#[derive(Clone)]
struct PopMaskedData<T: Default + Copy> {
    array: [T; 64],
    pop_mask: u64,
}

impl<T: Copy + Default> Default for PopMaskedData<T> {
    fn default() -> Self {
        Self {
            array: [T::default(); 64],
            pop_mask: 0,
        }
    }
}

impl<T: Copy + Default> PopMaskedData<T> {
    fn new(slice: &[T], pop_mask: u64) -> Self {
        let mut flat = [T::default(); 64];

        let mut var_pop_mask = pop_mask;

        for &value in slice {
            let next_pos = var_pop_mask.trailing_zeros();
            flat[next_pos as usize] = value;
            var_pop_mask &= var_pop_mask - 1;
        }

        Self {
            pop_mask,
            array: flat,
        }
    }

    fn as_compact(&self) -> arrayvec::ArrayVec<T, 64> {
        let mut array = arrayvec::ArrayVec::new();

        let mut var_pop_mask = self.pop_mask;

        while var_pop_mask != 0 {
            let next_pos = var_pop_mask.trailing_zeros();
            array.push(self.array[next_pos as usize]);
            var_pop_mask &= var_pop_mask - 1;
        }
        array
    }

    fn set(&mut self, index: u32, value: Option<T>) {
        self.array[index as usize] = value.unwrap_or_default();
        self.pop_mask = set_bit(self.pop_mask, index, value.is_some());
    }
}

fn count_ones_variable(value: u64, index: u32) -> u32 {
    (value & ((1 << index) - 1)).count_ones()
}

fn set_bit(mask: u64, index: u32, value: bool) -> u64 {
    (mask & !(1 << index)) | ((value as u64) << index)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BoundingBox {
    min: glam::UVec3,
    max: glam::UVec3,
}

impl BoundingBox {
    fn from_levels(num_levels: u8) -> Self {
        Self {
            min: glam::UVec3::splat(0),
            max: glam::UVec3::splat(4_u32.pow(num_levels as u32)),
        }
    }

    fn get_intersection(&self, other: &Self) -> Option<BoundingBox> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);

        if min.cmplt(max).all() {
            Some(BoundingBox { min, max })
        } else {
            None
        }
    }
}

#[test]
fn test_tree() {
    {
        let tree = Tree64::new(&[1, 1, 1, 1], [2, 2, 1]);
        dbg!(&tree.data, &tree.nodes);
        assert_eq!(&*tree.data, &[1; 4]);
        assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
        assert_eq!({ tree.nodes[0].pop_mask }, 0b00110011);

        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    assert_eq!(tree.get_value_at([x, y, z]), if z == 1 { 0 } else { 1 });
                }
            }
        }
    }

    {
        let array = &[1, 1, 1, 1, 1, 0, 1, 1];
        let tree = Tree64::new(array, [2, 2, 2]);
        assert_eq!(&*tree.data, &[1; 7]);
        assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
        assert_eq!(
            { tree.nodes[0].pop_mask },
            0b00000000001100010000000000110011,
            "{:064b}",
            { tree.nodes[0].pop_mask }
        );

        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    assert_eq!(
                        tree.get_value_at([x, y, z]),
                        array[x as usize + y as usize * 2 + z as usize * 4]
                    );
                }
            }
        }
    }

    {
        let mut values = [1; 64];
        values[63] = 0;

        let tree = Tree64::new(&values, [4, 4, 4]);
        assert_eq!(&*tree.data, &[1; 63]);
        assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
        assert_eq!({ tree.nodes[0].pop_mask }, (!0 & !(1 << 63)));

        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    assert_eq!(
                        tree.get_value_at([x, y, z]),
                        if x == 3 && y == 3 && z == 3 { 0 } else { 1 }
                    );
                }
            }
        }
    }

    {
        let mut values = [1; 64];
        values[63] = 0;
        let tree = Tree64::new(&values, [4, 4, 4]);

        let mut data = Vec::new();
        tree.serialize(&mut data).unwrap();
        let tree2 = Tree64::deserialize(io::Cursor::new(&data)).unwrap();
        assert_eq!(&*tree.data, &*tree2.data);
        assert_eq!(&*tree.nodes, &*tree2.nodes);
        assert_eq!(tree.edits, tree2.edits);

        let mut tree = tree;
        let ranges = tree.modify([2, 2, 3], 0);
        assert_eq!(ranges.nodes, 1..2);
        assert!(ranges.data.is_empty(), "{:?}", tree.data);

        let mut tree = tree;
        let ranges = tree.modify([0, 0, 0], 2);
        assert_eq!(ranges.nodes, 2..3);
        assert_eq!(ranges.data.len(), 64 - 2);
    }
}

#[test]
fn single_node_modifications() {
    {
        let values = [1; 4 * 4 * 4];
        let mut tree = Tree64::new(&values, [4; 3]);

        let ranges = tree.modify([3, 3, 3], 0);
        assert_eq!(ranges.nodes.len(), 1);
        assert!(ranges.data.is_empty());
    }

    {
        let values = [1; 4 * 4 * 4];
        let mut tree = Tree64::new(&values, [4; 3]);

        let ranges = tree.modify([3, 3, 3], 2);
        dbg!(&tree.data[ranges.data.clone()]);
        assert_eq!(ranges.data.len(), 1);
        assert_eq!(ranges.nodes.len(), 1);
    }

    {
        let values = [1; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);
        let updated_nodes = 64 + 1;

        let ranges = tree.modify([5, 5, 5], 0);
        assert_eq!(ranges.nodes.len(), updated_nodes);
        assert!(ranges.data.is_empty());

        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    assert_eq!(
                        tree.get_value_at([x, y, z]),
                        if x == 5 && y == 5 && z == 5 { 0 } else { 1 }
                    );
                }
            }
        }
    }

    {
        let values = [1; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);
        let updated_nodes = 64 + 64 + 1;

        let ranges = tree.modify([63, 63, 63], 0);
        assert!(ranges.data.is_empty());
        assert_eq!(ranges.nodes.len(), updated_nodes);
    }

    {
        let values = [1; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);
        let updated_nodes = 64 + 64 + 1;

        let ranges = tree.modify([63, 63, 63], 2);
        assert_eq!(ranges.data.len(), 1);
        assert_eq!(ranges.nodes.len(), updated_nodes);
    }

    {
        let values = [1; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);
        let updated_nodes = 64 + 64 + 1;

        let ranges = tree.modify([0, 0, 0], 2);
        assert_eq!(ranges.data.len(), 64);
        assert_eq!(ranges.nodes.len(), updated_nodes);
    }
}

#[test]
fn modifications_in_box() {
    {
        let values = [1; 4 * 4 * 4];
        let mut tree = Tree64::new(&values, [4; 3]);

        let ranges = tree.modify_nodes_in_box([1; 3], [2; 3], 0);
        assert_eq!(ranges.nodes.len(), 1);
        assert!(ranges.data.is_empty());
    }

    {
        let values = [1; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([1; 3], [3; 3], 0);
        assert_eq!(ranges.nodes.len(), 64 + 1);
        assert!(ranges.data.is_empty());
    }

    {
        let values = [1; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], 2);
        assert_eq!(ranges.nodes.len(), 64 + 1);
        assert_eq!(ranges.data.len(), 64);
    }

    {
        let values = [1; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], 2);

        assert_eq!(ranges.data.len(), 64);
        assert_eq!(ranges.nodes.len(), 64 + 1);
    }

    {
        let values = [1; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [4; 3], 2);
        assert_eq!(ranges.nodes.len(), 64 + 1);
    }
}

#[test]
fn advanced_modifications_in_box() {
    {
        let values = [1; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], 2);
        dbg!(tree.data);

        assert_eq!(ranges.data.len(), 64);
        assert_ne!(ranges.nodes.len(), 64 + 1);
    }

    {
        let values = [1; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], 2);
        dbg!(tree.data);

        assert_eq!(ranges.data.len(), 64);
        assert_ne!(ranges.nodes.len(), 64 + 1);
    }
}

#[test]
fn modifications_on_empty_spaces() {
    {
        let mut values = [0; 16 * 16 * 16];
        values[3 * 16 + 3 * 4 + 3] = 1;
        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify([0; 3], 1);
        assert_eq!(
            ranges,
            UpdatedRanges {
                nodes: 2..5,
                data: 1..1,
            }
        );
    }

    {
        let values = [0; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);
        dbg!(&tree.nodes);

        let ranges = tree.modify([0; 3], 1);
        dbg!(&tree.nodes);
        assert_eq!(
            ranges,
            UpdatedRanges {
                nodes: 1..3,
                data: 0..1,
            }
        );
    }

    {
        let values = [0; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);
        dbg!(&tree.nodes);

        let ranges = tree.modify([0; 3], 1);
        dbg!(&tree.nodes);
        assert_eq!(
            ranges,
            UpdatedRanges {
                nodes: 1..4,
                data: 0..1,
            }
        );
    }

    {
        let values = [0; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);
        dbg!(&tree.nodes);

        let ranges = tree.modify_nodes_in_box([0; 3], [16; 3], 1);
        dbg!(&tree.nodes);
        assert_eq!(
            ranges,
            UpdatedRanges {
                // 64 new leaf nodes, 1 non-leaf node and 1 root node.
                nodes: 1..1 + 64 + 1 + 1,
                data: 0..64,
            }
        );
    }

    {
        let values = [0; 4 * 4 * 4];

        let mut tree = Tree64::new(&values, [4; 3]);
        dbg!(&tree.nodes);

        let ranges = tree.modify_nodes_in_box([2; 3], [4; 3], 1);
        dbg!(&tree.nodes);
        assert_eq!(
            ranges,
            UpdatedRanges {
                data: 0..8,
                nodes: 1..2,
            }
        );
    }

    {
        let values = [0; 16 * 16 * 16];

        let mut tree = Tree64::new(&values, [16; 3]);
        dbg!(&tree.nodes);

        let ranges = tree.modify_nodes_in_box([2; 3], [4; 3], 1);
        dbg!(&tree.nodes);
        assert_eq!(
            ranges,
            UpdatedRanges {
                data: 0..8,
                nodes: 1..3,
            }
        );
    }

    {
        let values = [0; 16 * 16 * 16];

        let mut tree = Tree64::new(&values, [16; 3]);
        dbg!(&tree.nodes);

        let ranges = tree.modify_nodes_in_box([6; 3], [8; 3], 1);
        dbg!(&tree.nodes);
        assert_eq!(
            ranges,
            UpdatedRanges {
                data: 0..8,
                nodes: 1..3,
            }
        );
    }

    {
        let values = [0; 16 * 16 * 16];

        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [5; 3], 1);
        assert_eq!(
            ranges,
            UpdatedRanges {
                data: 0..64,
                nodes: 1..1 + 8 + 1,
            }
        );
    }

    {
        let values = [0; 16 * 16 * 16];

        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([4; 3], [12; 3], 1);
        assert_eq!(
            ranges,
            UpdatedRanges {
                data: 0..64,
                nodes: 1..1 + 8 + 1,
            }
        );
    }

    {
        let values = [1; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([1; 3], [3; 3], 0);
        assert_eq!(ranges.nodes.len(), 64 + 1);
        assert!(ranges.data.is_empty());
    }

    {
        let values = [1; 16 * 16 * 16];
        let mut tree = Tree64::new(&values, [16; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [16; 3], 0);
        assert_eq!(ranges.nodes.len(), 1);
        assert!(ranges.data.is_empty());
    }
}

#[test]
fn outside_modifications() {
    let mut tree = Tree64::new(&[], [0; 3]);
    assert_eq!(
        tree.modify_nodes_in_box([64; 3], [65; 3], 1),
        UpdatedRanges {
            nodes: 1..9,
            data: 0..1
        }
    );

    let mut tree = Tree64::new(&[], [0; 3]);
    assert_eq!(tree.modify_nodes_in_box([-100; 3], [2; 3], 1).data, 0..64);
}

#[test]
fn modify_box_wierd_sizes() {
    {
        let values = [0; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);

        tree.modify([10; 3], 1);

        for x in 0..64 {
            for y in 0..64 {
                for z in 0..64 {
                    assert_eq!(
                        tree.get_value_at([x, y, z]),
                        if x == 10 && y == 10 && z == 10 { 1 } else { 0 },
                        "{} {} {}",
                        x,
                        y,
                        z
                    );
                }
            }
        }
    }

    {
        let values = [0; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);

        tree.modify_nodes_in_box([0; 3], [20; 3], 1);

        for x in 0..64 {
            for y in 0..64 {
                for z in 0..64 {
                    assert_eq!(
                        tree.get_value_at([x, y, z]),
                        if x < 20 && y < 20 && z < 20 { 1 } else { 0 },
                        "{:?}",
                        (x, y, z)
                    );
                }
            }
        }
    }

    for start in 0..44 {
        let values = [0; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);

        tree.modify_nodes_in_box([start; 3], [start + 20, start + 20, start + 20], 1);

        let start = start as u32;

        for x in 0..64 {
            for y in 0..64 {
                for z in 0..64 {
                    assert_eq!(
                        tree.get_value_at([x, y, z]),
                        if x >= start
                            && x < (start + 20)
                            && y >= start
                            && y < (start + 20)
                            && z >= start
                            && z < (start + 20)
                        {
                            1
                        } else {
                            0
                        },
                        "{:?} {:?}",
                        (x, y, z),
                        start
                    );
                }
            }
        }
    }
}

#[test]
fn test_tiny() {
    let tree = Tree64::new(&[0], [1; 3]);
}

#[test]
#[ignore = "missing sponza file"]
fn test_sponza_editing() {
    let mut tree =
        Tree64::deserialize(std::fs::File::open("../voxviewer/assets/sponza.tree64").unwrap())
            .unwrap();
    let min = glam::UVec3::new(611, 304, 486);
    let max = min + 40;
    for x in min.x - 2..max.x + 2 {
        for y in min.y - 2..max.y + 2 {
            for z in min.z - 2..max.z + 2 {
                assert_eq!(tree.get_value_at([x, y, z]), 0);
            }
        }
    }
    tree.modify_nodes_in_box(min.as_ivec3().into(), max.as_ivec3().into(), 1);
    for x in min.x - 3..max.x + 3 {
        for y in min.y - 3..max.y + 3 {
            for z in min.z - 3..max.z + 3 {
                let expected = if z >= min.z
                    && z < max.z
                    && x >= min.x
                    && x < max.x
                    && y >= min.y
                    && y < max.y
                {
                    1
                } else {
                    0
                };
                assert_eq!(tree.get_value_at([x, y, z]), expected);
            }
        }
    }
    tree.edits.undo();
    for x in min.x - 2..max.x + 2 {
        for y in min.y - 2..max.y + 2 {
            for z in min.z - 2..max.z + 2 {
                assert_eq!(tree.get_value_at([x, y, z]), 0);
            }
        }
    }
}

#[test]
fn trillion_voxel_deletion() {
    let mut tree = Tree64::new(&[], [0; 3]);
    tree.modify_nodes_in_box([0; 3], [10000; 3], 1);
    let range = tree.modify_nodes_in_box([1; 3], [10000 - 1; 3], 0);
    assert_eq!(range.data.len(), 0);
    assert_eq!(range.nodes.len(), 2095);
}
