use std::fmt::Debug;
use std::hash::BuildHasher;
use std::io;

#[derive(Debug, Clone, Copy, PartialOrd, Ord, Eq, PartialEq)]
struct CompactRange {
    start: u32,
    length: u8,
}

impl CompactRange {
    fn end(&self) -> u32 {
        self.start + self.length as u32
    }

    fn as_range(&self) -> std::ops::Range<usize> {
        self.start as usize..self.start as usize + self.length as usize
    }
}

#[derive(Default, Debug)]
pub struct VecWithCaching<T> {
    pub inner: Vec<T>,
    cache: hashbrown::HashTable<CompactRange>,
}

impl<T: std::hash::Hash + Clone + PartialEq + Debug> VecWithCaching<T> {
    fn from_vec(vec: Vec<T>) -> Self {
        Self {
            inner: vec,
            cache: Default::default(),
        }
    }

    fn insert<F: FnMut(&[T]) -> Option<u32>>(&mut self, slice: &[T], mut try_find: F) -> u32 {
        let hasher = fnv::FnvBuildHasher::default();

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

#[derive(Default, Debug)]
pub struct VecStats {
    cache_hits: usize,
    cache_bytes_saved: usize,
    search_hits: usize,
    search_bytes_saved: usize,
    overlapping_saved: usize,
}

#[derive(Default, Debug)]
pub struct Tree64Stats {
    value: VecStats,
    nodes: VecStats,
}

#[repr(C, packed)]
#[derive(
    Default, Debug, Clone, Copy, std::hash::Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Tree64Node {
    pub is_leaf_and_ptr: u32,
    pub pop_mask: u64,
}

impl Tree64Node {
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

        Some((self.pop_mask & ((1 << index) - 1)).count_ones())
    }

    fn range(&self) -> std::ops::Range<usize> {
        self.ptr() as usize..self.ptr() as usize + self.pop_mask.count_ones() as usize
    }
}

#[derive(Default, Debug)]
struct Tree64GC {
    touched_nodes: fnv::FnvHashSet<u32>,
    data_spans: Vec<CompactRange>,
    node_spans: Vec<CompactRange>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Tree64Edits {
    roots_and_num_levels: Vec<(u32, u8)>,
    index: u32,
}

impl Tree64Edits {
    fn push(&mut self, index: u32, num_levels: u8) {
        if !self.roots_and_num_levels.is_empty() {
            if (index, num_levels) == self.current() {
                return;
            }

            // Ensure that history is linear.
            while self.can_redo() {
                self.roots_and_num_levels.pop();
            }
        }

        self.roots_and_num_levels.push((index, num_levels));
        self.index = self.roots_and_num_levels.len() as u32 - 1;
    }

    fn current(&self) -> (u32, u8) {
        self.roots_and_num_levels[self.index as usize]
    }

    pub fn can_undo(&self) -> bool {
        self.index > 0
    }

    pub fn can_redo(&self) -> bool {
        self.index < (self.roots_and_num_levels.len() as u32 - 1)
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
    pub nodes: VecWithCaching<Tree64Node>,
    pub data: VecWithCaching<u8>,
    stats: Tree64Stats,
    pub edits: Tree64Edits,
}

#[test]
fn test_tree_node_size() {
    assert_eq!(std::mem::size_of::<Tree64Node>(), 4 + 8);
}

impl Tree64 {
    pub fn collect_garbage(&self) {
        let root_node_index = self.root_node_index_and_num_levels().0;
        let mut gc = Tree64GC::default();
        let mut stack = vec![root_node_index];

        while let Some(node_index) = stack.pop() {
            if !gc.touched_nodes.insert(node_index) {
                continue;
            }

            let node = &self.nodes.inner[node_index as usize];

            if node.pop_mask == 0 {
                continue;
            }

            let range = CompactRange {
                start: node.is_leaf_and_ptr >> 1,
                length: node.pop_mask.count_ones() as u8,
            };

            if node.is_leaf() {
                gc.data_spans.push(range);
            } else {
                gc.node_spans.push(range);

                for i in 0..range.length {
                    stack.push(range.start + i as u32);
                }
            }
        }

        gc.data_spans.sort_unstable();
        gc.node_spans.sort_unstable();

        let mut previous = 0;

        for range in &gc.data_spans {
            if range.start > previous {
                dbg!("{:?}", previous..range.start);
            }

            previous = range.start + range.end();
        }

        let mut previous = 0;

        for range in &gc.node_spans {
            if range.start > previous {
                dbg!("{:?}", previous..range.start);
            }

            previous = range.end();
        }
    }

    fn insert_values(&mut self, values: &[u8]) -> u32 {
        if values.is_empty() {
            return 0;
        }

        let mut hit_cache = true;
        let old_length = self.data.inner.len();

        let index = self.data.insert(values, |_data| {
            hit_cache = false;

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

        let new_length = self.data.inner.len();

        if old_length != new_length {
            let added = new_length - old_length;
            self.stats.value.overlapping_saved += values.len() - added;
        }

        index
    }

    fn insert_nodes(&mut self, nodes: &[Tree64Node]) -> u32 {
        if nodes.is_empty() {
            return 0;
        }

        let mut hit_cache = true;
        let old_length = self.nodes.inner.len();

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

        let new_length = self.nodes.inner.len();

        if old_length != new_length {
            let added = new_length - old_length;
            self.stats.nodes.overlapping_saved += nodes.len() - added;
        }

        index
    }

    pub fn new(array: &[u8], dims: [usize; 3]) -> Self {
        let mut scale = dims[0].max(dims[1]).max(dims[2]).next_power_of_two() as u32;
        if scale.ilog2() % 2 == 1 {
            scale *= 2;
        }
        let num_levels = scale.ilog(4) as _;
        let mut this = Self {
            nodes: Default::default(),
            data: Default::default(),
            stats: Default::default(),
            edits: Tree64Edits::default(),
        };
        let root = this.insert(array, dims, [0; 3], scale);
        let root_node_index = this.insert_nodes(&[root]);
        this.edits.push(root_node_index, num_levels);
        {
            dbg!(&this.stats);
            dbg!(
                this.nodes.inner.len() * std::mem::size_of::<Tree64Node>(),
                this.data.inner.len()
            );
        }
        this
    }

    pub fn root_node_index_and_num_levels(&self) -> (u32, u8) {
        self.edits.current()
    }

    fn push_new_root_node(&mut self, node: Tree64Node, num_levels: u8) {
        let index = self.insert_nodes(&[node]);
        self.edits.push(index, num_levels);
    }

    pub fn expand(&mut self) -> std::ops::Range<usize> {
        let (root_node_index, num_levels) = self.root_node_index_and_num_levels();
        let current_len = self.nodes.inner.len();

        let new_nodes = [self.nodes.inner[root_node_index as usize]; 64];
        let children = self.insert_nodes(&new_nodes);
        self.push_new_root_node(Tree64Node::new(false, children, !0), num_levels + 1);

        current_len..self.nodes.inner.len()
    }

    fn get_leaves_for_node(&self, node: Tree64Node) -> arrayvec::ArrayVec<u8, 64> {
        debug_assert!(node.is_leaf());
        let slice = &self.data.inner[node.range()];
        let mut nodes = arrayvec::ArrayVec::<_, 64>::new();
        nodes.try_extend_from_slice(slice).unwrap();
        nodes
    }

    fn get_children_for_node(&self, node: Tree64Node) -> arrayvec::ArrayVec<Tree64Node, 64> {
        debug_assert!(!node.is_leaf());
        let slice = &self.nodes.inner[node.range()];
        let mut nodes = arrayvec::ArrayVec::<_, 64>::new();
        nodes.try_extend_from_slice(slice).unwrap();
        nodes
    }

    pub fn modify(&mut self, at: [u32; 3], value: u8) -> UpdatedRanges {
        self.modify_nodes_in_box(at, [at[0] + 1, at[1] + 1, at[2] + 1], value)
    }

    pub fn modify_nodes_in_box(
        &mut self,
        min: [u32; 3],
        max: [u32; 3],
        value: u8,
    ) -> UpdatedRanges {
        let num_data = self.data.inner.len();
        let num_nodes = self.nodes.inner.len();

        let (root_index, num_levels) = self.root_node_index_and_num_levels();

        let bbox = BoundingBox {
            min: min.into(),
            max: max.into(),
        };

        let root = self.nodes.inner[root_index as usize];
        let size = 4_u32.pow(num_levels as _);

        let root_bbox = BoundingBox {
            min: glam::UVec3::splat(0),
            max: glam::UVec3::splat(size),
        };

        if root.is_leaf() {
            let new_root = self.modify_leaf_node(root, bbox, root_bbox.min, value);
            self.push_new_root_node(new_root, num_levels);

            return UpdatedRanges {
                data: num_data..self.data.inner.len(),
                nodes: num_nodes..self.nodes.inner.len(),
            };
        }

        let children_data_for_node = |this: &Self, node| PopMaskedData {
            array: this.get_children_for_node(node),
            pop_mask: node.pop_mask,
        };

        #[derive(Clone)]
        struct StackItem {
            node_min_pos: glam::UVec3,
            node_level: u8,
            corresponding_node_index: Option<u32>,
            children: PopMaskedData<Tree64Node>,
            parent_and_index: Option<(u32, u8)>,
        }

        let mut stack: Vec<StackItem> = vec![StackItem {
            node_min_pos: root_bbox.min,
            node_level: num_levels,
            corresponding_node_index: Some(root_index),
            children: children_data_for_node(self, root),
            parent_and_index: None,
        }];

        let mut last_parent = None;

        let mut new_root = Tree64Node::default();

        while let Some(mut item) = stack.last().cloned() {
            let index = stack.len() as u32 - 1;
            let child_size = 4_u32.pow(item.node_level as u32 - 1);
            let children_are_leaves = child_size == 4;

            let going_up = last_parent == Some(index);
            let mut any_pushed = false;

            last_parent = item.parent_and_index.map(|(parent, _)| parent);

            for x in 0..4 {
                for y in 0..4 {
                    for z in 0..4 {
                        let node_bbox = BoundingBox {
                            min: item.node_min_pos + child_size * glam::UVec3::new(x, y, z),
                            max: item.node_min_pos + child_size * (glam::UVec3::new(x, y, z) + 1),
                        };

                        let child_index = x + y * 4 + z * 16;

                        match node_bbox.get_intersection(&bbox) {
                            None => continue,
                            Some(intersection)
                                if intersection == node_bbox && !children_are_leaves =>
                            {
                                item.children.set(
                                    child_index,
                                    if value == 0 {
                                        None
                                    } else {
                                        let mut child_node = Tree64Node::new(
                                            true,
                                            self.insert_values(&[value; 64]),
                                            !0,
                                        );

                                        for _ in 0..item.node_level.saturating_sub(2) {
                                            child_node = Tree64Node::new(
                                                false,
                                                self.insert_nodes(&[child_node; 64]),
                                                !0,
                                            );
                                        }

                                        Some(child_node)
                                    },
                                );
                                continue;
                            }
                            Some(_) => {}
                        };

                        let corresponding_child_index =
                            item.corresponding_node_index.and_then(|node_index| {
                                self.nodes.inner[node_index as usize]
                                    .get_index_for_child(child_index)
                            });
                        let child = if let Some(child_index) = corresponding_child_index {
                            self.nodes.inner[child_index as usize]
                        } else {
                            Tree64Node::empty(children_are_leaves)
                        };

                        if !children_are_leaves && !going_up {
                            any_pushed = true;
                            stack.push(StackItem {
                                node_min_pos: node_bbox.min,
                                node_level: item.node_level - 1,
                                corresponding_node_index: corresponding_child_index,
                                children: children_data_for_node(self, child),
                                parent_and_index: Some((index, child_index as u8)),
                            });
                        }

                        if children_are_leaves {
                            item.children.set(
                                child_index,
                                Some(self.modify_leaf_node(child, bbox, node_bbox.min, value))
                                    .filter(|node| node.pop_mask != 0),
                            );
                        }
                    }
                }
            }

            if !any_pushed {
                let node = Tree64Node::new(
                    false,
                    self.insert_nodes(&item.children.array),
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
            }
        }

        self.push_new_root_node(new_root, num_levels);

        UpdatedRanges {
            data: num_data..self.data.inner.len(),
            nodes: num_nodes..self.nodes.inner.len(),
        }
    }

    fn modify_leaf_node(
        &mut self,
        node: Tree64Node,
        bbox: BoundingBox,
        node_pos: glam::UVec3,
        value: u8,
    ) -> Tree64Node {
        let node_bbox = BoundingBox {
            min: node_pos,
            max: node_pos + 4,
        };

        debug_assert!(node.is_leaf());

        let mut intersection = match bbox.get_intersection(&node_bbox) {
            Some(intersection) => intersection,
            None => return node,
        };

        let new_min = intersection.min % 4;
        intersection.max = new_min + (intersection.max - intersection.min);
        intersection.min = new_min;

        let mut pop_masked_data = PopMaskedData {
            array: self.get_leaves_for_node(node),
            pop_mask: node.pop_mask,
        };
        for x in intersection.min.x..intersection.max.x {
            for y in intersection.min.y..intersection.max.y {
                for z in intersection.min.z..intersection.max.z {
                    let index = x + y * 4 + z * 16;
                    pop_masked_data.set(index, Some(value).filter(|&v| v != 0));
                }
            }
        }

        Tree64Node::new(
            true,
            self.insert_values(&pop_masked_data.array),
            pop_masked_data.pop_mask,
        )
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

            Tree64Node::new(true, self.insert_values(&vec), bitmask)
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

            Tree64Node::new(false, self.insert_nodes(&nodes), bitmask)
        }
    }

    pub fn serialize<W: io::Write>(&self, mut writer: W) -> io::Result<()> {
        let (root_node_index, num_levels) = self.root_node_index_and_num_levels();
        writer.write_all(&[num_levels])?;
        writer.write_all(&root_node_index.to_le_bytes())?;
        writer.write_all(&(self.nodes.inner.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.data.inner.len() as u32).to_le_bytes())?;
        writer.write_all(bytemuck::cast_slice(&self.nodes.inner))?;
        writer.write_all(bytemuck::cast_slice(&self.data.inner))?;
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
            nodes: VecWithCaching::from_vec(vec![Tree64Node::default(); num_nodes as usize]),
            data: VecWithCaching::from_vec(vec![0; num_data as usize]),
            stats: Default::default(),
            edits: Tree64Edits {
                roots_and_num_levels: vec![(root_node_index, num_levels)],
                index: 0,
            },
        };
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.nodes.inner))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.data.inner))?;
        Ok(this)
    }
}

fn extend_overlapping<T: PartialEq + Clone + Debug>(vec: &mut Vec<T>, data: &[T]) -> usize {
    for i in (1..=data.len()).rev() {
        let slice_to_match = &data[..i];

        if !vec.ends_with(slice_to_match) {
            continue;
        }

        let pointer = vec.len() - slice_to_match.len();
        vec.extend_from_slice(&data[i..]);
        return pointer;
    }

    let pointer = vec.len();
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
    assert_eq!(&x.array[..], &[1; 64]);
    for i in (0..64).rev() {
        x.set(i, Some(2));
    }
    assert_eq!(&x.array[..], &[2; 64]);
    let mut x = PopMaskedData::<u8>::default();
    for i in (0..64).rev() {
        x.set(i, Some(3));
    }
    assert_eq!(&x.array[..], &[3; 64]);
    for i in (0..64).rev() {
        x.set(i, None);
    }
    assert_eq!(&x.array[..], &[]);
}

#[derive(Default, Clone)]
struct PopMaskedData<T> {
    array: arrayvec::ArrayVec<T, 64>,
    pop_mask: u64,
}

impl<T> PopMaskedData<T> {
    fn set(&mut self, index: u32, value: Option<T>) {
        let occupied = (self.pop_mask >> index) & 1 != 0;
        let array_index = (self.pop_mask & ((1 << index) - 1)).count_ones() as usize;

        match (value, occupied) {
            (Some(value), true) => self.array[array_index] = value,
            (Some(value), false) => {
                self.pop_mask ^= 1 << index;
                self.array.insert(array_index, value);
            }
            (None, true) => {
                self.pop_mask ^= 1 << index;
                self.array.remove(array_index);
            }
            (None, false) => {}
        }
    }
}

#[test]
fn test_tree() {
    {
        let tree = Tree64::new(&[1, 1, 1, 1], [2, 2, 1]);
        dbg!(&tree.data.inner, &tree.nodes.inner);
        assert_eq!(tree.data.inner, &[1; 4]);
        assert_eq!({ tree.nodes.inner[0].is_leaf_and_ptr }, 1);
        assert_eq!({ tree.nodes.inner[0].pop_mask }, 0b00110011);
    }

    {
        let tree = Tree64::new(&[1, 1, 1, 1, 1, 0, 1, 1], [2, 2, 2]);
        assert_eq!(tree.data.inner, &[1; 7]);
        assert_eq!({ tree.nodes.inner[0].is_leaf_and_ptr }, 1);
        assert_eq!(
            { tree.nodes.inner[0].pop_mask },
            0b00000000001100010000000000110011,
            "{:064b}",
            { tree.nodes.inner[0].pop_mask }
        );
    }

    {
        let mut values = [1; 64];
        values[63] = 0;

        let tree = Tree64::new(&values, [4, 4, 4]);
        assert_eq!(tree.data.inner, &[1; 63]);
        assert_eq!({ tree.nodes.inner[0].is_leaf_and_ptr }, 1);
        assert_eq!({ tree.nodes.inner[0].pop_mask }, (!0 & !(1 << 63)));
    }

    {
        let mut values = [1; 64];
        values[63] = 0;
        let tree = Tree64::new(&values, [4, 4, 4]);

        let mut data = Vec::new();
        tree.serialize(&mut data).unwrap();
        let tree2 = Tree64::deserialize(io::Cursor::new(&data)).unwrap();
        assert_eq!(tree.data.inner, tree2.data.inner);
        assert_eq!(tree.nodes.inner, tree2.nodes.inner);
        assert_eq!(tree.edits, tree2.edits);

        let mut tree = tree;
        let ranges = tree.modify([2, 2, 3], 0);
        assert_eq!(ranges.nodes, 1..2);
        assert!(ranges.data.is_empty(), "{:?}", tree.data.inner);

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
        dbg!(&tree.data.inner[ranges.data.clone()]);
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
        dbg!(tree.data.inner);

        assert_eq!(ranges.data.len(), 64);
        assert_ne!(ranges.nodes.len(), 64 + 1);
    }

    {
        let values = [1; 64 * 64 * 64];
        let mut tree = Tree64::new(&values, [64; 3]);

        let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], 2);
        dbg!(tree.data.inner);

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
        dbg!(&tree.nodes.inner);

        let ranges = tree.modify([0; 3], 1);
        dbg!(&tree.nodes.inner);
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
        dbg!(&tree.nodes.inner);

        let ranges = tree.modify([0; 3], 1);
        dbg!(&tree.nodes.inner);
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
        dbg!(&tree.nodes.inner);

        let ranges = tree.modify_nodes_in_box([0; 3], [16; 3], 1);
        dbg!(&tree.nodes.inner);
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
        dbg!(&tree.nodes.inner);

        let ranges = tree.modify_nodes_in_box([2; 3], [4; 3], 1);
        dbg!(&tree.nodes.inner);
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
        dbg!(&tree.nodes.inner);

        let ranges = tree.modify_nodes_in_box([2; 3], [4; 3], 1);
        dbg!(&tree.nodes.inner);
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
        dbg!(&tree.nodes.inner);

        let ranges = tree.modify_nodes_in_box([6; 3], [8; 3], 1);
        dbg!(&tree.nodes.inner);
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct BoundingBox {
    min: glam::UVec3,
    max: glam::UVec3,
}

impl BoundingBox {
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
