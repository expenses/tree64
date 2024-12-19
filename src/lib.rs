use pyo3::prelude::*;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::collections::HashSet;

use arrays::{Array3D, ArrayPair};
use pattern_matching::{match_pattern, OverlappingRegexIter, Permutation};
use rand::{rngs::SmallRng, Rng, SeedableRng};

mod arrays;
mod bespoke_regex;
mod pattern_matching;
mod python;
mod write_vox;

mod palette {
    include!(concat!(env!("OUT_DIR"), "/palette.rs"));
}

#[derive(Clone, Debug)]
struct Node<T> {
    ty: NodeTy<T>,
    settings: NodeSettings,
}

#[derive(Clone, Debug)]
enum NodeTy<T> {
    // Try to apply a rewrite rule
    Rule(T),
    // Try to apply a child node in sequential order
    Markov(Vec<Node<T>>),
    // Try to apply a child node in random order
    One {
        children: Vec<Node<T>>,
        node_index_storage: Vec<usize>,
    },
    Sequence(Vec<Node<T>>),
    All(Vec<T>),
    Prl(Vec<T>),
}

fn map_node<T, U, F: FnMut(T) -> U>(node: Node<T>, mut map: &mut F) -> Node<U> {
    Node {
        ty: match node.ty {
            NodeTy::Rule(rule) => NodeTy::Rule(map(rule)),
            NodeTy::All(rules) => NodeTy::All(rules.into_iter().map(&mut map).collect::<Vec<_>>()),
            NodeTy::Prl(rules) => NodeTy::Prl(rules.into_iter().map(&mut map).collect::<Vec<_>>()),
            NodeTy::Markov(children) => NodeTy::Markov(
                children
                    .into_iter()
                    .map(|node| map_node(node, map))
                    .collect(),
            ),
            NodeTy::One { children, .. } => NodeTy::One {
                children: children
                    .into_iter()
                    .map(|node| map_node(node, map))
                    .collect(),
                node_index_storage: Vec::new(),
            },
            NodeTy::Sequence(children) => NodeTy::Sequence(
                children
                    .into_iter()
                    .map(|node| map_node(node, map))
                    .collect(),
            ),
        },
        settings: node.settings.clone(),
    }
}

#[pyclass]
#[derive(Default, Clone, Debug)]
struct NodeSettings {
    count: Option<u32>,
}
#[pymethods]
impl NodeSettings {
    #[new]
    #[pyo3(signature =  (count = None))]
    fn new(count: Option<u32>) -> Self {
        Self { count }
    }
}

fn get_unique_values(slice: &[u8]) -> HashSet<u8> {
    let mut values = [false; 255];

    for &v in slice.iter() {
        values[v as usize] = true;
    }

    values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if *v { Some(i as u8) } else { None })
        .collect()
}

fn execute_root_node<'a>(
    root: Node<Replace>,
    state: &mut Array3D<&mut [u8]>,
    rng: &mut SmallRng,
    callback: Option<Box<dyn Fn(u32) + 'a>>,
) {
    let mut replaces = Vec::new();
    let mut root = Node {
        settings: Default::default(),
        ty: NodeTy::Sequence(vec![map_node(root, &mut |replace| {
            let index = replaces.len();
            replaces.push(replace);
            index
        })]),
    };

    // Record which pattern outputs effect which pattern inputs.
    let mut interactions = Vec::with_capacity(replaces.len());

    for i in 0..replaces.len() {
        let prev = &replaces[i];

        let replacment_interactions: Vec<bool> = (0..replaces.len())
            .map(|j| {
                let next = &replaces[j];
                !next.from_values.is_disjoint(&prev.to_values)
            })
            .collect();

        interactions.push(replacment_interactions);
    }

    let unique_values = get_unique_values(state.inner);

    replaces.par_iter_mut().for_each(|rep| {
        // Don't bother searching for patterns that don't exist in the state.
        // Note: if a pattern is _all_ wildcards then this doesn't work.
        if !rep.from_values.is_subset(&unique_values) {
            return;
        }
        rep.store_initial_matches(state);
    });

    let mut updated = Vec::new();
    execute_node(
        &mut root,
        state,
        &mut replaces,
        &interactions,
        rng,
        &mut updated,
        callback.as_deref(),
    );
}

fn execute_node<'a>(
    node: &mut Node<usize>,
    state: &mut Array3D<&mut [u8]>,
    replaces: &mut [Replace],
    interactions: &[Vec<bool>],
    rng: &mut SmallRng,
    updated: &mut Vec<u32>,
    callback: Option<&(dyn Fn(u32) + 'a)>,
) -> bool {
    match &mut node.ty {
        NodeTy::All(indices) => {
            let mut any_applied = false;

            for index in indices.iter() {
                let replace = &mut replaces[*index];
                replace.store_initial_matches(state);

                for &m in &replace.potential_matches {
                    let permutation = &replace.permutations[m.permutation as usize];

                    if !match_pattern(permutation, state, m.index) {
                        continue;
                    }

                    if rng.gen_range(0.0..1.0) > replace.settings.chance {
                        continue;
                    }

                    for (i, v) in permutation
                        .to
                        .non_wildcard_values_in_state(state.width(), state.height())
                    {
                        any_applied = true;
                        state.put(m.index as usize + i, v);
                    }
                }
            }

            for (i, replace) in replaces.iter_mut().enumerate() {
                // Skip if there is not any interactions between the patterns.
                if !indices.iter().any(|index| interactions[*index][i]) {
                    continue;
                }
                replace.store_initial_matches(state);
            }

            any_applied
        }
        NodeTy::Prl(indices) => {
            let mut any_applied = false;

            for index in indices.iter() {
                replaces[*index].store_initial_matches(state);
            }

            for index in indices.iter() {
                let replace = &mut replaces[*index];

                for &m in &replace.potential_matches {
                    let permutation = &replace.permutations[m.permutation as usize];

                    if rng.gen_range(0.0..1.0) > replace.settings.chance {
                        continue;
                    }

                    for (i, v) in permutation
                        .to
                        .non_wildcard_values_in_state(state.width(), state.height())
                    {
                        any_applied = true;
                        state.put(m.index as usize + i, v);
                    }
                }
            }

            for (i, replace) in replaces.iter_mut().enumerate() {
                // Skip if there is not any interactions between the patterns.
                if !indices.iter().any(|index| interactions[*index][i]) {
                    continue;
                }
                replace.store_initial_matches(state);
            }

            any_applied
        }

        NodeTy::Rule(index) => {
            let applied = {
                updated.clear();

                if replaces[*index].get_match_and_update_state(state, rng, updated) {
                    for (i, replace) in replaces.iter_mut().enumerate() {
                        if !interactions[*index][i] {
                            continue;
                        }
                        replace.update_matches(state, updated);
                    }

                    true
                } else {
                    false
                }
            };

            if applied {
                if let Some(callback) = callback {
                    callback(0);
                }
            }

            applied
        }
        NodeTy::Markov(nodes) => {
            for node in nodes {
                if execute_node(node, state, replaces, interactions, rng, updated, callback) {
                    return true;
                }
            }

            false
        }
        NodeTy::Sequence(nodes) => {
            let mut applied = false;
            for node in nodes {
                for rule_iter in 1.. {
                    if !execute_node(node, state, replaces, interactions, rng, updated, callback) {
                        break;
                    } else {
                        applied = true;
                    }

                    if let Some(count) = node.settings.count {
                        if rule_iter >= count {
                            break;
                        }
                    }
                }
            }

            applied
        }
        NodeTy::One {
            children,
            node_index_storage,
        } => {
            node_index_storage.clear();
            node_index_storage.extend(0..children.len());

            sample_until_valid(node_index_storage, rng, |&node_index, rng| {
                execute_node(
                    &mut children[node_index],
                    state,
                    replaces,
                    interactions,
                    rng,
                    updated,
                    callback,
                )
            })
            .is_some()
        }
    }
}

#[pyfunction]
fn index_for_colour(colour: char) -> Option<u8> {
    palette::CHARS_TO_INDEX
        .get(colour as usize)
        .filter(|&&value| value != 255)
        .copied()
}

pub fn send_image(
    client: &mut tev_client::TevClient,
    values: &mut Vec<f32>,
    linear_palette: &[[f32; 3]],
    name: &str,
    slice: &[u8],
    width: u32,
    height: u32,
) {
    values.resize(slice.len() * 3, 0.0);

    for i in 0..slice.len() {
        let colour = &linear_palette[slice[i] as usize];
        values[i * 3..(i + 1) * 3].copy_from_slice(colour);
    }

    client
        .send(tev_client::PacketCreateImage {
            image_name: name,
            grab_focus: false,
            width,
            height,
            channel_names: &["R", "G", "B"],
        })
        .unwrap();
    client
        .send(tev_client::PacketUpdateImage {
            image_name: name,
            grab_focus: false,
            channel_names: &["R", "G", "B"],
            channel_offsets: &[0, 1, 2],
            channel_strides: &[3, 3, 3],
            x: 0,
            y: 0,
            width,
            height,
            data: values,
        })
        .unwrap();
}

#[pymodule]
fn markov(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::write_vox, m)?)?;
    m.add_function(wrap_pyfunction!(python::rep, m)?)?;
    m.add_function(wrap_pyfunction!(python::colour_image, m)?)?;
    m.add_function(wrap_pyfunction!(python::mesh_voxels, m)?)?;
    m.add_function(wrap_pyfunction!(python::map_2d, m)?)?;
    m.add_function(wrap_pyfunction!(python::map_3d, m)?)?;
    m.add_function(wrap_pyfunction!(index_for_colour, m)?)?;
    m.add_function(wrap_pyfunction!(python::find_closest_pairs, m)?)?;
    m.add_class::<python::Pattern>()?;
    m.add_class::<python::TevClient>()?;
    m.add_class::<python::One>()?;
    m.add_class::<python::Markov>()?;
    m.add_class::<python::Sequence>()?;
    m.add_class::<python::Palette>()?;
    m.add_class::<python::Wfc>()?;
    m.add_class::<python::Tileset>()?;
    m.add_class::<python::Prl>()?;
    m.add_class::<python::All>()?;
    m.add_class::<NodeSettings>()?;

    m.add(
        "PICO8_PALETTE",
        python::Palette::new(palette::COLOURS.to_vec()),
    )?;
    m.add("PALETTE_CHARS_TO_INDEX", palette::CHARS_TO_INDEX.to_vec())?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Match {
    index: u32,
    permutation: u8,
}

// Keep sampling random values from a list until a valid one is found.
// Values are removed when sampled.
fn sample_until_valid<T, F: FnMut(&T, &mut SmallRng) -> bool>(
    values: &mut Vec<T>,
    rng: &mut SmallRng,
    mut valid: F,
) -> Option<(T, u32)> {
    let mut iters = 0;
    while !values.is_empty() {
        let index = rng.gen_range(0..values.len());
        let value = values.swap_remove(index);
        iters += 1;
        if valid(&value, rng) {
            return Some((value, iters));
        }
    }

    None
}

// todo: paths, convolutions, do in python. rrmove callback index.

const WILDCARD: u8 = 255;

#[derive(Default, Clone)]
struct ReplaceSettings {
    chance: f32,
}

struct Replace {
    permutations: Vec<Permutation>,
    potential_matches: Vec<Match>,
    from_values: HashSet<u8>,
    to_values: HashSet<u8>,
    reinit: bool,
    settings: ReplaceSettings,
}

impl Replace {
    fn new(
        from: Array3D,
        mut to: Array3D,
        shuffles: &[[usize; 3]],
        flips: &[[bool; 3]],
        settings: ReplaceSettings,
        state: &Array3D<&mut [u8]>,
    ) -> Self {
        let dims = from.dims();
        let [width, height, depth] = dims;
        assert_eq!(from.dims(), to.dims());

        // Small optimization to reduce the interaction between patterns.
        for i in 0..from.inner.len() {
            if from.inner[i] == to.inner[i] {
                to.inner[i] = WILDCARD;
            }
        }

        let pair = ArrayPair { to, from };

        let from_values: HashSet<u8> = pair
            .from
            .inner
            .iter()
            .copied()
            .filter(|&v| v != WILDCARD)
            .collect();
        let to_values: HashSet<u8> = pair
            .to
            .inner
            .iter()
            .copied()
            .filter(|&v| v != WILDCARD)
            .collect();

        // Get a set of unique permutations.
        let mut permutations = HashSet::new();

        for &[x_mapping, y_mapping, z_mapping] in shuffles {
            for &[flip_x, flip_y, flip_z] in flips {
                permutations.insert(pair.permute(
                    dims[x_mapping],
                    dims[y_mapping],
                    dims[z_mapping],
                    |x, y, z| {
                        let array = [
                            if flip_x { width - 1 - x } else { x },
                            if flip_y { height - 1 - y } else { y },
                            if flip_z { depth - 1 - z } else { z },
                        ];
                        (array[x_mapping], array[y_mapping], array[z_mapping])
                    },
                ));
            }
        }

        Self {
            permutations: permutations
                .into_iter()
                .map(|p| Permutation::new(state, p))
                .collect(),
            potential_matches: Default::default(),
            from_values,
            to_values,
            reinit: false,
            settings,
        }
    }

    fn store_initial_matches(&mut self, state: &Array3D<&mut [u8]>) {
        self.potential_matches = self
            .permutations
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, permutation)| {
                OverlappingRegexIter::new(&permutation.bespoke_regex, state.inner)
                    .filter(|&index| {
                        state.shape_is_inbounds(
                            index,
                            permutation.width(),
                            permutation.height(),
                            permutation.depth(),
                        )
                    })
                    .map(move |index| Match {
                        index: index as u32,
                        permutation: i as u8,
                    })
            })
            .collect();
    }

    fn get_match_and_update_state(
        &mut self,
        state: &mut Array3D<&mut [u8]>,
        rng: &mut SmallRng,
        updated: &mut Vec<u32>,
    ) -> bool {
        if self.reinit {
            self.store_initial_matches(state);
            self.reinit = false;
        }

        let m = match sample_until_valid(&mut self.potential_matches, rng, |&m, _| {
            match_pattern(&self.permutations[m.permutation as usize], state, m.index)
        }) {
            Some((m, iters)) => {
                // If it took more than 500 attempts to find a match
                // then just re initialize the whole thing next time.
                self.reinit = iters > 500;
                m
            }
            None => return false,
        };

        let to = &self.permutations[m.permutation as usize].to;

        for (i, v) in to.non_wildcard_values_in_state(state.width(), state.height()) {
            state.put(m.index as usize + i, v);
            updated.push(m.index + i as u32);
        }
        true
    }

    fn update_matches(&mut self, state: &Array3D<&mut [u8]>, updated_cells: &[u32]) {
        for &index in updated_cells {
            for (i, permutation) in self.permutations.iter().enumerate() {
                for (x, y, z) in permutation.coords() {
                    let offset =
                        (state.width() * state.height() * z + state.width() * y + x) as u32;
                    if offset > index {
                        continue;
                    }
                    let index = index - offset;
                    if !match_pattern(permutation, state, index) {
                        continue;
                    }
                    self.potential_matches.push(Match {
                        index,
                        permutation: i as u8,
                    });
                }
            }
        }
    }
}
