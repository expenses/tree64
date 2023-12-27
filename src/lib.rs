use pyo3::prelude::*;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::collections::HashSet;

use arrays::{Array2D, ArrayPair};
use pattern_matching::{match_pattern, OverlappingRegexIter, Permutation};
use rand::{rngs::SmallRng, Rng, SeedableRng};

mod arrays;
mod bespoke_regex;
mod pattern_matching;
mod python;

#[derive(Clone, Debug)]
enum Node<T> {
    // Try to apply a rewrite rule
    Rule(T),
    // Try to apply a child node in sequential order
    Markov(Vec<Node<T>>),
    // Try to apply a child node in random order
    One(Vec<Node<T>>),
}

fn map_node<T, U, F: Fn(&T) -> U>(node: &Node<T>, map: &F) -> Node<U> {
    match node {
        Node::Rule(rule) => Node::Rule(map(rule)),
        Node::Markov(children) => {
            Node::Markov(children.iter().map(|node| map_node(node, map)).collect())
        }
        Node::One(children) => Node::One(children.iter().map(|node| map_node(node, map)).collect()),
    }
}

fn index_node(node: Node<Replace>, replaces: &mut Vec<Replace>) -> IndexNode {
    match node {
        Node::Rule(rep) => {
            let index = replaces.len();
            replaces.push(rep);
            IndexNode::Rule(index)
        }
        Node::Markov(nodes) => IndexNode::Markov(
            nodes
                .into_iter()
                .map(|node| index_node(node, replaces))
                .collect(),
        ),
        Node::One(nodes) => IndexNode::One {
            children: nodes
                .into_iter()
                .map(|node| index_node(node, replaces))
                .collect(),
            node_index_storage: Vec::new(),
        },
    }
}

enum IndexNode {
    Rule(usize),
    Markov(Vec<IndexNode>),
    One {
        children: Vec<IndexNode>,
        node_index_storage: Vec<usize>,
    },
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
    state: &mut Array2D<&mut [u8]>,
    rng: &mut SmallRng,
    max_iterations: Option<u32>,
    callback: Option<Box<dyn Fn(u32) + 'a>>,
) {
    let mut replaces = Vec::new();
    let mut root = index_node(root, &mut replaces);

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

    let unique_values = get_unique_values(&state.inner);

    replaces.par_iter_mut().for_each(|rep| {
        // Don't bother searching for patterns that don't exist in the state.
        // Note: if a pattern is _all_ wildcards then this doesn't work.
        if !rep.from_values.is_subset(&unique_values) {
            return;
        }
        rep.store_initial_matches(state);
    });

    let mut updated = Vec::new();

    for rule_iter in 1.. {
        if !execute_node(
            &mut root,
            state,
            &mut replaces,
            &interactions,
            rng,
            &mut updated,
        ) {
            return;
        }

        if let Some(callback) = callback.as_ref() {
            callback(rule_iter);
        }

        if let Some(max_iterations) = max_iterations {
            if rule_iter >= max_iterations {
                return;
            }
        }
    }
}

fn execute_node(
    node: &mut IndexNode,
    state: &mut Array2D<&mut [u8]>,
    replaces: &mut [Replace],
    interactions: &[Vec<bool>],
    rng: &mut SmallRng,
    updated: &mut Vec<u32>,
) -> bool {
    match node {
        IndexNode::Rule(index) => {
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
        }
        IndexNode::Markov(nodes) => {
            for node in nodes {
                if execute_node(node, state, replaces, interactions, rng, updated) {
                    return true;
                }
            }

            false
        }
        IndexNode::One {
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
                )
            })
            .is_some()
        }
    }
}

// From https://github.com/mxgmn/MarkovJunior#rewrite-rules,
// but swapped around to have B then W then R.
pub const PALETTE: [char; 16] = [
    'B', // Black
    'W', // White
    'R', // Red
    'I', // Dark blue
    'P', // Dark purple
    'E', // Dark green
    'N', // Brown
    'D', // Dark grey (Dead)
    'A', // Light grey (Alive)
    'O', // Orange
    'Y', // Yellow
    'G', // Green
    'U', // Blue
    'S', // Lavender
    'K', // Pink
    'F', // Light peach
];

// https://pico-8.fandom.com/wiki/Palette
pub const SRGB_PALETTE_VALUES: [[u8; 3]; 16] = [
    [0, 0, 0],
    [255, 241, 232],
    [255, 0, 7],
    [29, 43, 83],
    [126, 37, 83],
    [0, 135, 81],
    [171, 82, 54],
    [95, 87, 79],
    [194, 195, 199],
    [255, 163, 0],
    [255, 236, 39],
    [0, 228, 54],
    [41, 173, 255],
    [131, 118, 156],
    [255, 119, 168],
    [255, 204, 170],
];

fn index_for_colour(colour: char) -> Option<u8> {
    PALETTE.iter().position(|&c| c == colour).map(|v| v as u8)
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;

    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub fn send_image(
    client: &mut tev_client::TevClient,
    values: &mut Vec<f32>,
    name: &str,
    slice: &[u8],
    width: u32,
    height: u32,
) {
    let colours = SRGB_PALETTE_VALUES.map(|v| v.map(srgb_to_linear));

    values.resize(slice.len() * 3, 0.0);

    for i in 0..slice.len() {
        let colour = &colours[slice[i] as usize];
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
fn markov(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::rep, m)?)?;
    m.add_function(wrap_pyfunction!(python::rep_all, m)?)?;
    m.add_function(wrap_pyfunction!(python::index_for_colour, m)?)?;
    m.add_function(wrap_pyfunction!(python::colour_image, m)?)?;
    m.add_class::<python::PatternWithOptions>()?;
    m.add_class::<python::TevClient>()?;
    m.add_class::<python::One>()?;
    m.add_class::<python::Markov>()?;
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

fn pattern_from_chars(chars: &str) -> ([u8; 100], usize, usize) {
    let mut pattern = [0; 100];
    let mut i = 0;
    let mut row_width = None;
    for c in chars.chars() {
        if c == ' ' || c == '\n' {
            continue;
        }

        if c == ',' {
            if row_width.is_none() {
                row_width = Some(i);
            }
            continue;
        }

        pattern[i] = match c {
            '*' => WILDCARD,
            _ => match c.to_digit(10) {
                Some(digit) => digit as u8,
                None => index_for_colour(c).unwrap(),
            },
        };
        i += 1;
    }

    (pattern, i, row_width.unwrap_or(i))
}

fn execute_rule_all(state: &mut Array2D<&mut [u8]>, replaces: &mut [Replace]) {
    for rep in replaces.iter_mut() {
        rep.store_initial_matches(state);

        for &m in &rep.potential_matches {
            if !match_pattern(&rep.permutations[m.permutation as usize], state, m.index) {
                continue;
            }

            let to = &rep.permutations[m.permutation as usize].to;

            for (i, v) in to.non_wildcard_values_in_state(state.width()) {
                state.put(m.index as usize + i, v);
            }
        }
    }
}

const WILDCARD: u8 = 255;

struct Replace {
    permutations: Vec<Permutation>,
    potential_matches: Vec<Match>,
    from_values: HashSet<u8>,
    to_values: HashSet<u8>,
    reinit: bool,
}

impl Replace {
    fn new(
        from: &[u8],
        to: &mut [u8],
        width: usize,
        allow_rot90: bool,
        allow_vertical_flip: bool,
        state: &Array2D<&mut [u8]>,
    ) -> Self {
        let height = from.len() / width;

        // Small optimization to reduce the interaction between patterns.
        for i in 0..from.len() {
            if from[i] == to[i] {
                to[i] = WILDCARD;
            }
        }

        let pair = ArrayPair {
            to: Array2D::new(to, width),
            from: Array2D::new(from, width),
        };

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

        permutations.insert(pair.permute(width, height, |x, y| (width - 1 - x, y)));
        if allow_vertical_flip {
            permutations.insert(pair.permute(width, height, |x, y| (x, height - 1 - y)));
            permutations
                .insert(pair.permute(width, height, |x, y| (width - 1 - x, height - 1 - y)));
        }
        if allow_rot90 {
            permutations.insert(pair.permute(height, width, |x, y| (y, x)));
            permutations.insert(pair.permute(height, width, |x, y| (y, width - 1 - x)));
            permutations.insert(pair.permute(height, width, |x, y| (height - 1 - y, x)));
            permutations
                .insert(pair.permute(height, width, |x, y| (height - 1 - y, width - 1 - x)));
        }
        permutations.insert(pair);

        Self {
            permutations: permutations
                .into_iter()
                .map(|p| Permutation::new(state, p))
                .collect(),
            potential_matches: Default::default(),
            from_values,
            to_values,
            reinit: false,
        }
    }

    fn from_string(
        string: &str,
        allow_rot90: bool,
        allow_vertical_flip: bool,
        state: &Array2D<&mut [u8]>,
    ) -> Self {
        let (from, to) = string.split_once('=').unwrap();
        let (from, from_len, from_width) = pattern_from_chars(from);
        let (mut to, to_len, to_width) = pattern_from_chars(to);
        assert_eq!(from_width, to_width);
        Self::new(
            &from[..from_len],
            &mut to[..to_len],
            from_width,
            allow_rot90,
            allow_vertical_flip,
            state,
        )
    }

    fn store_initial_matches(&mut self, state: &Array2D<&mut [u8]>) {
        self.potential_matches = self
            .permutations
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, permutation)| {
                OverlappingRegexIter::new(&permutation.bespoke_regex, state.inner)
                    .filter(|index| (index % state.width() + permutation.width()) <= state.width())
                    .map(move |index| Match {
                        index: index as u32,
                        permutation: i as u8,
                    })
            })
            .collect();
    }

    fn get_match_and_update_state(
        &mut self,
        state: &mut Array2D<&mut [u8]>,
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

        for (i, v) in to.non_wildcard_values_in_state(state.width()) {
            state.put(m.index as usize + i, v);
            updated.push(m.index + i as u32);
        }
        true
    }

    fn update_matches(&mut self, state: &Array2D<&mut [u8]>, updated_cells: &[u32]) {
        for &index in updated_cells {
            for (i, permutation) in self.permutations.iter().enumerate() {
                for x in 0..permutation.width() {
                    for y in 0..permutation.height() {
                        let index = index - x as u32 - (state.width() * y) as u32;
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
}
