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
    m.add_class::<python::PatternWithOptions>()?;
    m.add_class::<python::TevClient>()?;
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
) -> Option<T> {
    while !values.is_empty() {
        let index = rng.gen_range(0..values.len());
        let value = values.swap_remove(index);
        if valid(&value, rng) {
            return Some(value);
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

fn execute_rule(
    state: &mut Array2D<&mut [u8]>,
    rng: &mut SmallRng,
    replaces: &mut [Replace],
    n_times_or_inf: u32,
    stop_after_nth_replace: Option<usize>,
) {
    // Record which pattern outputs effect which pattern inputs.
    let mut interactions = Vec::with_capacity(replaces.len());

    for i in 0..replaces.len() {
        let prev = &replaces[i];

        let replacment_interactions: Vec<bool> = (0..replaces.len())
            .map(|j| {
                let next = &replaces[j];
                next.from_values.contains(&WILDCARD)
                    || !next.from_values.is_disjoint(&prev.to_values)
            })
            .collect();

        interactions.push(replacment_interactions);
    }

    replaces.par_iter_mut().for_each(|rep| {
        rep.store_initial_matches(state);
    });

    let mut updated = Vec::new();
    let mut rule_indices = Vec::new();

    for rule_iter in 1.. {
        let mut rule_finished = true;

        rule_indices.extend(0..stop_after_nth_replace.unwrap_or(replaces.len()));

        // Try all the non-fallback rules first. Randomly select a valid one.
        sample_until_valid(&mut rule_indices, rng, |&rule_index, rng| {
            updated.clear();

            if replaces[rule_index].get_match_and_update_state(state, rng, &mut updated) {
                for (i, replace) in replaces.iter_mut().enumerate() {
                    if !interactions[rule_index][i] {
                        continue;
                    }
                    replace.update_matches(state, &updated);
                }

                rule_finished = false;
                true
            } else {
                false
            }
        });

        if rule_finished {
            if let Some(nth) = stop_after_nth_replace {
                for i in nth..replaces.len() {
                    updated.clear();
                    if replaces[i].get_match_and_update_state(state, rng, &mut updated) {
                        for (j, replace) in replaces.iter_mut().enumerate() {
                            if !interactions[i][j] {
                                continue;
                            }
                            replace.update_matches(state, &updated);
                        }

                        rule_finished = false;
                        break;
                    }
                }
            }
        }

        if rule_finished || rule_iter == n_times_or_inf {
            return;
        }
    }
}

fn execute_rule_all(state: &mut Array2D<&mut [u8]>, replaces: &mut [Replace]) {
    for rep in replaces.iter_mut() {
        rep.store_initial_matches(state);

        for &m in &rep.potential_matches {
            if !match_pattern(&rep.permutations[m.permutation as usize], state, m.index) {
                continue;
            }

            let to = &rep.permutations[m.permutation as usize].to;

            for y in 0..to.height {
                for x in 0..to.width {
                    let v = to.inner[to.width * y + x];
                    let i = m.index as usize + x + y * state.width;
                    if v == WILDCARD {
                        continue;
                    }
                    state.put(i, v);
                }
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
}

impl Replace {
    fn new(
        from: &[u8],
        to: &[u8],
        width: usize,
        allow_rot90: bool,
        allow_vertical_flip: bool,
        state: &Array2D<&mut [u8]>,
    ) -> Self {
        let height = from.len() / width;

        let pair = ArrayPair {
            to: Array2D::new(to, width),
            from: Array2D::new(from, width),
        };

        let from_values: HashSet<u8> = pair.from.inner.iter().copied().collect();
        let to_values: HashSet<u8> = pair.to.inner.iter().copied().collect();

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
        let (to, to_len, to_width) = pattern_from_chars(to);
        assert_eq!(from_width, to_width);
        Self::new(
            &from[..from_len],
            &to[..to_len],
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
                    .filter(|index| (index % state.width + permutation.to.width) <= state.width)
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
        let m = match sample_until_valid(&mut self.potential_matches, rng, |&m, _| {
            match_pattern(&self.permutations[m.permutation as usize], state, m.index)
        }) {
            Some(m) => m,
            None => return false,
        };

        let to = &self.permutations[m.permutation as usize].to;

        for y in 0..to.height {
            for x in 0..to.width {
                let v = to.inner[to.width * y + x];
                let i = m.index as usize + x + y * state.width;
                if v == WILDCARD {
                    continue;
                }
                if state.put(i, v) {
                    updated.push(i as u32);
                }
            }
        }

        true
    }

    fn update_matches(&mut self, state: &Array2D<&mut [u8]>, updated_cells: &[u32]) {
        for &index in updated_cells {
            for (i, permutation) in self.permutations.iter().enumerate() {
                for x in 0..permutation.to.width {
                    for y in 0..permutation.to.height {
                        let index = index - x as u32 - (state.width * y) as u32;
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
