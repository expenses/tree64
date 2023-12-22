use pyo3::prelude::*;

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

pub fn send_image(client: &mut tev_client::TevClient, values: &mut Vec<f32>, name: &str, slice: &[u8], width: u32, height: u32) {
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
            data: &values,
        })
        .unwrap();
}

pub mod python {
    use super::*;
    use pyo3::types::PyList;

    #[pyclass]
    pub struct TevClient {
        inner: tev_client::TevClient,
        values: Vec<f32>,
    }

    #[pymethods]
    impl TevClient {
        #[new]
        fn new() -> Self {
            Self {
                inner: tev_client::TevClient::wrap(std::net::TcpStream::connect("127.0.0.1:14158").unwrap()),
                values: Vec::new()
            }
        }

        pub fn send_image(&mut self, name: &str, array: numpy::borrow::PyReadonlyArray2<u8>) {
            let dims = array.dims();
            let slice = array.as_slice().unwrap();

            super::send_image(&mut self.inner, &mut self.values, name, slice, dims[0] as u32, dims[1] as u32);
        }
    }

    #[pyclass]
    #[derive(Clone)]
    pub struct PatternWithOptions {
        pattern: String,
        allow_rot90: bool,
    }

    #[pymethods]
    impl PatternWithOptions {
        #[new]
        fn new(pattern: String, allow_rot90: Option<bool>) -> Self {
            Self {
                pattern,
                allow_rot90: allow_rot90.unwrap_or(true),
            }
        }
    }

    #[pyfunction]
    pub fn rep(
        mut array: numpy::borrow::PyReadwriteArray2<u8>,
        patterns: &PyList,
        priority_after: Option<usize>,
        times: Option<u32>,
    ) {
        let mut replaces = Vec::new();

        for pattern in patterns {
            let pattern =   if let Ok(pattern) = pattern.extract::<PatternWithOptions>() {
                pattern
            } else {
                PatternWithOptions::new(pattern.extract::<&str>().unwrap().to_string(), None)
            };

            replaces.push(Replace::from_string(&pattern.pattern, pattern.allow_rot90));
        }

        let mut array_2d = Array2D {
            width: array.dims()[0],
            height: array.dims()[1],
            inner: array.as_slice_mut().unwrap(),
        };

        let mut rng = rand::rngs::SmallRng::from_entropy();

        execute_rule(
            &mut array_2d,
            &mut rng,
            &mut replaces,
            times.unwrap_or(0),
            priority_after,
        );
    }

    #[pyfunction]
    pub fn rep_all(mut array: numpy::borrow::PyReadwriteArray2<u8>, patterns: &PyList) {
        let mut replaces = Vec::new();

        for pattern in patterns {
            replaces.push(Replace::from_string(pattern.extract::<&str>().unwrap(), true));
        }

        let mut array_2d = Array2D {
            width: array.dims()[0],
            height: array.dims()[1],
            inner: array.as_slice_mut().unwrap(),
        };

        execute_rule_all(&mut array_2d, &mut replaces);
    }

    #[pyfunction]
    pub fn index_for_colour(colour: char) -> Option<u8> {
        super::index_for_colour(colour)
    }
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

use rand::SeedableRng;

struct Array2D<T = Vec<u8>> {
    inner: T,
    width: usize,
    height: usize,
}

impl Array2D<Vec<u8>> {
    fn new(slice: &[u8], width: usize) -> Self {
        Self {
            inner: slice.to_owned(),
            width: width,
            height: slice.len() / width,
        }
    }
}

trait MutByteSlice: std::ops::Deref<Target = [u8]> + std::ops::DerefMut {}

impl<T> MutByteSlice for T where T: std::ops::Deref<Target = [u8]> + std::ops::DerefMut {}

impl<T: MutByteSlice> Array2D<T> {
    fn is_symmetrical(&self) -> bool {
        self.inner.iter().eq(self.inner.iter().rev())
    }

    fn iter_locations_for_dir(&self, dir: Direction) -> impl Iterator<Item = glam::I16Vec2> + '_ {
        (0..self.width)
            .flat_map(|x| (0..self.height).map(move |y| (x, y)))
            .map(move |(x, y)| dir.delta() * x as i16 + dir.rot_90().delta() * y as i16)
    }

    fn iter_match_locations_and_values(
        &self,
        m: Match,
    ) -> impl Iterator<Item = (glam::I16Vec2, u8)> + '_ {
        (0..self.width)
            .flat_map(|x| (0..self.height).map(move |y| (x, y)))
            .filter_map(move |(x, y)| {
                let value = self.inner[y * self.width + x];

                if value != WILDCARD {
                    let position = m.pos + m.dir.delta() * x as i16 + m.dir.rot_90().delta() * y as i16;
                    Some((position, value))
                } else {
                    None
                }
            })
    }

    fn get(&self, coord: glam::I16Vec2) -> Option<u8> {
        if coord.x < 0 || coord.x >= self.width as i16 || coord.y < 0 {
            return None;
        }

        let index = coord.y as usize * self.width + coord.x as usize;
        self.inner.get(index).copied()
    }

    fn put(&mut self, coord: glam::I16Vec2, value: u8) -> bool {
        let index = coord.y as usize * self.width + coord.x as usize;
        let previous_value = self.inner[index];
        self.inner[index] = value;
        previous_value != value
    }
}

fn can_replace_pattern<T: MutByteSlice>(pattern: &Array2D, state: &Array2D<T>, m: Match) -> bool {
    for (pos, v) in pattern.iter_match_locations_and_values(m) {
        if state.get(pos) != Some(v) {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy, Debug)]
enum Direction {
    L2R,
    R2L,
    B2T,
    T2B,
}

impl Direction {
    fn rot_90(&self) -> Direction {
        match self {
            Self::L2R => Self::T2B,
            Self::T2B => Self::R2L,
            Self::R2L => Self::B2T,
            Self::B2T => Self::L2R,
        }
    }

    fn delta(&self) -> glam::I16Vec2 {
        match self {
            Self::L2R => glam::I16Vec2::new(1, 0),
            Self::R2L => glam::I16Vec2::new(-1, 0),
            Self::B2T => glam::I16Vec2::new(0, -1),
            Self::T2B => glam::I16Vec2::new(0, 1),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Match {
    pos: glam::I16Vec2,
    dir: Direction,
}

// Keep sampling random values from a list until a valid one is found.
// Values are removed when sampled.
fn sample_until_valid<T, R: rand::Rng, F: Fn(&T) -> bool>(
    values: &mut Vec<T>,
    rng: &mut R,
    valid: F,
) -> Option<T> {
    while !values.is_empty() {
        let index = rng.gen_range(0..values.len());
        let value = values.swap_remove(index);
        if valid(&value) {
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
                None => index_for_colour(c).unwrap()
            }
        };
        i += 1;
    }

    (pattern, i, row_width.unwrap_or(i))
}

fn execute_rule<R: rand::Rng, T: MutByteSlice>(
    state: &mut Array2D<T>,
    rng: &mut R,
    replaces: &mut [Replace],
    n_times_or_inf: u32,
    stop_after_nth_replace: Option<usize>,
) {
    // Record which pattern outputs effect which pattern inputs.
    let mut interactions = Vec::with_capacity(replaces.len());

    for i in 0..replaces.len() {
        let prev = &replaces[i];

        // Todo: this has pretty bad time complexity for large patterns.
        let replacment_interactions: Vec<bool> = (0..replaces.len())
            .map(|j| {
                let next = &replaces[j];
                next.from.inner.contains(&WILDCARD) ||
                next.from
                    .inner
                    .iter()
                    .any(|next_value| prev.to.inner.contains(&next_value))
            })
            .collect();

        interactions.push(replacment_interactions);
    }

    for rep in replaces.iter_mut() {
        rep.store_initial_matches(&state);
    }

    let mut updated = Vec::new();

    for rule_iter in 1.. {
        let mut rule_finished = true;

        for i in 0..replaces.len() {
            updated.clear();

            if replaces[i].get_match_and_update_state(state, rng, &mut updated) {
                rule_finished = false;

                for j in 0..replaces.len() {
                    if !interactions[i][j] {
                        continue;
                    }
                    replaces[j].update_matches(state, &updated);
                }

                if let Some(n) = stop_after_nth_replace {
                    if n <= i {
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

fn execute_rule_all<T: MutByteSlice>(state: &mut Array2D<T>, replaces: &mut [Replace]) {
    for rep in replaces.iter_mut() {
        rep.store_initial_matches(&state);

        for &m in &rep.potential_matches {
            if !can_replace_pattern(&rep.from, &state, m) {
                continue;
            }

            for (pos, v) in rep.to.iter_match_locations_and_values(m) {
                state.put(pos, v);
            }
        }
    }
}

const WILDCARD: u8 = 255;

struct Replace {
    to: Array2D,
    from: Array2D,
    potential_matches: Vec<Match>,
    applicable_directions: &'static [Direction],
}

impl Replace {
    fn new(from: &[u8], to: &[u8], width: usize, allow_rot90: bool) -> Self {
        let from = Array2D::new(from, width);

        Self {
            to: Array2D::new(to, width),
            applicable_directions: if from.is_symmetrical() {
                if allow_rot90 {
                    &[Direction::L2R, Direction::T2B]
                } else {
                    &[Direction::L2R]
                }
            } else if allow_rot90 {
                &[
                    Direction::L2R,
                    Direction::T2B,
                    Direction::R2L,
                    Direction::B2T,
                ]
            } else {
                &[
                    Direction::L2R,
                    Direction::R2L,
                ]
            },
            from,
            potential_matches: Default::default(),
        }
    }

    fn from_string(string: &str, allow_rot90: bool) -> Self {
        let (from, to) = string.split_once('=').unwrap();
        let (from, from_len, from_width) = pattern_from_chars(from);
        let (to, to_len, to_width) = pattern_from_chars(to);
        assert_eq!(from_width, to_width);
        Self::new(&from[..from_len], &to[..to_len], from_width, allow_rot90)
    }

    fn store_initial_matches<T: MutByteSlice>(&mut self, state: &Array2D<T>) {
        for &dir in self.applicable_directions {
            for x in 0..state.width {
                for y in 0..state.height {
                    let m = Match {
                        pos: glam::I16Vec2::new(x as i16, y as i16),
                        dir,
                    };

                    if !can_replace_pattern(&self.from, state, m) {
                        continue;
                    }

                    self.potential_matches.push(m);
                }
            }
        }
    }

    fn get_match_and_update_state<T: MutByteSlice, R: rand::Rng>(
        &mut self,
        state: &mut Array2D<T>,
        rng: &mut R,
        updated: &mut Vec<glam::I16Vec2>,
    ) -> bool {
        let m = match sample_until_valid(&mut self.potential_matches, rng, |&m| {
            can_replace_pattern(&self.from, state, m)
        }) {
            Some(m) => m,
            None => return false,
        };

        for (pos, v) in self.to.iter_match_locations_and_values(m) {
            if v == WILDCARD {
                continue;
            }

            if state.put(pos, v) {
                updated.push(pos);
            }
        }

        true
    }

    fn update_matches<T: MutByteSlice>(
        &mut self,
        state: &Array2D<T>,
        updated_cells: &[glam::I16Vec2],
    ) {
        for &pos in updated_cells {
            for &dir in self.applicable_directions {
                // todo: this can lead to cases of biasing. Think of the case where a 2x2 cube has been placed.
                // By performing updated_cells.len() (4) x num from cells (4) = 16 checks, we're checking the middle cube 4 times.
                // We should only be doing (2 + (2-1))x(2+ (2-1)) = 9 checks.
                for rel_pos in self.from.iter_locations_for_dir(dir) {
                    let new_match = Match {
                        pos: pos - rel_pos,
                        dir,
                    };

                    if !can_replace_pattern(&self.from, state, new_match) {
                        continue;
                    }

                    self.potential_matches.push(new_match);
                }
            }
        }
    }
}
