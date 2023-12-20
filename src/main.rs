use rand::{Rng, SeedableRng};
use std::io::{BufRead, Read};

mod util;

use util::*;

struct State {
    inner: Vec<u8>,
    dim: usize,
}

impl State {
    fn new(dim: usize) -> Self {
        Self {
            inner: vec![0; dim * dim],
            dim,
        }
    }

    fn row(&self, index: usize) -> &[u8] {
        &self.inner[index * self.dim..(index + 1) * self.dim]
    }

    fn put(&mut self, coord: glam::IVec2, value: u8) -> bool {
        let index = coord.y as usize * self.dim + coord.x as usize;
        let previous_value = self.inner[index];
        self.inner[index] = value;
        previous_value != value
    }
}

const COLOURS: &[[f32; 3]] = &[
    [0.0; 3],
    [1.0; 3],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.5, 0.0],
    [1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0],
];

fn update_values(state: &State, values: &mut [f32]) {
    for i in 0..state.inner.len() {
        let colour = &COLOURS[state.inner[i] as usize];
        values[i * 3..(i + 1) * 3].copy_from_slice(colour);
    }
}

fn can_replace_pattern(pattern: &[u8], state: &State, m: Match) -> bool {
    for (i, v) in pattern.iter().enumerate() {
        let pos = m.pos + delta_for_rule(m.rule) * i as i32;

        if pos.y >= state.dim as i32 || pos.y < 0 {
            return false;
        }

        if state.row(pos.y as usize).get(pos.x as usize) != Some(&v) {
            return false;
        }
    }
    true
}

fn test_against_update(state: &mut State, matches: &mut Matches, updated: &[glam::IVec2], pattern: &[u8]) {
    for &pos in updated {
        for rule in 0..4 {
            for j in 0 .. pattern.len() {
                let new_match = Match { pos: pos - delta_for_rule(rule) * j as i32, rule };
                matches.try_push(new_match, |m| can_replace_pattern(pattern, &state, m));
            }
        }
    }
}

/*
fn put_all(state: &mut State, free_locations: &FreeLocations, replace: &Replace) {
    for &(pos, delta) in free_locations {
        for (i, value) in replace.to_as_slice().iter().enumerate() {
            state.put(pos + delta * i as i32, *value);
        }
    }
}

fn sample_and_put<R: rand::Rng>(
    state: &mut State,
    free_locations: &FreeLocations,
    rng: &mut R,
    replace: &Replace,
) -> bool {
    if !free_locations.is_empty() {
        let index = rng.gen_range(0..free_locations.len());
        let (pos, delta) = free_locations[index];

        for (i, value) in replace.to_as_slice().iter().enumerate() {
            state.put(pos + delta * i as i32, *value);
        }

        true
    } else {
        false
    }
}

fn join4<A: FnOnce() + Send, B: FnOnce() + Send, C: FnOnce() + Send, D: FnOnce() + Send>(
    a: A,
    b: B,
    c: C,
    d: D,
) {
    rayon::join(|| rayon::join(a, b), || rayon::join(c, d));
}

type FreeLocations = Vec<(glam::IVec2, glam::IVec2)>;
*/

fn delta_for_rule(rule: u8) -> glam::IVec2 {
    match rule {
        0 => glam::IVec2::new(1, 0),
        1 => glam::IVec2::new(0, 1),
        2 => glam::IVec2::new(-1, 0),
        3 => glam::IVec2::new(0, -1),
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, Debug)]
struct Match {
    pos: glam::IVec2,
    rule: u8,
}

#[derive(Default)]
struct Matches {
    matches: Vec<Match>,
}

impl Matches {
    fn try_push<F: Fn(Match) -> bool>(&mut self, m: Match, valid: F) {
        if valid(m) {
            self.matches.push(m);
        }
    }

    fn get_valid_match<R: Rng, F: Fn(Match) -> bool>(
        &mut self,
        rng: &mut R,
        valid: F,
    ) -> Option<Match> {
        let mut i = 0;
        while !self.matches.is_empty() {
            let mut index = rng.gen_range(0..self.matches.len());

            let m = self.matches.swap_remove(index);

            if valid(m) {
                return Some(m);
            }

            i += 1;
        }

        None
    }
}
/*
fn test_replace(state: &State, replace: &Replace, free_locations: &mut FreeLocations) {
    let dim = state.dim;
    let offset = replace.len as i32 - 1;

    free_locations.clear();

    for y in 0..dim {
        let row = state.row(y);

        for x in replace.finder.find_iter(row) {
            free_locations
                .push((glam::IVec2::new(x as i32, y as i32), glam::IVec2::new(1, 0)))
        }
    }

    if replace.len == 1 {
        return;
    }

    for x in 0..dim {
        let row = state.col(x);

        for y in replace.finder.find_iter(row) {
            free_locations
                .push((glam::IVec2::new(x as i32, y as i32), glam::IVec2::new(0, 1)))
        }
    }

    for y in 0..dim {
        let row = state.row(y);

        for x in replace.finder_rev.find_iter(row) {
            free_locations.push((
                glam::IVec2::new(x as i32 + offset, y as i32),
                glam::IVec2::new(-1, 0),
            ))
        }
    }

    for x in 0..dim {
        let row = state.col(x);

        for y in replace.finder_rev.find_iter(row) {
            free_locations.push((
                glam::IVec2::new(x as i32, y as i32 + offset),
                glam::IVec2::new(0, -1),
            ))
        }
    }
}

fn pattern_from_chars(chars: &str) -> [u8; MAX_VALUES] {
    let mut pattern = [0; MAX_VALUES];
    for (i, c) in chars.chars().enumerate() {
        pattern[i] = c.to_digit(10).unwrap() as _;
    }
    pattern
}

enum Mode {
    Normal,
    Times(u32),
    All,
    Priority,
}
*/
fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).unwrap();

    let mut command = std::process::Command::new("tev").spawn();
    std::thread::sleep(std::time::Duration::from_millis(16));

    let mut client = tev_client::TevClient::wrap(std::net::TcpStream::connect("127.0.0.1:14158")?);

    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut state = State::new(2048);

    /*
    let input = std::fs::read_to_string(&model).unwrap();
    let mut patterns = Vec::new();
    let mut rules = Vec::new();
    let mut mode = Mode::Normal;

    let mut push_rule = |patterns: &mut Vec<Replace>, mode: &mut Mode| {
        let patterns = std::mem::take(patterns);

        match *mode {
            Mode::Normal => {
                rules.push(Rule::Once(patterns));
            }
            Mode::Times(times) => {
                rules.push(Rule::n_times(patterns, times));
            }
            Mode::All => {
                rules.push(Rule::All(patterns));
            }
            Mode::Priority => {
                rules.push(Rule::Priority(patterns));
            }
        }
        *mode = Mode::Normal;
    };

    for line in input.lines() {
        if line.is_empty() {
            push_rule(&mut patterns, &mut mode);
            continue;
        } else if line.starts_with("-") {
            let line = &line[2..];
            if line.starts_with("times") {
                let (_, count) = line.split_once(" = ").unwrap();
                mode = Mode::Times(count.parse().unwrap())
            } else if line.starts_with("all") {
                mode = Mode::All;
            } else if line.starts_with("priority") {
                mode = Mode::Priority;
            }
            continue;
        }

        for pattern in line.split(' ') {
            let (from, to) = pattern.split_once('=').unwrap();
            patterns.push(Replace::new(
                &pattern_from_chars(from)[..from.len()],
                &pattern_from_chars(to)[..to.len()],
            ))
        }
    }
    if !patterns.is_empty() {
        push_rule(&mut patterns, &mut mode);
    }
    */
    let mut values = vec![0.0_f32; state.dim * state.dim * 3];
    /*let mut free_locations = FreeLocations::default();

    rules.reverse();
    let mut rules_stack = rules;
    */
    let update_freq = 1000000;

    let mut matches = Matches::default();

    state.put(glam::IVec2::new(128, 128), 1);

    for i in 0..3000000 {
        /*let rule_finished = {
            let rule = match rules_stack.last_mut() {
                Some(rule) => rule,
                None => {
                    break;
                }
            };

            match rule {
                Rule::OnceNTimes {
                    replaces,
                    total,
                    current,
                } => {
                    for replace in replaces {
                        test_replace(&state, &replace, &mut free_locations);
                        sample_and_put(&mut state, &free_locations, &mut rng, &replace);
                    }

                    *current += 1;

                    *current == *total
                }
                Rule::All(replaces) => {
                    for replace in replaces {
                        test_replace(&state, &replace, &mut free_locations);
                        put_all(&mut state, &free_locations, &replace);
                    }
                    true
                }
                Rule::Once(replaces) => {
                    let mut finished = true;

                    for replace in replaces {
                        test_replace(&state, &replace, &mut free_locations);

                        if sample_and_put(&mut state, &free_locations, &mut rng, &replace) {
                            finished = false;
                        }
                    }

                    finished
                }
                Rule::Priority(replaces) => {
                    let mut finished = true;

                    for replace in replaces {
                        test_replace(&state, &replace, &mut free_locations);

                        if sample_and_put(&mut state, &free_locations, &mut rng, &replace) {
                            finished = false;
                            break;
                        }
                    }

                    finished
                }
            }
        };

        if rule_finished {
            rules_stack.pop();
        }*/

        let from = &[1, 0];
        let to = &[1, 1];

        if i == 0 {
            for x in 0 .. state.dim {
                for y in 0 .. state.dim {
                    for rule in 0 .. 4 {
                        let m = Match {
                            pos: glam::IVec2::new(x as i32, y as i32),
                            rule,
                        };

                        matches.try_push(m, |m| can_replace_pattern(from, &state, m));
                    }
                }
            }
        } else {
            let mat = match matches.get_valid_match(&mut rng, |m| can_replace_pattern(from, &state, m)) {
                Some(m) => m,
                None => break,
            };

            let mut updated = Vec::new();

            for (i, v) in to.iter().enumerate() {
                let pos = mat.pos + delta_for_rule(mat.rule) * i as i32;
                if state.put(pos, *v) {
                    updated.push(pos);
                }
            }

            test_against_update(&mut state, &mut matches, &updated, from);
        }

        if i % update_freq == 0 {
            update_values(&state, &mut values);
            send_image(&mut client, &format!("step {}", i), state.dim as _, &values);
        }

    }

    update_values(&state, &mut values);
    send_image(&mut client, "final", state.dim as _, &values);

    Ok(())
}

const MAX_VALUES: usize = 10;
const MAX_COLOURS: usize = 10;

#[derive(Default)]
struct Pattern {
    values: [u8; MAX_VALUES],
    len: u8,
}

impl Pattern {
    fn new(slice: &[u8]) -> Self {
        assert!(slice.len() <= MAX_VALUES);
        let mut this = Self::default();
        this.values[..slice.len()].copy_from_slice(slice);
        this.len = slice.len() as u8;
        this
    }

    fn as_slice(&self) -> &[u8] {
        &self.values[..self.len as usize]
    }
}
type LocationsForColour = [Vec<glam::IVec2>; MAX_COLOURS];

struct Replace {
    finder: memchr::memmem::Finder<'static>,
    finder_rev: memchr::memmem::Finder<'static>,
    input: LocationsForColour,
    output: LocationsForColour,
    to: [u8; MAX_VALUES],
    len: u8,
}

impl Replace {
    fn new(from: &[u8], to: &[u8]) -> Self {
        assert_eq!(from.len(), to.len());
        assert!(from.len() <= MAX_VALUES);

        let mut to_arr = [0; MAX_VALUES];
        to_arr[..to.len()].copy_from_slice(to);

        let from_rev: Vec<u8> = from.iter().rev().copied().collect();

        let mut input = LocationsForColour::default();
        let mut output = LocationsForColour::default();

        for (i, v) in from.iter().enumerate() {
            input[*v as usize].push(glam::IVec2::new(i as i32, 0));
        }

        for (i, v) in to.iter().enumerate() {
            output[*v as usize].push(glam::IVec2::new(i as i32, 0));
        }

        Self {
            finder: memchr::memmem::Finder::new(from).into_owned(),
            finder_rev: memchr::memmem::Finder::new(&from_rev).into_owned(),
            to: to_arr,
            input,
            output,
            len: from.len() as u8,
        }
    }

    fn to_as_slice(&self) -> &[u8] {
        &self.to[..self.len as usize]
    }
}

enum Rule {
    All(Vec<Replace>),
    OnceNTimes {
        replaces: Vec<Replace>,
        total: u32,
        current: u32,
    },
    Once(Vec<Replace>),
    Priority(Vec<Replace>),
}

impl Rule {
    fn n_times(replaces: Vec<Replace>, total: u32) -> Self {
        Self::OnceNTimes {
            replaces,
            total,
            current: 0,
        }
    }
}
