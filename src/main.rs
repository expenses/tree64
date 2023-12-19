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

    fn get(&self, coord: glam::IVec2) -> Option<u8> {
        self.inner
            .get(coord.y as usize * self.dim + coord.x as usize)
            .copied()
    }

    fn get_mut(&mut self, coord: glam::IVec2) -> Option<&mut u8> {
        self.inner
            .get_mut(coord.y as usize * self.dim + coord.x as usize)
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

fn put_all(state: &mut State, free_locations: &FreeLocations, replace: &Replace) {
    dbg!(free_locations.iter().count());
    dbg!(state.inner.iter().filter(|&&v| v == 3).count());
    for (pos, delta) in free_locations.iter() {
        for (i, value) in replace.to_as_slice().iter().enumerate() {
            *state.get_mut(pos + delta * i as i32).unwrap() = *value;
        }
    }
}

fn sample_and_put<R: rand::Rng>(
    state: &mut State,
    free_locations: &FreeLocations,
    rng: &mut R,
    replace: &Replace,
) -> bool {
    if let Some((pos, delta)) = free_locations.sample(rng) {
        for (i, value) in replace.to_as_slice().iter().enumerate() {
            *state.get_mut(pos + delta * i as i32).unwrap() = *value;
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

fn matches<F: Fn(usize) -> glam::IVec2>(state: &State, pattern: &[u8], func: F) -> bool {
    pattern
        .iter()
        .enumerate()
        .all(|(i, v)| state.get(func(i)).unwrap() == *v)
}

fn test_replace(state: &State, replace: &Replace, free_locations: &mut FreeLocations) {
    free_locations.clear();

    let slice = replace.from_as_slice();

    let dim = state.dim;

    join4(
        || {
            for y in 0..dim {
                for x in 0..dim + 1 - slice.len() {
                    if matches(state, slice, |i| glam::IVec2::new((x + i) as i32, y as i32)) {
                        free_locations
                            .l2r
                            .push(glam::IVec2::new(x as i32, y as i32));
                    }
                }
            }
        },
        || {
            for y in 0..dim {
                for x in slice.len() - 1..dim {
                    if matches(state, slice, |i| glam::IVec2::new((x - i) as i32, y as i32)) {
                        free_locations
                            .r2l
                            .push(glam::IVec2::new(x as i32, y as i32));
                    }
                }
            }
        },
        || {
            for y in slice.len() - 1..dim {
                for x in 0..dim {
                    if matches(state, slice, |i| glam::IVec2::new(x as i32, (y - i) as i32)) {
                        free_locations
                            .t2b
                            .push(glam::IVec2::new(x as i32, y as i32));
                    }
                }
            }
        },
        || {
            for y in 0..dim + 1 - slice.len() {
                for x in 0..dim {
                    if matches(state, slice, |i| glam::IVec2::new(x as i32, (y + i) as i32)) {
                        free_locations
                            .b2t
                            .push(glam::IVec2::new(x as i32, y as i32));
                    }
                }
            }
        },
    );
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

fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).unwrap();

    let mut command = std::process::Command::new("tev").spawn();
    std::thread::sleep(std::time::Duration::from_millis(16));

    let mut client = tev_client::TevClient::wrap(std::net::TcpStream::connect("127.0.0.1:14158")?);

    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut state = State::new(1024);

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
                dbg!(&count);
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

    let mut values = vec![0.0_f32; state.dim * state.dim * 3];
    let mut free_locations = FreeLocations::default();

    rules.reverse();
    let mut rules_stack = rules;

    let update_freq = 1000;

    for i in 0.. {
        let rule_finished = {
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
        }

        if rule_finished || i % update_freq == 0 {
            update_values(&state, &mut values);
            send_image(&mut client, &format!("step {}", i), state.dim as _, &values);
        }
    }

    send_image(&mut client, "final", state.dim as _, &values);

    Ok(())
}

const MAX_VALUES: usize = 10;

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

#[derive(Default)]
struct Replace {
    from: [u8; MAX_VALUES],
    to: [u8; MAX_VALUES],
    len: u8,
}

impl Replace {
    fn new(from: &[u8], to: &[u8]) -> Self {
        assert_eq!(from.len(), to.len());
        assert!(from.len() <= MAX_VALUES);
        let mut this = Self::default();
        this.from[..from.len()].copy_from_slice(from);
        this.to[..to.len()].copy_from_slice(to);
        this.len = from.len() as u8;
        this
    }

    fn from_as_slice(&self) -> &[u8] {
        &self.from[..self.len as usize]
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
