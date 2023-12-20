use rand::SeedableRng;

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

    fn get(&self, coord: glam::I16Vec2) -> Option<u8> {
        if coord.x < 0 || coord.x >= self.dim as i16 {
            return None;
        }

        let index = coord.y as usize * self.dim + coord.x as usize;
        self.inner.get(index).copied()
    }

    fn put(&mut self, coord: glam::I16Vec2, value: u8) -> bool {
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

fn can_replace_pattern(pattern: &[u8], state: &State, m: Match) -> bool {
    for (i, v) in pattern.iter().enumerate() {
        let pos = m.pos + m.dir.delta() * i as i16;

        if state.get(pos) != Some(*v) {
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
    fn delta(&self) -> glam::I16Vec2 {
        match self {
            Self::L2R => glam::I16Vec2::new(1, 0),
            Self::R2L => glam::I16Vec2::new(-1, 0),
            Self::B2T => glam::I16Vec2::new(0, 1),
            Self::T2B => glam::I16Vec2::new(0, -1),
        }
    }

    fn enumerate() -> [Self; 4] {
        [Self::L2R, Self::R2L, Self::B2T, Self::T2B]
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

fn pattern_from_chars(chars: &str) -> [u8; 10] {
    let mut pattern = [0; 10];
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

    let _command = std::process::Command::new("tev").spawn();
    std::thread::sleep(std::time::Duration::from_millis(16));

    let mut client = tev_client::TevClient::wrap(std::net::TcpStream::connect("127.0.0.1:14158")?);

    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut state = State::new(1024);

    let input = std::fs::read_to_string(model).unwrap();
    let mut patterns = Vec::new();
    let mut rules = Vec::new();
    let mut mode = Mode::Normal;

    let mut push_rule = |patterns: &mut Vec<Replace>, mode: &mut Mode| {
        if patterns.is_empty() {
            return;
        }

        let patterns = std::mem::take(patterns);

        match *mode {
            Mode::Normal => {
                rules.push(Rule::Once {
                    replaces: patterns,
                    n_times_or_inf: 0,
                    stop_after_first_replace: false,
                });
            }
            Mode::Times(times) => {
                rules.push(Rule::Once {
                    replaces: patterns,
                    n_times_or_inf: times,
                    stop_after_first_replace: false,
                });
            }
            Mode::All => {
                rules.push(Rule::All(patterns));
            }
            Mode::Priority => {
                rules.push(Rule::Once {
                    replaces: patterns,
                    n_times_or_inf: 0,
                    stop_after_first_replace: true,
                });
            }
        }
        *mode = Mode::Normal;
    };

    for line in input.lines() {
        if line.starts_with('-') {
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

        push_rule(&mut patterns, &mut mode);

        if line.is_empty() {
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
    push_rule(&mut patterns, &mut mode);

    let mut values = vec![0.0_f32; state.dim * state.dim * 3];

    let update_freq = 1_000_000;

    let mut current_rule_iter = 0;
    let mut updated = Vec::new();

    // Flip the rules into a stack that we pop rules off of when finished.
    // This allows us to conserve memory by dropping the reasonably large
    // `potential_matches` values when they are no longer used.
    rules.reverse();

    for i in 0..100_000_000 {
        let mut rule_finished = true;

        let end = rules.len() - 1;

        match rules.get_mut(end) {
            Some(Rule::Once {
                replaces,
                n_times_or_inf,
                stop_after_first_replace,
            }) => {
                if current_rule_iter == 0 {
                    for rep in replaces.iter_mut() {
                        rep.store_initial_matches(&state);
                    }
                }

                for i in 0..replaces.len() {
                    updated.clear();

                    if replaces[i].get_match_and_update_state(&mut state, &mut rng, &mut updated) {
                        rule_finished = false;

                        for rep in replaces.iter_mut() {
                            rep.update_matches(&state, &updated);
                        }

                        if *stop_after_first_replace {
                            break;
                        }
                    }
                }

                current_rule_iter += 1;

                rule_finished |= current_rule_iter == *n_times_or_inf;
            }
            Some(Rule::All(replaces)) => {
                for rep in replaces.iter_mut() {
                    rep.store_initial_matches(&state);

                    for &m in &rep.potential_matches {
                        if !can_replace_pattern(&rep.from, &state, m) {
                            continue;
                        }

                        for (i, v) in rep.to.iter().enumerate() {
                            let pos = m.pos + m.dir.delta() * i as i16;
                            state.put(pos, *v);
                        }
                    }
                }
            }
            None => break,
        };

        if rule_finished {
            rules.pop();
            current_rule_iter = 0;
        }

        if rule_finished || i % update_freq == 0 {
            send_image(
                &mut client,
                &format!("step {}", i),
                state.dim as _,
                &mut values,
                &state,
            );
        }
    }

    send_image(&mut client, "final", state.dim as _, &mut values, &state);

    Ok(())
}

struct Replace {
    to: Vec<u8>,
    from: Vec<u8>,
    potential_matches: Vec<Match>,
}

impl Replace {
    fn new(from: &[u8], to: &[u8]) -> Self {
        Self {
            to: to.to_owned(),
            from: from.to_owned(),
            potential_matches: Default::default(),
        }
    }

    fn store_initial_matches(&mut self, state: &State) {
        for x in 0..state.dim {
            for y in 0..state.dim {
                for dir in Direction::enumerate() {
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

    fn get_match_and_update_state<R: rand::Rng>(
        &mut self,
        state: &mut State,
        rng: &mut R,
        updated: &mut Vec<glam::I16Vec2>,
    ) -> bool {
        let m = match sample_until_valid(&mut self.potential_matches, rng, |&m| {
            can_replace_pattern(&self.from, state, m)
        }) {
            Some(m) => m,
            None => return false,
        };

        for (i, v) in self.to.iter().enumerate() {
            let pos = m.pos + m.dir.delta() * i as i16;
            if state.put(pos, *v) {
                updated.push(pos);
            }
        }

        true
    }

    fn update_matches(&mut self, state: &State, updated_cells: &[glam::I16Vec2]) {
        for &pos in updated_cells {
            for dir in Direction::enumerate() {
                // check the whole pattern against the updated cells.
                for j in 0..self.from.len() {
                    let new_match = Match {
                        pos: pos - dir.delta() * j as i16,
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

enum Rule {
    All(Vec<Replace>),
    Once {
        replaces: Vec<Replace>,
        // 0 for inf.
        n_times_or_inf: u32,
        // Aka prioritizing / markov chains
        stop_after_first_replace: bool,
    },
}

fn send_image(
    client: &mut tev_client::TevClient,
    name: &str,
    dim: u32,
    values: &mut [f32],
    state: &State,
) {
    for i in 0..state.inner.len() {
        let colour = &COLOURS[state.inner[i] as usize];
        values[i * 3..(i + 1) * 3].copy_from_slice(colour);
    }

    client
        .send(tev_client::PacketCreateImage {
            image_name: name,
            grab_focus: false,
            width: dim,
            height: dim,
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
            width: dim,
            height: dim,
            data: values,
        })
        .unwrap();
}
