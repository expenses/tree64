use rand::SeedableRng;
use std::rc::Rc;

struct Array2D {
    inner: Vec<u8>,
    width: usize,
    height: usize,
}

impl Array2D {
    fn new(slice: &[u8], width: usize) -> Self {
        Self {
            inner: slice.to_owned(),
            width: width,
            height: slice.len() / width,
        }
    }

    fn empty_square(dim: usize) -> Self {
        Self {
            inner: vec![0; dim * dim],
            width: dim,
            height: dim,
        }
    }

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
            .map(move |(x, y)| {
                let value = self.inner[y * self.width + x];
                let position = m.pos + m.dir.delta() * x as i16 + m.dir.rot_90().delta() * y as i16;
                (position, value)
            })
    }

    fn get(&self, coord: glam::I16Vec2) -> Option<u8> {
        if coord.x < 0 || coord.x >= self.width as i16 {
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

fn can_replace_pattern(pattern: &Array2D, state: &Array2D, m: Match) -> bool {
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

        pattern[i] = c.to_digit(10).unwrap() as _;
        i += 1;
    }

    (pattern, i, row_width.unwrap_or(i))
}

fn execute_rule<R: rand::Rng>(
    state: &mut Array2D,
    rng: &mut R,
    replaces: &mut [Replace],
    n_times_or_inf: u32,
    stop_after_first_replace: bool,
) {
    // Record which pattern outputs effect which pattern inputs.
    let mut interactions = Vec::with_capacity(replaces.len());

    for i in 0..replaces.len() {
        let prev = &replaces[i];

        let mut replacment_interactions: Vec<bool> = (0..replaces.len())
            .map(|j| {
                let next = &replaces[j];
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

                if stop_after_first_replace {
                    break;
                }
            }
        }

        if rule_finished || rule_iter == n_times_or_inf {
            return;
        }
    }
}

fn execute_rule_all<R: rand::Rng>(
    state: &mut Array2D,
    rng: &mut R,
    updated: &mut Vec<glam::I16Vec2>,
    replaces: &mut [Replace],
) {
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

fn parse_value_from_table_as_pattern(
    value: mlua::Value,
    index: i64,
) -> Result<Replace, mlua::Error> {
    let pattern_str = match value.as_str() {
        Some(string) => string,
        None => {
            return Err(mlua::Error::BadArgument {
                pos: index as usize,
                name: None,
                to: None,
                cause: mlua::Error::runtime("integer arguments must be strings").into(),
            })
        }
    };
    let (from, to) = pattern_str.split_once('=').unwrap();
    let (from, from_len, from_width) = pattern_from_chars(from);
    let (to, to_len, to_width) = pattern_from_chars(to);
    assert_eq!(from_width, to_width);
    Ok(Replace::new(&from[..from_len], &to[..to_len], from_width))
}

struct GlobalState {
    state: Array2D,
    rng: rand::rngs::SmallRng,
    tev_client: tev_client::TevClient,
    values: Vec<f32>,
    i: usize,
}

fn main() -> anyhow::Result<()> {
    let lua = mlua::Lua::new();

    let mut rng = rand::rngs::SmallRng::from_entropy();
    let mut state = Array2D::empty_square(4096);

    let _command = std::process::Command::new("tev").spawn();
    std::thread::sleep(std::time::Duration::from_millis(16));

    let mut client = tev_client::TevClient::wrap(std::net::TcpStream::connect("127.0.0.1:14158")?);

    let state = Rc::new(std::cell::RefCell::new(GlobalState {
        values: vec![0.0_f32; state.width * state.height * 3],
        state: state,
        rng: rng,
        tev_client: client,
        i: 0,
    }));

    let state_c = state.clone();

    lua.globals()
        .set(
            "rep",
            mlua::Function::wrap_mut(move |_lua, table: mlua::Table| {
                let mut times = 0;
                let mut patterns = Vec::new();
                let mut priority = false;

                table.for_each::<mlua::Value, mlua::Value>(|key, value| {
                    match key {
                        mlua::Value::Integer(index) => {
                            patterns.push(parse_value_from_table_as_pattern(value, index)?);
                        }
                        mlua::Value::String(option) => match option.to_str().unwrap() {
                            "times" => {
                                times = value.as_u32().unwrap();
                            }
                            "priority" => {
                                priority = value.as_boolean().unwrap();
                            }
                            _ => panic!(),
                        },
                        other => panic!("{:?}", other),
                    }
                    Ok(())
                })?;

                let state = &mut *state_c.borrow_mut();

                execute_rule(
                    &mut state.state,
                    &mut state.rng,
                    &mut patterns,
                    times,
                    priority,
                );

                send_image(
                    &mut state.tev_client,
                    &format!("step {}", state.i),
                    &mut state.values,
                    &state.state,
                );
                state.i += 1;

                Ok(())
            }),
        )
        .unwrap();

    lua.globals()
        .set(
            "rep_all",
            mlua::Function::wrap_mut(move |_lua, table: mlua::Table| {
                let mut patterns = Vec::new();
                let mut updated = Vec::new();

                table.for_each::<mlua::Value, mlua::Value>(|key, value| {
                    match key {
                        mlua::Value::Integer(index) => {
                            patterns.push(parse_value_from_table_as_pattern(value, index)?);
                        }
                        other => panic!("{:?}", other),
                    }
                    Ok(())
                })?;

                let state = &mut *state.borrow_mut();

                execute_rule_all(
                    &mut state.state,
                    &mut state.rng,
                    &mut updated,
                    &mut patterns,
                );

                send_image(
                    &mut state.tev_client,
                    &format!("step {}", state.i),
                    &mut state.values,
                    &state.state,
                );
                state.i += 1;

                Ok(())
            }),
        )
        .unwrap();

    let model = std::env::args().nth(1).unwrap();
    let txt = std::fs::read_to_string(model).unwrap();

    if let Err(error) = lua.load(txt).exec() {
        println!("{}", error);
    };

    Ok(())
}

struct Replace {
    to: Array2D,
    from: Array2D,
    is_symmetrical: bool,
    potential_matches: Vec<Match>,
    applicable_directions: &'static [Direction],
}

impl Replace {
    fn new(from: &[u8], to: &[u8], width: usize) -> Self {
        let from = Array2D::new(from, width);

        let is_symmetrical = from.is_symmetrical();

        Self {
            to: Array2D::new(to, width),
            is_symmetrical,
            applicable_directions: if is_symmetrical {
                &[Direction::L2R, Direction::T2B]
            } else {
                &[
                    Direction::L2R,
                    Direction::T2B,
                    Direction::R2L,
                    Direction::B2T,
                ]
            },
            from,
            potential_matches: Default::default(),
        }
    }

    fn store_initial_matches(&mut self, state: &Array2D) {
        for x in 0..state.width {
            for y in 0..state.height {
                for &dir in self.applicable_directions {
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
        state: &mut Array2D,
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
            if state.put(pos, v) {
                updated.push(pos);
            }
        }

        true
    }

    fn update_matches(&mut self, state: &Array2D, updated_cells: &[glam::I16Vec2]) {
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

fn send_image(client: &mut tev_client::TevClient, name: &str, values: &mut [f32], state: &Array2D) {
    for i in 0..state.inner.len() {
        let colour = &COLOURS[state.inner[i] as usize];
        values[i * 3..(i + 1) * 3].copy_from_slice(colour);
    }

    client
        .send(tev_client::PacketCreateImage {
            image_name: name,
            grab_focus: false,
            width: state.width as _,
            height: state.height as _,
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
            width: state.width as _,
            height: state.height as _,
            data: values,
        })
        .unwrap();
}
