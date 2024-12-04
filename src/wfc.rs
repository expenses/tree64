use crate::{
    arrays::{compose, decompose},
    Array3D,
};
use rand::{prelude::SliceRandom, rngs::SmallRng, Rng, SeedableRng};

type Wave = u64;

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
    NegX = 3,
    NegY = 4,
    NegZ = 5,
}

impl Axis {
    const ALL: [Self; 6] = [
        Self::X,
        Self::Y,
        Self::Z,
        Self::NegX,
        Self::NegY,
        Self::NegZ,
    ];

    fn opp(&self) -> Axis {
        match self {
            Self::X => Self::NegX,
            Self::Y => Self::NegY,
            Self::Z => Self::NegZ,
            Self::NegX => Self::X,
            Self::NegY => Self::Y,
            Self::NegZ => Self::Z,
        }
    }
}

#[derive(Default, Debug)]
struct Tile {
    connections: [Wave; 6],
}

impl Tile {
    fn connect(&mut self, other: usize, axis: Axis) {
        self.connections[axis as usize] |= 1 << other;
    }
}

struct Wfc {
    tiles: arrayvec::ArrayVec<Tile, { Wave::BITS as _ }>,
    array: Vec<Wave>,
    width: usize,
    height: usize,
    entropy_to_indices: [indexmap::IndexSet<usize>; { Wave::BITS as usize - 2 }],
    tile_list: arrayvec::ArrayVec<u8, { Wave::BITS as _ }>,
    stack: Vec<(usize, Wave)>,
}

impl Wfc {
    fn new(size: (usize, usize, usize)) -> Self {
        let (width, height, depth) = size;
        Self {
            tiles: Default::default(),
            tile_list: Default::default(),
            entropy_to_indices: [(); { Wave::BITS as usize - 2 }].map(|()| Default::default()),
            array: vec![0; width * height * depth],
            width,
            height,
            stack: Vec::new(),
        }
    }

    fn depth(&self) -> usize {
        self.array.len() / self.width / self.height
    }

    fn add(&mut self) -> usize {
        let index = self.tiles.len();
        self.tiles.push(Tile::default());
        index
    }

    fn connect(&mut self, from: usize, to: usize, axises: &[Axis]) {
        for &axis in axises {
            self.tiles[from].connect(to, axis);
            self.tiles[to].connect(from, axis.opp());
        }
    }

    fn setup_state(&mut self) {
        let wave = Wave::MAX >> (Wave::BITS as usize - self.tiles.len());
        for value in &mut self.array {
            *value = wave;
        }
        for i in 0..self.array.len() {
            self.entropy_to_indices[self.tiles.len() - 2].insert(i);
        }
    }

    fn fill_tile_list_from_value(&mut self, value: Wave) {
        self.tile_list.clear();

        for i in (0..Wave::BITS) {
            if ((value >> i) & 1) == 0 {
                continue;
            }

            self.tile_list.push(i as _);
        }
    }

    fn find_lowest_entropy(&mut self, rng: &mut SmallRng) -> Option<(usize, u8)> {
        let lowest_entropy_set = match self.entropy_to_indices.iter().find(|set| !set.is_empty()) {
            Some(set) => set,
            None => return None,
        };

        let index = rng.gen_range(0..lowest_entropy_set.len());
        let index = *lowest_entropy_set.get_index(index).unwrap();

        let value = self.array[index];

        self.fill_tile_list_from_value(value);

        let tile = *self.tile_list.choose(rng).unwrap();

        Some((index, tile))
    }

    fn collapse_all(&mut self, rng: &mut SmallRng) {
        while let Some((index, tile)) = self.find_lowest_entropy(rng) {
            self.collapse(index, tile);
        }
    }

    fn collapse(&mut self, index: usize, tile: u8) {
        self.set(index, 1 << tile);
    }

    fn set(&mut self, index: usize, remaining_possible_states: Wave) {
        self.stack.clear();
        self.stack.push((index, remaining_possible_states));

        while let Some((index, remaining_possible_states)) = self.stack.pop() {
            let old = self.array[index];
            self.array[index] &= remaining_possible_states;
            let new = self.array[index];

            if old == new || new == 0 {
                continue;
            }

            self.entropy_to_indices[old.count_ones() as usize - 2].swap_remove(&index);
            if new.count_ones() > 1 {
                self.entropy_to_indices[new.count_ones() as usize - 2].insert(index);
            }

            for axis in Axis::ALL {
                let (mut x, mut y, mut z) = decompose(index, self.width, self.height);
                match axis {
                    Axis::X if x < self.width - 1 => x += 1,
                    Axis::Y if y < self.height - 1 => y += 1,
                    Axis::Z if z < self.depth() - 1 => z += 1,
                    Axis::NegX if x > 0 => x -= 1,
                    Axis::NegY if y > 0 => y -= 1,
                    Axis::NegZ if z > 0 => z -= 1,
                    _ => continue,
                };

                let index = compose(x, y, z, self.width, self.height);

                let mut valid = 0;

                self.fill_tile_list_from_value(new);

                for &tile in self.tile_list.iter() {
                    valid |= self.tiles[tile as usize].connections[axis as usize]
                }

                self.stack.push((index, valid));
            }
        }
    }

    fn all_collapsed(&self) -> bool {
        self.array.iter().all(|&value| value.count_ones() == 1)
    }

    fn values(&self) -> Vec<u8> {
        self.array
            .iter()
            .map(|&value| value.trailing_zeros() as u8)
            .collect()
    }
}

#[test]
fn test() {
    let mut rng = SmallRng::from_entropy();

    {
        let mut wfc = Wfc::new((1000, 1000, 1));
        let sea = wfc.add();
        let beach = wfc.add();
        let grass = wfc.add();
        wfc.connect(sea, sea, &Axis::ALL);
        wfc.connect(sea, beach, &Axis::ALL);
        wfc.connect(beach, beach, &Axis::ALL);
        wfc.connect(beach, grass, &Axis::ALL);
        wfc.connect(grass, grass, &Axis::ALL);

        assert_eq!(wfc.tiles[sea].connections, [3; 6]);

        wfc.setup_state();

        assert!(!wfc.all_collapsed());
        wfc.collapse_all(&mut rng);
        assert!(
            wfc.all_collapsed(),
            "{:?}",
            &wfc.array.iter().map(|v| v.count_ones()).collect::<Vec<_>>()
        );
    }

    // Wait until there's a collapse failure due to beaches not being able to connect to beaches.
    loop {
        let mut wfc = Wfc::new((10, 10, 1));
        let sea = wfc.add();
        let beach = wfc.add();
        let grass = wfc.add();
        wfc.connect(sea, sea, &Axis::ALL);
        wfc.connect(sea, beach, &Axis::ALL);
        //wfc.connect(beach, beach, &Axis::ALL);
        wfc.connect(beach, grass, &Axis::ALL);
        wfc.connect(grass, grass, &Axis::ALL);

        assert_eq!(wfc.tiles[sea].connections, [3; 6]);

        wfc.setup_state();

        assert!(!wfc.all_collapsed());
        wfc.collapse_all(&mut rng);

        if !wfc.all_collapsed() {
            // Make sure that at least one state has collapsed properly (aka that the error hasn't spread).
            assert!(wfc.array.iter().any(|&v| v.count_ones() == 1));
            break;
        }
    }
}
