use super::*;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::PyTupleMethods;
use pyo3::types::{PyFunction, PyTuple};

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
            inner: tev_client::TevClient::wrap(
                std::net::TcpStream::connect("127.0.0.1:14158").unwrap(),
            ),
            values: Vec::new(),
        }
    }

    pub fn send_image(
        &mut self,
        name: &str,
        array: numpy::borrow::PyReadonlyArray2<u8>,
        palette: &Palette,
    ) {
        let dims = array.dims();
        let slice = array.as_slice().unwrap();

        super::send_image(
            &mut self.inner,
            &mut self.values,
            &palette.linear,
            name,
            slice,
            dims[0] as u32,
            dims[1] as u32,
        );
    }
}

fn split_pattern_string(pattern: &str) -> PyResult<(&str, &str)> {
    pattern
        .split_once('=')
        .ok_or_else(|| PyTypeError::new_err("missing '=' in pattern string"))
}

const ALL_FLIPS: [[bool; 3]; 8] = [
    [false, false, false],
    [false, false, true],
    [false, true, false],
    [false, true, true],
    [true, false, false],
    [true, false, true],
    [true, true, false],
    [true, true, true],
];

const ALL_SHUFFLES: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[pyclass]
#[derive(Clone)]
pub struct Pattern {
    from: Array3D,
    to: Array3D,
    options: PatternOptions,
}

#[pyclass]
#[derive(Clone)]
pub struct PatternOptions {
    shuffles: Vec<[usize; 3]>,
    flips: Vec<[bool; 3]>,
    settings: ReplaceSettings,
    node_settings: NodeSettings,
}
impl Pattern {
    fn new_from_pattern(pattern: PatternInput) -> PyResult<Self> {
        let (from, to) = pattern.arrays()?;
        Ok(Self {
            from,
            to,
            options: PatternOptions {
                shuffles: ALL_SHUFFLES.to_vec(),
                flips: ALL_FLIPS.to_vec(),

                settings: ReplaceSettings {
                    apply_all: false,
                    chance: 10.0,
                },
                node_settings: Default::default(),
            },
        })
    }
}

#[pymethods]
impl Pattern {
    #[new]
    #[pyo3(signature = (pattern, shuffles = None, flips = None, apply_all = None, chance = None, node_settings = None))]
    fn new(
        pattern: PatternInput,
        shuffles: Option<Vec<[usize; 3]>>,
        flips: Option<Vec<[bool; 3]>>,
        apply_all: Option<bool>,
        chance: Option<f32>,
        node_settings: Option<NodeSettings>,
    ) -> PyResult<Self> {
        let mut pattern = Self::new_from_pattern(pattern)?;
        pattern.options.flips = flips.unwrap_or(pattern.options.flips);
        pattern.options.shuffles = shuffles.unwrap_or(pattern.options.shuffles);
        pattern.options.settings.apply_all =
            apply_all.unwrap_or(pattern.options.settings.apply_all);
        pattern.options.settings.chance = chance.unwrap_or(pattern.options.settings.chance);
        pattern.options.node_settings = node_settings.unwrap_or(pattern.options.node_settings);
        Ok(pattern)
    }
}

type PatternList = Vec<Node<Pattern>>;

fn parse_pattern_list(list: Bound<PyTuple>) -> PyResult<PatternList> {
    list.iter()
        .map(|item| {
            item.extract::<PythonNode>()
                .and_then(|python_node| python_node.convert())
        })
        .collect::<PyResult<Vec<Node<Pattern>>>>()
}

#[derive(Clone)]
#[pyclass]
pub struct One(PatternList, NodeSettings);

#[pymethods]
impl One {
    #[new]
    #[pyo3(signature = (*list, settings=None))]
    fn new(list: Bound<PyTuple>, settings: Option<NodeSettings>) -> PyResult<Self> {
        Ok(Self(
            parse_pattern_list(list)?,
            settings.unwrap_or_default(),
        ))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Markov(PatternList, NodeSettings);

#[pymethods]
impl Markov {
    #[new]
    #[pyo3(signature = (*list, settings=None))]
    fn new(list: Bound<PyTuple>, settings: Option<NodeSettings>) -> PyResult<Self> {
        Ok(Self(
            parse_pattern_list(list)?,
            settings.unwrap_or_default(),
        ))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Sequence(PatternList, NodeSettings);

#[pymethods]
impl Sequence {
    #[new]
    #[pyo3(signature = (*list, settings=None))]
    fn new(list: Bound<PyTuple>, settings: Option<NodeSettings>) -> PyResult<Self> {
        Ok(Self(
            parse_pattern_list(list)?,
            settings.unwrap_or_default(),
        ))
    }
}

#[derive(FromPyObject)]
pub enum PatternInput<'a> {
    String(String),
    TwoStrings(String, String),
    TwoArrays(Array<'a>, Array<'a>),
}

impl<'a> PatternInput<'a> {
    fn arrays(self) -> PyResult<(Array3D, Array3D)> {
        match self {
            Self::String(string) => {
                let (from, to) = split_pattern_string(&string)?;
                Self::TwoStrings(from.to_string(), to.to_string()).arrays()
            }
            Self::TwoStrings(from, to) => Ok((string_to_array(&from), string_to_array(&to))),
            Self::TwoArrays(from, to) => Ok((array_to_owned(from), array_to_owned(to))),
        }
    }
}

fn array_to_owned(array: Array) -> Array3D {
    match array {
        Array::D2(array) => Array3D::new_from(
            array.as_slice().unwrap().to_vec(),
            array.dims()[1],
            array.dims()[0],
            1,
        ),
        Array::D3(array) => Array3D::new_from(
            array.as_slice().unwrap().to_vec(),
            array.dims()[2],
            array.dims()[1],
            array.dims()[0],
        ),
    }
}

fn string_to_array(string: &str) -> Array3D {
    let mut width = None;
    let mut list = Vec::new();

    for c in string.chars() {
        if c == ' ' || c == '\n' {
            continue;
        }
        if c == ',' {
            if width.is_none() {
                width = Some(list.len());
            }
            continue;
        }

        list.push(match c {
            '*' => WILDCARD,
            _ => match c.to_digit(10) {
                Some(digit) => digit as u8,
                None => PALETTE.iter().position(|&v| v == c).unwrap() as _,
            },
        });
    }

    if let Some(width) = width {
        let height = list.len() / width;
        Array3D::new_from(list, width, height, 1)
    } else {
        let width = list.len();
        Array3D::new_from(list, width, 1, 1)
    }
}

#[derive(FromPyObject)]
pub enum PythonNode<'a> {
    One(One),
    Markov(Markov),
    Sequence(Sequence),
    Pattern(PatternInput<'a>),
    PatternWithOptions(Pattern),
}

impl<'a> PythonNode<'a> {
    fn convert(self) -> PyResult<Node<Pattern>> {
        Ok(match self {
            Self::One(list) => Node {
                settings: list.1,
                ty: NodeTy::One(list.0),
            },
            Self::Markov(list) => Node {
                settings: list.1,
                ty: NodeTy::Markov(list.0),
            },
            Self::Sequence(list) => Node {
                settings: list.1,
                ty: NodeTy::Sequence(list.0),
            },
            Self::Pattern(pattern) => {
                Self::PatternWithOptions(Pattern::new_from_pattern(pattern)?).convert()?
            }
            Self::PatternWithOptions(pattern) => Node {
                settings: pattern.options.node_settings.clone(),
                ty: NodeTy::Rule(pattern),
            },
        })
    }
}

#[derive(FromPyObject)]
pub enum Array<'a> {
    D2(numpy::borrow::PyReadonlyArray2<'a, u8>),
    D3(numpy::borrow::PyReadonlyArray3<'a, u8>),
}

#[pyfunction]
#[pyo3(signature = (array, node, callback = None))]
pub fn rep(array: Array, node: PythonNode, callback: Option<&Bound<PyFunction>>) -> PyResult<()> {
    let mut array_2d = match &array {
        Array::D2(array) => {
            Array3D::new_from(
                // I don't like doing this, but it's the only way to get a callback
                // function working afaik.
                unsafe { array.as_slice_mut().unwrap() },
                array.dims()[1],
                array.dims()[0],
                1,
            )
        }
        Array::D3(array) => {
            Array3D::new_from(
                // I don't like doing this, but it's the only way to get a callback
                // function working afaik.
                unsafe { array.as_slice_mut().unwrap() },
                array.dims()[2],
                array.dims()[1],
                array.dims()[0],
            )
        }
    };

    let node = node.convert()?;

    let node = map_node(node, &|pattern| {
        Replace::new(
            pattern.from,
            pattern.to,
            &pattern.options.shuffles,
            &pattern.options.flips,
            pattern.options.settings.clone(),
            &array_2d,
        )
    });

    let mut rng = rand::rngs::SmallRng::from_entropy();

    let callback = callback.map(|callback| {
        Box::new(|iteration| {
            callback.call1((iteration,)).unwrap();
        }) as _
    });

    execute_root_node(node, &mut array_2d, &mut rng, callback);
    Ok(())
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;

    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[pyclass]
pub struct Wfc(crate::wfc::Wfc);

#[pymethods]
impl Wfc {
    #[new]
    fn new(dims: (usize, usize, usize)) -> Self {
        Self(crate::wfc::Wfc::new(dims))
    }

    fn add(&mut self, prob: f32) -> usize {
        self.0.add(prob)
    }

    fn connect_to_all(&mut self, tile: usize) {
        self.0.connect_to_all(tile)
    }

    fn connect(&mut self, from: usize, to: usize, axises: Vec<String>) {
        let axises = axises
            .into_iter()
            .map(|string| match &(string.to_lowercase())[..] {
                "x" => crate::wfc::Axis::X,
                "y" => crate::wfc::Axis::Y,
                "z" => crate::wfc::Axis::Z,
                "negx" => crate::wfc::Axis::NegX,
                "negy" => crate::wfc::Axis::NegY,
                "negz" => crate::wfc::Axis::NegZ,
                _ => panic!(),
            })
            .collect::<Vec<_>>();
        self.0.connect(from, to, &axises)
    }

    fn setup_state(&mut self) {
        self.0.setup_state()
    }

    fn values<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, numpy::array::PyArray3<u8>>> {
        numpy::array::PyArray1::from_vec(py, self.0.values()).reshape((
            self.0.depth(),
            self.0.height(),
            self.0.width(),
        ))
    }

    fn collapse_all(&mut self) {
        let mut rng = rand::rngs::SmallRng::from_entropy();

        self.0.collapse_all(&mut rng)
    }

    fn all_collapsed(&self) -> bool {
        self.0.all_collapsed()
    }

    fn collapse(&mut self, index: usize, tile: u8) {
        self.0.collapse(index, tile)
    }

    fn find_lowest_entropy(&mut self) -> Option<(usize, u8)> {
        let mut rng = rand::rngs::SmallRng::from_entropy();
        self.0.find_lowest_entropy(&mut rng)
    }
}

#[pyclass]
pub struct Palette {
    #[pyo3(get)]
    linear: Vec<[f32; 3]>,
    #[pyo3(get)]
    srgb: Vec<[u8; 3]>,
}

#[pymethods]
impl Palette {
    #[new]
    fn new(srgb: Vec<[u8; 3]>) -> Self {
        Self {
            linear: srgb.iter().map(|col| col.map(srgb_to_linear)).collect(),
            srgb,
        }
    }
}

#[pyfunction]
pub fn colour_image(
    mut output: numpy::borrow::PyReadwriteArray3<u8>,
    input: numpy::borrow::PyReadonlyArray2<u8>,
    palette: &Palette,
) {
    let input_slice = input.as_slice().unwrap();
    let output_slice = output.as_slice_mut().unwrap();

    let height = input.dims()[0];
    let width = input.dims()[1];

    for y in 0..height {
        for x in 0..width {
            let colour = palette.srgb[input_slice[y * width + x] as usize];
            output_slice[(y * width + x) * 3..(y * width + x + 1) * 3].copy_from_slice(&colour);
        }
    }
}

#[repr(transparent)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct ByteVoxel(u8);

impl block_mesh::Voxel for ByteVoxel {
    fn get_visibility(&self) -> block_mesh::VoxelVisibility {
        if self.0 == 0 {
            block_mesh::VoxelVisibility::Empty
        } else {
            block_mesh::VoxelVisibility::Opaque
        }
    }
}

impl block_mesh::MergeVoxel for ByteVoxel {
    type MergeValue = u8;

    fn merge_value(&self) -> Self::MergeValue {
        self.0
    }
}

#[pyfunction]
pub fn mesh_voxels(array: Array) -> (Vec<[f32; 3]>, Vec<u8>, Vec<u32>) {
    let (slice, dims) = match &array {
        Array::D2(array) => (
            array.as_slice().unwrap(),
            [array.dims()[1], array.dims()[0], 1],
        ),
        Array::D3(array) => (
            array.as_slice().unwrap(),
            [array.dims()[2], array.dims()[1], array.dims()[0]],
        ),
    };

    let dims = dims.map(|x| x as u32);

    let voxels: &[ByteVoxel] = bytemuck::cast_slice(slice);

    let mut buffer = block_mesh::GreedyQuadsBuffer::new(0);

    block_mesh::greedy_quads(
        voxels,
        &ndshape::RuntimeShape::<u32, 3>::new(dims),
        [0; 3],
        [dims[0] - 1, dims[1] - 1, dims[2] - 1],
        &block_mesh::RIGHT_HANDED_Y_UP_CONFIG.faces,
        &mut buffer,
    );

    let mut positions = Vec::new();
    let mut colours = Vec::new();
    let mut indices = Vec::new();

    for (i, group) in buffer.quads.groups.into_iter().enumerate() {
        let face = block_mesh::RIGHT_HANDED_Y_UP_CONFIG.faces[i];

        let flip_winding = i == 1 || i == 2 || i == 3;

        for quad in group.into_iter() {
            let index =
                quad.minimum[0] + quad.minimum[1] * dims[0] + quad.minimum[2] * dims[0] * dims[1];
            let value = slice[index as usize];

            let face_positions = face.quad_mesh_positions(&quad, 1.0);

            colours.push(value);

            let index = positions.len() as u32;
            if flip_winding {
                indices.extend_from_slice(&[index, index + 2, index + 3, index + 1]);
            } else {
                indices.extend_from_slice(&[index, index + 1, index + 3, index + 2]);
            }

            for position in face_positions {
                positions.push(position);
            }
        }
    }

    (positions, colours, indices)
}
