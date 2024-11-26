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

    pub fn send_image(&mut self, name: &str, array: numpy::borrow::PyReadonlyArray2<u8>) {
        let dims = array.dims();
        let slice = array.as_slice().unwrap();

        super::send_image(
            &mut self.inner,
            &mut self.values,
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

#[pyclass]
#[derive(Clone)]
pub struct PatternWithOptions {
    from: Vec<String>,
    to: Vec<String>,
    options: PatternOptions,
}

#[pyclass]
#[derive(Clone)]
pub struct PatternOptions {
    allow_dimension_shuffling: bool,
    allow_flip: bool,
    settings: ReplaceSettings,
    node_settings: NodeSettings,
}

#[pymethods]
impl PatternWithOptions {
    #[new]
    fn new(
        pattern: Pattern,
        allow_dimension_shuffling: Option<bool>,
        allow_flip: Option<bool>,
        apply_all: Option<bool>,
        chance: Option<f32>,
        node_settings: Option<NodeSettings>,
    ) -> PyResult<Self> {
        let (from, to) = pattern.strings()?;
        Ok(Self {
            from,
            to,
            options: PatternOptions {
                allow_dimension_shuffling: allow_dimension_shuffling.unwrap_or(true),
                allow_flip: allow_flip.unwrap_or(true),
                settings: ReplaceSettings {
                    apply_all: apply_all.unwrap_or(false),
                    chance: chance.unwrap_or(10.0),
                },
                node_settings: node_settings.unwrap_or_default(),
            },
        })
    }
}

type PatternList = Vec<Node<PatternWithOptions>>;

fn parse_pattern_list(list: Bound<PyTuple>) -> PyResult<PatternList> {
    list.iter()
        .map(|item| {
            item.extract::<PythonNode>()
                .and_then(|python_node| python_node.convert())
        })
        .collect::<PyResult<Vec<Node<PatternWithOptions>>>>()
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
enum Pattern<'a> {
    String(String),
    TwoStrings(String, String),
    TwoLists(Bound<'a, PyTuple>, Bound<'a, PyTuple>),
}

impl<'a> Pattern<'a> {
    fn strings(self) -> PyResult<(Vec<String>, Vec<String>)> {
        match self {
            Self::String(string) => {
                let (from, to) = split_pattern_string(&string)?;
                Ok((vec![from.to_string()], vec![to.to_string()]))
            }
            Self::TwoLists(from, to) => Ok((
                from.iter()
                    .map(|from| from.extract::<String>())
                    .collect::<PyResult<Vec<_>>>()?,
                to.iter()
                    .map(|to| to.extract::<String>())
                    .collect::<PyResult<Vec<_>>>()?,
            )),
            Self::TwoStrings(from, to) => Ok((vec![from], vec![to])),
        }
    }
}

#[derive(FromPyObject)]
pub enum PythonNode<'a> {
    One(One),
    Markov(Markov),
    Sequence(Sequence),
    Pattern(Pattern<'a>),
    PatternWithOptions(PatternWithOptions),
}

impl<'a> PythonNode<'a> {
    fn convert(self) -> PyResult<Node<PatternWithOptions>> {
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
            Self::Pattern(pattern) => Self::PatternWithOptions(PatternWithOptions::new(
                pattern, None, None, None, None, None,
            )?)
            .convert()?,
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
pub fn rep(array: Array, node: PythonNode, callback: Option<&Bound<PyFunction>>) -> PyResult<()> {
    let mut array_2d = match &array {
        Array::D2(array) => {
            Array2D::new_from(
                // I don't like doing this, but it's the only way to get a callback
                // function working afaik.
                unsafe { array.as_slice_mut().unwrap() },
                array.dims()[1],
                array.dims()[0],
                1,
            )
        }
        Array::D3(array) => {
            Array2D::new_from(
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

    let node = map_node(&node, &|pattern| {
        let shuffles: &[[usize; 3]] = if pattern.options.allow_dimension_shuffling {
            &[
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ]
        } else {
            &[[0, 1, 2]]
        };

        let flips: &[[bool; 3]] = if pattern.options.allow_flip {
            &[
                [false, false, false],
                [false, false, true],
                [false, true, false],
                [false, true, true],
                [true, false, false],
                [true, false, true],
                [true, true, false],
                [true, true, true],
            ]
        } else {
            &[[false; 3]]
        };

        Replace::from_layers(
            &pattern.from,
            &pattern.to,
            shuffles,
            flips,
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

#[pyfunction]
pub fn index_for_colour(colour: char) -> Option<u8> {
    super::index_for_colour(colour)
}

#[pyfunction]
pub fn colour_image(
    mut output: numpy::borrow::PyReadwriteArray3<u8>,
    input: numpy::borrow::PyReadonlyArray2<u8>,
) {
    let input_slice = input.as_slice().unwrap();
    let output_slice = output.as_slice_mut().unwrap();

    let height = input.dims()[0];
    let width = input.dims()[1];

    for y in 0..height {
        for x in 0..width {
            let colour = SRGB_PALETTE_VALUES[input_slice[y * width + x] as usize];
            output_slice[(y * width + x) * 3..(y * width + x + 1) * 3].copy_from_slice(&colour);
        }
    }
}
