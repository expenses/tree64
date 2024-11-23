use super::*;
use pyo3::exceptions::PyTypeError;
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
    allow_dimension_shuffling: bool,
    allow_flip: bool,
    apply_all: bool,
}

#[pymethods]
impl PatternWithOptions {
    #[new]
    fn new(
        pattern: Pattern,
        allow_dimension_shuffling: Option<bool>,
        allow_flip: Option<bool>,
        apply_all: Option<bool>,
    ) -> PyResult<Self> {
        let (from, to) = pattern.strings()?;
        Ok(Self {
            from,
            to,
            allow_dimension_shuffling: allow_dimension_shuffling.unwrap_or(true),
            allow_flip: allow_flip.unwrap_or(true),
            apply_all: apply_all.unwrap_or(false),
        })
    }
}

type PatternList = Vec<Node<PatternWithOptions>>;

fn parse_pattern_list(list: &PyTuple) -> PyResult<PatternList> {
    list.iter()
        .map(|item| {
            item.extract::<PythonNode>()
                .and_then(|python_node| python_node.convert())
        })
        .collect::<PyResult<Vec<Node<PatternWithOptions>>>>()
}

#[derive(Clone)]
#[pyclass]
pub struct One(PatternList);

#[pymethods]
impl One {
    #[new]
    #[pyo3(signature = (*list))]
    fn new(list: &PyTuple) -> PyResult<Self> {
        Ok(Self(parse_pattern_list(list)?))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Markov(PatternList);

#[pymethods]
impl Markov {
    #[new]
    #[pyo3(signature = (*list))]
    fn new(list: &PyTuple) -> PyResult<Self> {
        Ok(Self(parse_pattern_list(list)?))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Sequence(PatternList);

#[pymethods]
impl Sequence {
    #[new]
    #[pyo3(signature = (*list))]
    fn new(list: &PyTuple) -> PyResult<Self> {
        Ok(Self(parse_pattern_list(list)?))
    }
}

#[derive(FromPyObject)]
enum Pattern<'a> {
    String(String),
    TwoStrings(String, String),
    TwoLists(&'a PyTuple, &'a PyTuple),
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
            Self::One(list) => Node::One(list.0),
            Self::Markov(list) => Node::Markov(list.0),
            Self::Sequence(list) => Node::Sequence(list.0),
            Self::Pattern(pattern) => {
                Node::Rule(PatternWithOptions::new(pattern, None, None, None)?)
            }
            Self::PatternWithOptions(pattern) => Node::Rule(pattern),
        })
    }
}

#[derive(FromPyObject)]
pub enum Array<'a> {
    D2(numpy::borrow::PyReadonlyArray2<'a, u8>),
    D3(numpy::borrow::PyReadonlyArray3<'a, u8>),
}

#[pyfunction]
pub fn rep(
    array: Array,
    node: PythonNode,
    times: Option<u32>,
    callback: Option<&PyFunction>,
) -> PyResult<()> {
    let mut array_2d = match &array {
        Array::D2(array) => {
            Array2D::new_from(
                // I don't like doing this, but it's the only way to get a callback
                // function working afaik.
                unsafe { array.as_slice_mut().unwrap() },
                array.dims()[0],
                array.dims()[1],
                1,
            )
        }
        Array::D3(array) => {
            Array2D::new_from(
                // I don't like doing this, but it's the only way to get a callback
                // function working afaik.
                unsafe { array.as_slice_mut().unwrap() },
                array.dims()[0],
                array.dims()[1],
                array.dims()[2],
            )
        }
    };

    let node = node.convert()?;

    let node = map_node(&node, &|pattern| {
        Replace::from_layers(
            &pattern.from,
            &pattern.to,
            pattern.allow_dimension_shuffling,
            pattern.allow_flip,
            pattern.apply_all,
            &array_2d,
        )
    });

    let mut rng = rand::rngs::SmallRng::from_entropy();

    let callback = callback.map(|callback| {
        Box::new(|iteration| {
            callback.call1((iteration,)).unwrap();
        }) as _
    });

    execute_root_node(node, &mut array_2d, &mut rng, times, callback);

    Ok(())
}

#[pyfunction]
pub fn rep_all(
    mut array: numpy::borrow::PyReadwriteArray2<u8>,
    patterns: &PyTuple,
    chance: Option<f32>,
) -> PyResult<()> {
    let width = array.dims()[0];
    let height = array.dims()[1];
    let mut array_2d = Array2D::new_from(array.as_slice_mut().unwrap(), width, height, 1);

    let mut replaces = Vec::new();

    for pattern in patterns {
        let pattern =
            PatternWithOptions::new(pattern.extract::<Pattern>().unwrap(), None, None, None)?;
        replaces.push(Replace::from_layers(
            &pattern.from,
            &pattern.to,
            pattern.allow_dimension_shuffling,
            pattern.allow_flip,
            pattern.apply_all,
            &array_2d,
        ));
    }

    let mut chance_and_rng = chance.map(|value| (rand::rngs::SmallRng::from_entropy(), value));

    execute_rule_all(
        &mut array_2d,
        &mut replaces,
        chance_and_rng.as_mut().map(|(rng, value)| (rng, *value)),
    );

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

    let width = input.dims()[0];
    let height = input.dims()[1];

    for y in 0..height {
        for x in 0..width {
            let colour = SRGB_PALETTE_VALUES[input_slice[y * width + x] as usize];
            output_slice[(y * width + x) * 3..(y * width + x + 1) * 3].copy_from_slice(&colour);
        }
    }
}
