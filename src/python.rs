use super::*;
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

#[pyclass]
#[derive(Clone)]
pub struct PatternWithOptions {
    pattern: String,
    allow_rot90: bool,
    allow_vertical_flip: bool,
}

#[pymethods]
impl PatternWithOptions {
    #[new]
    fn new(pattern: String, allow_rot90: Option<bool>, allow_vertical_flip: Option<bool>) -> Self {
        Self {
            pattern,
            allow_rot90: allow_rot90.unwrap_or(true),
            allow_vertical_flip: allow_vertical_flip.unwrap_or(true),
        }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct One {
    inner: Node<PatternWithOptions>,
}

#[pymethods]
impl One {
    #[new]
    #[pyo3(signature = (*list))]
    fn new(list: &PyTuple) -> PyResult<Self> {
        Ok(Self {
            inner: Node::One(
                list.iter()
                    .map(|item| {
                        item.extract::<PythonNode>()
                            .map(|python_node| python_node.convert())
                    })
                    .collect::<PyResult<Vec<Node<PatternWithOptions>>>>()?,
            ),
        })
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Markov {
    inner: Node<PatternWithOptions>,
}

#[pymethods]
impl Markov {
    #[new]
    #[pyo3(signature = (*list))]
    fn new(list: &PyTuple) -> PyResult<Self> {
        Ok(Self {
            inner: Node::Markov(
                list.iter()
                    .map(|item| {
                        item.extract::<PythonNode>()
                            .map(|python_node| python_node.convert())
                    })
                    .collect::<PyResult<Vec<Node<PatternWithOptions>>>>()?,
            ),
        })
    }
}

#[derive(FromPyObject)]
pub enum PythonNode {
    One(One),
    Markov(Markov),
    String(String),
    PatternWithOptions(PatternWithOptions),
}

impl PythonNode {
    fn convert(self) -> Node<PatternWithOptions> {
        match self {
            Self::One(one) => one.inner,
            Self::Markov(markov) => markov.inner,
            Self::String(string) => Node::Rule(PatternWithOptions::new(string, None, None)),
            Self::PatternWithOptions(pattern) => Node::Rule(pattern),
        }
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

    let node = node.convert();

    let node = map_node(&node, &|pattern| {
        Replace::from_string(
            &pattern.pattern,
            pattern.allow_rot90,
            pattern.allow_vertical_flip,
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
pub fn rep_all(mut array: numpy::borrow::PyReadwriteArray2<u8>, patterns: &PyTuple) {
    let width = array.dims()[0];
    let height = array.dims()[1];
    let mut array_2d = Array2D::new_from(array.as_slice_mut().unwrap(), width, height, 1);

    let mut replaces = Vec::new();

    for pattern in patterns {
        replaces.push(Replace::from_string(
            pattern.extract::<&str>().unwrap(),
            true,
            true,
            &array_2d,
        ));
    }

    execute_rule_all(&mut array_2d, &mut replaces);
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
