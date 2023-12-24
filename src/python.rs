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

#[pyfunction]
pub fn rep(
    mut array: numpy::borrow::PyReadwriteArray2<u8>,
    patterns: &PyList,
    priority_after: Option<usize>,
    times: Option<u32>,
    priority: Option<bool>,
) {
    let mut array_2d = Array2D {
        width: array.dims()[0],
        height: array.dims()[1],
        inner: array.as_slice_mut().unwrap(),
    };

    let mut replaces = Vec::new();

    for pattern in patterns {
        let pattern = if let Ok(pattern) = pattern.extract::<PatternWithOptions>() {
            pattern
        } else {
            PatternWithOptions::new(pattern.extract::<&str>().unwrap().to_string(), None, None)
        };

        replaces.push(Replace::from_string(
            &pattern.pattern,
            pattern.allow_rot90,
            pattern.allow_vertical_flip,
            &array_2d,
        ));
    }

    let mut rng = rand::rngs::SmallRng::from_entropy();

    execute_rule(
        &mut array_2d,
        &mut rng,
        &mut replaces,
        times.unwrap_or(0),
        priority_after.or(priority.filter(|&boolean| boolean).map(|_| 0)),
    );
}

#[pyfunction]
pub fn rep_all(mut array: numpy::borrow::PyReadwriteArray2<u8>, patterns: &PyList) {
    let mut array_2d = Array2D {
        width: array.dims()[0],
        height: array.dims()[1],
        inner: array.as_slice_mut().unwrap(),
    };

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
