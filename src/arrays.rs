#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Array2D<T = Vec<u8>> {
    pub inner: T,
    width: usize,
    height: usize,
    depth: usize,
}

impl<T> Array2D<T> {
    pub fn new_from(array: T, width: usize, height: usize, depth: usize) -> Self {
        Self {
            inner: array,
            width: width,
            height: height,
            depth,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

impl Array2D<Vec<u8>> {
    pub fn new(slice: &[u8], width: usize) -> Self {
        Self {
            inner: slice.to_owned(),
            width,
            height: slice.len() / width,
            depth: 1,
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = &[u8]> {
        self.inner.chunks_exact(self.width)
    }

    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.inner[y * self.width + x]
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = (usize, usize, u8)> + '_ {
        self.rows()
            .enumerate()
            .flat_map(|(y, row)| row.iter().enumerate().map(move |(x, v)| (x, y, *v)))
    }

    #[inline]
    pub fn non_wildcard_values(&self) -> impl Iterator<Item = (usize, usize, u8)> + '_ {
        self.values()
            .filter(|&(_, _, value)| value != crate::WILDCARD)
    }

    #[inline]
    pub fn non_wildcard_values_in_state(
        &self,
        state_width: usize,
    ) -> impl Iterator<Item = (usize, u8)> + '_ {
        self.non_wildcard_values()
            .map(move |(x, y, value)| (x + y * state_width, value))
    }

    pub fn permute<F: Fn(usize, usize) -> (usize, usize)>(
        &self,
        width: usize,
        height: usize,
        remap: F,
    ) -> Array2D {
        let mut array = Array2D {
            inner: vec![0; width * height],
            width,
            height,
            depth: 1,
        };

        for (x, y, value) in self.values() {
            let (x, y) = remap(x, y);
            array.inner[y * width + x] = value;
        }

        array
    }
}

impl Array2D<&mut [u8]> {
    pub fn put(&mut self, index: usize, value: u8) {
        self.inner[index] = value;
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ArrayPair {
    pub to: Array2D,
    pub from: Array2D,
}

impl ArrayPair {
    pub fn permute<F: Fn(usize, usize) -> (usize, usize)>(
        &self,
        width: usize,
        height: usize,
        remap: F,
    ) -> Self {
        Self {
            to: self.to.permute(width, height, &remap),
            from: self.from.permute(width, height, &remap),
        }
    }
}
