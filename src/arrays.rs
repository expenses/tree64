#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Array2D<T = Vec<u8>> {
    pub inner: T,
    pub width: usize,
    pub height: usize,
}

impl Array2D<Vec<u8>> {
    pub fn new(slice: &[u8], width: usize) -> Self {
        Self {
            inner: slice.to_owned(),
            width,
            height: slice.len() / width,
        }
    }
}

pub trait MutByteSlice: std::ops::Deref<Target = [u8]> + std::ops::DerefMut + Sync {}

impl<T> MutByteSlice for T where T: std::ops::Deref<Target = [u8]> + std::ops::DerefMut + Sync {}

impl<T: MutByteSlice> Array2D<T> {
    pub fn put(&mut self, index: usize, value: u8) -> bool {
        let previous_value = self.inner[index];
        self.inner[index] = value;
        previous_value != value
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
        };

        for x in 0..self.width {
            for y in 0..self.height {
                let value = self.inner[y * self.width + x];
                let (x, y) = remap(x, y);
                array.inner[y * width + x] = value;
            }
        }

        array
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
