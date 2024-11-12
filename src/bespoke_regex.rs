#[derive(Clone, Debug)]
pub enum LiteralsOrWildcards {
    Literal(Vec<u8>),
    Wildcards(usize),
}

pub fn compress_values(values: &[LiteralsOrWildcards]) -> Vec<LiteralsOrWildcards> {
    let mut prev = None;
    let mut compressed = Vec::new();

    for value in values {
        match prev.as_mut() {
            None => prev = Some(value.clone()),
            Some(prev) => match (prev, value) {
                (
                    LiteralsOrWildcards::Wildcards(count),
                    LiteralsOrWildcards::Wildcards(additional),
                ) => *count += additional,
                (
                    LiteralsOrWildcards::Literal(literals),
                    LiteralsOrWildcards::Literal(additional),
                ) => literals.extend_from_slice(additional),
                (prev, _) => {
                    compressed.push(std::mem::replace(prev, value.clone()));
                }
            },
        }
    }

    if let Some(prev) = prev.take() {
        compressed.push(prev);
    }

    compressed
}

pub struct BespokeRegex {
    skip_initial_wildcards: usize,
    initial_literal: memchr::memmem::Finder<'static>,
    remainder: Vec<LiteralsOrWildcards>,
}

impl BespokeRegex {
    pub fn new(values: &[LiteralsOrWildcards]) -> Self {
        let mut skip_initial_wildcards = 0;
        let values = compress_values(values);

        let mut offset = 0;

        if let &LiteralsOrWildcards::Wildcards(count) = &values[0] {
            skip_initial_wildcards = count;
            offset += 1;
        };

        let initial_literal = match &values[offset] {
            LiteralsOrWildcards::Literal(literal) => literal,
            LiteralsOrWildcards::Wildcards(_) => panic!("Bad input"),
        };

        Self {
            skip_initial_wildcards,
            initial_literal: memchr::memmem::Finder::new(&initial_literal).into_owned(),
            remainder: values[offset + 1..].to_vec(),
        }
    }

    pub fn find(&self, slice: &[u8]) -> Option<usize> {
        let mut offset = 0;

        'outer: while (offset + self.skip_initial_wildcards) < slice.len() {
            let mut test_slice = &slice[offset + self.skip_initial_wildcards..];

            let start = match self.initial_literal.find(test_slice) {
                Some(start) => start,
                None => return None,
            };

            test_slice = &test_slice[start + self.initial_literal.needle().len()..];

            offset += start;

            for item in &self.remainder {
                match item {
                    LiteralsOrWildcards::Wildcards(count) => {
                        if *count > test_slice.len() {
                            return None;
                        }
                        test_slice = &test_slice[*count..];
                    }
                    LiteralsOrWildcards::Literal(literal) => {
                        if !test_slice.starts_with(literal) {
                            offset += 1;
                            continue 'outer;
                        }

                        test_slice = &test_slice[literal.len()..];
                    }
                }
            }

            return Some(offset);
        }

        None
    }

    pub fn is_immediate_match(&self, mut slice: &[u8]) -> bool {
        slice = &slice[self.skip_initial_wildcards..];
        if !slice.starts_with(self.initial_literal.needle()) {
            return false;
        }
        slice = &slice[self.initial_literal.needle().len()..];
        for item in &self.remainder {
            match item {
                LiteralsOrWildcards::Wildcards(count) => slice = &slice[*count..],
                LiteralsOrWildcards::Literal(literal) => {
                    if !slice.starts_with(literal) {
                        return false;
                    }

                    slice = &slice[literal.len()..];
                }
            }
        }

        true
    }
}
