use crate::arrays::{Array2D, ArrayPair};
use crate::{bespoke_regex, WILDCARD};

pub struct OverlappingRegexIter<'a> {
    regex: &'a bespoke_regex::BespokeRegex,
    haystack: &'a [u8],
    offset: usize,
}

impl<'a> OverlappingRegexIter<'a> {
    pub fn new(regex: &'a bespoke_regex::BespokeRegex, haystack: &'a [u8]) -> Self {
        Self {
            regex,
            haystack,
            offset: 0,
        }
    }
}

impl<'a> Iterator for OverlappingRegexIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.regex.find(&self.haystack[self.offset..]) {
            Some(start) => {
                let index = self.offset + start;
                self.offset += start + 1;
                Some(index)
            }
            None => None,
        }
    }
}

pub struct Permutation {
    //regex: regex::bytes::Regex,
    pub bespoke_regex: bespoke_regex::BespokeRegex,
    pattern_len: usize,
    pub to: Array2D,
}

impl Permutation {
    pub fn new(state: &Array2D<&mut [u8]>, pair: ArrayPair) -> Self {
        //let mut regex = String::new();
        let mut bespoke_values = Vec::new();
        let mut pattern_len = 0;

        for y in 0..pair.from.height {
            for x in 0..pair.from.width {
                let index = y * pair.from.width + x;
                let value = pair.from.inner[index];

                if value == WILDCARD {
                    bespoke_values.push(bespoke_regex::LiteralsOrWildcards::Wildcards(1));
                } else {
                    bespoke_values.push(bespoke_regex::LiteralsOrWildcards::Literal(vec![value]));
                }
            }

            pattern_len += pair.from.width;

            if y < pair.from.height - 1 {
                //regex += &format!(r".{{{}}}", state.width - pair.from.width);
                bespoke_values.push(bespoke_regex::LiteralsOrWildcards::Wildcards(
                    state.width - pair.from.width,
                ));
                pattern_len += state.width - pair.from.width;
            }
        }

        Self {
            pattern_len,
            /*
            regex: regex::bytes::RegexBuilder::new(&string)
                .unicode(false)
                .dot_matches_new_line(true)
                .build()
                .unwrap(),
            */
            bespoke_regex: bespoke_regex::BespokeRegex::new(&bespoke_values),
            to: pair.to,
        }
    }
}

pub fn match_pattern(regex: &Permutation, state: &Array2D<&mut [u8]>, index: u32) -> bool {
    let end = index as usize + regex.pattern_len;
    if end > state.inner.len() || (index as usize % state.width + regex.to.width) > state.width {
        return false;
    }
    //regex.regex.is_match(&state.inner[index as usize..end])
    regex
        .bespoke_regex
        .is_immediate_match(&state.inner[index as usize..end])
}
