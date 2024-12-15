use std::env;
use std::fs::*;
use std::io::*;
use std::path::*;
use xml::*;

fn parse_hex_color(hex: &str) -> [u8; 3] {
    [
        u8::from_str_radix(&hex[0..2], 16).unwrap(),
        u8::from_str_radix(&hex[2..4], 16).unwrap(),
        u8::from_str_radix(&hex[4..6], 16).unwrap(),
    ]
}

fn main() {
    // Consume the MarkovJunior palette file and output a list of character
    // and srgb values.

    // The palette file is extended from the pico8 palette
    // at https://pico-8.fandom.com/wiki/Palette. I want to have
    // it go B then W then R though so I've reordered those.
    let reorderings = ['B', 'W', 'R'];

    let mut palette_values = Vec::new();
    let mut offset = reorderings.len();

    let mut char_to_index = [255; 255];

    for (i, x) in reorderings.iter().enumerate() {
        char_to_index[(*x) as usize] = i as u8;
    }

    let file = File::open("MarkovJunior/resources/palette.xml").unwrap();

    let mut reader = ParserConfig::default()
        .ignore_root_level_whitespace(true)
        .ignore_comments(false)
        .cdata_to_characters(true)
        .coalesce_characters(true)
        .create_reader(file);

    loop {
        let reader_event = reader.next().unwrap();

        match reader_event {
            xml::reader::XmlEvent::EndDocument => break,
            xml::reader::XmlEvent::StartElement {
                name, attributes, ..
            } => {
                if name.local_name == "color" {
                    let mut palette_char = None;
                    let mut hex_colour = None;

                    for attribute in attributes {
                        if attribute.name.local_name == "symbol" {
                            palette_char = Some(attribute.value.parse().unwrap());
                        }
                        if attribute.name.local_name == "value" {
                            hex_colour = Some(parse_hex_color(&attribute.value));
                        }
                    }

                    let palette_char = palette_char.unwrap();
                    let hex_colour = hex_colour.unwrap();

                    let index = if !reorderings.contains(&palette_char) {
                        let index = offset;
                        offset += 1;
                        index
                    } else {
                        reorderings.iter().position(|&c| c == palette_char).unwrap()
                    };

                    while index >= palette_values.len() {
                        palette_values.push([0; 3]);
                    }

                    palette_values[index] = hex_colour;

                    char_to_index[palette_char as usize] = index as u8;
                }
            }
            _ => {}
        }
    }

    // slightly hacky, but as we're using 'B'/0 as both transparent and black
    // we need a way to have opaque black
    char_to_index['@' as usize] = palette_values.len() as u8;
    palette_values.push([0; 3]);

    for digit in 0..=9 {
        char_to_index[char::from_digit(digit, 10).unwrap() as usize] = digit as u8;
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("palette.rs");

    let mut output = File::create(dest_path).unwrap();
    writeln!(
        output,
        "pub const CHARS_TO_INDEX: &[u8] = &{:?};",
        char_to_index
    )
    .unwrap();
    writeln!(
        output,
        "pub const COLOURS: &[[u8;3]] = &{:?};",
        palette_values
    )
    .unwrap();

    println!("cargo::rerun-if-changed=build.rs");
}
