use std::io::Write;

pub fn write_vox(
    filename: &str,
    slice: &[u8],
    w: usize,
    h: usize,
    d: usize,
) -> std::io::Result<()> {
    let chunks_w = (w - 1) / 256 + 1;
    let chunks_h = (h - 1) / 256 + 1;
    let chunks_d = (d - 1) / 256 + 1;

    let mut chunks = vec![Vec::new(); chunks_w * chunks_h * chunks_d];

    for (i, v) in slice.iter().copied().enumerate().filter(|&(_, v)| v != 0) {
        let x = i % w;
        let y = (i / w) % h;
        let z = i / w / h;

        let chunk_x = x / 256;
        let chunk_y = y / 256;
        let chunk_z = z / 256;

        let x = x % 256;
        let y = y % 256;
        let z = z % 256;

        chunks[chunk_x + chunk_y * chunks_w + chunk_z * chunks_w * chunks_h]
            .push([x as u8, y as u8, z as u8, v]);
    }

    let mut output = std::fs::File::create(filename)?;

    enum Content<'a> {
        Raw(&'a [u8]),
        Xyzi(&'a [[u8; 4]]),
        #[allow(dead_code)]
        Rgba(&'a [[u8; 4]; 255]),
        Transform {
            id: i32,
            child_id: i32,
            layer: i32,
            translation: String,
            //transforms: &'a [(&'a [u8], &'a [u8])],
        },
        Group {
            id: i32,
            children: &'a [i32],
        },
        Shape {
            id: i32,
            model: i32, //models: &'a [i32],
        },
    }

    impl<'a> Content<'a> {
        fn size(&self) -> u32 {
            match self {
                Self::Raw(bytes) => bytes.len() as u32,
                Self::Xyzi(xyzi) => xyzi.len() as u32 * 4 + 4,
                Self::Rgba(_) => 255 * 4,
                Self::Group { children, .. } => {
                    // id + empty dict + num_children
                    3 * 4 + children.len() as u32 * 4
                }
                Self::Shape { .. } => 3 * 4 + (/*models.len() as u32*/1) * 8,
                Self::Transform { translation, .. } => {
                    let transforms: &[(&[u8], &[u8])] = &[(b"_t", translation.as_bytes())];
                    7 * 4
                        + transforms
                            .iter()
                            .map(|(k, v)| k.len() as u32 + v.len() as u32 + 8)
                            .sum::<u32>()
                }
            }
        }
    }

    struct FileChunk<'a> {
        name: &'a [u8; 4],
        content: Content<'a>,
        children: &'a [FileChunk<'a>],
    }

    impl<'a> FileChunk<'a> {
        fn size(&self) -> u32 {
            4 + 4 + 4 + self.content.size()
        }
    }

    let dims = [w as u32, h as u32, d as u32];

    let mut sections = vec![
        FileChunk {
            name: b"SIZE",
            content: Content::Raw(bytemuck::cast_slice(&dims)),
            children: &[],
        },
        // root transform
        FileChunk {
            name: b"nTRN",
            content: Content::Transform {
                id: 0,
                child_id: 1,
                layer: -1,
                translation: "0 0 0".to_string(),
            },
            children: &[],
        },
    ];

    let mut chunk_ids = Vec::new();

    let mut id = 2;
    for chunk in chunks.iter() {
        if chunk.is_empty() {
            continue;
        }
        chunk_ids.push(id);
        chunk_ids.push(id + 1);
        id += 2;
    }

    // root group
    sections.push(FileChunk {
        name: b"nGRP",
        content: Content::Group {
            id: 1,
            children: &chunk_ids,
        },
        children: &[],
    });

    //transforms: &[(b"_t", &[0x33, 0x20, 0x31, 0x32, 0x20, 0x35])],

    let mut id = 2;
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.is_empty() {
            continue;
        }

        let chunk_x = i % chunks_w;
        let chunk_y = (i / chunks_w) % chunks_h;
        let chunk_z = i / chunks_w / chunks_h;

        sections.push(FileChunk {
            name: b"XYZI",
            content: Content::Xyzi(&chunk),
            children: &[],
        });
        sections.push(FileChunk {
            name: b"nTRN",
            content: Content::Transform {
                id,
                child_id: id + 1,
                layer: 0,
                translation: format!("{} {} {}", chunk_x * 256, chunk_y * 256, chunk_z * 256),
            },
            children: &[],
        });
        sections.push(FileChunk {
            name: b"nSHP",
            content: Content::Shape {
                id: id + 1,
                model: i as i32,
            },
            children: &[],
        });
        id += 2;
    }
    let main = FileChunk {
        name: b"MAIN",
        content: Content::Raw(&[]),
        children: &sections,
    };

    fn write_chunk(chunk: &FileChunk, output: &mut std::fs::File) -> std::io::Result<()> {
        output.write(chunk.name)?;
        output.write(&chunk.content.size().to_le_bytes())?;
        output.write(
            &chunk
                .children
                .iter()
                .map(|child| child.size())
                .sum::<u32>()
                .to_le_bytes(),
        )?;
        match chunk.content {
            Content::Raw(bytes) => {
                output.write(bytes)?;
            }
            Content::Rgba(rgba) => {
                output.write(bytemuck::cast_slice(rgba))?;
            }
            Content::Xyzi(xyzi) => {
                output.write(&(xyzi.len() as u32).to_le_bytes())?;
                output.write(bytemuck::cast_slice(xyzi))?;
            }
            Content::Group { id, children } => {
                output.write(&id.to_le_bytes())?;
                output.write(&0_u32.to_le_bytes())?;
                output.write(&(children.len() as u32).to_le_bytes())?;
                for child_id in children {
                    output.write(&child_id.to_le_bytes())?;
                }
            }
            Content::Shape { id, model } => {
                output.write(&id.to_le_bytes())?;
                output.write(&0_u32.to_le_bytes())?;
                output.write(&/*(models.len() as u32)*/1_u32.to_le_bytes())?;
                //for model in models {
                output.write(&model.to_le_bytes())?;
                output.write(&0_u32.to_le_bytes())?;
                //}
            }
            Content::Transform {
                id,
                child_id,
                layer,
                ref translation,
            } => {
                let transforms: &[(&[u8], &[u8])] = &[(b"_t", translation.as_bytes())];

                output.write(&id.to_le_bytes())?;
                output.write(&0_u32.to_le_bytes())?;
                output.write(&child_id.to_le_bytes())?;
                output.write(&(-1_i32).to_le_bytes())?;
                // layer
                output.write(&layer.to_le_bytes())?;
                // num_frames
                output.write(&1_u32.to_le_bytes())?;
                // frame dictionary
                output.write(&(transforms.len() as u32).to_le_bytes())?;
                for (k, v) in transforms.iter() {
                    output.write(&(k.len() as u32).to_le_bytes())?;
                    output.write(k)?;
                    output.write(&(v.len() as u32).to_le_bytes())?;
                    output.write(v)?;
                }
            }
        };
        for child in chunk.children.iter() {
            write_chunk(child, output)?;
        }
        Ok(())
    }

    output.write(b"VOX ")?;
    output.write(&150_u32.to_le_bytes())?;

    write_chunk(&main, &mut output)?;
    Ok(())
}
