use std::io::{Read, Write};

pub fn write_zvox(
    filename: &str,
    slice: &[u8],
    width: usize,
    height: usize,
    depth: usize,
    palette: [[u8; 4]; 256]
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(filename)?;

    file.write_all("zvox".as_bytes())?;
    file.write_all(&1_u32.to_le_bytes())?;
    file.write_all(&(width as u32).to_le_bytes())?;
    file.write_all(&(height as u32).to_le_bytes())?;
    file.write_all(&(depth as u32).to_le_bytes())?;
    file.write_all(bytemuck::cast_slice(&palette))?;

    let mut file = zstd::stream::write::Encoder::new(file, 0)?.auto_finish();

    file.write_all(slice)?;

    Ok(())
}

pub fn read_zvox(filename: &str) -> std::io::Result<(Vec<u8>, u32, u32, u32, [[u8; 4]; 256])> {
    let mut file = std::fs::File::open(filename)?;

    let mut read_4 = || {
        let mut bytes = [0_u8; 4];
        file.read_exact(&mut bytes)?;
        std::io::Result::Ok(bytes)
    };

    let identifier = read_4()?;
    assert_eq!(&identifier, b"zvox");
    let version = u32::from_le_bytes(read_4()?);
    assert_eq!(version, 1);
    let width = u32::from_le_bytes(read_4()?);
    let height = u32::from_le_bytes(read_4()?);
    let depth = u32::from_le_bytes(read_4()?);

    let mut palette = [[0_u8; 4]; 256];
    file.read_exact(bytemuck::cast_slice_mut(&mut palette))?;
    //dbg!(palette);

    let mut array = vec![0; width as usize * height as usize * depth as usize];
    zstd::Decoder::new(file)?.read_exact(&mut array)?;
    Ok((array, width, height, depth, palette))
}
