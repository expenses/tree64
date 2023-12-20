pub fn send_image(client: &mut tev_client::TevClient, name: &str, dim: u32, values: &[f32]) {
    client
        .send(tev_client::PacketCreateImage {
            image_name: name,
            grab_focus: false,
            width: dim,
            height: dim,
            channel_names: &["R", "G", "B"],
        })
        .unwrap();
    client
        .send(tev_client::PacketUpdateImage {
            image_name: name,
            grab_focus: false,
            channel_names: &["R", "G", "B"],
            channel_offsets: &[0, 1, 2],
            channel_strides: &[3, 3, 3],
            x: 0,
            y: 0,
            width: dim,
            height: dim,
            data: values,
        })
        .unwrap();
}
