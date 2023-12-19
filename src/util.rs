#[derive(Default)]
pub struct FreeLocations {
    pub l2r: Vec<glam::IVec2>,
    pub r2l: Vec<glam::IVec2>,
    pub b2t: Vec<glam::IVec2>,
    pub t2b: Vec<glam::IVec2>,
}

impl FreeLocations {
    pub fn clear(&mut self) {
        self.l2r.clear();
        self.r2l.clear();
        self.b2t.clear();
        self.t2b.clear();
    }

    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Option<(glam::IVec2, glam::IVec2)> {
        let mut total = self.l2r.len() + self.r2l.len() + self.b2t.len() + self.t2b.len();

        if total == 0 {
            return None;
        }

        let mut index = rng.gen_range(0..total);

        if index < self.l2r.len() {
            return Some((self.l2r[index], glam::IVec2::new(1, 0)));
        }

        index -= self.l2r.len();

        if index < self.r2l.len() {
            return Some((self.r2l[index], glam::IVec2::new(-1, 0)));
        }

        index -= self.r2l.len();

        if index < self.b2t.len() {
            return Some((self.b2t[index], glam::IVec2::new(0, 1)));
        }

        index -= self.b2t.len();

        Some((self.t2b[index], glam::IVec2::new(0, -1)))
    }

    pub fn iter(&self) -> impl Iterator<Item = (glam::IVec2, glam::IVec2)> + '_ {
        self.l2r
            .iter()
            .copied()
            .map(|pos| (pos, glam::IVec2::new(1, 0)))
            .chain(
                self.r2l
                    .iter()
                    .copied()
                    .map(|pos| (pos, glam::IVec2::new(-1, 0))),
            )
            .chain(
                self.b2t
                    .iter()
                    .copied()
                    .map(|pos| (pos, glam::IVec2::new(0, 1))),
            )
            .chain(
                self.t2b
                    .iter()
                    .copied()
                    .map(|pos| (pos, glam::IVec2::new(0, -1))),
            )
    }
}

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
            data: &values,
        })
        .unwrap();
}
