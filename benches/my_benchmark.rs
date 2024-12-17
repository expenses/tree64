use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use markov::wfc::*;
use rand::{rngs::SmallRng, SeedableRng};

fn benchmark_wave_size<Wave: WaveNum, const BITS: usize>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let mut tileset = Tileset::<Wave, BITS>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(0.5);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    tileset.connect(beach, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);

    let mut rng = SmallRng::from_entropy();

    for i in [5, 10, 25, 50, 100, 250, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new(format!("{}_shannon", BITS), i),
            i,
            |b, &i| {
                b.iter(|| {
                    let mut wfc = tileset.create_wfc::<ShannonEntropy>((i, 100, 1));
                    wfc.collapse_all(&mut rng)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}_linear", BITS), i),
            i,
            |b, &i| {
                b.iter(|| {
                    let mut wfc = tileset.create_wfc::<LinearEntropy>((i, 100, 1));
                    wfc.collapse_all(&mut rng)
                })
            },
        );
    }
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wfc");
    benchmark_wave_size::<u8, 8>(&mut group);
    benchmark_wave_size::<u16, 16>(&mut group);
    benchmark_wave_size::<u32, 32>(&mut group);
    benchmark_wave_size::<u64, 64>(&mut group);
    benchmark_wave_size::<u128, 128>(&mut group);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
