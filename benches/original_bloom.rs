#[macro_use]
extern crate criterion;
#[macro_use]
extern crate itertools;
extern crate approximate;

use approximate::filters::bloom::original::Bloom;
use criterion::{Criterion, ParameterizedBenchmark, Throughput};

static BYTES: [usize; 4] = [2048, 4096, 8192, 16384];
static ERRORS: [f64; 4] = [0.1, 0.01, 0.001, 0.0001];
static TOTAL_HASHES: [u16; 3] = [2, 4, 8];
static TOTAL_INSERTS: [usize; 4] = [10, 100, 1000, 10_000];

fn creation(c: &mut Criterion) {
    c.bench(
        "insertion",
        ParameterizedBenchmark::new(
            "insertion",
            |b, (&bytes, &error_bound, &total_hashes)| {
                b.iter(|| {
                    let _ = Bloom::<usize, _>::new(bytes, error_bound, total_hashes);
                })
            },
            iproduct!(BYTES.iter(), ERRORS.iter(), TOTAL_HASHES.iter()),
        ).throughput(|(bytes, _, _)| Throughput::Elements(**bytes as u32)),
    );
}

fn insertion(c: &mut Criterion) {
    c.bench(
        "insertion",
        ParameterizedBenchmark::new(
            "insertion",
            |b, (&bytes, &error_bound, &total_hashes, &total_inserts)| {
                b.iter(|| {
                    if let Ok(mut bloom) = Bloom::<usize, _>::new(bytes, error_bound, total_hashes)
                    {
                        for i in 0..total_inserts {
                            let _ = bloom.insert(&i);
                        }
                    }
                })
            },
            iproduct!(
                BYTES.iter(),
                ERRORS.iter(),
                TOTAL_HASHES.iter(),
                TOTAL_INSERTS.iter()
            ),
        ).throughput(|(_, _, _, &total_inserts)| Throughput::Elements(total_inserts as u32)),
    );
}

criterion_group!{
    name = original_bloom_benches;
    config = Criterion::default();
    targets = insertion, creation
}
criterion_main!(original_bloom_benches);
