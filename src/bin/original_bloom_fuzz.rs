#[macro_use]
extern crate honggfuzz;
extern crate approximate;
extern crate arbitrary;

use approximate::filters::bloom::original::Bloom;
use arbitrary::*;

enum Operation<T> {
    Insert(T),
    Contains(T),
}

impl<T> Arbitrary for Operation<T>
where
    T: Arbitrary,
{
    fn arbitrary<U: Unstructured + ?Sized>(u: &mut U) -> Result<Self, U::Error> {
        let var: u8 = Arbitrary::arbitrary(u)?;
        match var % 2 {
            0 => {
                let val: T = Arbitrary::arbitrary(u)?;
                Ok(Operation::Insert(val))
            }
            1 => {
                let val: T = Arbitrary::arbitrary(u)?;
                Ok(Operation::Contains(val))
            }
            _ => unreachable!(),
        }
    }
}

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            let mut ring = RingBuffer::new(data, 4048).unwrap();

            let total_hashes: usize = Arbitrary::arbitrary(&mut ring).unwrap();
            let total_hashes = total_hashes % 8;
            let mut hash_factors: Vec<usize> = Vec::with_capacity(total_hashes);
            for _ in 0..total_hashes {
                hash_factors.push(Arbitrary::arbitrary(&mut ring).unwrap());
            }
            let error_bound: f64 = Arbitrary::arbitrary(&mut ring).unwrap();
            let bytes: u16 = Arbitrary::arbitrary(&mut ring).unwrap();
            let bytes: usize = bytes as usize;

            let limit: u16 = Arbitrary::arbitrary(&mut ring).unwrap();
            if let Ok(mut bloom) =
                Bloom::<u32, _>::new_with_hash_factors(bytes, error_bound, hash_factors)
            {
                for _ in 0..(limit % 10_000) {
                    match Arbitrary::arbitrary(&mut ring).unwrap() {
                        Operation::Insert(val) => {
                            let _ = bloom.insert(&val);
                        }
                        Operation::Contains(val) => {
                            let _ = bloom.is_member(&val);
                        }
                    }
                }
            }
        })
    }
}
