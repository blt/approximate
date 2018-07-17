//! The original 'method 2' bloom filter

use bitvec::{BigEndian, BitVec};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use twox_hash::RandomXxHashBuilder;

///
pub struct Bloom<K, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    hash_builder: S,
    // Bit writes are spaced according to `a*H(x) mod 2^q` where `2^q` is the
    // period of the hash function `H(x)` and `a` is an odd value chosen from
    // `2^q`. This thanks to "An Improved Construction for Counting Bloom
    // Filters" by Bonomi et al.
    capacity: usize,
    multiplicative_factors: Vec<u64>,
    data: BitVec<BigEndian, u8>,
    phantom: PhantomData<K>,
}

#[inline]
fn is_nonzero_even(x: u64) -> bool {
    if x == 0 {
        false
    } else {
        (x % 2) == 0
    }
}

impl<K> Bloom<K, RandomXxHashBuilder>
where
    K: Hash + Eq,
{
    // TODO(blt) -- I am unhappy with the duplication in the initializers. I
    // wonder, can I define these in terms of each other somehow?
    pub fn with_capacity(capacity: usize) -> Bloom<K, RandomXxHashBuilder> {
        let total_factors = 4;
        let mut factors: Vec<u64> = Vec::with_capacity(total_factors);
        let range = Uniform::from(0..u64::max_value());
        let mut rng = thread_rng();
        let capacity = (total_factors * 2) * capacity; // total_factors * 2 because we only pick odds
        while factors.len() < total_factors {
            let fct = range.sample(&mut rng) % (capacity as u64);
            if is_nonzero_even(fct) {
                continue;
            }
            if !factors.contains(&fct) {
                factors.push(fct)
            }
        }
        Bloom::with_capacity_and_factors(capacity, factors)
    }

    pub fn with_capacity_and_factors(
        capacity: usize,
        factors: Vec<u64>,
    ) -> Bloom<K, RandomXxHashBuilder> {
        assert!(!factors.is_empty());
        let total_factors = factors.len();
        for fct in &factors {
            assert!(!is_nonzero_even(*fct))
        }
        let capacity = (total_factors * 2) * capacity; // total_factors * 2 because we only pick odds
        let mut data = BitVec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(false);
        }
        assert_eq!(capacity, data.len());
        println!("CAPACITY: {:?} | FACTORS: {:?}", capacity, factors);

        Bloom {
            hash_builder: RandomXxHashBuilder::default(),
            data,
            capacity,
            multiplicative_factors: factors,
            phantom: PhantomData,
        }
    }
}

impl<K, S> Bloom<K, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    // pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Bloom<K, S> {
    //     let total_factors = 4;
    //     let mut factors: Vec<u64> = Vec::with_capacity(total_factors);
    //     let range = Uniform::from(0..u64::max_value());
    //     let mut rng = thread_rng();
    //     while factors.len() < total_factors {
    //         let fct = range.sample(&mut rng);
    //         if !factors.contains(&fct) {
    //             factors.push(fct)
    //         }
    //     }
    //     let capacity = total_factors * capacity;
    //     let mut data = BitVec::with_capacity(capacity);
    //     for _ in 0..capacity {
    //         data.push(false);
    //     }

    //     Bloom {
    //         hash_builder,
    //         data,
    //         multiplicative_factors: factors,
    //         phantom: PhantomData,
    //     }
    // }

    // pub fn with_capacity_and_hasher_and_factors(
    //     capacity: usize,
    //     hash_builder: S,
    //     multiplicative_factors: Vec<u64>,
    // ) -> Bloom<K, S> {
    //     let capacity = multiplicative_factors.len() * capacity;
    //     let mut data = BitVec::with_capacity(capacity);
    //     for _ in 0..capacity {
    //         data.push(false);
    //     }
    //     // TODO(blt) -- must check that all multiplicative_factors are unique
    //     Bloom {
    //         hash_builder,
    //         data,
    //         multiplicative_factors,
    //         phantom: PhantomData,
    //     }
    // }

    pub fn insert(&mut self, key: &K) -> bool {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base = hasher.finish();
        let mut member = false;
        let cap = self.capacity as u64;
        for idx in self
            .multiplicative_factors
            .iter()
            .map(|x| x.wrapping_mul(base) % cap)
        {
            println!("HERE I START: {:?} | {:?} | {:?}", base, idx, cap);
            member |= self.data.get(idx as usize);
            println!("MEMBER: {:?}", member);
            self.data.set(idx as usize, true);
        }
        member
    }

    pub fn is_member(&self, key: &K) -> bool {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base = hasher.finish();
        let mut member = false;
        let cap = self.capacity as u64;
        for idx in self
            .multiplicative_factors
            .iter()
            .map(|x| x.wrapping_mul(base) % cap)
        {
            member |= self.data.get(idx as usize);
        }
        member
    }

    #[cfg(test)]
    pub fn total_factors(&self) -> usize {
        self.multiplicative_factors.len()
    }

    #[cfg(test)]
    pub fn total_ones(&self) -> usize {
        self.data.count_one()
    }
}

// Properties

// - On insertion to a fresh Bloom there should only be
//   multiplicative_factors.len() worth bits lit up

// - No false negatives, only false positives

// - Actual false-positive rate is observed

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    #[test]
    pub fn explicit_factors_total_lit() {
        let capacity = 1;
        let entry = 0;
        let mut bloom = Bloom::with_capacity_and_factors(capacity, vec![0, 1, 3]);
        // After many inserts we may have a false positive result here. The
        // first insertion into the bloom filter is the only one we can be
        // sure will be totally accurate.
        assert_eq!(false, bloom.insert(&entry));
        assert_eq!(4, bloom.total_factors());
    }

    #[test]
    pub fn factors_total_lit() {
        fn inner(capacity: usize, entry: u16, mut factors: Vec<u64>) -> TestResult {
            factors.sort();
            factors.dedup();
            for fct in factors {
                if is_nonzero_even(fct) {
                    return TestResult::discard();
                }
            }

            if capacity == 0 {
                return TestResult::discard();
            }
            let mut bloom = Bloom::with_capacity(capacity);
            // After many inserts we may have a false positive result here. The
            // first insertion into the bloom filter is the only one we can be
            // sure will be totally accurate.
            assert_eq!(false, bloom.insert(&entry));
            assert_eq!(4, bloom.total_factors());

            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(usize, u16, Vec<u64>) -> TestResult);
    }
}
