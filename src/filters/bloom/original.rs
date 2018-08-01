//! The original 'method 2' bloom filter

use bitvec::{BigEndian, BitVec};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use twox_hash::RandomXxHashBuilder;

#[derive(Debug)]
pub enum Error {
    /// Not enough underlying bytes have been allocated to satisfy the error
    /// bound for the number of hash functions demanded.
    InsufficientCapacity,
    /// Error guard out of bounds, must be > 0.0 and < 1.0
    GuardOutOfBounds,
    /// When supplying hash factors the vector must not be empty
    HashFactorsEmpty,
}

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
    len: usize,
    hash_factors: Vec<usize>,
    data: BitVec<BigEndian, u8>,
    phantom: PhantomData<K>,
}

#[inline]
fn is_nonzero_even(x: usize) -> bool {
    if x == 0 {
        false
    } else {
        (x % 2) == 0
    }
}

#[inline]
fn capacity(bytes: usize, error_bound: f64, total_hashes: u16) -> usize {
    let bytes = bytes as f64;
    let total_hashes = f64::from(total_hashes);
    let res = -((bytes * (1.0 - error_bound.powf(1.0 / total_hashes)).ln()) / total_hashes);
    assert!(!res.is_sign_negative());
    res as usize
}

impl<K> Bloom<K, RandomXxHashBuilder>
where
    K: Hash + Eq,
{
    pub fn new(bytes: usize, error_bound: f64, total_hashes: u16) -> Result<Self, Error> {
        let max_capacity = capacity(bytes, error_bound, total_hashes);
        if max_capacity == 0 {
            return Err(Error::InsufficientCapacity);
        }
        let mut hash_factors: Vec<usize> = Vec::with_capacity(total_hashes as usize);

        // We populate hash_factors with random factors, taking care to make
        // sure they do not overlap and are odd. The oddness decreases the
        // likelyhood that we will hash to the same bits. Doesn't eliminate it,
        // mind, just makes it less likely.
        let range = Uniform::from(0..usize::max_value());
        let mut rng = thread_rng();
        let capacity = (total_hashes as usize * 2) * max_capacity; // total_factors * 2 because we only pick odds
        while hash_factors.len() < total_hashes as usize {
            let fct = range.sample(&mut rng) % capacity;
            if !is_nonzero_even(fct) {
                continue;
            }
            if !hash_factors.contains(&fct) {
                hash_factors.push(fct)
            }
        }
        Self::new_with_hash_factors(bytes, error_bound, hash_factors)
    }

    pub fn new_with_hash_factors(
        bytes: usize,
        error_bound: f64,
        hash_factors: Vec<usize>,
    ) -> Result<Self, Error> {
        if hash_factors.is_empty() {
            return Err(Error::HashFactorsEmpty);
        }
        if (error_bound >= 0.0) || (error_bound <= 1.0) {
            return Err(Error::GuardOutOfBounds);
        }

        let total_hashes = hash_factors.len();
        assert!(total_hashes <= u16::max_value() as usize);
        let max_capacity = capacity(bytes, error_bound, total_hashes as u16);
        assert!(max_capacity != 0);
        for fct in &hash_factors {
            assert!(is_nonzero_even(*fct))
        }
        let capacity = (total_hashes * 2) * max_capacity; // total_factors * 2 because we only pick odds
        let mut data = BitVec::with_capacity(max_capacity);
        for _ in 0..capacity {
            data.push(false);
        }
        assert_eq!(capacity, data.len());

        Ok(Self {
            hash_builder: RandomXxHashBuilder::default(),
            data,
            capacity,
            len: 0,
            hash_factors,
            phantom: PhantomData,
        })
    }
}

impl<K, S> Bloom<K, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    pub fn new_with_hash_factors_and_hasher(
        bytes: usize,
        error_bound: f64,
        hash_factors: Vec<usize>,
        hash_builder: S,
    ) -> Result<Self, Error> {
        if hash_factors.is_empty() {
            return Err(Error::HashFactorsEmpty);
        }
        if (error_bound >= 0.0) || (error_bound <= 1.0) {
            return Err(Error::GuardOutOfBounds);
        }

        let total_hashes = hash_factors.len();
        assert!(total_hashes <= u16::max_value() as usize);
        let max_capacity = capacity(bytes, error_bound, total_hashes as u16);
        for fct in &hash_factors {
            assert!(!is_nonzero_even(*fct))
        }
        let capacity = (total_hashes * 2) * max_capacity; // total_factors * 2 because we only pick odds
        let mut data = BitVec::with_capacity(max_capacity);
        for _ in 0..capacity {
            data.push(false);
        }
        assert_eq!(capacity, data.len());

        Ok(Self {
            hash_builder,
            data,
            capacity,
            len: 0,
            hash_factors,
            phantom: PhantomData,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn insert(&mut self, key: &K) -> bool {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base = hasher.finish() as usize;
        let mut member = false;
        let cap = self.capacity;
        for idx in self.hash_factors.iter().map(|x| x.wrapping_mul(base) % cap) {
            member &= self.data.get(idx as usize);
            self.data.set(idx as usize, true);
        }
        if !member {
            self.len += 1;
        }
        member
    }

    pub fn is_member(&self, key: &K) -> bool {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base = hasher.finish();
        let mut member = false;
        for idx in self
            .hash_factors
            .iter()
            .map(|x| x.wrapping_mul(base as usize) % self.capacity)
        {
            member &= self.data.get(idx as usize);
        }
        member
    }

    #[cfg(test)]
    pub fn total_factors(&self) -> usize {
        self.hash_factors.len()
    }

    #[cfg(test)]
    pub fn total_ones(&self) -> usize {
        self.data.count_one()
    }
}

// - No false negatives, only false positives

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    // Here we demonstrate that we can never observe a false negative result
    // from is_member.
    #[test]
    pub fn prop_never_false_negatives() {
        fn inner(
            bytes: usize,
            error_bound: f64,
            entries: Vec<u16>,
            mut factors: Vec<usize>,
        ) -> TestResult {
            factors.sort();
            factors.dedup();
            if factors.is_empty() {
                return TestResult::discard();
            }
            for fct in &factors {
                if !is_nonzero_even(*fct) {
                    return TestResult::discard();
                }
            }

            if bytes == 0 || (error_bound <= 0.0) || (error_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Bloom::new_with_hash_factors(bytes, error_bound, factors);
            if bloom.is_err() {
                return TestResult::discard();
            }
            let mut bloom = bloom.unwrap();
            for entry in &entries {
                bloom.insert(entry);
            }
            for entry in entries {
                assert!(bloom.is_member(&entry));
            }

            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<usize>) -> TestResult);
    }

    // Here we determine that after an insertion there is at least 1 bit lit in
    // the underlying byte array and no more than factors.len().
    //
    // No claim can be made on whether an insertion will return a false positive
    // or not, even if that insertion is the first into the set. That is on
    // account of one of the 'factor' hashes available may map to the same
    // underlying bit.
    #[test]
    pub fn prop_lit_interior_bits_inequality() {
        fn inner(
            bytes: usize,
            error_bound: f64,
            entries: Vec<u16>,
            mut factors: Vec<usize>,
        ) -> TestResult {
            factors.sort();
            factors.dedup();
            let factors_len = factors.len();
            if factors.is_empty() {
                return TestResult::discard();
            }
            for fct in &factors {
                if !is_nonzero_even(*fct) {
                    return TestResult::discard();
                }
            }

            if bytes == 0 || (error_bound <= 0.0) || (error_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Bloom::new_with_hash_factors(bytes, error_bound, factors);
            if bloom.is_err() {
                return TestResult::discard();
            }
            let mut bloom = bloom.unwrap();
            let mut entries_inserted = 0;
            for entry in entries {
                bloom.insert(&entry);
                entries_inserted += 1;
                assert!(bloom.total_factors() <= entries_inserted * factors_len);
                assert!(bloom.total_factors() >= 1);
            }

            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<usize>) -> TestResult);
    }

    // Here we determine that 'len' is always the same as the `entries.len`
    // after all insertions have completed.
    #[test]
    pub fn prop_len_equality() {
        fn inner(
            bytes: usize,
            error_bound: f64,
            entries: Vec<u16>,
            mut factors: Vec<usize>,
        ) -> TestResult {
            factors.sort();
            factors.dedup();
            if factors.is_empty() {
                return TestResult::discard();
            }
            for fct in &factors {
                if !is_nonzero_even(*fct) {
                    return TestResult::discard();
                }
            }

            if bytes == 0 || (error_bound <= 0.0) || (error_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Bloom::new_with_hash_factors(bytes, error_bound, factors);
            if bloom.is_err() {
                return TestResult::discard();
            }
            let mut bloom = bloom.unwrap();
            for entry in &entries {
                bloom.insert(entry);
            }
            assert_eq!(bloom.len(), entries.len());

            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<usize>) -> TestResult);
    }
}
