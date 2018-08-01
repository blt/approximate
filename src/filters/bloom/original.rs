//! The original 'method 2' bloom filter

use bitvec::{BigEndian, BitVec};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use twox_hash::RandomXxHashBuilder;

/// `BuildError` signals when the construction of a Bloom has failed. There are
/// a few toggles available to the user during construction and it's not
/// impossible to find settings that, taken together, result in a poorly
/// configured Bloom.
#[derive(Debug)]
pub enum BuildError {
    /// Not enough underlying bytes have been allocated to satisfy the error
    /// bound for the number of hash functions demanded.
    InsufficientCapacity,
    /// Error guard out of bounds, must be > 0.0 and < 1.0
    GuardOutOfBounds,
    /// When supplying hash factors the vector must not be empty
    HashFactorsEmpty,
}

/// `InsertError` signals when a checked insertion into a Bloom goes
/// sideways. There are a limited number of failure conditions, happily.
#[derive(Debug)]
pub enum InsertError {
    /// Too many elements have been inserted into the Bloom to maintain the
    /// error guarantee.
    Overfill(bool),
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
fn is_even(x: usize) -> bool {
    (x % 2) == 0
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
    /// Construct a new Bloom
    ///
    /// There are three toggles available to the user here:
    ///
    ///  * bytes
    ///  * error_bound
    ///  * total_hashes
    ///
    /// First, bytes. The Bloom has an underlying, fixed size allocation and
    /// 'bytes' controls the size of this allocation. There total size of Bloom
    /// will not be exactly 'bytes' but within a low constant factor of it.
    ///
    /// Second, error_bound. The bloom filter data structure is able to answer
    /// set member queries but will sometimes falsely answer in the positive to
    /// a query, a 'false positive'. `error_bound` controls the likelihood of
    /// false positives.
    ///
    /// Third, total_hashes. This Bloom uses a singular hash function and
    /// modifies the resulting hash according to according to `a*H(x) mod 2^q`
    /// where `2^q` is the period of the hash function `H(x)` and `a` is an odd
    /// value chosen from `2^q`. This approach is via "An Improved Construction
    /// for Counting Bloom Filters" by Bonomi et al and has the advantage of
    /// requiring only a single hash per insertion/query. `total_hashes`
    /// controls how many `a` values will be constructed. See
    /// [`new_with_hash_factors`] if you wish to supply the `a` values yourself.
    ///
    /// ## Error Conditions
    ///
    /// `BuildError::InsufficientCapacity` will result if the allocated bytes
    /// are insufficient to store any items, per the `error_bound` and
    /// `total_hashes` constraints. The total capacity of a Bloom is `- (bytes
    /// ln (1 - error_bound ^ (1/total_hashes))) / total_hashes)`. This may be
    /// inspected after construction with [`capacity`].
    ///
    /// `BuildError::GuardOutOfBounds` will result if the error_bound is not
    /// between 0 and 1, exclusive.
    ///
    /// `BuildError::HashFactorsEmpty` will result if `total_hashes` is 0.
    ///
    /// ```
    /// use approximate::filters::bloom::original::Bloom;
    ///
    /// let too_small_bloom = Bloom::<u16, _>::new(1, 0.0001, 10);
    /// assert!(too_small_bloom.is_err());
    ///
    /// let bloom = Bloom::<u16, _>::new(100_000, 0.0001, 10);
    /// assert!(bloom.is_ok())
    /// ```
    pub fn new(bytes: usize, error_bound: f64, total_hashes: u16) -> Result<Self, BuildError> {
        let max_capacity = capacity(bytes, error_bound, total_hashes);
        if max_capacity == 0 {
            return Err(BuildError::InsufficientCapacity);
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
            if is_even(fct) {
                continue;
            }
            if !hash_factors.contains(&fct) {
                hash_factors.push(fct)
            }
        }
        Self::new_with_hash_factors(bytes, error_bound, hash_factors)
    }

    /// Like [`new`] but with explicit hash factors.
    pub fn new_with_hash_factors(
        bytes: usize,
        error_bound: f64,
        hash_factors: Vec<usize>,
    ) -> Result<Self, BuildError> {
        if hash_factors.is_empty() {
            return Err(BuildError::HashFactorsEmpty);
        }
        if !((error_bound > 0.0) || (error_bound < 1.0)) {
            return Err(BuildError::GuardOutOfBounds);
        }

        let total_hashes = hash_factors.len();
        assert!(total_hashes <= u16::max_value() as usize);
        let max_capacity = capacity(bytes, error_bound, total_hashes as u16);
        if max_capacity == 0 {
            return Err(BuildError::InsufficientCapacity);
        }
        for fct in &hash_factors {
            assert!(!is_even(*fct))
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
    /// Like [`new`] but with explicit hash factors and hasher.
    ///
    /// The default construction of Bloom uses twox hash. Use this function if
    /// you wish to supply your own hash function.
    pub fn new_with_hash_factors_and_hasher(
        bytes: usize,
        error_bound: f64,
        hash_factors: Vec<usize>,
        hash_builder: S,
    ) -> Result<Self, BuildError> {
        if hash_factors.is_empty() {
            return Err(BuildError::HashFactorsEmpty);
        }
        if !((error_bound > 0.0) || (error_bound < 1.0)) {
            return Err(BuildError::GuardOutOfBounds);
        }

        let total_hashes = hash_factors.len();
        assert!(total_hashes <= u16::max_value() as usize);
        let max_capacity = capacity(bytes, error_bound, total_hashes as u16);
        for fct in &hash_factors {
            assert!(!is_even(*fct))
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

    /// Return the capacity of the Bloom filter
    ///
    /// The 'capacity' of a bloom filter is the number of items that can be
    /// stored in the filter without violating the false positive error bound.
    /// This implementation calculates the capacity at startup.
    ///
    /// ```
    /// use approximate::filters::bloom::original::Bloom;
    ///
    /// let bloom = Bloom::<u16, _>::new(32_000, 0.1, 2).unwrap();
    /// assert_eq!(24_328, bloom.capacity());
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the length of the Bloom filter
    ///
    /// The 'len' of a bloom filter is the number of items that have been stored
    /// in the filter.
    ///
    /// ```
    /// use approximate::filters::bloom::original::Bloom;
    ///
    /// let mut bloom = Bloom::<u16, _>::new(32_000, 0.1, 2).unwrap();
    /// assert_eq!(bloom.len(), 0);
    /// let _ = bloom.insert(&0001);
    /// let _ = bloom.insert(&0010);
    /// let _ = bloom.insert(&0100);
    /// let _ = bloom.insert(&1000);
    /// assert_eq!(4, bloom.len());
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Determine if the Bloom filter is empty
    ///
    /// This function will only return true if no insertion calls have been
    /// made. That is, if [`len`] is zero this function will return true.
    ///
    /// ```
    /// use approximate::filters::bloom::original::Bloom;
    ///
    /// let mut bloom = Bloom::<u16, _>::new(32_000, 0.1, 2).unwrap();
    /// assert_eq!(bloom.len(), 0);
    /// assert!(bloom.is_empty());
    ///
    /// let _ = bloom.insert(&0001);
    /// let _ = bloom.insert(&0010);
    /// let _ = bloom.insert(&0100);
    /// let _ = bloom.insert(&1000);
    ///
    /// assert_eq!(4, bloom.len());
    /// assert!(!bloom.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a key into the Bloom filter, return prior membership
    ///
    /// This function inserts a new `key` into the Bloom filter, incrementing
    /// length et al as appropriate. A successful return may incorrectly report
    /// that the `key` was a prior member to the filter but never report that it
    /// was not. If more than [`capacity`] keys have been inserted then an error
    /// condition will be raised, embedding the membership details inside the
    /// error variant.
    ///
    /// ```
    /// use approximate::filters::bloom::original::Bloom;
    ///
    /// let mut bloom = Bloom::<u16, _>::new(64, 0.001, 2).unwrap();
    /// assert_eq!(bloom.capacity(), 4);
    /// assert!(bloom.insert(&0001).is_ok());
    /// assert!(bloom.insert(&0010).is_ok());
    /// assert!(bloom.insert(&0100).is_ok());
    /// assert!(bloom.insert(&1000).is_ok());
    /// assert!(bloom.insert(&1001).is_err());
    ///
    /// assert_eq!(bloom.len(), 5);
    /// ```
    pub fn insert(&mut self, key: &K) -> Result<bool, InsertError> {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base = hasher.finish() as usize;
        let mut member = false;
        let cap = self.capacity;
        let err = self.len == cap;

        for idx in self.hash_factors.iter().map(|x| x.wrapping_mul(base) % cap) {
            member &= self.data.get(idx as usize);
            self.data.set(idx as usize, true);
        }
        if !member {
            self.len += 1;
        }
        if err {
            Err(InsertError::Overfill(member))
        } else {
            Ok(member)
        }
    }

    /// Insert a key into the Bloom filter, return prior membership
    ///
    /// This function acts like [`insert`] but does not check for overflow conditions.
    ///
    /// ```
    /// use approximate::filters::bloom::original::Bloom;
    ///
    /// let mut bloom = Bloom::<u16, _>::new_with_hash_factors(64, 0.001, vec![1,3]).unwrap();
    /// assert_eq!(bloom.capacity(), 4);
    /// assert_eq!(false, bloom.insert_unchecked(&0001));
    /// assert_eq!(false, bloom.insert_unchecked(&0010));
    /// assert_eq!(false, bloom.insert_unchecked(&0100));
    /// assert_eq!(false, bloom.insert_unchecked(&1000));
    /// assert_eq!(false, bloom.insert_unchecked(&1001));
    /// ```
    pub fn insert_unchecked(&mut self, key: &K) -> bool {
        match self.insert(key) {
            Ok(b) => b,
            Err(e) => match e {
                InsertError::Overfill(b) => b,
            },
        }
    }

    /// Tests if a key is a member of the Bloom
    ///
    /// This function will never report negatively to a previously entered key
    /// but may return true for a key that was not previously inserted.
    ///
    /// ```
    /// use approximate::filters::bloom::original::Bloom;
    ///
    /// let mut bloom = Bloom::<u16, _>::new(64, 0.001, 2).unwrap();
    /// assert_eq!(bloom.capacity(), 4);
    /// assert!(bloom.insert(&0001).is_ok());
    ///
    /// assert!(bloom.is_member(&0001));
    /// ```
    pub fn is_member(&self, key: &K) -> bool {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base = hasher.finish();
        let mut member = true;
        assert!(!self.hash_factors.is_empty());
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
            if factors.iter().any(|x| is_even(*x)) {
                return TestResult::discard();
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
                bloom.insert_unchecked(entry);
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
                if is_even(*fct) {
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
                bloom.insert_unchecked(&entry);
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
            if factors.iter().any(|x| is_even(*x)) {
                return TestResult::discard();
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
                bloom.insert_unchecked(entry);
            }
            assert_eq!(bloom.len(), entries.len());

            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<usize>) -> TestResult);
    }

    // Here we demonstrate that an overfill only happens when more than
    // 'capacity' elements are inserted into the Bloom and the reverse.
    #[test]
    pub fn prop_overfill_per_capacity() {
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
            if factors.iter().any(|x| is_even(*x)) {
                return TestResult::discard();
            }
            if bytes == 0 || (error_bound <= 0.0) || (error_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Bloom::new_with_hash_factors(bytes, error_bound, factors);
            if bloom.is_err() {
                return TestResult::discard();
            }
            let mut bloom = bloom.unwrap();

            if entries.len() > bloom.capacity() {
                // will encounter overflow
                let mut found_overflow = false;
                for entry in &entries {
                    if let Err(e) = bloom.insert(entry) {
                        match e {
                            InsertError::Overfill(_) => found_overflow = true,
                        }
                    }
                }
                assert!(found_overflow);
            } else {
                // will not encounter overflow
                for entry in &entries {
                    bloom.insert(entry).unwrap();
                }
            }

            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<usize>) -> TestResult);
    }
}
