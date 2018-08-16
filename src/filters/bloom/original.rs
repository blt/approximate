//! Dr. Burton Bloom's 'method 2' filter
//!
//! This implementation of Bloom's filter is from "Space/Time Trade-offs in Hash
//! Coding with Allowable Errors", specifically the 'method 2'
//! implementation. The filter uses k hash functions -- adapted somewhat from
//! Bloom's original _actual_ k hash functions to a single hash function smeared
//! around per "An Improved Construction for Counting Bloom Filters" by Bonomi
//! et al -- to map a hashed key onto a bit array, writing at least 1 and at
//! most k '1' bits into the array on insertion. Lookups are performed by
//! checking the same indexes: the key is present in the filter if all bits are
//! '1', else not.
//!
//! Bloom filters may possibly answer queries with a false positive. The rate at
//! which this occurs is given, in this implementation, by the user as
//! `error_guard`. See [`Bloom::new`] for full details.
use bitvec::{BigEndian, BitVec};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::f64::consts::{E, LN_2};
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use twox_hash::RandomXxHashBuilder;

struct Parameters {
    bits: Option<usize>, // realistically, isize::max_value() as usize
    capacity: Option<usize>,
    false_positive_bound: Option<f64>,
    total_hashes: Option<u8>,
}

impl Parameters {
    pub fn new(
        bits: Option<usize>,
        capacity: Option<usize>,
        false_positive_bound: Option<f64>,
        total_hashes: Option<u8>,
    ) -> Self {
        Parameters {
            bits,
            capacity,
            false_positive_bound,
            total_hashes,
        }
    }

    pub fn bits(&self) -> Result<usize, BuildError> {
        // where k = total_hashes
        //       m = bits in the filter
        //       n = number of items in the filter
        //       p = probability of false positives
        //
        //     _          _
        //     | n * ln p |
        // m = | -------- |
        //     | (ln 2)^2 |
        if self.false_positive_bound.is_none() || self.capacity.is_none() {
            return Err(BuildError::UnableToComputeByteLimit);
        }
        let n = self.capacity.unwrap() as f64;
        let p = self.false_positive_bound.unwrap() as f64;

        let m: f64 = ((n * p.ln()) / (LN_2 * LN_2)).ceil();
        let bits: usize = m as usize;

        if bits == 0 {
            Err(BuildError::InsufficientCapacity)
        } else {
            Ok(bits)
        }
    }

    pub fn capacity(&self) -> Result<usize, BuildError> {
        // where k = total_hashes
        //       m = bits in the filter
        //       n = number of items in the filter
        //       p = probability of false positives
        // and
        //       p = (1 - (1-(1/m))^(k*n))^k
        // then
        //                  (1 / k)
        //     ln ( 1 - p ^         )
        // n = ----------------------
        //                 m - 1
        //        k * ln ( ----- )
        //                   m
        if self.false_positive_bound.is_none() || self.total_hashes.is_none() || self.bits.is_none()
        {
            return Err(BuildError::UnableToComputeCapacity);
        }
        let p = self.false_positive_bound.unwrap();
        let k = self.total_hashes.unwrap() as f64;
        let m = self.bits.unwrap() as f64;

        let num = (1.0 - p.powf(1.0 / k)).ln();
        let dem = k * ((m - 1.0) / m).ln();
        let n = (num / dem).ceil() as usize;
        if n == 0 {
            Err(BuildError::InsufficientCapacity)
        } else {
            Ok(n)
        }
    }

    pub fn false_positive_bound(&self) -> Result<f64, BuildError> {
        // where k = total_hashes
        //       m = bits in the filter
        //       n = number of items in the filter
        //       p = probability of false positives
        //
        //      /                      \ k
        //     |         - (k * n) / m |
        // p = | 1 - e ^               |
        //     |                       |
        //     \                       /
        if self.capacity.is_none() || self.total_hashes.is_none() || self.bits.is_none() {
            return Err(BuildError::UnableToComputeErrorBound);
        }
        let k = self.total_hashes.unwrap() as f64;
        let m = self.bits.unwrap() as f64;
        let n = self.capacity.unwrap() as f64;
        let p = (1.0 - E.powf(-((k * n) / m))).powf(k);
        if !((p > 0.0) && (p < 1.0)) {
            Err(BuildError::GuardOutOfBounds)
        } else {
            Ok(p)
        }
    }

    pub fn total_hashes(&self) -> Result<u8, BuildError> {
        // where k = total_hashes
        //       m = bits in the filter
        //       n = number of items in the filter
        //       p = probability of false positives
        //
        //              m * ln(2)
        //  k = round ( --------- )
        //                  n
        if self.bits.is_none() || self.capacity.is_none() {
            return Err(BuildError::UnableToComputeTotalHashes);
        }
        let n = self.capacity.unwrap() as f64;
        let m = self.bits.unwrap() as f64;
        let k = ((m * LN_2) / n).round();
        let total_hashes = k as u8;
        if total_hashes == 0 {
            Err(BuildError::HashFactorsEmpty)
        } else {
            Ok(total_hashes)
        }
    }
}

/// TODO
pub struct Builder<K, S = RandomXxHashBuilder>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    bits: Option<usize>,               // maximum bits to allocate, less overhead
    capacity: Option<usize>,           // estimated total elements to be stored
    false_positive_bound: Option<f64>, // false positive rate
    hash_builder: S,                   // the hasher to use
    total_hash_factors: Option<u8>,    // total number of hashing factors
    hash_factors: Option<Vec<u64>>,    // explicit hash factors
    phantom: PhantomData<K>,
}

impl<K> Builder<K, RandomXxHashBuilder>
where
    K: Hash + Eq,
{
    /// Create a new Builder
    ///
    /// The purpose of the builder is to allow the user to configure the Bloom
    /// to their liking. There are a few knobs to twist on this data structure,
    /// some imply others or allow their derivation at optimal settings.
    pub fn new() -> Self {
        Builder {
            bits: None,
            capacity: None,
            false_positive_bound: None,
            hash_builder: RandomXxHashBuilder::default(),
            total_hash_factors: None,
            hash_factors: None,
            phantom: PhantomData,
        }
    }
}

impl<K, S> Builder<K, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Create a new Builder with a user-supplied hasher
    ///
    /// The purpose of the builder is to allow the user to configure the Bloom
    /// to their liking. There are a few knobs to twist on this data structure,
    /// some imply others or allow their derivation at optimal settings.
    pub fn with_hasher(hash_builder: S) -> Self {
        Builder {
            bits: None,
            capacity: None,
            false_positive_bound: None,
            hash_builder,
            total_hash_factors: None,
            hash_factors: None,
            phantom: PhantomData,
        }
    }

    /// Introduce a maximum allocation limit onto Bloom
    ///
    /// The `bits` parameter acts as a constraint on the maximum allowable
    /// bits to be allocated for the Bloom's underlying bit array. Depending on
    /// whether the user calls [`Bloom::strict_bits`] or not the resulting
    /// Bloom may have less than `bits` bits reserved for use.
    pub fn bits(mut self, bits: usize) -> Self {
        self.bits = Some(bits);
        self
    }

    /// Estimated capacity of Bloom
    ///
    /// A Bloom can only hold a finite number of distinct elements in itself
    /// before violating the user-defined error bound on false positives. This
    /// data structure cannot estimate the capacity based on other parameters
    /// and, so, the user must supply an estimate.
    ///
    /// This is a MANDATORY call
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    /// Set the error bound for false positives
    ///
    /// A Bloom filter always answers queries with some chance for false
    /// positive responses. This parameter bounds that error, with an important
    /// caveat. The error bound only holds so long as the user does not insert
    /// more than [`capacity`] distinct elements into the Bloom.
    pub fn false_positive_bound(mut self, false_positive_bound: f64) -> Self {
        self.false_positive_bound = Some(false_positive_bound);
        self
    }

    /// Set the number of hash factors for Bloom
    ///
    /// Explicitly declare the number of hashing factors to use but allow the
    /// factors to be auto-discovered. Incompatible with a call to
    /// [`Bloom::hash_factors`].
    pub fn total_hash_factors(mut self, total_hash_factors: u8) -> Self {
        self.total_hash_factors = Some(total_hash_factors);
        self
    }

    /// Set the hash factors explicitly for Bloom
    ///
    /// Explicitly declare the hashing factors to use in Bloom. This is
    /// incompatible with a call to [`Bloom::total_hash_factors`]. Each hash
    /// factor must be distinct, non-zero and odd.
    pub fn hash_factors(mut self, hash_factors: Vec<u64>) -> Self {
        self.hash_factors = Some(hash_factors);
        self
    }

    /// Create a `Bloom<K, S>` from a `Builder<K, S>`
    ///
    /// There are three toggles available to the user here:
    ///
    ///  * bits
    ///  * false_positive_bound
    ///  * total_hashes
    ///
    /// First, bits. The Bloom has an underlying, fixed size allocation and
    /// 'bits' controls the size of this allocation. There total size of Bloom
    /// will not be exactly 'bits' but within a low constant factor of it.
    ///
    /// Second, false_positive_bound. The bloom filter data structure is able to answer
    /// set member queries but will sometimes falsely answer in the positive to
    /// a query, a 'false positive'. `false_positive_bound` controls the likelihood of
    /// false positives.
    ///
    /// Third, total_hashes. This Bloom uses a singular hash function and
    /// modifies the resulting hash according to according to `a*H(x) mod 2^q`
    /// where `2^q` is the period of the hash function `H(x)` and `a` is an odd
    /// value chosen from `2^q`. This approach is via "An Improved Construction
    /// for Counting Bloom Filters" by Bonomi et al and has the advantage of
    /// requiring only a single hash per insertion/query. `total_hashes`
    /// controls how many `a` values will be constructed. See
    /// [`Bloom::new_with_hash_factors`] if you wish to supply the `a` values
    /// yourself.
    ///
    /// ## Error Conditions
    ///
    /// `BuildError::InsufficientCapacity` will result if the allocated bits
    /// are insufficient to store any items, per the `false_positive_bound` and
    /// `total_hashes` constraints. The total capacity of a Bloom is `- (bits
    /// ln (1 - false_positive_bound ^ (1/total_hashes))) / total_hashes)`. This may be
    /// inspected after construction with [`capacity`].
    ///
    /// `BuildError::GuardOutOfBounds` will result if the false_positive_bound is not
    /// between 0 and 1, exclusive.
    ///
    /// `BuildError::HashFactorsEmpty` will result if `total_hashes` is 0.
    ///
    /// ```
    /// use approximate::filters::bloom::original::{Builder, Bloom};
    ///
    /// let too_small_bloom: Result<Bloom<u16, _>, _> = Builder::new().false_positive_bound(0.0001).bits(1).freeze();
    /// assert!(too_small_bloom.is_err());
    ///
    /// let bloom: Result<Bloom<u16, _>, _> = Builder::new().false_positive_bound(0.0001).bits(100_000).freeze();
    /// assert!(bloom.is_ok())
    /// ```
    pub fn freeze(self) -> Result<Bloom<K, S>, BuildError> {
        let total_hashes = if let Some(total_hashes) = self.total_hash_factors {
            Some(total_hashes)
        } else {
            if let Some(ref hash_factors) = self.hash_factors {
                let total_hashes = hash_factors.len();
                if total_hashes > u8::max_value() as usize {
                    return Err(BuildError::TooManyHashFactors);
                }
                Some(total_hashes as u8)
            } else {
                None
            }
        };
        let parameters = Parameters::new(
            self.bits,
            self.capacity,
            self.false_positive_bound,
            total_hashes,
        );
        let bits;
        let capacity;
        let false_positive_bound;
        let total_hashes;
        match (
            self.bits,
            self.false_positive_bound,
            self.total_hash_factors,
            self.capacity,
        ) {
            (Some(_), Some(_), Some(_), None) => {
                // compute 'capacity'
                bits = self.bits.unwrap();
                capacity = parameters.capacity()?;
                false_positive_bound = self.false_positive_bound.unwrap();
                total_hashes = self.total_hash_factors.unwrap();
            }
            (None, Some(_), Some(_), Some(_)) => {
                // compute 'bits'
                bits = parameters.bits()?;
                capacity = self.capacity.unwrap();
                false_positive_bound = self.false_positive_bound.unwrap();
                total_hashes = self.total_hash_factors.unwrap();
            }
            (Some(_), None, Some(_), Some(_)) => {
                // compute 'false_positive_bound'
                bits = self.bits.unwrap();
                capacity = self.capacity.unwrap();
                false_positive_bound = parameters.false_positive_bound()?;
                total_hashes = self.total_hash_factors.unwrap();
            }
            (Some(_), None, None, Some(_)) | (Some(_), Some(_), None, Some(_)) => {
                // compute 'total_hashes'
                bits = self.bits.unwrap();
                capacity = self.capacity.unwrap();
                false_positive_bound = self.false_positive_bound.unwrap_or(0.01);
                total_hashes = parameters.total_hashes()?;
            }
            _ => {
                return Err(BuildError::InsufficientParameters);
            }
        }
        if !((false_positive_bound > 0.0) && (false_positive_bound < 1.0)) {
            return Err(BuildError::GuardOutOfBounds);
        }
        if total_hashes == 0 {
            return Err(BuildError::HashFactorsEmpty);
        }
        let hash_factors = if let Some(hash_factors) = self.hash_factors {
            if hash_factors.is_empty() {
                return Err(BuildError::HashFactorsEmpty);
            }
            if hash_factors.len() < usize::from(total_hashes) {
                return Err(BuildError::SuppliedHashFactorsBelowOptimal);
            }
            hash_factors
        } else {
            let mut hash_factors: Vec<u64> = Vec::with_capacity(total_hashes as usize);
            // We populate hash_factors with random factors, taking care to make
            // sure they do not overlap and are odd. The oddness decreases the
            // likelyhood that we will hash to the same bits. Doesn't eliminate it,
            // mind, just makes it less likely.
            let range = Uniform::from(0..usize::max_value());
            let mut rng = thread_rng();
            let capacity = (total_hashes as usize * 2) * capacity;
            while hash_factors.len() < total_hashes as usize {
                let fct = (range.sample(&mut rng) % capacity) as u64;
                if is_even(fct) {
                    continue;
                }
                if !hash_factors.contains(&fct) {
                    hash_factors.push(fct)
                }
            }
            hash_factors
        };
        if hash_factors.is_empty() {
            return Err(BuildError::HashFactorsEmpty);
        } else {
            for fct in &hash_factors {
                if is_even(*fct) {
                    return Err(BuildError::HashFactorsZeroOrEven);
                }
            }
        }
        let mut data = BitVec::with_capacity(bits);
        for _ in 0..bits {
            data.push(false);
        }
        assert_eq!(bits, data.len());
        Ok(Bloom {
            hash_builder: self.hash_builder,
            capacity,
            len: 0,
            hash_factors: hash_factors.iter().map(|x| *x as usize).collect(),
            data,
            phantom: PhantomData,
        })
    }
}

/// `BuildError` signals when the construction of a Bloom has failed. There are
/// a few toggles available to the user during construction and it's not
/// impossible to find settings that, taken together, result in a poorly
/// configured Bloom.
#[derive(Debug)]
pub enum BuildError {
    /// If 'bits' is not supplied then 'false_positive_bound', 'total_hashes' and
    /// 'capacity' must be.
    UnableToComputeByteLimit,
    /// If 'capacity' is not supplied then 'false_positive_bound', 'total_hashes' and
    /// 'bits' must be.
    UnableToComputeCapacity,
    /// If 'false_positive_bound' is not supplied then 'capacity', 'total_hashes' and
    /// 'bits' must be.
    UnableToComputeErrorBound,
    /// If 'hash_factors' or 'total_hash_factors' is not supplied then 'bits'
    /// and 'capacity' must be.
    UnableToComputeTotalHashes,
    /// If 'hash_factors' is supplied and 'total_hash_factors' is computed to be
    /// greater than the length of this value, based on 'byte', 'false_positive_bound'
    /// settings.
    SuppliedHashFactorsBelowOptimal,
    /// A minimal number of parameters must be provided when building a new
    /// Bloom. In this instance not enough were available.
    InsufficientParameters,

    /// User must supply an estimated capacity for the structure.
    MissingCapacity,
    /// Not enough underlying bits have been allocated to satisfy the error
    /// bound for the number of hash functions demanded.
    InsufficientCapacity,
    /// Error guard out of bounds, must be > 0.0 and < 1.0
    GuardOutOfBounds,
    /// When supplying hash factors the vector must not be empty
    HashFactorsEmpty,
    /// When supplying hash factors the factors themselves must be non-zero and
    /// odd
    HashFactorsZeroOrEven,
    /// When supplying hash factors any more than u16 is nuts. Realistically
    /// values much closer to 8 are going to be optimial.
    TooManyHashFactors,
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
    data: BitVec<BigEndian, u64>,
    phantom: PhantomData<K>,
}

#[inline]
fn is_even(x: u64) -> bool {
    (x % 2) == 0
}

impl<K, S> Bloom<K, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Return the capacity of the Bloom filter
    ///
    /// The 'capacity' of a bloom filter is the number of items that can be
    /// stored in the filter without violating the false positive error bound.
    /// This implementation calculates the capacity at startup.
    ///
    /// ```
    /// use approximate::filters::bloom::original::{Bloom, Builder};
    ///
    /// let bloom: Bloom<u16, _> = Builder::new()
    ///     .bits(32_000)
    ///     .false_positive_bound(0.001)
    ///     .total_hash_factors(12)
    ///     .freeze().unwrap();
    /// assert_eq!(2204, bloom.capacity());
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
    /// use approximate::filters::bloom::original::{Bloom, Builder};
    ///
    /// let mut bloom: Bloom<u16, _> = Builder::new()
    ///         .bits(32_000)
    ///         .false_positive_bound(0.1)
    ///         .total_hash_factors(32)
    ///         .freeze().unwrap();
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
    /// made. That is, if [`Bloom::len`] is zero this function will return true.
    ///
    /// ```
    /// use approximate::filters::bloom::original::{Builder, Bloom};
    ///
    /// let mut bloom: Bloom<u16, _> = Builder::new()
    ///         .bits(32_000)
    ///         .false_positive_bound(0.1)
    ///         .total_hash_factors(32)
    ///         .freeze().unwrap();
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
    /// use approximate::filters::bloom::original::{Builder, Bloom};
    ///
    /// let mut bloom: Bloom<u16, _> = Builder::new()
    ///         .bits(38)
    ///         .false_positive_bound(0.1)
    ///         .freeze().unwrap();
    /// assert_eq!(bloom.capacity(), 8);
    /// assert!(bloom.insert(&0001).is_ok());
    /// assert!(bloom.insert(&0010).is_ok());
    /// assert!(bloom.insert(&0011).is_ok());
    /// assert!(bloom.insert(&0010).is_ok());
    /// assert!(bloom.insert(&0110).is_ok());
    /// assert!(bloom.insert(&0111).is_ok());
    /// assert!(bloom.insert(&0101).is_ok());
    /// assert!(bloom.insert(&0100).is_ok());
    /// assert!(bloom.insert(&1100).is_err());
    /// assert!(bloom.insert(&1101).is_err());
    /// assert!(bloom.insert(&1111).is_err());
    /// assert!(bloom.insert(&1110).is_err());
    /// assert!(bloom.insert(&1010).is_err());
    /// assert!(bloom.insert(&1011).is_err());
    /// assert!(bloom.insert(&1001).is_err());
    /// assert!(bloom.insert(&1000).is_err());
    ///
    /// assert_eq!(bloom.len(), 16);
    /// ```
    pub fn insert(&mut self, key: &K) -> Result<bool, InsertError> {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base = hasher.finish() as usize;
        let mut member = false;
        let cap = self.capacity;
        let err = self.len >= cap;

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
    /// This function acts like [`Bloom::insert`] but does not check for
    /// overflow conditions.
    ///
    /// ```
    /// use approximate::filters::bloom::original::{Builder, Bloom};
    ///
    /// let mut bloom: Bloom<u16, _> = Builder::new()
    ///         .bits(64)
    ///         .false_positive_bound(0.001)
    ///         .hash_factors(vec![1,3])
    ///         .freeze().unwrap();
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
    /// let mut bloom = Bloom::<u16, _>::new(64, 0.001).unwrap();
    /// assert_eq!(bloom.capacity(), 80);
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

    /// Return the number of hashes used by this filter
    ///
    /// If the user created the Bloom by supplying an explicit vector of hash
    /// factors then this will be the length of that vector. Else, it will be
    /// the optimal -- in the sense of maximizing capacity -- number of hashes
    /// computed as -lg(false_positive_bound).
    ///
    /// ```
    /// use approximate::filters::bloom::original::{Builder, Bloom};
    ///
    /// let mut bloom: Bloom<u16, _> = Builder::new()
    ///         .bits(1024)
    ///         .false_positive_bound(0.1)
    ///         .total_hash_factors(4)
    ///         .freeze().unwrap();
    /// assert_eq!(bloom.capacity(), 212);
    /// assert_eq!(bloom.total_hashes(), 4);
    /// ```
    pub fn total_hashes(&self) -> usize {
        self.hash_factors.len()
    }

    /// Clear the filter
    ///
    /// This function resets all storage held by the filter, save the supplied
    /// (or computed) hash functions.
    ///
    /// ```
    /// use approximate::filters::bloom::original::{Builder, Bloom};
    ///
    /// let mut bloom: Bloom<u16, _> = Builder::new()
    ///         .bits(38)
    ///         .false_positive_bound(0.1)
    ///         .total_hash_factors(2)
    ///         .freeze().unwrap();
    ///
    /// assert_eq!(bloom.capacity(), 8);
    /// assert!(bloom.insert(&0001).is_ok());
    /// assert!(bloom.insert(&0010).is_ok());
    /// assert!(bloom.insert(&0011).is_ok());
    /// assert!(bloom.insert(&0010).is_ok());
    /// assert!(bloom.insert(&0110).is_ok());
    /// assert!(bloom.insert(&0111).is_ok());
    /// assert!(bloom.insert(&0101).is_ok());
    /// assert!(bloom.insert(&0100).is_ok());
    /// assert!(bloom.insert(&1100).is_err());
    /// assert!(bloom.insert(&1101).is_err());
    /// assert!(bloom.insert(&1111).is_err());
    /// assert!(bloom.insert(&1110).is_err());
    /// assert!(bloom.insert(&1010).is_err());
    /// assert!(bloom.insert(&1011).is_err());
    /// assert!(bloom.insert(&1001).is_err());
    /// assert!(bloom.insert(&1000).is_err());
    ///
    /// assert_eq!(bloom.len(), 16);
    /// bloom.clear();
    ///
    /// assert_eq!(bloom.len(), 0);
    /// assert!(!bloom.is_member(&1010));
    /// ```
    pub fn clear(&mut self) -> () {
        self.len = 0;
        self.data.clear();
        for _ in 0..self.capacity {
            self.data.push(false);
        }
    }

    #[cfg(test)]
    pub fn total_ones(&self) -> usize {
        self.data.count_one()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    // Here we demonstrate that we can never observe a false negative result
    // from is_member.
    #[test]
    pub fn prop_never_false_negatives() {
        fn inner(
            bits: usize,
            false_positive_bound: f64,
            entries: Vec<u16>,
            mut factors: Vec<u64>,
        ) -> TestResult {
            factors.sort();
            factors.dedup();
            if factors.is_empty() {
                return TestResult::discard();
            }
            if factors.iter().any(|x| is_even(*x)) {
                return TestResult::discard();
            }
            if bits == 0 || (false_positive_bound <= 0.0) || (false_positive_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Builder::new()
                .bits(bits)
                .false_positive_bound(false_positive_bound)
                .hash_factors(factors)
                .freeze();
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
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<u64>) -> TestResult);
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
            bits: usize,
            false_positive_bound: f64,
            entries: Vec<u16>,
            mut factors: Vec<u64>,
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

            if bits == 0 || (false_positive_bound <= 0.0) || (false_positive_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Builder::new()
                .bits(bits)
                .false_positive_bound(false_positive_bound)
                .hash_factors(factors)
                .freeze();
            if bloom.is_err() {
                return TestResult::discard();
            }
            let mut bloom = bloom.unwrap();
            let mut entries_inserted = 0;
            for entry in entries {
                bloom.insert_unchecked(&entry);
                entries_inserted += 1;
                assert!(bloom.total_hashes() <= entries_inserted * factors_len);
                assert!(bloom.total_hashes() >= 1);
            }

            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<u64>) -> TestResult);
    }

    // Here we determine that 'len' is always the same as the `entries.len`
    // after all insertions have completed.
    #[test]
    pub fn prop_len_equality() {
        fn inner(
            bits: usize,
            false_positive_bound: f64,
            entries: Vec<u16>,
            mut factors: Vec<u64>,
        ) -> TestResult {
            factors.sort();
            factors.dedup();
            if factors.is_empty() {
                return TestResult::discard();
            }
            if factors.iter().any(|x| is_even(*x)) {
                return TestResult::discard();
            }
            if bits == 0 || (false_positive_bound <= 0.0) || (false_positive_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Builder::new()
                .bits(bits)
                .false_positive_bound(false_positive_bound)
                .hash_factors(factors)
                .freeze();
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
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<u64>) -> TestResult);
    }

    // Here we demonstrate that an overfill only happens when more than
    // 'capacity' elements are inserted into the Bloom and the reverse.
    #[test]
    pub fn prop_overfill_per_capacity() {
        fn inner(
            bits: usize,
            false_positive_bound: f64,
            entries: Vec<u16>,
            mut factors: Vec<u64>,
        ) -> TestResult {
            factors.sort();
            factors.dedup();
            if factors.is_empty() {
                return TestResult::discard();
            }
            if factors.iter().any(|x| is_even(*x)) {
                return TestResult::discard();
            }
            if bits == 0 || (false_positive_bound <= 0.0) || (false_positive_bound >= 1.0) {
                return TestResult::discard();
            }
            let bloom = Builder::new()
                .bits(bits)
                .false_positive_bound(false_positive_bound)
                .hash_factors(factors)
                .freeze();
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
        QuickCheck::new().quickcheck(inner as fn(usize, f64, Vec<u16>, Vec<u64>) -> TestResult);
    }
}
