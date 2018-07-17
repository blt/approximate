//! # approximate - a crate of approximate algorithms
//!
//! This crate contains many approximate algorithms and data structures for
//! counting, set membership and the like. These structures use statistical
//! techniques to return approximate answers to questions: they may be wrong, but we
//! can qualify how wrong they are. Approximate structures save on space and time
//! compared to their totally accurate analogs.
//!
//! ## Ambitions
//!
//! The ambitions of this crate are to be a warehouse for approximation structures,
//! many of which use similar internal techniques. Any approximation is welcome. If
//! a broad category is not represented here that is only because I've not got
//! around to needing it for work or being interested enough to plug away
//! understanding the research to build something useful.
//!
//! The structures in this crate refer to their research and are property tested
//! according to whatever the inventors could prove about the structures.

extern crate bitvec;
extern crate rand;
extern crate twox_hash;

#[cfg(test)]
extern crate quickcheck;

#[deny(warnings)]
#[deny(bad_style)]
#[deny(future_incompatible)]
#[deny(nonstandard_style)]
#[deny(rust_2018_compatibility)]
#[deny(rust_2018_idioms)]
#[deny(unused)]
#[cfg_attr(feature = "cargo-clippy", deny(clippy))]
#[cfg_attr(feature = "cargo-clippy", deny(clippy_pedantic))]
#[cfg_attr(feature = "cargo-clippy", deny(clippy_perf))]
#[cfg_attr(feature = "cargo-clippy", deny(clippy_style))]
#[cfg_attr(feature = "cargo-clippy", deny(clippy_complexity))]
#[cfg_attr(feature = "cargo-clippy", deny(clippy_correctness))]
#[cfg_attr(feature = "cargo-clippy", deny(clippy_cargo))]
pub mod filters;
