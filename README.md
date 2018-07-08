# approximate - a crate of approximate algorithms

This crate contains many approximate algorithms and data structures for
counting, set membership and the like. These structures use statistical
techniques to return approximate answers to questions: they may be wrong, but we
can qualify how wrong they are. Approximate structures save on space and time
compared to their totally accurate analogs.

## Ambitions

The ambitions of this crate are to be a warehouse for approximation structures,
many of which use similar internal techniques. Any approximation is welcome. If
a broad category is not represented here that is only because I've not got
around to needing it for work or being interested enough to plug away
understanding the research to build something useful.

The structures in this crate refer to their research and are property tested
according to whatever the inventors could prove about the structures.
