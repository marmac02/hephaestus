Hephaestus for Rust
===================

This fork contains Hephaestus extended for Rust support.

To run Hephaestus use the command:
python3 hephaestus.py --language rust --transformations 0 --iterations [N] -P

Branch rust-support contains the generator without capturing closure (Fn FnMut, FnOnce) support.

Branch capturing-closures contains the generator with capturing clousres, is functional and can be run.
However, it still contains redundant code (it is quite messy).
Note that when Hephaestus is run for more iterations, it is highly likely that the compiler bug rustc#127525 will occur 
(usually manifesting in "if-else have incompatible types" error message, but could be others for sligtly different cases, so this
has to be checked on a case-by-case basis).

The two branches will be eventually merged, potentially a command line option to disable capturing closures will be added.