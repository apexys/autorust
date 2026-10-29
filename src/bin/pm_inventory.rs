//! Prints the codegen pipeline `PatternMatcher` inventory (one canonical
//! signature per rule) as machine-readable lines, for diffing against tinygrad
//! (`py/pm_inventory.py`).
//!
//! Format:
//!   PM `<name>` `<rule_count>`
//!   R `<index>` `<signature>`
fn main() {
    for (name, sigs) in autorust::codegen::pipeline::pm_inventory() {
        println!("PM {} {}", name, sigs.len());
        for (i, s) in sigs.iter().enumerate() {
            println!("R {} {}", i, s);
        }
    }
}
