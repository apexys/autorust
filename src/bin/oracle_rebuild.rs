//! `cargo run --bin oracle_rebuild [file]` — reads a JSON graph (file or stdin),
//! rebuilds it through the interned graph, writes the canonical JSON back to
//! stdout. Pure round-trip oracle for the M0 gate.

use std::io::{Read, Write};

fn main() {
    let mut input = String::new();
    match std::env::args().nth(1) {
        Some(path) => {
            input = std::fs::read_to_string(&path).expect("failed to read graph file");
        }
        None => {
            std::io::stdin()
                .read_to_string(&mut input)
                .expect("failed to read stdin");
        }
    }
    match autorust::rebuild(&input) {
        Ok(out) => {
            print!("{out}");
            std::io::stdout().flush().unwrap();
        }
        Err(e) => {
            eprintln!("rebuild failed: {e}");
            std::process::exit(1);
        }
    }
}
