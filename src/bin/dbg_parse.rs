//! Debug helper: parse a JSON graph file and print the exact codec error.
fn main() {
    let path = std::env::args().nth(1).expect("usage: dbg_parse <file.json>");
    let j = std::fs::read_to_string(&path).expect("read");
    match autorust::uop::parse_flat(&j) {
        Err(e) => {
            println!("parse_flat error: {e}");
            std::process::exit(1);
        }
        Ok(fg) => match autorust::uop::Graph::from_flat(&fg) {
            Err(e) => {
                println!("from_flat error: {e}");
                std::process::exit(2);
            }
            Ok(g) => {
                let out = autorust::uop::to_string_flat(&g.to_flat());
                println!("OK {} bytes out", out.len());
            }
        },
    }
}
