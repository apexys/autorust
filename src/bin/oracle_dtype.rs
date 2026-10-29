//! `cargo run --bin oracle_dtype` — prints Rust dtype-table answers for the
//! differential tests. Line protocol (tab separated):
//!   DT <attr>                       (one per dtype in the closed set)
//!   LU <a> <b> <least-upper>        (all ordered pairs over ALL + weak dtypes)
//!   CL <a> <b> <0|1>                (can_lossless_cast)
//!   MINMAX <attr> <min> <max>       (Python-repr-ish: ints decimal, inf/-inf/nan)
//!   TRUNC <attr> <value> <result>   (truncate table over a value sweep)

use autorust::dtype::*;

fn fmt_const(v: &ConstVal) -> String {
    match v {
        ConstVal::Bool(b) => format!("bool:{}", if *b { "True" } else { "False" }),
        ConstVal::Int(i) => format!("int:{}", i),
        ConstVal::Big(b) => format!("int:{}", b.to_decimal()),
        ConstVal::Float(f) => {
            if f.is_nan() {
                "float:nan".into()
            } else if f.is_infinite() {
                format!("float:{}", if *f > 0.0 { "inf" } else { "-inf" })
            } else {
                format!("float:{}", f)
            }
        }
        ConstVal::Invalid => "invalid".into(),
    }
}

fn main() {
    // closed set for the differential: dtypes.all + weakint/weakfloat
    let mut all: Vec<DTypeId> = ALL.to_vec();
    all.push(WEAKINT);
    all.push(WEAKFLOAT);

    for &d in &all {
        println!("DT\t{}", d.attr());
    }

    for &a in &all {
        for &b in &all {
            let lu = least_upper_dtype(&[a, b]);
            println!("LU\t{}\t{}\t{}", a.attr(), b.attr(), lu.attr());
            println!(
                "CL\t{}\t{}\t{}",
                a.attr(),
                b.attr(),
                if can_lossless_cast(a, b) { 1 } else { 0 }
            );
        }
    }

    for &d in &all {
        let (mn, mx) = (dtype_min(d), dtype_max(d));
        println!(
            "MINMAX\t{}\t{}\t{}",
            d.attr(),
            fmt_const(&mn),
            fmt_const(&mx)
        );
    }

    let vals: [f64; 15] = [
        0.0,
        -0.0,
        1.5,
        2.5,
        -3.75,
        300.0,
        -300.0,
        1e9,
        1e30,
        -1e30,
        65519.0,
        1e-8,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for &d in &all {
        // skip dtypes without a truncate entry (mirrors Python's truncate dict keys)
        if !(in_set(FLOATS, d) || in_set(INTS, d) || is_bool(d)) {
            continue;
        }
        for &v in &vals {
            let res = truncate(d, v);
            println!("TRUNC\t{}\t{}\t{}", d.attr(), fmt_val(v), fmt_const(&res));
        }
    }
}

fn fmt_val(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v.is_infinite() {
        if v > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        format!("{}", v)
    }
}
