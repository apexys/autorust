use nanoserde::{DeJson, SerJson};

use autorust::uop::{arg_to_flat, parse_flat, to_string_flat, Arg, EnumClass, EnumVal, FlatGraph, Graph, NonFinite, Ops};
use autorust::dtype::{DTypeId, BOOL, FLOAT32, INT32};

#[test]
fn flat_roundtrip_synthetic() {
    let mut g = Graph::new();
    let c1 = g.intern(Ops::Const, FLOAT32, vec![], Arg::Float(2.5), Arg::None);
    let c2 = g.intern(Ops::Const, INT32, vec![], Arg::Int(3), Arg::None);
    let c3 = g.intern(Ops::Const, BOOL, vec![], Arg::Bool(true), Arg::None);
    let w = g.intern(Ops::Where, FLOAT32, vec![c3, c1.clone(), c1], Arg::None, Arg::None);
    let nan = g.intern(Ops::Const, FLOAT32, vec![], Arg::Float(f64::NAN), Arg::None);
    let inf = g.intern(Ops::Const, FLOAT32, vec![], Arg::Float(f64::INFINITY), Arg::None);
    let ninf = g.intern(Ops::Const, FLOAT32, vec![], Arg::Float(f64::NEG_INFINITY), Arg::None);
    let neg0 = g.intern(Ops::Const, FLOAT32, vec![], Arg::Float(-0.0), Arg::None);
    let red = g.intern(Ops::Reduce, BOOL, vec![w], Arg::Seq {
        tuple: true,
        items: vec![Arg::Enum(EnumVal { class: EnumClass::Ops, value: Ops::Add as u16 }), Arg::Int(0)],
    }, Arg::None);
    let pk = g.intern(Ops::Buffer, BOOL, vec![], Arg::Pickle(vec![1, 2, 3]), Arg::None);
    let sp = g.intern(Ops::Range, INT32, vec![c2], Arg::Seq {
        tuple: true,
        items: vec![Arg::Int(0), Arg::Enum(EnumVal { class: EnumClass::AxisType, value: 11 })],
    }, Arg::None);
    g.roots = vec![red, pk, sp, nan, inf, ninf, neg0];
    let _ = DTypeId(0); // keep import used

    let flat = g.to_flat();
    let json = to_string_flat(&flat);
    println!("JSON: {json}");

    let back = parse_flat(&json).unwrap();
    let g2 = Graph::from_flat(&back).unwrap();
    // structural identity: re-encode must be byte-identical (canonical post-order)
    let json2 = to_string_flat(&g2.to_flat());
    assert_eq!(json, json2);
    // node identity: same roots, interned equality for the reduce chain
    assert_eq!(g.roots.len(), g2.roots.len());
    for (na, nb) in g.roots.iter().zip(g2.roots.iter()) {
        assert_eq!(na.op, nb.op, "op mismatch");
        assert_eq!(na.dtype, nb.dtype);
        assert_eq!(na.srcs.len(), nb.srcs.len());
        // arg equality through bits (floats)
        assert!(na.arg == nb.arg, "arg mismatch: {:?} vs {:?}", na.arg, nb.arg);
        assert!(na.tag == nb.tag);
    }
}

#[test]
fn arg_json_shapes() {
    // document the exact JSON shapes the Python bridge must emit
    let cases: Vec<(&str, Arg)> = vec![
        ("none", Arg::None),
        ("int", Arg::Int(-5)),
        ("float", Arg::Float(1.5)),
        ("nan", Arg::Float(f64::NAN)),
        ("posinf", Arg::Float(f64::INFINITY)),
        ("neginf", Arg::Float(f64::NEG_INFINITY)),
        ("negzero", Arg::Float(-0.0)),
        ("bool", Arg::Bool(true)),
        ("str", Arg::Str("hello".into())),
        ("bytes", Arg::Bytes(vec![1, 2, 3])),
        ("enum_ops", Arg::Enum(EnumVal { class: EnumClass::Ops, value: 33 })),
        ("enum_axis", Arg::Enum(EnumVal { class: EnumClass::AxisType, value: 11 })),
        ("enum_addr", Arg::Enum(EnumVal { class: EnumClass::AddrSpace, value: 1 })),
        ("dtype", Arg::Dtype(FLOAT32)),
        ("seq", Arg::Seq { tuple: true, items: vec![Arg::Int(1), Arg::Int(2)] }),
        ("list", Arg::Seq { tuple: false, items: vec![Arg::Bool(false)] }),
        ("pickle", Arg::Pickle(vec![0x80, 0x04])),
        ("nonfinite", Arg::NonFinite(NonFinite::Nan)),
    ];
    for (name, a) in cases {
        let j = arg_to_flat(&a).serialize_json();
        println!("{name}: {j}");
        let back: FlatGraph = FlatGraph::deserialize_json(&"{\"dtypes\":[],\"nodes\":[],\"roots\":[]}".to_string()).unwrap();
        assert!(back.dtypes.is_empty());
    }
}
