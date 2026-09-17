fn main() {
    for (name, pm) in [
        ("symbolic", autorust::uop::sym::symbolic().rules().iter().enumerate()),
    ] {
        let _ = name;
        for (i, r) in pm {
            let ops: Vec<String> = r.pat.op.iter().map(|v| v.iter().map(|o| o.pyname().rsplit('.').next().unwrap().to_lowercase()).collect::<Vec<_>>().join("|")).collect();
            println!("{:3} ops={} name={:?}", i, ops.join(","), r.pat.name.unwrap_or("-"));
        }
    }
}
