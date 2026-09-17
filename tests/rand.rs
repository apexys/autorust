//! RNG golden tests: the Threefry graph `rand`/`uniform`/`kaiming_uniform` must
//! reproduce tinygrad's `Tensor.manual_seed(...)` outputs bit-for-bit (within
//! f32 rounding). Goldens captured from the reference `tinygrad/` tree.

use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

fn close(a: &[f32], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "len mismatch: {a:?} vs {b:?}");
    for (&x, &y) in a.iter().zip(b) {
        let y = y as f32;
        assert!((x - y).abs() <= 1e-6, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn rand_sequence_matches_tinygrad() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(42);
    let r4 = Tensor::rand(&mut ctx, &[4], None, None).to_list_fresh::<f32>(&mut ctx);
    let r23 = Tensor::rand(&mut ctx, &[2, 3], None, None).to_list_fresh::<f32>(&mut ctx);
    let r5 = Tensor::rand(&mut ctx, &[5], None, None).to_list_fresh::<f32>(&mut ctx);
    close(&r4, &[0.5334206819534302, 0.6701551675796509, 0.7630789279937744, 0.4320552349090576]);
    close(&r23, &[0.28115880489349365, 0.2776648998260498, 0.4473385810852051, 0.33798515796661377, 0.8195433616638184, 0.24290215969085693]);
    close(&r5, &[0.7045223712921143, 0.9117947816848755, 0.08823657035827637, 0.0228884220123291, 0.03821361064910889]);
}

#[test]
fn rand_counter_continuity() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(99);
    let a = Tensor::rand(&mut ctx, &[3], None, None).to_list_fresh::<f32>(&mut ctx);
    let b = Tensor::rand(&mut ctx, &[3], None, None).to_list_fresh::<f32>(&mut ctx);
    close(&a, &[0.9903250932693481, 0.132806658744812, 0.07761192321777344]);
    close(&b, &[0.2560626268386841, 0.45948076248168945, 0.7808310985565186]);
}

#[test]
fn uniform_matches_tinygrad() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(123);
    let u = Tensor::uniform(&mut ctx, &[5], Some(-2.0), Some(3.0), None, None).to_list_fresh::<f32>(&mut ctx);
    close(&u, &[2.338348388671875, -1.274613857269287, -0.8167816400527954, -0.5347211360931396, -1.291520118713379]);
}

#[test]
fn kaiming_uniform_matches_tinygrad() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(7);
    let k = Tensor::kaiming_uniform(&mut ctx, &[4, 8], 0.01, None).to_list_fresh::<f32>(&mut ctx);
    let exp: Vec<f64> = vec![
        -0.7744928002357483, 0.584154486656189, -0.27595221996307373, 0.27652373909950256, -0.244951531291008, 0.06952708959579468, -0.13325102627277374, -0.8336478471755981,
        0.8555353283882141, -0.3269983232021332, -0.03378861024975777, -0.4892677664756775, 0.8458175659179688, 0.8266645669937134, -0.7752709984779358, -0.8003231883049011,
    ];
    close(&k[..16], &exp);
}

fn close_tol(a: &[f32], b: &[f64], tol: f32) {
    assert_eq!(a.len(), b.len(), "len mismatch: {a:?} vs {b:?}");
    for (&x, &y) in a.iter().zip(b) {
        let y = y as f32;
        assert!((x - y).abs() <= tol, "value mismatch: {x} vs {y}");
    }
}

#[test]
fn randn_matches_tinygrad() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(42);
    let n23 = Tensor::randn(&mut ctx, &[2, 3], None, None).to_list_fresh::<f32>(&mut ctx);
    close_tol(&n23, &[1.95764, -0.18594, 1.64042, -0.76466, -0.86949, -0.43788], 1e-4);
    let n6 = Tensor::randn(&mut ctx, &[6], None, None).to_list_fresh::<f32>(&mut ctx);
    close_tol(&n6, &[-0.73821, 1.51643, -0.30645, -0.78621, 0.54107, -0.43943], 1e-4);
    Tensor::manual_seed(100);
    let n4 = Tensor::randn(&mut ctx, &[4], None, None).to_list_fresh::<f32>(&mut ctx);
    close_tol(&n4, &[1.28954, -0.69803, 0.38132, -1.55418], 1e-4);
}

#[test]
fn randn_counter_continuity() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(42);
    let a = Tensor::randn(&mut ctx, &[3], None, None).to_list_fresh::<f32>(&mut ctx);
    let b = Tensor::randn(&mut ctx, &[3], None, None).to_list_fresh::<f32>(&mut ctx);
    close_tol(&a, &[-0.36704, 1.18429, 0.73243], 1e-4);
    close_tol(&b, &[0.04003, -0.7341, 1.50839], 1e-4);
}

#[test]
fn rand_then_randn_continuity() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(42);
    let r = Tensor::rand(&mut ctx, &[3], None, None).to_list_fresh::<f32>(&mut ctx);
    let n = Tensor::randn(&mut ctx, &[3], None, None).to_list_fresh::<f32>(&mut ctx);
    close(&r, &[0.5334206819534302, 0.6701551675796509, 0.7630789279937744]);
    close_tol(&n, &[0.39645, 0.65455, 1.61238], 1e-4);
}

#[test]
fn randint_matches_tinygrad() {
    let mut ctx = UOpCtx::new();
    Tensor::manual_seed(42);
    let ri = Tensor::randint(&mut ctx, &[4], -5, 5, None, None).to_list_fresh::<i32>(&mut ctx);
    assert_eq!(ri, vec![0, 1, 2, -1]);
    Tensor::manual_seed(42);
    let ri2 = Tensor::randint(&mut ctx, &[2, 3], 0, 10, None, None).to_list_fresh::<i32>(&mut ctx);
    assert_eq!(ri2, vec![3, 0, 1, 1, 5, 3]);
    Tensor::manual_seed(42);
    let a = Tensor::randint(&mut ctx, &[3], 0, 100, None, None).to_list_fresh::<i32>(&mut ctx);
    let b = Tensor::randint(&mut ctx, &[3], 0, 100, None, None).to_list_fresh::<i32>(&mut ctx);
    assert_eq!(a, vec![53, 67, 76]);
    assert_eq!(b, vec![85, 16, 77]);
}
