//! Signal-processing Tensor ops: DFT/FFT, rfft, STFT and window functions.
//!
//! References are host f64 DFT evaluations; f64 tensors keep the tolerance
//! tight, and a separate f32 case covers the compute-in-f32 path.

use autorust::dtype;
use autorust::engine::session::Session;
use autorust::tensor::signal::{blackman_window, hamming_window, hann_window};
use autorust::tensor::{Ctx, Tensor};

fn tensor(ctx: &Ctx, vals: &[f64], shape: &[i64]) -> Tensor {
    Tensor::from_slice(ctx, vals, Some(dtype::FLOAT64), None).reshape(shape.to_vec())
}

fn complex(re: &Tensor, im: &Tensor) -> Tensor {
    re.unsqueeze(-1).cat(&[im.unsqueeze(-1)], -1)
}

fn read(t: &Tensor, sess: &mut Session) -> Vec<f64> {
    t.to_list::<f64>(sess)
}

fn naive_dft(re: &[f64], im: &[f64], inverse: bool) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    assert_eq!(im.len(), n);
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut out_r = vec![0.0; n];
    let mut out_i = vec![0.0; n];
    for k in 0..n {
        for t in 0..n {
            let a = sign * std::f64::consts::TAU * (k as f64) * (t as f64) / (n as f64);
            let (s, c) = a.sin_cos();
            out_r[k] += re[t] * c - im[t] * s;
            out_i[k] += re[t] * s + im[t] * c;
        }
        if inverse {
            out_r[k] /= n as f64;
            out_i[k] /= n as f64;
        }
    }
    (out_r, out_i)
}

fn close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "[{i}] {a} != {b} (diff {})",
            (a - b).abs()
        );
    }
}

#[test]
fn dft_and_fft_match_naive() {
    let ctx = Ctx::new();
    let mut sess = Session::new();

    for n in [6i64, 8] {
        let re: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let im = vec![0.0; n as usize];
        let x = tensor(&ctx, &re, &[n]);
        let y = if (n as u64).is_power_of_two() {
            x.fft(0, false)
        } else {
            x.dft(0, false)
        };
        let vals = read(&y, &mut sess);
        let (wr, wi) = naive_dft(&re, &im, false);
        let mut want = Vec::new();
        for k in 0..n as usize {
            want.push(wr[k]);
            want.push(wi[k]);
        }
        close(&vals, &want, 1e-9);
    }

    // Batched transform along the last axis.
    let re = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.5];
    let x = tensor(&ctx, &re, &[2, 4]);
    let y = x.fft(1, false);
    let vals = read(&y, &mut sess);
    let mut want = Vec::new();
    for row in 0..2 {
        let (wr, wi) = naive_dft(&re[row * 4..row * 4 + 4], &[0.0; 4], false);
        for k in 0..4 {
            want.push(wr[k]);
            want.push(wi[k]);
        }
    }
    close(&vals, &want, 1e-9);
}

#[test]
fn dft_along_leading_axis() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    // Shape [3, 2]; transform the 3 samples of each column.
    let re = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = tensor(&ctx, &re, &[3, 2]);
    let y = x.dft(0, false);
    let vals = read(&y, &mut sess);
    let mut want = Vec::new();
    for col in 0..2 {
        let column: Vec<f64> = (0..3).map(|r| re[r * 2 + col]).collect();
        let (wr, wi) = naive_dft(&column, &[0.0; 3], false);
        for k in 0..3 {
            want.push(wr[k]);
            want.push(wi[k]);
        }
    }
    // The output layout is [3, 2, 2]: sample, column, complex.
    let mut ordered = Vec::with_capacity(vals.len());
    for k in 0..3 {
        for col in 0..2 {
            ordered.push(vals[(k * 2 + col) * 2]);
            ordered.push(vals[(k * 2 + col) * 2 + 1]);
        }
    }
    let mut want_ordered = Vec::with_capacity(want.len());
    for k in 0..3 {
        for col in 0..2 {
            want_ordered.push(want[col * 6 + k * 2]);
            want_ordered.push(want[col * 6 + k * 2 + 1]);
        }
    }
    close(&ordered, &want_ordered, 1e-9);
}

#[test]
fn cfft_inverse_round_trip() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let n = 8i64;
    let re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let im: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
    let z = complex(&tensor(&ctx, &re, &[n]), &tensor(&ctx, &im, &[n]));
    let forward = z.cfft(0, false);
    let back = forward.cfft(0, true);
    let vals = read(&back, &mut sess);
    let mut want = Vec::with_capacity(2 * n as usize);
    for i in 0..n as usize {
        want.push(re[i]);
        want.push(im[i]);
    }
    close(&vals, &want, 1e-9);
}

#[test]
fn cdft_complex_matches_naive() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let n = 5i64;
    let re = vec![1.0, -2.0, 0.5, 3.0, 0.0];
    let im = vec![0.0, 1.5, -0.5, 2.0, -1.0];
    let z = complex(&tensor(&ctx, &re, &[n]), &tensor(&ctx, &im, &[n]));
    let y = z.cdft(0, false);
    let vals = read(&y, &mut sess);
    let (wr, wi) = naive_dft(&re, &im, false);
    let mut want = Vec::new();
    for k in 0..n as usize {
        want.push(wr[k]);
        want.push(wi[k]);
    }
    close(&vals, &want, 1e-9);
}

#[test]
fn rfft_is_onesided_dft() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let n = 8i64;
    let re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.4).sin() + 1.0).collect();
    let y = tensor(&ctx, &re, &[n]).rfft(0);
    let vals = read(&y, &mut sess);
    let (wr, wi) = naive_dft(&re, &[0.0; 8], false);
    let bins = (n / 2 + 1) as usize;
    let mut want = Vec::new();
    for k in 0..bins {
        want.push(wr[k]);
        want.push(wi[k]);
    }
    close(&vals, &want, 1e-9);
}

#[test]
fn stft_matches_frame_dfts() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let length = 10i64;
    let frame = 3i64; // non-power-of-two: exercises the DFT fallback
    let step = 3i64;
    let signal: Vec<f64> = (0..length)
        .map(|i| (i as f64 * 0.5).cos() + 0.2 * i as f64)
        .collect();
    let x = tensor(&ctx, &signal, &[1, length]);
    let y = x.stft(frame, step, None, true);
    let vals = read(&y, &mut sess);
    let frames = 1 + (length - frame) / step;
    let bins = frame / 2 + 1;
    assert_eq!(vals.len(), (frames * bins * 2) as usize);
    for f in 0..frames {
        let start = (f * step) as usize;
        let slice = &signal[start..start + frame as usize];
        let (wr, wi) = naive_dft(slice, &vec![0.0; frame as usize], false);
        for k in 0..bins as usize {
            let idx = ((f * bins + k as i64) * 2) as usize;
            assert!(
                (vals[idx] - wr[k]).abs() <= 1e-8,
                "frame {f} bin {k} real {} != {}",
                vals[idx],
                wr[k]
            );
            assert!(
                (vals[idx + 1] - wi[k]).abs() <= 1e-8,
                "frame {f} bin {k} imag {} != {}",
                vals[idx + 1],
                wi[k]
            );
        }
    }
}

#[test]
fn stft_full_spectrum_non_power_of_two() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let signal = vec![1.0, -1.0, 2.0, 0.5, -0.5, 1.5];
    let x = tensor(&ctx, &signal, &[1, 6]);
    let y = x.stft(3, 3, None, false);
    let vals = read(&y, &mut sess);
    let frames = 2i64;
    let bins = 3i64;
    assert_eq!(vals.len(), (frames * bins * 2) as usize);
    for f in 0..frames {
        let start = (f * 3) as usize;
        let slice = &signal[start..start + 3];
        let (wr, wi) = naive_dft(slice, &[0.0; 3], false);
        for k in 0..bins as usize {
            let idx = ((f * bins + k as i64) * 2) as usize;
            assert!((vals[idx] - wr[k]).abs() < 1e-9);
            assert!((vals[idx + 1] - wi[k]).abs() < 1e-9);
        }
    }
}

#[test]
fn windows_match_formulas() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let n = 8i64;
    let hann = hann_window(&ctx, n, true, Some(dtype::FLOAT64));
    let hamming = hamming_window(&ctx, n, true, Some(dtype::FLOAT64));
    let blackman = blackman_window(&ctx, n, true, Some(dtype::FLOAT64));
    let hann_s = hann_window(&ctx, n, false, Some(dtype::FLOAT64));
    let vals_hann = read(&hann, &mut sess);
    let vals_hamming = read(&hamming, &mut sess);
    let vals_blackman = read(&blackman, &mut sess);
    let vals_hann_s = read(&hann_s, &mut sess);
    for i in 0..n as usize {
        let p = std::f64::consts::TAU * i as f64 / n as f64;
        let s = std::f64::consts::TAU * i as f64 / (n - 1) as f64;
        assert!((vals_hann[i] - (0.5 - 0.5 * p.cos())).abs() < 1e-12);
        assert!((vals_hamming[i] - (0.54 - 0.46 * p.cos())).abs() < 1e-12);
        assert!((vals_blackman[i] - (0.42 - 0.5 * p.cos() + 0.08 * (2.0 * p).cos())).abs() < 1e-12);
        assert!((vals_hann_s[i] - (0.5 - 0.5 * s.cos())).abs() < 1e-12);
    }
}

#[test]
fn f32_path_matches_f64() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let n = 16i64;
    let re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let re32: Vec<f32> = re.iter().map(|v| *v as f32).collect();
    let x = Tensor::from_slice(&ctx, &re32, Some(dtype::FLOAT32), None).reshape([n]);
    let y = x.fft(0, false);
    let got = y.to_list::<f32>(&mut sess);
    let (wr, wi) = naive_dft(&re, &[0.0; 16], false);
    let mut want = Vec::new();
    for k in 0..n as usize {
        want.push(wr[k] as f32);
        want.push(wi[k] as f32);
    }
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "[{i}] {a} != {b} (diff {})",
            (a - b).abs()
        );
    }
}

#[test]
#[should_panic(expected = "power-of-two")]
fn fft_rejects_non_power_of_two() {
    let ctx = Ctx::new();
    let x = tensor(&ctx, &[1.0, 2.0, 3.0], &[3]);
    let _ = x.fft(0, false);
}
