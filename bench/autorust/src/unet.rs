//! A small UNet (64×64, four levels) for op coverage the CNN/ViT rows miss:
//! 3x3 convolutions at 64², GroupNorm, max-pool downsampling, transposed
//! convolution, nearest-neighbour upsample and skip `cat`.

use autorust::tensor::nn::{self, Conv2d, ConvTranspose2d, GroupNorm};
use autorust::tensor::{Ctx, Tensor};

struct Dc {
    c1: Conv2d,
    g1: GroupNorm,
    c2: Conv2d,
    g2: GroupNorm,
}

impl Dc {
    fn new(ctx: &Ctx, cin: i64, cout: i64) -> Dc {
        let g = cout.min(8);
        Dc {
            c1: Conv2d::new(ctx, cin, cout, &[3, 3], 1, &[(1, 1), (1, 1)], 1, 1, true),
            g1: GroupNorm::new(ctx, g, cout, 1e-5, true),
            c2: Conv2d::new(ctx, cout, cout, &[3, 3], 1, &[(1, 1), (1, 1)], 1, 1, true),
            g2: GroupNorm::new(ctx, g, cout, 1e-5, true),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let y = nn::relu(&self.g1.forward(&self.c1.forward(x)));
        nn::relu(&self.g2.forward(&self.c2.forward(&y)))
    }
}

fn pool(x: &Tensor) -> Tensor {
    x.max_pool2d(&[2, 2], None, &[1, 1], &[0], false, false).0
}

fn up(x: &Tensor, size: i64) -> Tensor {
    x.interpolate(&[size, size], "nearest", false)
}

/// The UNet (3→32→64→128→256, back to 3 channels).
pub struct Unet {
    e1: Dc,
    e2: Dc,
    e3: Dc,
    bottom: Dc,
    up3: ConvTranspose2d,
    d3: Dc,
    d2: Dc,
    d1: Dc,
    out: Conv2d,
}

impl Unet {
    pub fn new(ctx: &Ctx) -> Unet {
        Unet {
            e1: Dc::new(ctx, 3, 32),
            e2: Dc::new(ctx, 32, 64),
            e3: Dc::new(ctx, 64, 128),
            bottom: Dc::new(ctx, 128, 256),
            up3: ConvTranspose2d::new(ctx, 256, 128, &[2, 2], 2, &[(0, 0), (0, 0)], 0, 1, 1, true),
            d3: Dc::new(ctx, 256, 128),
            d2: Dc::new(ctx, 192, 64),
            d1: Dc::new(ctx, 96, 32),
            out: Conv2d::new(ctx, 32, 3, &[1, 1], 1, &[(0, 0), (0, 0)], 1, 1, true),
        }
    }

    /// x: `[B, 3, 64, 64]` → `[B, 3, 64, 64]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let e1 = self.e1.forward(x);
        let e2 = self.e2.forward(&pool(&e1));
        let e3 = self.e3.forward(&pool(&e2));
        let b = self.bottom.forward(&pool(&e3));
        let u3 = self.up3.forward(&b);
        let d3 = self.d3.forward(&u3.cat(&[e3], 1));
        let d2 = self.d2.forward(&up(&d3, 32).cat(&[e2], 1));
        let d1 = self.d1.forward(&up(&d2, 64).cat(&[e1], 1));
        self.out.forward(&d1)
    }
}
