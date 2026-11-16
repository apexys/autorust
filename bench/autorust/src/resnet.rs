//! ResNet-18/50 and MobileNetV2 at 224², the conv-heavy counterparts to the
//! MNIST CNN: real channel counts, strided 7x7/3x3 convolutions, residual
//! adds, and (MobileNet) depthwise/grouped conv with no GEMM lowering.

use autorust::engine::session::Session;
use autorust::tensor::nn::{self, BatchNorm, Conv2d, Linear, Module};
use autorust::tensor::{Ctx, Tensor};

fn prefixed<'a>(
    prefix: &str,
    it: impl IntoIterator<Item = (String, Tensor)> + 'a,
) -> Vec<(String, Tensor)> {
    it.into_iter()
        .map(|(n, t)| (format!("{prefix}.{n}"), t))
        .collect()
}

/// Conv2d + BatchNorm (+ ReLU at the call site).
struct Cb {
    conv: Conv2d,
    bn: BatchNorm,
    /// BN folded into `conv` for inference (see [`Cb::fold`]).
    folded: bool,
    /// Allow the fused tensor-core 3x3 path. A bottleneck's 3x3 sits between
    /// two 1x1 convs, so its input is never realized and taking the host path
    /// forces a mid-graph realize that fragments the schedule (measured 13x
    /// slower on ResNet-50). Basic blocks chain 3x3 -> 3x3 and keep it on.
    fused: bool,
}

impl Cb {
    fn new(ctx: &Ctx, cin: i64, cout: i64, k: i64, stride: i64, pad: i64, groups: i64) -> Cb {
        Cb {
            conv: Conv2d::new(
                ctx,
                cin,
                cout,
                &[k, k],
                stride,
                &[(pad, pad), (pad, pad)],
                1,
                groups,
                false,
            ),
            bn: BatchNorm::new(ctx, cout, 1e-5, true, true, 0.1),
            folded: false,
            fused: true,
        }
    }

    /// Fold the eval-mode BN into the conv (inference only) so the conv store
    /// is a plain bias(+ReLU) GEMM the host GEMM can accept.
    fn fold(&mut self, sess: &mut Session) -> bool {
        self.folded = nn::fold_bn_eval(&mut self.conv, &self.bn, sess);
        self.folded
    }

    fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        let tc3x3 = !training
            && self.fused
            && self.conv.groups == 1
            && self.conv.kernel_size == [3, 3]
            && self.conv.stride == 1
            && self.conv.dilation == 1
            && self.conv.padding == [1, 1, 1, 1]
            && match std::env::var("AUTORUST_CONV3X3_TC") {
                Ok(v) => v != "0" && !v.is_empty(),
                Err(_) => {
                    x.ctx().borrow().settings.math_mode == autorust::settings::MathMode::Fast
                }
            };
        let y = if tc3x3 {
            {
                let mut c = x.ctx().borrow_mut();
                x.realize_device_ctx(&mut c, sess);
            }
            self.conv.forward_with(x, sess)
        } else {
            self.conv.forward(x)
        };
        if self.folded && !training {
            return y;
        }
        self.bn.forward(&y, sess, training)
    }

    fn named_params(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("conv", self.conv.named_params());
        v.extend(prefixed("bn", self.bn.named_params()));
        v
    }

    fn into_params(mut self) -> Cb {
        self.conv = self.conv.into_params();
        self.bn = self.bn.into_params();
        self
    }
}

/// ResNet basic block (18/34).
struct Basic {
    c1: Cb,
    c2: Cb,
    down: Option<Cb>,
}

impl Basic {
    fn new(ctx: &Ctx, cin: i64, cout: i64, stride: i64) -> Basic {
        let down = (stride != 1 || cin != cout).then(|| Cb::new(ctx, cin, cout, 1, stride, 0, 1));
        Basic {
            c1: Cb::new(ctx, cin, cout, 3, stride, 1, 1),
            c2: Cb::new(ctx, cout, cout, 3, 1, 1, 1),
            down,
        }
    }

    fn fold_bn(&mut self, sess: &mut Session) {
        self.c1.fold(sess);
        self.c2.fold(sess);
        if let Some(d) = &mut self.down {
            d.fold(sess);
        }
    }

    fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        let id = match &self.down {
            Some(d) => d.forward(x, sess, training),
            None => x.clone(),
        };
        let y = nn::relu(&self.c1.forward(x, sess, training));
        let y = self.c2.forward(&y, sess, training);
        nn::relu(&y.add(&id))
    }

    fn named_params(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("c1", self.c1.named_params());
        v.extend(prefixed("c2", self.c2.named_params()));
        if let Some(d) = &self.down {
            v.extend(prefixed("down", d.named_params()));
        }
        v
    }

    fn into_params(self) -> Basic {
        Basic {
            c1: self.c1.into_params(),
            c2: self.c2.into_params(),
            down: self.down.map(Cb::into_params),
        }
    }
}

/// ResNet bottleneck (50/101/152): 1x1 -> 3x3 -> 1x1, expansion 4.
struct Bottleneck {
    c1: Cb,
    c2: Cb,
    c3: Cb,
    down: Option<Cb>,
}

impl Bottleneck {
    fn new(ctx: &Ctx, cin: i64, cout: i64, stride: i64) -> Bottleneck {
        let mid = cout / 4;
        let down = (stride != 1 || cin != cout).then(|| Cb::new(ctx, cin, cout, 1, stride, 0, 1));
        // Only c2 is 3x3, and its input is the 1x1 c1 output: the host path
        // would force a realize there and regress the whole model.
        let mut c2 = Cb::new(ctx, mid, mid, 3, stride, 1, 1);
        c2.fused = false;
        Bottleneck {
            c1: Cb::new(ctx, cin, mid, 1, 1, 0, 1),
            c2,
            c3: Cb::new(ctx, mid, cout, 1, 1, 0, 1),
            down,
        }
    }

    fn fold_bn(&mut self, sess: &mut Session) {
        self.c1.fold(sess);
        self.c2.fold(sess);
        self.c3.fold(sess);
        if let Some(d) = &mut self.down {
            d.fold(sess);
        }
    }

    fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        let id = match &self.down {
            Some(d) => d.forward(x, sess, training),
            None => x.clone(),
        };
        let y = nn::relu(&self.c1.forward(x, sess, training));
        let y = nn::relu(&self.c2.forward(&y, sess, training));
        let y = self.c3.forward(&y, sess, training);
        nn::relu(&y.add(&id))
    }

    fn named_params(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("c1", self.c1.named_params());
        v.extend(prefixed("c2", self.c2.named_params()));
        v.extend(prefixed("c3", self.c3.named_params()));
        if let Some(d) = &self.down {
            v.extend(prefixed("down", d.named_params()));
        }
        v
    }

    fn into_params(self) -> Bottleneck {
        Bottleneck {
            c1: self.c1.into_params(),
            c2: self.c2.into_params(),
            c3: self.c3.into_params(),
            down: self.down.map(Cb::into_params),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum Block {
    Basic(Basic),
    Bottleneck(Bottleneck),
}

impl Block {
    fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        match self {
            Block::Basic(b) => b.forward(x, sess, training),
            Block::Bottleneck(b) => b.forward(x, sess, training),
        }
    }
    fn fold_bn(&mut self, sess: &mut Session) {
        match self {
            Block::Basic(b) => b.fold_bn(sess),
            Block::Bottleneck(b) => b.fold_bn(sess),
        }
    }
    fn named_params(&self) -> Vec<(String, Tensor)> {
        match self {
            Block::Basic(b) => b.named_params(),
            Block::Bottleneck(b) => b.named_params(),
        }
    }
    fn into_params(self) -> Block {
        match self {
            Block::Basic(b) => Block::Basic(b.into_params()),
            Block::Bottleneck(b) => Block::Bottleneck(b.into_params()),
        }
    }
}

/// `torchvision.models.resnet18` / `resnet50` (1000 classes).
pub struct ResNet {
    stem: Cb,
    stages: Vec<Vec<Block>>,
    fc: Linear,
}

impl ResNet {
    /// `depth` is 18 or 50; `width` scales every channel count (1.0 =
    /// torchvision, 0.25 = the small ONNX asset).
    pub fn new(ctx: &Ctx, depth: usize, classes: i64, width: f64) -> ResNet {
        let w = |c: i64| -> i64 { (((c as f64) * width) as i64 / 8 * 8).max(8) };
        let (layers, bottleneck) = match depth {
            18 => (vec![2usize, 2, 2, 2], false),
            50 => (vec![3usize, 4, 6, 3], true),
            other => panic!("ResNet depth {other} not benchmarked"),
        };
        let stem = Cb::new(ctx, 3, w(64), 7, 2, 3, 1);
        let mut stages = Vec::new();
        let mut cin = w(64);
        for (i, &n) in layers.iter().enumerate() {
            let cout = w(64 << i);
            let mut blocks = Vec::new();
            for b in 0..n {
                let stride = if i > 0 && b == 0 { 2 } else { 1 };
                blocks.push(if bottleneck {
                    Block::Bottleneck(Bottleneck::new(ctx, cin, cout, stride))
                } else {
                    Block::Basic(Basic::new(ctx, cin, cout, stride))
                });
                cin = cout;
            }
            stages.push(blocks);
        }
        ResNet {
            stem,
            stages,
            fc: Linear::new(ctx, cin, classes, true),
        }
    }

    /// Fold every BN into its conv for inference (no-op for training callers).
    pub fn fold_bn(&mut self, sess: &mut Session) {
        self.stem.fold(sess);
        for stage in &mut self.stages {
            for b in stage {
                b.fold_bn(sess);
            }
        }
    }

    pub fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        let mut y = nn::relu(&self.stem.forward(x, sess, training));
        let (y2, _) = y.max_pool2d(&[3, 3], Some(&[2, 2]), &[1, 1], &[1, 1], false, false);
        y = y2;
        for stage in &self.stages {
            for b in stage {
                y = b.forward(&y, sess, training);
            }
        }
        let pooled = y.mean(Some(&[2usize, 3]), false);
        self.fc.forward(&pooled.flatten(1, -1))
    }
}

impl Module for ResNet {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let mut v = prefixed("stem", self.stem.named_params());
        for (i, stage) in self.stages.iter().enumerate() {
            for (j, b) in stage.iter().enumerate() {
                v.extend(prefixed(
                    &format!("stage{i}.{j}"),
                    Box::new(b.named_params().into_iter()),
                ));
            }
        }
        v.extend(prefixed("fc", self.fc.named_params()));
        Box::new(v.into_iter())
    }

    fn into_params(mut self) -> Self {
        self.stem = self.stem.into_params();
        self.stages = self
            .stages
            .into_iter()
            .map(|s| s.into_iter().map(Block::into_params).collect())
            .collect();
        self.fc = self.fc.into_params();
        self
    }
}

/// MobileNetV2 inverted residual: 1x1 expand -> 3x3 depthwise -> 1x1 project.
struct InvRes {
    expand: Option<Cb>,
    dw: Cb,
    project: Cb,
    add: bool,
}

impl InvRes {
    fn new(ctx: &Ctx, cin: i64, cout: i64, stride: i64, expand: i64) -> InvRes {
        let mid = cin * expand;
        InvRes {
            expand: (expand != 1).then(|| Cb::new(ctx, cin, mid, 1, 1, 0, 1)),
            dw: Cb::new(ctx, mid, mid, 3, stride, 1, mid),
            project: Cb::new(ctx, mid, cout, 1, 1, 0, 1),
            add: stride == 1 && cin == cout,
        }
    }

    fn fold_bn(&mut self, sess: &mut Session) {
        if let Some(e) = &mut self.expand {
            e.fold(sess);
        }
        self.dw.fold(sess);
        self.project.fold(sess);
    }

    fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        let y = match &self.expand {
            Some(e) => nn::relu(&e.forward(x, sess, training)).minimum(6.0),
            None => x.clone(),
        };
        let y = nn::relu(&self.dw.forward(&y, sess, training)).minimum(6.0);
        let y = self.project.forward(&y, sess, training);
        if self.add { x.add(&y) } else { y }
    }

    fn named_params(&self) -> Vec<(String, Tensor)> {
        let mut v = Vec::new();
        if let Some(e) = &self.expand {
            v.extend(prefixed("expand", e.named_params()));
        }
        v.extend(prefixed("dw", self.dw.named_params()));
        v.extend(prefixed("project", self.project.named_params()));
        v
    }

    fn into_params(self) -> InvRes {
        InvRes {
            expand: self.expand.map(Cb::into_params),
            dw: self.dw.into_params(),
            project: self.project.into_params(),
            add: self.add,
        }
    }
}

/// `torchvision.models.mobilenet_v2` (1000 classes).
pub struct MobileNet {
    first: Cb,
    blocks: Vec<InvRes>,
    last: Cb,
    fc: Linear,
}

impl MobileNet {
    pub fn new(ctx: &Ctx, classes: i64) -> MobileNet {
        let first = Cb::new(ctx, 3, 32, 3, 2, 1, 1);
        // (expansion, out, stride) per inverted-residual block.
        let cfg: &[(i64, i64, i64)] = &[
            (1, 16, 1),
            (6, 24, 2),
            (6, 24, 1),
            (6, 32, 2),
            (6, 32, 1),
            (6, 32, 1),
            (6, 64, 2),
            (6, 64, 1),
            (6, 64, 1),
            (6, 64, 1),
            (6, 96, 1),
            (6, 96, 1),
            (6, 96, 1),
            (6, 160, 2),
            (6, 160, 1),
            (6, 160, 1),
            (6, 320, 1),
        ];
        let mut cin = 32;
        let blocks = cfg
            .iter()
            .map(|&(e, cout, s)| {
                let b = InvRes::new(ctx, cin, cout, s, e);
                cin = cout;
                b
            })
            .collect();
        MobileNet {
            first,
            blocks,
            last: Cb::new(ctx, 320, 1280, 1, 1, 0, 1),
            fc: Linear::new(ctx, 1280, classes, true),
        }
    }

    /// Fold every BN into its conv for inference.
    pub fn fold_bn(&mut self, sess: &mut Session) {
        self.first.fold(sess);
        for b in &mut self.blocks {
            b.fold_bn(sess);
        }
        self.last.fold(sess);
    }

    pub fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        let mut y = nn::relu(&self.first.forward(x, sess, training)).minimum(6.0);
        for b in &self.blocks {
            y = nn::relu(&b.forward(&y, sess, training)).minimum(6.0);
        }
        let y = nn::relu(&self.last.forward(&y, sess, training)).minimum(6.0);
        let pooled = y.mean(Some(&[2usize, 3]), false);
        self.fc.forward(&pooled.flatten(1, -1))
    }
}

impl Module for MobileNet {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let mut v = prefixed("first", self.first.named_params());
        for (i, b) in self.blocks.iter().enumerate() {
            v.extend(prefixed(
                &format!("b{i}"),
                Box::new(b.named_params().into_iter()),
            ));
        }
        v.extend(prefixed("last", self.last.named_params()));
        v.extend(prefixed("fc", self.fc.named_params()));
        Box::new(v.into_iter())
    }

    fn into_params(mut self) -> Self {
        self.first = self.first.into_params();
        self.blocks = self.blocks.into_iter().map(InvRes::into_params).collect();
        self.last = self.last.into_params();
        self.fc = self.fc.into_params();
        self
    }
}
