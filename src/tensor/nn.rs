//! M7: the `nn` module — neural network layers (nn/__init__.py).
//! Mirrors tinygrad's nn classes: Linear, Conv2d, BatchNorm, LayerNorm, GroupNorm,
//! and helper activations (relu, gelu, softmax, etc.).

use crate::dtype::ConstVal;
use crate::engine::session::Session;
use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

/// A neural-network module: anything that owns trainable parameters.
///
/// Mirrors tinygrad's `nn.state` parameter collection, but expresses it as a
/// recursive trait so a model can name its params for safetensors round-tripping
/// (e.g. `"l1.weight"`). A `Module` returns its params as `(name, tensor)`
/// pairs; nested modules are flattened by prefixing with the child path via
/// [`scoped`]. `Tensor` itself carries no name — naming lives entirely here.
pub trait Module {
    /// This module's trainable params as `(relative_name, tensor)` pairs.
    /// Implementations return names *local* to the module (e.g. `"weight"`);
    /// parents prefix them with the child path using [`scoped`].
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_>;

    /// Just the parameter tensors, dropping the names. This is what an
    /// optimizer is constructed from (tinygrad `nn.state.get_parameters`).
    fn params(&self) -> impl Iterator<Item = Tensor> + '_ {
        self.named_params().map(|(_name, p)| p)
    }

    /// Materialize every parameter into a buffer-backed leaf (see
    /// [`Tensor::to_param`]), consuming and rebuilding the module. Recurse
    /// through nested modules. Do this once before training so the optimizer can
    /// store updates into the parameter buffers in place ([`Tensor::assign`]).
    ///
    /// This is the port's stand-in for tinygrad realizing parameters as buffers
    /// with stable identity; a raw init expression would be recomputed (and the
    /// update lost) on every realize.
    fn into_params(self, ctx: &mut UOpCtx) -> Self
    where
        Self: Sized;
}

/// Prefix every param name in `it` with `parent` + `"."`, for nesting a child
/// module's [`Module::named_params`] under a path segment.
///
/// ```ignore
/// scoped("l1", self.l1.named_params())
/// ```
pub fn scoped<'a>(parent: &'a str, it: impl Iterator<Item = (String, Tensor)> + 'a) -> impl Iterator<Item = (String, Tensor)> + 'a {
    it.map(move |(n, t)| (format!("{parent}.{n}"), t))
}

impl Module for Linear {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        self.weight = self.weight.to_param(ctx);
        self.bias = self.bias.map(|b| b.to_param(ctx));
        self
    }
}

/// The `(name, tensor)` pairs shared by every layer with a `weight` and an
/// optional `bias` (`Linear`, `Conv2d`, …), mirroring tinygrad's attribute names.
fn weight_bias(weight: &Tensor, bias: &Option<Tensor>) -> Box<dyn Iterator<Item = (String, Tensor)> + 'static> {
    let w = std::iter::once(("weight".to_string(), weight.clone()));
    let b = bias.clone().map(|b| ("bias".to_string(), b));
    Box::new(w.chain(b))
}

/// Map the optional `weight`/`bias` of an affine layer to buffer-backed leaves.
fn affine_into_params(weight: Option<Tensor>, bias: Option<Tensor>, ctx: &mut UOpCtx) -> (Option<Tensor>, Option<Tensor>) {
    (weight.map(|w| w.to_param(ctx)), bias.map(|b| b.to_param(ctx)))
}

/// `(name, tensor)` pairs for a layer whose `weight`/`bias` are optional
/// (`BatchNorm`, `LayerNorm`, `GroupNorm`).
fn optional_weight_bias(weight: &Option<Tensor>, bias: &Option<Tensor>) -> Box<dyn Iterator<Item = (String, Tensor)> + 'static> {
    let w = weight.clone().map(|w| ("weight".to_string(), w));
    let b = bias.clone().map(|b| ("bias".to_string(), b));
    Box::new(IntoIterator::into_iter(w).chain(b))
}

/// `nn::Linear`: `x.linear(weight.transpose(), bias)`.
/// weight shape: (out_features, in_features), bias shape: (out_features,).
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Create a new Linear layer with Kaiming uniform initialization (tinygrad
    /// `nn.Linear.__init__`: `kaiming_uniform(out, in, a=sqrt(5))` for weight,
    /// `uniform(in, -1/sqrt(in), 1/sqrt(in))` for bias).
    pub fn new(ctx: &mut UOpCtx, in_features: i64, out_features: i64, bias: bool) -> Self {
        let weight = Tensor::kaiming_uniform(ctx, &[out_features, in_features], 5.0f64.sqrt(), None);
        let bias = if bias {
            let bound = 1.0 / f64::sqrt(in_features as f64);
            Some(Tensor::uniform(ctx, &[out_features], Some(-bound), Some(bound), None, None))
        } else {
            None
        };
        Linear { weight, bias }
    }

    /// Forward pass: `x.linear(self.weight.transpose(), self.bias)`.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let w = self.weight.transpose(ctx, 0, 1);
        x.linear(ctx, &w, self.bias.as_ref())
    }
}

/// `nn::Conv2d`: 2D convolution.
/// weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w).
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub groups: i64,
    pub stride: i64,
    pub dilation: i64,
    pub padding: Vec<i64>,
}

impl Conv2d {
    /// Create a new Conv2d layer.
    /// kernel_size: (kh, kw), padding: (ph_lo, ph_hi, pw_lo, pw_hi).
    pub fn new(
        ctx: &mut UOpCtx,
        in_channels: i64,
        out_channels: i64,
        kernel_size: &[i64],
        stride: i64,
        padding: &[(i64, i64); 2],
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        assert_eq!(kernel_size.len(), 2, "Conv2d: kernel_size must be (kh, kw)");
        assert_eq!(padding.len(), 2, "Conv2d: padding must be ((ph_lo, ph_hi), (pw_lo, pw_hi))");
        assert_eq!(in_channels % groups, 0, "in_channels must be divisible by groups");
        assert_eq!(out_channels % groups, 0, "out_channels must be divisible by groups");

        let (kh, kw) = (kernel_size[0], kernel_size[1]);
        let scale = 1.0 / f64::sqrt((in_channels * kh * kw) as f64);

        // weight: (out_channels, in_channels/groups, kh, kw)
        let weight_shape = [out_channels, in_channels / groups, kh, kw];
        let weight = Tensor::uniform(ctx, &weight_shape, Some(-scale), Some(scale), None, None);
        let bias = if bias {
            Some(Tensor::uniform(ctx, &[out_channels], Some(-scale), Some(scale), None, None))
        } else {
            None
        };

        // Flatten padding to (0,0,..., ph_lo, ph_hi, pw_lo, pw_hi) for non-batch dims
        let padding_flat = vec![padding[0].0, padding[0].1, padding[1].0, padding[1].1];

        Conv2d {
            weight,
            bias,
            groups,
            stride,
            dilation,
            padding: padding_flat,
        }
    }

    /// Forward pass: `x.conv2d(self.weight, self.bias, ...)`.
    /// Input: (batch, channels, height, width) or (channels, height, width).
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        x.conv2d(
            ctx,
            &self.weight,
            self.bias.as_ref(),
            self.groups,
            self.stride,
            self.dilation,
            &self.padding,
            None,
        )
    }
}

impl Module for Conv2d {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        self.weight = self.weight.to_param(ctx);
        self.bias = self.bias.map(|b| b.to_param(ctx));
        self
    }
}

/// `nn::BatchNorm`: batch normalization (nn/__init__.py:8-64).
pub struct BatchNorm {
    pub eps: f64,
    pub track_running_stats: bool,
    pub momentum: f64,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub running_mean: Option<Tensor>,
    pub running_var: Option<Tensor>,
    pub num_batches_tracked: Option<Tensor>,
}

impl BatchNorm {
    /// Create a new BatchNorm layer.
    /// sz: feature size, eps: epsilon for numerical stability.
    pub fn new(ctx: &mut UOpCtx, sz: i64, eps: f64, affine: bool, track_running_stats: bool, momentum: f64) -> Self {
        let weight = if affine { Some(Tensor::ones(ctx, [sz])) } else { None };
        let bias = if affine { Some(Tensor::zeros(ctx, [sz])) } else { None };
        // `num_batches_tracked = Tensor.zeros(dtype='long').is_param_(False)`
        let num_batches_tracked = Some(Tensor::full(ctx, &[] as &[i64], ConstVal::Int(0), Some(crate::dtype::INT64), None));
        let (running_mean, running_var) = if track_running_stats {
            (Some(Tensor::zeros(ctx, [sz])), Some(Tensor::ones(ctx, [sz])))
        } else {
            (None, None)
        };

        BatchNorm {
            eps,
            track_running_stats,
            momentum,
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked,
        }
    }

    /// `BatchNorm.calc_stats` (nn/__init__.py:27-35):
    /// `shape_mask = [1, -1, *([1]*(x.ndim-2))]`; in eval with tracked stats it
    /// returns the (reshaped) running stats, otherwise the batch mean/variance.
    pub fn calc_stats(&self, x: &Tensor, ctx: &mut UOpCtx, training: bool) -> (Tensor, Tensor) {
        let x_shape: Vec<SVal> = x.shape(ctx).into_vec();
        let ndim = x_shape.len();
        let mut shape_mask: Vec<crate::uop::sint::SVal> = vec![crate::uop::sint::SVal::I(1), crate::uop::sint::SVal::I(-1)];
        shape_mask.extend(std::iter::repeat_n(crate::uop::sint::SVal::I(1), ndim - 2));
        if self.track_running_stats && !training {
            let running_mean = self.running_mean.clone().unwrap();
            let running_var = self.running_var.clone().unwrap().reshape(ctx, &shape_mask);
            return (running_mean, running_var);
        }
        let reduce_axes: Vec<usize> = (0..ndim).filter(|&i| i != 1).collect();
        let batch_mean = x.mean(ctx, Some(&reduce_axes), false);
        // `y = (x - batch_mean.detach().reshape(shape_mask))` — d(var)/d(mean) = 0
        let mean_detached = batch_mean.detach(ctx).reshape(ctx, &shape_mask);
        let y = x.sub(ctx, &mean_detached);
        let batch_var = y.square(ctx).mean(ctx, Some(&reduce_axes), false);
        (batch_mean, batch_var)
    }

    /// `BatchNorm.__call__` (nn/__init__.py:37-46). In training with tracked
    /// stats, the running mean/variance (and batch counter) are updated in place
    /// via [`Tensor::assign`] (hence `sess`); eval reads them.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx, sess: &mut Session, training: bool) -> Tensor {
        let (batch_mean, batch_var) = self.calc_stats(x, ctx, training);

        if self.track_running_stats && training {
            let x_shape: Vec<SVal> = x.shape(ctx).into_vec();
            let numel: i64 = x_shape.iter().map(|s| s.clone().as_i64().unwrap_or(0)).product();
            let channels = x_shape[1].clone().as_i64().unwrap_or(0);
            // running_mean.assign((1-momentum)*running_mean + momentum*batch_mean.detach())
            let running_mean = self.running_mean.clone().unwrap();
            let mean_term = batch_mean.detach(ctx).mul(ctx, self.momentum);
            let new_mean = running_mean.mul(ctx, 1.0 - self.momentum).add(ctx, &mean_term);
            running_mean.assign(ctx, sess, &new_mean);
            // running_var.assign((1-momentum)*running_var + momentum*numel/(numel-C)*batch_var.detach())
            let factor = self.momentum * numel as f64 / (numel - channels) as f64;
            let running_var = self.running_var.clone().unwrap();
            let var_term = batch_var.detach(ctx).mul(ctx, factor);
            let new_var = running_var.mul(ctx, 1.0 - self.momentum).add(ctx, &var_term);
            running_var.assign(ctx, sess, &new_var);
            // num_batches_tracked += 1
            if let Some(nbt) = &self.num_batches_tracked {
                let updated = nbt.add(ctx, 1i64);
                nbt.assign(ctx, sess, &updated);
            }
        }

        let invstd = batch_var.add(ctx, self.eps).rsqrt(ctx);
        x.batchnorm(ctx, self.weight.as_ref(), self.bias.as_ref(), &batch_mean, &invstd, &[1])
    }

    /// Training-mode convenience (`forward` with `training = true`).
    pub fn forward_train(&self, x: &Tensor, ctx: &mut UOpCtx, sess: &mut Session) -> Tensor {
        self.forward(x, ctx, sess, true)
    }

    /// Inference-mode convenience (`forward` with `training = false`).
    pub fn forward_inference(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let (batch_mean, batch_var) = self.calc_stats(x, ctx, false);
        let invstd = batch_var.add(ctx, self.eps).rsqrt(ctx);
        x.batchnorm(ctx, self.weight.as_ref(), self.bias.as_ref(), &batch_mean, &invstd, &[1])
    }
}

impl Module for BatchNorm {
    /// Trainable params only (`weight`/`bias`). `running_mean`/`running_var`/
    /// `num_batches_tracked` are non-trainable buffers (`is_param_(False)` in
    /// tinygrad) and are not collected by the optimizer.
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias, ctx);
        // Running stats are buffer-backed leaves too (they are assigned in place
        // during training), but they stay out of `named_params` (non-trainable).
        self.running_mean = self.running_mean.map(|t| t.to_param(ctx));
        self.running_var = self.running_var.map(|t| t.to_param(ctx));
        self.num_batches_tracked = self.num_batches_tracked.map(|t| t.to_param(ctx));
        self
    }
}

/// `nn::LayerNorm`: layer normalization.
pub struct LayerNorm {
    pub normalized_shape: Vec<i64>,
    pub eps: f64,
    pub affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    /// Create a new LayerNorm layer.
    /// normalized_shape: shape of normalized dimensions (usually just the last dim).
    pub fn new(ctx: &mut UOpCtx, normalized_shape: Vec<i64>, eps: f64, affine: bool) -> Self {
        let weight = if affine { Some(Tensor::ones(ctx, &normalized_shape)) } else { None };
        let bias = if affine { Some(Tensor::zeros(ctx, &normalized_shape)) } else { None };

        LayerNorm {
            normalized_shape,
            eps,
            affine,
            weight,
            bias,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        // Normalize over the last len(normalized_shape) dimensions
        let ndim = x.shape(ctx).len();
        let norm_dim_count = self.normalized_shape.len();
        let start_axis = ndim - norm_dim_count;

        // Build axis list
        let axis: Vec<usize> = (start_axis..ndim).collect();
        let mean = x.mean(ctx, Some(&axis), true);
        let diff = x.sub(ctx, &(mean));
        let var = diff.square(ctx).mean(ctx, Some(&axis), true);
        let eps_tensor = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(self.eps), None);
        let invstd = var.add(ctx, &(eps_tensor)).sqrt(ctx).reciprocal(ctx);
        let normalized = diff.mul(ctx, &(invstd));

        if self.affine {
            // tinygrad `LayerNorm.__call__`: `x * self.weight + self.bias`.
            let norm_shape = self._norm_shape(ndim);
            let w = self.weight.as_ref().unwrap().reshape(ctx, &norm_shape);
            let b = self.bias.as_ref().unwrap().reshape(ctx, &norm_shape);
            normalized.mul(ctx, &w).add(ctx, &b)
        } else {
            normalized
        }
    }

    fn _norm_shape(&self, total_ndim: usize) -> Vec<crate::uop::sint::SVal> {
        let mut shape = vec![crate::uop::sint::SVal::I(1); total_ndim - self.normalized_shape.len()];
        shape.extend(self.normalized_shape.iter().map(|&s| crate::uop::sint::SVal::I(s as i128)));
        shape
    }
}

impl Module for LayerNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias, ctx);
        self
    }
}

/// `nn::GroupNorm`: group normalization.
pub struct GroupNorm {
    pub num_groups: i64,
    pub num_channels: i64,
    pub eps: f64,
    pub affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    /// Create a new GroupNorm layer.
    pub fn new(ctx: &mut UOpCtx, num_groups: i64, num_channels: i64, eps: f64, affine: bool) -> Self {
        assert_eq!(num_channels % num_groups, 0, "num_channels must be divisible by num_groups");
        let weight = if affine { Some(Tensor::ones(ctx, [num_channels])) } else { None };
        let bias = if affine { Some(Tensor::zeros(ctx, [num_channels])) } else { None };

        GroupNorm {
            num_groups,
            num_channels,
            eps,
            affine,
            weight,
            bias,
        }
    }

    /// Forward pass (nn/__init__.py:195-203): reshape to
    /// `(N, num_groups, -1)`, layer-norm the last axis, reshape back, then apply
    /// the per-channel affine `x * weight.reshape(1,-1,1..) + bias.reshape(...)`.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let x_shape: Vec<SVal> = x.shape(ctx).into_vec();
        let ndim = x_shape.len();
        let batch = x_shape[0].clone();
        let x = x
            .reshape(ctx, vec![batch, SVal::I(self.num_groups as i128), SVal::I(-1)])
            .layernorm(ctx, -1, self.eps)
            .reshape(ctx, x_shape);
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let mut aff_shape: Vec<SVal> = vec![SVal::I(1), SVal::I(-1)];
                aff_shape.extend(std::iter::repeat_n(SVal::I(1), ndim - 2));
                let w_r = w.reshape(ctx, aff_shape.clone());
                let b_r = b.reshape(ctx, aff_shape);
                x.mul(ctx, &w_r).add(ctx, &b_r)
            }
            _ => x,
        }
    }
}

impl Module for GroupNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias, ctx);
        self
    }
}

/// `nn::InstanceNorm`: instance normalization (nn/__init__.py:209-233). Like
/// `GroupNorm` with `num_groups = num_features`: reshape to
/// `(N, num_features, -1)`, layer-norm the last axis, reshape back, then apply
/// the per-channel affine `x * weight.reshape(1,-1,1..) + bias.reshape(...)`.
pub struct InstanceNorm {
    pub num_features: i64,
    pub eps: f64,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl InstanceNorm {
    /// Create a new InstanceNorm layer.
    pub fn new(ctx: &mut UOpCtx, num_features: i64, eps: f64, affine: bool) -> Self {
        let weight = if affine { Some(Tensor::ones(ctx, [num_features])) } else { None };
        let bias = if affine { Some(Tensor::zeros(ctx, [num_features])) } else { None };
        InstanceNorm { num_features, eps, weight, bias }
    }

    /// Forward pass (nn/__init__.py:230-233).
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let x_shape: Vec<SVal> = x.shape(ctx).into_vec();
        let ndim = x_shape.len();
        let batch = x_shape[0].clone();
        let x = x
            .reshape(ctx, vec![batch, SVal::I(self.num_features as i128), SVal::I(-1)])
            .layernorm(ctx, -1, self.eps)
            .reshape(ctx, x_shape);
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let mut aff_shape: Vec<SVal> = vec![SVal::I(1), SVal::I(-1)];
                aff_shape.extend(std::iter::repeat_n(SVal::I(1), ndim - 2));
                let w_r = w.reshape(ctx, aff_shape.clone());
                let b_r = b.reshape(ctx, aff_shape);
                x.mul(ctx, &w_r).add(ctx, &b_r)
            }
            _ => x,
        }
    }
}

impl Module for InstanceNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias, ctx);
        self
    }
}

/// `nn::LayerNorm2d`: layer norm over a channel-first 2D image
/// (nn/__init__.py:263-279). Normalizes over the channel axis by permuting
/// `(N,C,H,W) -> (N,H,W,C)`, applying the inner `LayerNorm`, then permuting back.
pub struct LayerNorm2d {
    pub norm: LayerNorm,
}

impl LayerNorm2d {
    /// Create a new LayerNorm2d layer (subclasses `LayerNorm(num_channels)`).
    pub fn new(ctx: &mut UOpCtx, num_channels: i64, eps: f64, elementwise_affine: bool) -> Self {
        LayerNorm2d { norm: LayerNorm::new(ctx, vec![num_channels], eps, elementwise_affine) }
    }

    /// `LayerNorm2d.__call__` (nn/__init__.py:279).
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let p = x.permute(ctx, &[0, 2, 3, 1]);
        self.norm.forward(&p, ctx).permute(ctx, &[0, 3, 1, 2])
    }
}

impl Module for LayerNorm2d {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        self.norm.named_params()
    }

    fn into_params(self, ctx: &mut UOpCtx) -> Self {
        LayerNorm2d { norm: self.norm.into_params(ctx) }
    }
}

/// `nn::RMSNorm`: root-mean-square normalization (nn/__init__.py:281-304).
pub struct RMSNorm {
    pub dim: i64,
    pub eps: f64,
    pub weight: Option<Tensor>,
}

impl RMSNorm {
    /// Create a new RMSNorm layer.
    pub fn new(ctx: &mut UOpCtx, dim: i64, eps: f64, elementwise_affine: bool) -> Self {
        let weight = if elementwise_affine { Some(Tensor::ones(ctx, [dim])) } else { None };
        RMSNorm { dim, eps, weight }
    }

    /// `RMSNorm._norm` (nn/__init__.py:300):
    /// `x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()`.
    pub fn norm(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let last = x.shape(ctx).len() - 1;
        let ms = x.square(ctx).mean(ctx, Some(&[last]), true);
        let eps_t = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(self.eps), None);
        let r = ms.add(ctx, &eps_t).rsqrt(ctx);
        x.mul(ctx, &r)
    }

    /// `RMSNorm.__call__` (nn/__init__.py:302-304): normalize in f32, cast back,
    /// then apply the (optional) elementwise affine.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let dt = x.dtype(ctx);
        let xf = x.cast(ctx, crate::dtype::FLOAT32);
        let normed = self.norm(&xf, ctx).cast(ctx, dt);
        match &self.weight {
            Some(w) => normed.mul(ctx, w),
            None => normed,
        }
    }
}

impl Module for RMSNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &None)
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        self.weight = self.weight.map(|w| w.to_param(ctx));
        self
    }
}

/// A list of modules, indexed like tinygrad `nn.state.get_state_dict` on a
/// Python `list` (`"0.weight"`, `"1.weight"`, …). Handy for `Vec<Conv2d>` or a
/// `Vec<Linear>` MLP without hand-listing every parameter.
impl<T: Module> Module for Vec<T> {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(
            self.iter()
                .enumerate()
                .flat_map(|(i, m)| m.named_params().map(move |(n, t)| (format!("{i}.{n}"), t))),
        )
    }

    fn into_params(self, ctx: &mut UOpCtx) -> Self {
        self.into_iter().map(|m| m.into_params(ctx)).collect()
    }
}

/// `_embedding_fwd` (nn/__init__.py:369-371): the non-atomic embedding lookup.
/// `(arange(vocab) == idx.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2)`.
pub fn _embedding_fwd(weight: &Tensor, idx: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let vocab = weight.shape(ctx).into_vec()[0].clone().as_i64().unwrap_or(0);
    let arange = Tensor::arange(ctx, 0.0, Some(vocab as f64), 1.0, Some(crate::dtype::default_int()));
    let idx_u = idx.unsqueeze(ctx, -1);
    let cond = arange.eq(ctx, &idx_u).unsqueeze(ctx, -1);
    let zero = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(0.0), Some(weight.dtype(ctx)));
    let selected = cond.where_(ctx, weight, zero);
    let ndim = selected.shape(ctx).len();
    selected.sum(ctx, Some(&[ndim - 2]), false, Some(weight.dtype(ctx)))
}

/// `nn::Embedding`: a lookup table (nn/__init__.py:377-396).
pub struct Embedding {
    pub vocab_size: i64,
    pub embed_size: i64,
    pub weight: Tensor,
}

impl Embedding {
    /// Create a new Embedding layer (`Tensor.glorot_uniform(vocab, embed)`).
    pub fn new(ctx: &mut UOpCtx, vocab_size: i64, embed_size: i64) -> Self {
        let weight = Tensor::glorot_uniform(ctx, &[vocab_size, embed_size], None);
        Embedding { vocab_size, embed_size, weight }
    }

    /// Forward pass: `Embedding.__call__` (nn/__init__.py:391-396). The atomic
    /// path is only used with `USE_ATOMICS`; this is `_embedding_fwd`.
    pub fn forward(&self, idx: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        if !crate::dtype::is_int(idx.dtype(ctx)) {
            panic!("TGERR: Expected integer dtype for index in embedding, got {:?}", idx.dtype(ctx));
        }
        _embedding_fwd(&self.weight, idx, ctx)
    }
}

impl Module for Embedding {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(std::iter::once(("weight".to_string(), self.weight.clone())))
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        self.weight = self.weight.to_param(ctx);
        self
    }
}

/// `nn::LSTMCell` (nn/__init__.py:398-420).
pub struct LSTMCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
}

impl LSTMCell {
    /// Create a new LSTMCell layer (uniform init in `[-stdv, stdv]`,
    /// `stdv = 1/sqrt(hidden_size)`; zero biases).
    pub fn new(ctx: &mut UOpCtx, input_size: i64, hidden_size: i64, bias: bool) -> Self {
        let stdv = 1.0 / (hidden_size as f64).sqrt();
        let weight_ih = Tensor::uniform(ctx, &[hidden_size * 4, input_size], Some(-stdv), Some(stdv), None, None);
        let weight_hh = Tensor::uniform(ctx, &[hidden_size * 4, hidden_size], Some(-stdv), Some(stdv), None, None);
        let bias_ih = if bias { Some(Tensor::zeros(ctx, [hidden_size * 4])) } else { None };
        let bias_hh = if bias { Some(Tensor::zeros(ctx, [hidden_size * 4])) } else { None };
        LSTMCell { weight_ih, weight_hh, bias_ih, bias_hh }
    }
}

impl LSTMCell {
    /// `LSTMCell.__call__` (nn/__init__.py:414-420). `hc` is the optional (h, c)
    /// hidden state; when absent it starts as zeros.
    pub fn forward(&self, x: &Tensor, hc: Option<(&Tensor, &Tensor)>, ctx: &mut UOpCtx) -> (Tensor, Tensor) {
        let (h, c) = match hc {
            Some((h, c)) => (h.clone(), c.clone()),
            None => {
                let batch = x.shape(ctx).into_vec()[0].clone();
                let hidden = self.weight_hh.shape(ctx).into_vec()[1].clone();
                let z = Tensor::full_const(ctx, vec![batch, hidden], ConstVal::Float(0.0), Some(x.dtype(ctx)));
                (z.clone(), z)
            }
        };
        let wih_t = self.weight_ih.transpose(ctx, 0, 1);
        let whh_t = self.weight_hh.transpose(ctx, 0, 1);
        let g1 = x.linear(ctx, &wih_t, self.bias_ih.as_ref());
        let g2 = h.linear(ctx, &whh_t, self.bias_hh.as_ref());
        let gates = g1.add(ctx, &g2);
        let parts = gates.chunk(ctx, 4, 1);
        let i = sigmoid(&parts[0], ctx);
        let f = sigmoid(&parts[1], ctx);
        let g = parts[2].tanh(ctx);
        let o = sigmoid(&parts[3], ctx);
        // new_c = f * hc[1] + i * g
        let ig = i.mul(ctx, &g);
        let new_c = f.mul(ctx, &c).add(ctx, &ig);
        let tanh_c = new_c.tanh(ctx);
        (o.mul(ctx, &tanh_c), new_c)
    }
}

impl Module for LSTMCell {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let w = vec![
            ("weight_ih".to_string(), self.weight_ih.clone()),
            ("weight_hh".to_string(), self.weight_hh.clone()),
        ];
        let b1 = self.bias_ih.clone().map(|b| ("bias_ih".to_string(), b));
        let b2 = self.bias_hh.clone().map(|b| ("bias_hh".to_string(), b));
        Box::new(w.into_iter().chain(b1).chain(b2))
    }

    fn into_params(mut self, ctx: &mut UOpCtx) -> Self {
        self.weight_ih = self.weight_ih.to_param(ctx);
        self.weight_hh = self.weight_hh.to_param(ctx);
        self.bias_ih = self.bias_ih.map(|b| b.to_param(ctx));
        self.bias_hh = self.bias_hh.map(|b| b.to_param(ctx));
        self
    }
}

// ***** Activations *****

/// `relu()`: max(x, 0).
pub fn relu(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let zero = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(0.0), None);
    x.maximum(ctx, &(zero))
}

/// `gelu()`: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let sqrt_2_over_pi = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float((2.0 / std::f64::consts::PI).sqrt()), None);
    let c3 = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(0.044715), None);
    let half = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(0.5), None);
    let one = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(1.0), None);
    let x3 = x.mul(ctx, x).mul(ctx, x); // x^3
    let c3x3 = c3.mul(ctx, &x3);
    let x_plus = x.add(ctx, &c3x3);
    let inner = sqrt_2_over_pi.mul(ctx, &x_plus);
    let inner = inner.tanh(ctx); // (exp(2i)-1)/(exp(2i)+1)
    let one_plus = one.add(ctx, &inner);
    let half_one_plus = half.mul(ctx, &one_plus);
    x.mul(ctx, &half_one_plus)
}

/// `softmax()`: exp(x - max) / sum(exp(x - max)).
pub fn softmax(x: &Tensor, ctx: &mut UOpCtx, axis: Option<usize>) -> Tensor {
    let axis_vec = axis.map(|a| vec![a]);
    let max_val = x.max(ctx, axis_vec.as_deref(), true);
    let x_shifted = x.sub(ctx, &max_val);
    let exp_x = x_shifted.exp(ctx);
    let sum_exp = exp_x.sum(ctx, axis_vec.as_deref(), true, None);
    let recip = sum_exp.reciprocal(ctx);
    exp_x.mul(ctx, &recip)
}

/// `log_softmax()`: softmax(log(x)) = x - max - log(sum(exp(x - max))).
pub fn log_softmax(x: &Tensor, ctx: &mut UOpCtx, axis: Option<usize>) -> Tensor {
    let axis_vec = axis.map(|a| vec![a]);
    let max_val = x.max(ctx, axis_vec.as_deref(), true);
    let x_shifted = x.sub(ctx, &max_val);
    let exp_x = x_shifted.exp(ctx);
    let sum_exp = exp_x.sum(ctx, axis_vec.as_deref(), true, None);
    let log_sum_exp = sum_exp.log(ctx);
    x_shifted.sub(ctx, &log_sum_exp)
}

/// `sigmoid()`: 1 / (1 + exp(-x)).
pub fn sigmoid(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let one = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(1.0), None);
    let neg_exp = x.neg(ctx).exp(ctx);
    let denom = one.add(ctx, &neg_exp);
    one.div(ctx, &denom)
}

/// `tanh()`: (exp(2x) - 1) / (exp(2x) + 1).
pub fn tanh(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let two = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(2.0), None);
    let one = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(1.0), None);
    let exp_2x = x.mul(ctx, &two).exp(ctx);
    let num = exp_2x.sub(ctx, &one);
    let den = exp_2x.add(ctx, &one);
    num.div(ctx, &den)
}

/// `sin()`: sine (already on Tensor).
pub fn sin(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.sin(ctx)
}

/// `cos()`: cosine — not directly available, use sin(x + pi/2).
pub fn cos(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let pi_over_2 = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(std::f64::consts::FRAC_PI_2), None);
    let shifted = x.add(ctx, &pi_over_2);
    sin(&shifted, ctx)
}

/// `abs()`: absolute value.
pub fn abs(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.abs(ctx)
}

/// `sqrt()`: square root.
pub fn sqrt(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.sqrt(ctx)
}

/// `rsqrt()`: 1 / sqrt(x).
pub fn rsqrt(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.sqrt(ctx).reciprocal(ctx)
}
