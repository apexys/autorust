//! M7: the `nn` module — neural network layers (nn/__init__.py).
//! Mirrors tinygrad's nn classes: Linear, Conv2d, BatchNorm, LayerNorm, GroupNorm,
//! and helper activations (relu, gelu, softmax, etc.).

use crate::dtype::ConstVal;
use crate::engine::session::Session;
use crate::tensor::{AsCtx, Tensor};
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

    /// Non-trainable buffers (`running_mean`/`running_var`/…). tinygrad's
    /// `get_state_dict` includes these; optimizers do not. Default: none.
    fn named_buffers(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(std::iter::empty())
    }

    /// The full state dict: parameters followed by buffers. This is what
    /// safetensors save/load uses (tinygrad `get_state_dict`).
    fn named_state(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(self.named_params().chain(self.named_buffers()))
    }

    /// Materialize every parameter into a buffer-backed leaf (see
    /// [`Tensor::to_param`]), consuming and rebuilding the module. Recurse
    /// through nested modules. Do this once before training so the optimizer can
    /// store updates into the parameter buffers in place ([`Tensor::assign`]).
    ///
    /// This is the port's stand-in for tinygrad realizing parameters as buffers
    /// with stable identity; a raw init expression would be recomputed (and the
    /// update lost) on every realize.
    fn into_params(self) -> Self
    where
        Self: Sized;
}

/// Prefix every param name in `it` with `parent` + `"."`, for nesting a child
/// module's [`Module::named_params`] under a path segment.
///
/// ```ignore
/// scoped("l1", self.l1.named_params())
/// ```
pub fn scoped<'a>(
    parent: &'a str,
    it: impl Iterator<Item = (String, Tensor)> + 'a,
) -> impl Iterator<Item = (String, Tensor)> + 'a {
    it.map(move |(n, t)| (format!("{parent}.{n}"), t))
}

impl Module for Linear {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self) -> Self {
        self.weight = self.weight.to_param();
        self.bias = self.bias.map(|b| b.to_param());
        self
    }
}

/// The `(name, tensor)` pairs shared by every layer with a `weight` and an
/// optional `bias` (`Linear`, `Conv2d`, …), mirroring tinygrad's attribute names.
fn weight_bias(
    weight: &Tensor,
    bias: &Option<Tensor>,
) -> Box<dyn Iterator<Item = (String, Tensor)> + 'static> {
    let w = std::iter::once(("weight".to_string(), weight.clone()));
    let b = bias.clone().map(|b| ("bias".to_string(), b));
    Box::new(w.chain(b))
}

/// Map the optional `weight`/`bias` of an affine layer to buffer-backed leaves.
fn affine_into_params(
    weight: Option<Tensor>,
    bias: Option<Tensor>,
) -> (Option<Tensor>, Option<Tensor>) {
    (weight.map(|w| w.to_param()), bias.map(|b| b.to_param()))
}

/// `(name, tensor)` pairs for a layer whose `weight`/`bias` are optional
/// (`BatchNorm`, `LayerNorm`, `GroupNorm`).
fn optional_weight_bias(
    weight: &Option<Tensor>,
    bias: &Option<Tensor>,
) -> Box<dyn Iterator<Item = (String, Tensor)> + 'static> {
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
    pub fn new(ctx: impl AsCtx, in_features: i64, out_features: i64, bias: bool) -> Self {
        ctx.with_ctx(|ctx| {
            let weight = Tensor::kaiming_uniform(
                &mut *ctx,
                &[out_features, in_features],
                5.0f64.sqrt(),
                None,
            );
            let bias = if bias {
                let bound = 1.0 / f64::sqrt(in_features as f64);
                Some(Tensor::uniform(
                    &mut *ctx,
                    &[out_features],
                    Some(-bound),
                    Some(bound),
                    None,
                    None,
                ))
            } else {
                None
            };
            Linear { weight, bias }
        })
    }

    /// Forward pass: `x.linear(self.weight.transpose(), self.bias)`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let w = self.weight.transpose_ctx(ctx, 0, 1);
        x.linear_ctx(ctx, &w, self.bias.as_ref())
    }
}

/// `nn::Conv2d`: N-D convolution (nn/__init__.py:80-109). `kernel_size` has one
/// entry per spatial dim, so a 1-tuple makes this a `Conv1d`.
/// weight shape: (out_channels, in_channels/groups, *kernel_size).
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub kernel_size: Vec<i64>,
    pub groups: i64,
    pub stride: i64,
    pub dilation: i64,
    pub padding: Vec<i64>,
}

impl Conv2d {
    /// Create a new Conv2d layer.
    /// kernel_size: one entry per spatial dim; padding: one `(lo, hi)` per dim.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: impl AsCtx,
        in_channels: i64,
        out_channels: i64,
        kernel_size: &[i64],
        stride: i64,
        padding: &[(i64, i64)],
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        ctx.with_ctx(|ctx| {
            assert_eq!(
                padding.len(),
                kernel_size.len(),
                "Conv2d: padding must have one (lo, hi) per spatial dim"
            );
            assert_eq!(
                in_channels % groups,
                0,
                "in_channels must be divisible by groups"
            );
            assert_eq!(
                out_channels % groups,
                0,
                "out_channels must be divisible by groups"
            );

            let kprod: i64 = kernel_size.iter().product();
            let scale = 1.0 / f64::sqrt((in_channels * kprod) as f64);

            // weight: (out_channels, in_channels/groups, *kernel_size)
            let mut weight_shape = vec![out_channels, in_channels / groups];
            weight_shape.extend_from_slice(kernel_size);
            let weight = Tensor::uniform(
                &mut *ctx,
                &weight_shape,
                Some(-scale),
                Some(scale),
                None,
                None,
            );
            let bias = if bias {
                Some(Tensor::uniform(
                    &mut *ctx,
                    &[out_channels],
                    Some(-scale),
                    Some(scale),
                    None,
                    None,
                ))
            } else {
                None
            };

            // Flatten to the explicit `2*dims` form the conv path expects.
            let padding_flat: Vec<i64> = padding.iter().flat_map(|(a, b)| [*a, *b]).collect();

            Conv2d {
                weight,
                bias,
                kernel_size: kernel_size.to_vec(),
                groups,
                stride,
                dilation,
                padding: padding_flat,
            }
        })
    }

    /// Forward pass: `x.conv2d(self.weight, self.bias, ...)`.
    /// Input: (batch, channels, height, width) or (channels, height, width).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        x.conv2d_ctx(
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

    fn into_params(mut self) -> Self {
        self.weight = self.weight.to_param();
        self.bias = self.bias.map(|b| b.to_param());
        self
    }
}

/// `nn.Conv1d` (nn/__init__.py:62-78) — a `Conv2d` with a 1-element kernel.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    ctx: &mut UOpCtx,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
    bias: bool,
) -> Conv2d {
    Conv2d::new(
        ctx,
        in_channels,
        out_channels,
        &[kernel_size],
        stride,
        &[(padding, padding)],
        dilation,
        groups,
        bias,
    )
}

/// `nn::ConvTranspose2d` (nn/__init__.py:130-154). Subclasses `Conv2d`, but the
/// weight has shape (in_channels, out_channels/groups, *kernel_size).
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub kernel_size: Vec<i64>,
    pub output_padding: i64,
    pub groups: i64,
    pub stride: i64,
    pub dilation: i64,
    pub padding: Vec<i64>,
}

impl ConvTranspose2d {
    /// Create a new ConvTranspose2d layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: impl AsCtx,
        in_channels: i64,
        out_channels: i64,
        kernel_size: &[i64],
        stride: i64,
        padding: &[(i64, i64)],
        output_padding: i64,
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        ctx.with_ctx(|ctx| {
            // super().__init__(...) then replace the weight with the transposed shape.
            let base = Conv2d::new(
                &mut *ctx,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            );
            let kprod: i64 = kernel_size.iter().product();
            let scale = 1.0 / f64::sqrt((in_channels * kprod) as f64);
            let mut weight_shape = vec![in_channels, out_channels / groups];
            weight_shape.extend_from_slice(kernel_size);
            let weight = Tensor::uniform(
                &mut *ctx,
                &weight_shape,
                Some(-scale),
                Some(scale),
                None,
                None,
            );
            ConvTranspose2d {
                weight,
                bias: base.bias,
                kernel_size: base.kernel_size,
                output_padding,
                groups,
                stride,
                dilation,
                padding: base.padding,
            }
        })
    }

    /// `ConvTranspose2d.__call__` (nn/__init__.py:153-154).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        x.conv_transpose2d_ctx(
            ctx,
            &self.weight,
            self.bias.as_ref(),
            self.groups,
            self.stride,
            self.dilation,
            &self.padding,
            self.output_padding,
        )
    }
}

impl Module for ConvTranspose2d {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self) -> Self {
        self.weight = self.weight.to_param();
        self.bias = self.bias.map(|b| b.to_param());
        self
    }
}

/// `nn.ConvTranspose1d` (nn/__init__.py:111-128) — a `ConvTranspose2d` with a
/// 1-element kernel.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    ctx: &mut UOpCtx,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: i64,
    padding: i64,
    output_padding: i64,
    dilation: i64,
    groups: i64,
    bias: bool,
) -> ConvTranspose2d {
    ConvTranspose2d::new(
        ctx,
        in_channels,
        out_channels,
        &[kernel_size],
        stride,
        &[(padding, padding)],
        output_padding,
        dilation,
        groups,
        bias,
    )
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
    pub fn new(
        ctx: impl AsCtx,
        sz: i64,
        eps: f64,
        affine: bool,
        track_running_stats: bool,
        momentum: f64,
    ) -> Self {
        ctx.with_ctx(|ctx| {
            let weight = if affine {
                Some(Tensor::ones(&mut *ctx, [sz]))
            } else {
                None
            };
            let bias = if affine {
                Some(Tensor::zeros(&mut *ctx, [sz]))
            } else {
                None
            };
            // `num_batches_tracked = Tensor.zeros(dtype='long').is_param_(False)`
            let num_batches_tracked = Some(Tensor::full(
                &mut *ctx,
                &[] as &[i64],
                ConstVal::Int(0),
                Some(crate::dtype::INT64),
                None,
            ));
            let (running_mean, running_var) = if track_running_stats {
                (
                    Some(Tensor::zeros(&mut *ctx, [sz])),
                    Some(Tensor::ones(&mut *ctx, [sz])),
                )
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
        })
    }

    /// `BatchNorm.calc_stats` (nn/__init__.py:27-35):
    /// `shape_mask = [1, -1, *([1]*(x.ndim-2))]`; in eval with tracked stats it
    /// returns the (reshaped) running stats, otherwise the batch mean/variance.
    pub fn calc_stats(&self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        self.calc_stats_ctx(x, &mut x.ctx().borrow_mut(), training)
    }
    #[doc(hidden)]
    pub fn calc_stats_ctx(&self, x: &Tensor, ctx: &mut UOpCtx, training: bool) -> (Tensor, Tensor) {
        let x_shape: Vec<SVal> = x.shape_ctx(ctx).into_vec();
        let ndim = x_shape.len();
        let mut shape_mask: Vec<crate::uop::sint::SVal> =
            vec![crate::uop::sint::SVal::I(1), crate::uop::sint::SVal::I(-1)];
        shape_mask.extend(std::iter::repeat_n(crate::uop::sint::SVal::I(1), ndim - 2));
        if self.track_running_stats && !training {
            let running_mean = self.running_mean.clone().unwrap();
            let running_var = self
                .running_var
                .clone()
                .unwrap()
                .reshape_ctx(ctx, &shape_mask);
            return (running_mean, running_var);
        }
        let reduce_axes: Vec<usize> = (0..ndim).filter(|&i| i != 1).collect();
        let batch_mean = x.mean_ctx(ctx, Some(&reduce_axes), false);
        // `y = (x - batch_mean.detach().reshape(shape_mask))` — d(var)/d(mean) = 0
        let mean_detached = batch_mean.detach_ctx(ctx).reshape_ctx(ctx, &shape_mask);
        let y = x.sub_ctx(ctx, &mean_detached);
        let batch_var = y.square_ctx(ctx).mean_ctx(ctx, Some(&reduce_axes), false);
        (batch_mean, batch_var)
    }

    /// `BatchNorm.__call__` (nn/__init__.py:37-46). In training with tracked
    /// stats, the running mean/variance (and batch counter) are updated in place
    /// via [`Tensor::assign`] (hence `sess`); eval reads them.
    pub fn forward(&self, x: &Tensor, sess: &mut Session, training: bool) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut(), sess, training)
    }
    #[doc(hidden)]
    pub fn forward_ctx(
        &self,
        x: &Tensor,
        ctx: &mut UOpCtx,
        sess: &mut Session,
        training: bool,
    ) -> Tensor {
        let (batch_mean, batch_var) = self.calc_stats_ctx(x, ctx, training);

        if self.track_running_stats && training {
            let x_shape: Vec<SVal> = x.shape_ctx(ctx).into_vec();
            let numel: i64 = x_shape
                .iter()
                .map(|s| s.clone().as_i64().unwrap_or(0))
                .product();
            let channels = x_shape[1].clone().as_i64().unwrap_or(0);
            // running_mean.assign((1-momentum)*running_mean + momentum*batch_mean.detach())
            let running_mean = self.running_mean.clone().unwrap();
            let mean_term = batch_mean.detach_ctx(ctx).mul_ctx(ctx, self.momentum);
            let new_mean = running_mean
                .mul_ctx(ctx, 1.0 - self.momentum)
                .add_ctx(ctx, &mean_term);
            running_mean.assign_ctx(ctx, sess, &new_mean);
            // running_var.assign((1-momentum)*running_var + momentum*numel/(numel-C)*batch_var.detach())
            let factor = self.momentum * numel as f64 / (numel - channels) as f64;
            let running_var = self.running_var.clone().unwrap();
            let var_term = batch_var.detach_ctx(ctx).mul_ctx(ctx, factor);
            let new_var = running_var
                .mul_ctx(ctx, 1.0 - self.momentum)
                .add_ctx(ctx, &var_term);
            running_var.assign_ctx(ctx, sess, &new_var);
            // num_batches_tracked += 1
            if let Some(nbt) = &self.num_batches_tracked {
                let updated = nbt.add_ctx(ctx, 1i64);
                nbt.assign_ctx(ctx, sess, &updated);
            }
        }

        let invstd = batch_var.add_ctx(ctx, self.eps).rsqrt_ctx(ctx);
        x.batchnorm_ctx(
            ctx,
            self.weight.as_ref(),
            self.bias.as_ref(),
            &batch_mean,
            &invstd,
            &[1],
        )
    }

    /// Training-mode convenience (`forward` with `training = true`).
    pub fn forward_train(&self, x: &Tensor, sess: &mut Session) -> Tensor {
        self.forward_train_ctx(x, &mut x.ctx().borrow_mut(), sess)
    }
    #[doc(hidden)]
    pub fn forward_train_ctx(&self, x: &Tensor, ctx: &mut UOpCtx, sess: &mut Session) -> Tensor {
        self.forward_ctx(x, ctx, sess, true)
    }

    /// Inference-mode convenience (`forward` with `training = false`).
    pub fn forward_inference(&self, x: &Tensor) -> Tensor {
        self.forward_inference_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_inference_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let (batch_mean, batch_var) = self.calc_stats_ctx(x, ctx, false);
        let invstd = batch_var.add_ctx(ctx, self.eps).rsqrt_ctx(ctx);
        x.batchnorm_ctx(
            ctx,
            self.weight.as_ref(),
            self.bias.as_ref(),
            &batch_mean,
            &invstd,
            &[1],
        )
    }
}

impl Module for BatchNorm {
    /// Trainable params only (`weight`/`bias`). `running_mean`/`running_var`/
    /// `num_batches_tracked` are non-trainable buffers (`is_param_(False)` in
    /// tinygrad) and are not collected by the optimizer.
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn named_buffers(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        let mut v: Vec<(String, Tensor)> = Vec::new();
        if let Some(t) = &self.running_mean {
            v.push(("running_mean".to_string(), t.clone()));
        }
        if let Some(t) = &self.running_var {
            v.push(("running_var".to_string(), t.clone()));
        }
        if let Some(t) = &self.num_batches_tracked {
            v.push(("num_batches_tracked".to_string(), t.clone()));
        }
        Box::new(v.into_iter())
    }

    fn into_params(mut self) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias);
        // Running stats are buffer-backed leaves too (they are assigned in place
        // during training), but they stay out of `named_params` (non-trainable).
        self.running_mean = self.running_mean.map(|t| t.to_param());
        self.running_var = self.running_var.map(|t| t.to_param());
        self.num_batches_tracked = self.num_batches_tracked.map(|t| t.to_param());
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
    pub fn new(ctx: impl AsCtx, normalized_shape: Vec<i64>, eps: f64, affine: bool) -> Self {
        ctx.with_ctx(|ctx| {
            let weight = if affine {
                Some(Tensor::ones(&mut *ctx, &normalized_shape))
            } else {
                None
            };
            let bias = if affine {
                Some(Tensor::zeros(&mut *ctx, &normalized_shape))
            } else {
                None
            };

            LayerNorm {
                normalized_shape,
                eps,
                affine,
                weight,
                bias,
            }
        })
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        // Normalize over the last len(normalized_shape) dimensions
        let ndim = x.shape_ctx(ctx).len();
        let norm_dim_count = self.normalized_shape.len();
        let start_axis = ndim - norm_dim_count;

        // Build axis list
        let axis: Vec<usize> = (start_axis..ndim).collect();
        let mean = x.mean_ctx(ctx, Some(&axis), true);
        let diff = x.sub_ctx(ctx, &(mean));
        let var = diff.square_ctx(ctx).mean_ctx(ctx, Some(&axis), true);
        let eps_tensor =
            Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(self.eps), None);
        let invstd = var
            .add_ctx(ctx, &(eps_tensor))
            .sqrt_ctx(ctx)
            .reciprocal_ctx(ctx);
        let normalized = diff.mul_ctx(ctx, &(invstd));

        if self.affine {
            // tinygrad `LayerNorm.__call__`: `x * self.weight + self.bias`.
            let norm_shape = self._norm_shape(ndim);
            let w = self.weight.as_ref().unwrap().reshape_ctx(ctx, &norm_shape);
            let b = self.bias.as_ref().unwrap().reshape_ctx(ctx, &norm_shape);
            normalized.mul_ctx(ctx, &w).add_ctx(ctx, &b)
        } else {
            normalized
        }
    }

    fn _norm_shape(&self, total_ndim: usize) -> Vec<crate::uop::sint::SVal> {
        let mut shape =
            vec![crate::uop::sint::SVal::I(1); total_ndim - self.normalized_shape.len()];
        shape.extend(
            self.normalized_shape
                .iter()
                .map(|&s| crate::uop::sint::SVal::I(s as i128)),
        );
        shape
    }
}

impl Module for LayerNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias);
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
    pub fn new(
        ctx: impl AsCtx,
        num_groups: i64,
        num_channels: i64,
        eps: f64,
        affine: bool,
    ) -> Self {
        ctx.with_ctx(|ctx| {
            assert_eq!(
                num_channels % num_groups,
                0,
                "num_channels must be divisible by num_groups"
            );
            let weight = if affine {
                Some(Tensor::ones(&mut *ctx, [num_channels]))
            } else {
                None
            };
            let bias = if affine {
                Some(Tensor::zeros(&mut *ctx, [num_channels]))
            } else {
                None
            };

            GroupNorm {
                num_groups,
                num_channels,
                eps,
                affine,
                weight,
                bias,
            }
        })
    }

    /// Forward pass (nn/__init__.py:195-203): reshape to
    /// `(N, num_groups, -1)`, layer-norm the last axis, reshape back, then apply
    /// the per-channel affine `x * weight.reshape(1,-1,1..) + bias.reshape(...)`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let x_shape: Vec<SVal> = x.shape_ctx(ctx).into_vec();
        let ndim = x_shape.len();
        let batch = x_shape[0].clone();
        let x = x
            .reshape_ctx(
                ctx,
                vec![batch, SVal::I(self.num_groups as i128), SVal::I(-1)],
            )
            .layernorm_ctx(ctx, -1, self.eps)
            .reshape_ctx(ctx, x_shape);
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let mut aff_shape: Vec<SVal> = vec![SVal::I(1), SVal::I(-1)];
                aff_shape.extend(std::iter::repeat_n(SVal::I(1), ndim - 2));
                let w_r = w.reshape_ctx(ctx, aff_shape.clone());
                let b_r = b.reshape_ctx(ctx, aff_shape);
                x.mul_ctx(ctx, &w_r).add_ctx(ctx, &b_r)
            }
            _ => x,
        }
    }
}

impl Module for GroupNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias);
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
    pub fn new(ctx: impl AsCtx, num_features: i64, eps: f64, affine: bool) -> Self {
        ctx.with_ctx(|ctx| {
            let weight = if affine {
                Some(Tensor::ones(&mut *ctx, [num_features]))
            } else {
                None
            };
            let bias = if affine {
                Some(Tensor::zeros(&mut *ctx, [num_features]))
            } else {
                None
            };
            InstanceNorm {
                num_features,
                eps,
                weight,
                bias,
            }
        })
    }

    /// Forward pass (nn/__init__.py:230-233).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let x_shape: Vec<SVal> = x.shape_ctx(ctx).into_vec();
        let ndim = x_shape.len();
        let batch = x_shape[0].clone();
        let x = x
            .reshape_ctx(
                ctx,
                vec![batch, SVal::I(self.num_features as i128), SVal::I(-1)],
            )
            .layernorm_ctx(ctx, -1, self.eps)
            .reshape_ctx(ctx, x_shape);
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let mut aff_shape: Vec<SVal> = vec![SVal::I(1), SVal::I(-1)];
                aff_shape.extend(std::iter::repeat_n(SVal::I(1), ndim - 2));
                let w_r = w.reshape_ctx(ctx, aff_shape.clone());
                let b_r = b.reshape_ctx(ctx, aff_shape);
                x.mul_ctx(ctx, &w_r).add_ctx(ctx, &b_r)
            }
            _ => x,
        }
    }
}

impl Module for InstanceNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &self.bias)
    }

    fn into_params(mut self) -> Self {
        (self.weight, self.bias) = affine_into_params(self.weight, self.bias);
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
    pub fn new(ctx: impl AsCtx, num_channels: i64, eps: f64, elementwise_affine: bool) -> Self {
        ctx.with_ctx(|ctx| LayerNorm2d {
            norm: LayerNorm::new(ctx, vec![num_channels], eps, elementwise_affine),
        })
    }

    /// `LayerNorm2d.__call__` (nn/__init__.py:279).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let p = x.permute_ctx(ctx, &[0, 2, 3, 1]);
        self.norm
            .forward_ctx(&p, ctx)
            .permute_ctx(ctx, &[0, 3, 1, 2])
    }
}

impl Module for LayerNorm2d {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        self.norm.named_params()
    }

    fn into_params(self) -> Self {
        LayerNorm2d {
            norm: self.norm.into_params(),
        }
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
    pub fn new(ctx: impl AsCtx, dim: i64, eps: f64, elementwise_affine: bool) -> Self {
        ctx.with_ctx(|ctx| {
            let weight = if elementwise_affine {
                Some(Tensor::ones(&mut *ctx, [dim]))
            } else {
                None
            };
            RMSNorm { dim, eps, weight }
        })
    }

    /// `RMSNorm._norm` (nn/__init__.py:300):
    /// `x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()`.
    pub fn norm(&self, x: &Tensor) -> Tensor {
        self.norm_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn norm_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let last = x.shape_ctx(ctx).len() - 1;
        let ms = x.square_ctx(ctx).mean_ctx(ctx, Some(&[last]), true);
        let eps_t = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(self.eps), None);
        let r = ms.add_ctx(ctx, &eps_t).rsqrt_ctx(ctx);
        x.mul_ctx(ctx, &r)
    }

    /// `RMSNorm.__call__` (nn/__init__.py:302-304): normalize in f32, cast back,
    /// then apply the (optional) elementwise affine.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_ctx(x, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let dt = x.dtype_ctx(ctx);
        let xf = x.cast_ctx(ctx, crate::dtype::FLOAT32);
        let normed = self.norm_ctx(&xf, ctx).cast_ctx(ctx, dt);
        match &self.weight {
            Some(w) => normed.mul_ctx(ctx, w),
            None => normed,
        }
    }
}

impl Module for RMSNorm {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        optional_weight_bias(&self.weight, &None)
    }

    fn into_params(mut self) -> Self {
        self.weight = self.weight.map(|w| w.to_param());
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

    fn into_params(self) -> Self {
        self.into_iter().map(|m| m.into_params()).collect()
    }
}

/// `_embedding_fwd` (nn/__init__.py:369-371): the non-atomic embedding lookup.
/// `(arange(vocab) == idx.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2)`.
pub fn _embedding_fwd(weight: &Tensor, idx: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let vocab = weight.shape_ctx(ctx).into_vec()[0]
        .clone()
        .as_i64()
        .unwrap_or(0);
    let arange = Tensor::arange(
        &mut *ctx,
        0.0,
        Some(vocab as f64),
        1.0,
        Some(crate::dtype::default_int()),
    );
    let idx_u = idx.unsqueeze_ctx(ctx, -1);
    let cond = arange.eq_ctx(ctx, &idx_u).unsqueeze_ctx(ctx, -1);
    let dt = weight.dtype_ctx(ctx);
    let zero = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(0.0), Some(dt));
    let selected = cond.where__ctx(ctx, weight, zero);
    let ndim = selected.shape_ctx(ctx).len();
    selected.sum_ctx(ctx, Some(&[ndim - 2]), false, Some(weight.dtype_ctx(ctx)))
}

/// `nn::Embedding`: a lookup table (nn/__init__.py:377-396).
pub struct Embedding {
    pub vocab_size: i64,
    pub embed_size: i64,
    pub weight: Tensor,
}

impl Embedding {
    /// Create a new Embedding layer (`Tensor.glorot_uniform(vocab, embed)`).
    pub fn new(ctx: impl AsCtx, vocab_size: i64, embed_size: i64) -> Self {
        ctx.with_ctx(|ctx| {
            let weight = Tensor::glorot_uniform(&mut *ctx, &[vocab_size, embed_size], None);
            Embedding {
                vocab_size,
                embed_size,
                weight,
            }
        })
    }

    /// Forward pass: `Embedding.__call__` (nn/__init__.py:391-396). The atomic
    /// path is only used with `USE_ATOMICS`; this is `_embedding_fwd`.
    pub fn forward(&self, idx: &Tensor) -> Tensor {
        self.forward_ctx(idx, &mut idx.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(&self, idx: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        if !crate::dtype::is_int(idx.dtype_ctx(ctx)) {
            panic!(
                "TGERR: Expected integer dtype for index in embedding, got {:?}",
                idx.dtype_ctx(ctx)
            );
        }
        _embedding_fwd(&self.weight, idx, ctx)
    }
}

impl Module for Embedding {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(std::iter::once(("weight".to_string(), self.weight.clone())))
    }

    fn into_params(mut self) -> Self {
        self.weight = self.weight.to_param();
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
    pub fn new(ctx: impl AsCtx, input_size: i64, hidden_size: i64, bias: bool) -> Self {
        ctx.with_ctx(|ctx| {
            let stdv = 1.0 / (hidden_size as f64).sqrt();
            let weight_ih = Tensor::uniform(
                &mut *ctx,
                &[hidden_size * 4, input_size],
                Some(-stdv),
                Some(stdv),
                None,
                None,
            );
            let weight_hh = Tensor::uniform(
                &mut *ctx,
                &[hidden_size * 4, hidden_size],
                Some(-stdv),
                Some(stdv),
                None,
                None,
            );
            let bias_ih = if bias {
                Some(Tensor::zeros(&mut *ctx, [hidden_size * 4]))
            } else {
                None
            };
            let bias_hh = if bias {
                Some(Tensor::zeros(&mut *ctx, [hidden_size * 4]))
            } else {
                None
            };
            LSTMCell {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
            }
        })
    }
}

impl LSTMCell {
    /// `LSTMCell.__call__` (nn/__init__.py:414-420). `hc` is the optional (h, c)
    /// hidden state; when absent it starts as zeros.
    pub fn forward(&self, x: &Tensor, hc: Option<(&Tensor, &Tensor)>) -> (Tensor, Tensor) {
        self.forward_ctx(x, hc, &mut x.ctx().borrow_mut())
    }
    #[doc(hidden)]
    pub fn forward_ctx(
        &self,
        x: &Tensor,
        hc: Option<(&Tensor, &Tensor)>,
        ctx: &mut UOpCtx,
    ) -> (Tensor, Tensor) {
        let (h, c) = match hc {
            Some((h, c)) => (h.clone(), c.clone()),
            None => {
                let batch = x.shape_ctx(ctx).into_vec()[0].clone();
                let hidden = self.weight_hh.shape_ctx(ctx).into_vec()[1].clone();
                let dt = x.dtype_ctx(ctx);
                let z = Tensor::full_const(
                    &mut *ctx,
                    vec![batch, hidden],
                    ConstVal::Float(0.0),
                    Some(dt),
                );
                (z.clone(), z)
            }
        };
        let wih_t = self.weight_ih.transpose_ctx(ctx, 0, 1);
        let whh_t = self.weight_hh.transpose_ctx(ctx, 0, 1);
        let g1 = x.linear_ctx(ctx, &wih_t, self.bias_ih.as_ref());
        let g2 = h.linear_ctx(ctx, &whh_t, self.bias_hh.as_ref());
        let gates = g1.add_ctx(ctx, &g2);
        let parts = gates.chunk_ctx(ctx, 4, 1);
        let i = sigmoid_ctx(&parts[0], ctx);
        let f = sigmoid_ctx(&parts[1], ctx);
        let g = parts[2].tanh_ctx(ctx);
        let o = sigmoid_ctx(&parts[3], ctx);
        // new_c = f * hc[1] + i * g
        let ig = i.mul_ctx(ctx, &g);
        let new_c = f.mul_ctx(ctx, &c).add_ctx(ctx, &ig);
        let tanh_c = new_c.tanh_ctx(ctx);
        (o.mul_ctx(ctx, &tanh_c), new_c)
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

    fn into_params(mut self) -> Self {
        self.weight_ih = self.weight_ih.to_param();
        self.weight_hh = self.weight_hh.to_param();
        self.bias_ih = self.bias_ih.map(|b| b.to_param());
        self.bias_hh = self.bias_hh.map(|b| b.to_param());
        self
    }
}

// ***** Activations *****

/// `relu()`: max(x, 0).
pub fn relu(x: &Tensor) -> Tensor {
    relu_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn relu_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let zero = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(0.0), None);
    x.maximum_ctx(ctx, &(zero))
}

/// `gelu()`: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu(x: &Tensor) -> Tensor {
    gelu_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn gelu_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let sqrt_2_over_pi = Tensor::full_const(
        &mut *ctx,
        &[] as &[i64],
        ConstVal::Float((2.0 / std::f64::consts::PI).sqrt()),
        None,
    );
    let c3 = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(0.044715), None);
    let half = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(0.5), None);
    let one = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(1.0), None);
    let x3 = x.mul_ctx(ctx, x).mul_ctx(ctx, x); // x^3
    let c3x3 = c3.mul_ctx(ctx, &x3);
    let x_plus = x.add_ctx(ctx, &c3x3);
    let inner = sqrt_2_over_pi.mul_ctx(ctx, &x_plus);
    let inner = inner.tanh_ctx(ctx); // (exp(2i)-1)/(exp(2i)+1)
    let one_plus = one.add_ctx(ctx, &inner);
    let half_one_plus = half.mul_ctx(ctx, &one_plus);
    x.mul_ctx(ctx, &half_one_plus)
}

/// `softmax()`: exp(x - max) / sum(exp(x - max)).
pub fn softmax(x: &Tensor, axis: Option<usize>) -> Tensor {
    softmax_ctx(x, &mut x.ctx().borrow_mut(), axis)
}
#[doc(hidden)]
pub fn softmax_ctx(x: &Tensor, ctx: &mut UOpCtx, axis: Option<usize>) -> Tensor {
    let axis_vec = axis.map(|a| vec![a]);
    let max_val = x.max_ctx(ctx, axis_vec.as_deref(), true);
    let x_shifted = x.sub_ctx(ctx, &max_val);
    let exp_x = x_shifted.exp_ctx(ctx);
    let sum_exp = exp_x.sum_ctx(ctx, axis_vec.as_deref(), true, None);
    let recip = sum_exp.reciprocal_ctx(ctx);
    exp_x.mul_ctx(ctx, &recip)
}

/// `log_softmax()`: softmax(log(x)) = x - max - log(sum(exp(x - max))).
pub fn log_softmax(x: &Tensor, axis: Option<usize>) -> Tensor {
    log_softmax_ctx(x, &mut x.ctx().borrow_mut(), axis)
}
#[doc(hidden)]
pub fn log_softmax_ctx(x: &Tensor, ctx: &mut UOpCtx, axis: Option<usize>) -> Tensor {
    let axis_vec = axis.map(|a| vec![a]);
    let max_val = x.max_ctx(ctx, axis_vec.as_deref(), true);
    let x_shifted = x.sub_ctx(ctx, &max_val);
    let exp_x = x_shifted.exp_ctx(ctx);
    let sum_exp = exp_x.sum_ctx(ctx, axis_vec.as_deref(), true, None);
    let log_sum_exp = sum_exp.log_ctx(ctx);
    x_shifted.sub_ctx(ctx, &log_sum_exp)
}

/// `sigmoid()`: 1 / (1 + exp(-x)).
pub fn sigmoid(x: &Tensor) -> Tensor {
    sigmoid_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn sigmoid_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let one = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(1.0), None);
    let neg_exp = x.neg_ctx(ctx).exp_ctx(ctx);
    let denom = one.add_ctx(ctx, &neg_exp);
    one.div_ctx(ctx, &denom)
}

/// `tanh()`: (exp(2x) - 1) / (exp(2x) + 1).
pub fn tanh(x: &Tensor) -> Tensor {
    tanh_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn tanh_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let two = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(2.0), None);
    let one = Tensor::full_const(&mut *ctx, &[] as &[i64], ConstVal::Float(1.0), None);
    let exp_2x = x.mul_ctx(ctx, &two).exp_ctx(ctx);
    let num = exp_2x.sub_ctx(ctx, &one);
    let den = exp_2x.add_ctx(ctx, &one);
    num.div_ctx(ctx, &den)
}

/// `sin()`: sine (already on Tensor).
pub fn sin(x: &Tensor) -> Tensor {
    sin_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn sin_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.sin_ctx(ctx)
}

/// `cos()`: cosine — not directly available, use sin(x + pi/2).
pub fn cos(x: &Tensor) -> Tensor {
    cos_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn cos_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    let pi_over_2 = Tensor::full_const(
        &mut *ctx,
        &[] as &[i64],
        ConstVal::Float(std::f64::consts::FRAC_PI_2),
        None,
    );
    let shifted = x.add_ctx(ctx, &pi_over_2);
    sin_ctx(&shifted, ctx)
}

/// `abs()`: absolute value.
pub fn abs(x: &Tensor) -> Tensor {
    abs_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn abs_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.abs_ctx(ctx)
}

/// `sqrt()`: square root.
pub fn sqrt(x: &Tensor) -> Tensor {
    sqrt_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn sqrt_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.sqrt_ctx(ctx)
}

/// `rsqrt()`: 1 / sqrt(x).
pub fn rsqrt(x: &Tensor) -> Tensor {
    rsqrt_ctx(x, &mut x.ctx().borrow_mut())
}
#[doc(hidden)]
pub fn rsqrt_ctx(x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
    x.sqrt_ctx(ctx).reciprocal_ctx(ctx)
}
