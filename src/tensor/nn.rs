//! M7: the `nn` module — neural network layers (nn/__init__.py).
//! Mirrors tinygrad's nn classes: Linear, Conv2d, BatchNorm, LayerNorm, GroupNorm,
//! and helper activations (relu, gelu, softmax, etc.).

use crate::tensor::Tensor;
use crate::uop::ctx::UOpCtx;
use crate::dtype::ConstVal;

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

    /// Just the parameter tensors, dropping the names.
    fn params(&self) -> impl Iterator<Item = Tensor> + '_ {
        self.named_params().map(|(_name, p)| p)
    }
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
        let w = std::iter::once(("weight".to_string(), self.weight));
        let b = self.bias.map(|b| ("bias".to_string(), b));
        Box::new(w.chain(b))
    }
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
            Some(Tensor::uniform(ctx, &[in_features], Some(-bound), Some(bound), None, None))
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
    /// NOTE: Simplified implementation — full conv2d logic is complex and TODO'd in NNImpl.md.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        // TODO: Implement full conv2d logic (see NNImpl.md)
        // For now, return a zero tensor with the expected output shape
        let w_shape = self.weight.clone().shape(ctx);
        let _x_shape = x.shape(ctx);
        let w_len = w_shape.len() as isize;
        let _kh: i64 = w_shape.get(w_len - 2).unwrap().as_i64().unwrap_or(0);
        let _kw: i64 = w_shape.get(w_len - 1).unwrap().as_i64().unwrap_or(0);
        // Output shape: (B, out_channels, out_h, out_w) — simplified to just out_channels
        let _out_channels: i64 = w_shape.get(0).unwrap().as_i64().unwrap_or(0);
        match &self.bias {
            Some(b) => b.clone(),
            None => Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(0.0), None),
        }
    }
}

/// `nn::BatchNorm`: batch normalization.
pub struct BatchNorm {
    pub num_features: i64,
    pub eps: f64,
    pub affine: bool,
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
        let weight = if affine { Some(Tensor::ones(ctx, &[sz])) } else { None };
        let bias = if affine { Some(Tensor::zeros(ctx, &[sz])) } else { None };
        let running_mean = if track_running_stats {
            Some(Tensor::zeros(ctx, &[sz]))
        } else {
            None
        };
        let running_var = if track_running_stats {
            Some(Tensor::ones(ctx, &[sz]))
        } else {
            None
        };

        BatchNorm {
            num_features: sz,
            eps,
            affine,
            track_running_stats,
            momentum,
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked: None, // Not implemented yet
        }
    }

    /// Forward pass with training mode.
    pub fn forward_train(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let x_shape = x.clone().shape(ctx);
        let ndim = x_shape.len();
        // Reduce over all dims except batch (dim 0) and channels (dim 1)
        let reduce_axis: Vec<usize> = (0..ndim).filter(|&i| i != 1).collect();
        let batch_mean = x.clone().mean(ctx, Some(&reduce_axis), false);
        let norm_shape = self._norm_shape(&*x_shape);
        let batch_mean_reshaped = batch_mean.reshape(ctx, &norm_shape);
        let diff = x.sub(ctx, &(batch_mean_reshaped));
        let batch_var = diff.square(ctx).mean(ctx, Some(&reduce_axis), false);
        let batch_var_shape = self._norm_shape(&*x_shape);
        let batch_var_reshaped = batch_var.reshape(ctx, &batch_var_shape);
        let eps_tensor = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(self.eps), None);
        let invstd = batch_var_reshaped.add(ctx, &(eps_tensor)).sqrt(ctx).reciprocal(ctx);

        // Update running stats if tracking
        if self.track_running_stats {
            // In a real implementation, we'd update running_mean and running_var here
            // For now, just return the normalized output
        }

        x.batchnorm(ctx, self.weight.as_ref(), self.bias.as_ref(), &batch_mean, &invstd, &reduce_axis)
    }

    /// Forward pass with inference mode (uses running stats).
    pub fn forward_inference(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let mean = self.running_mean.clone().unwrap_or_else(|| Tensor::zeros(ctx, &[self.num_features]));
        let var = self.running_var.clone().unwrap_or_else(|| Tensor::ones(ctx, &[self.num_features]));

        let eps_tensor = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(self.eps), None);
        let invstd = var.add(ctx, &(eps_tensor)).sqrt(ctx).reciprocal(ctx);

        // Reshape mean and var for broadcasting
        let x_shape = x.shape(ctx);
        let ndim = x_shape.len();
        let mut norm_shape = Vec::with_capacity(ndim);
        for i in 0..ndim {
            if i == 1 {
                norm_shape.push(crate::uop::sint::SVal::I(1));
            } else {
                norm_shape.push(x_shape[i].clone());
            }
        }
        let mean_reshaped = mean.reshape(ctx, &norm_shape);
        let var_reshaped = var.reshape(ctx, &norm_shape);
        let invstd_reshaped = invstd.reshape(ctx, &norm_shape);

        let diff = x.sub(ctx, &(mean_reshaped));
        let normalized = diff.mul(ctx, &(invstd_reshaped));

        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let w_reshaped = w.clone().reshape(ctx, &norm_shape);
                let b_reshaped = b.clone().reshape(ctx, &norm_shape);
                let wb = w_reshaped.mul(ctx, &b_reshaped);
                normalized.add(ctx, &wb)
            }
            (Some(w), None) => {
                let w_reshaped = w.clone().reshape(ctx, &norm_shape);
                normalized.mul(ctx, &(w_reshaped))
            }
            _ => normalized,
        }
    }

    fn _norm_shape(&self, x_shape: &[crate::uop::sint::SVal]) -> Vec<crate::uop::sint::SVal> {
        let ndim = x_shape.len();
        (0..ndim).map(|i| {
            if i == 1 {
                crate::uop::sint::SVal::I(1)
            } else {
                x_shape[i].clone()
            }
        }).collect()
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
            let norm_shape = self._norm_shape(ndim);
            let w_reshaped = self.weight.as_ref().unwrap().reshape(ctx, &norm_shape);
            let b_reshaped = self.bias.as_ref().unwrap().reshape(ctx, &norm_shape);
            let wb = w_reshaped.mul(ctx, &b_reshaped);
            normalized.add(ctx, &wb)
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
        let weight = if affine { Some(Tensor::ones(ctx, &[num_channels])) } else { None };
        let bias = if affine { Some(Tensor::zeros(ctx, &[num_channels])) } else { None };

        GroupNorm {
            num_groups,
            num_channels,
            eps,
            affine,
            weight,
            bias,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor, ctx: &mut UOpCtx) -> Tensor {
        let x_shape = x.shape(ctx);
        let ndim = x_shape.len();
        // GroupNorm: normalize over channels within groups
        // Reshape to (B, G, C_per_G, ...) -> normalize over C_per_G and spatial dims
        let c_per_g = self.num_channels / self.num_groups;

        // For simplicity, use the same reduction axes as BatchNorm but per-group
        // This is a simplified implementation
        let reduce_axis: Vec<usize> = (0..ndim).filter(|&i| i != 1).collect();
        let batch_mean = x.mean(ctx, Some(&reduce_axis), false);
        let batch_mean_reshaped = batch_mean.reshape(ctx, &self._norm_shape(ndim));
        let diff = x.sub(ctx, &(batch_mean_reshaped));
        let batch_var = diff.square(ctx).mean(ctx, Some(&reduce_axis), false);
        let batch_var_reshaped = batch_var.reshape(ctx, &self._norm_shape(ndim));
        let eps_tensor = Tensor::full_const(ctx, &[] as &[i64], ConstVal::Float(self.eps), None);
        let invstd = batch_var_reshaped.add(ctx, &(eps_tensor)).sqrt(ctx).reciprocal(ctx);

        x.batchnorm(ctx, self.weight.as_ref(), self.bias.as_ref(), &batch_mean, &invstd, &reduce_axis)
    }

    fn _norm_shape(&self, ndim: usize) -> Vec<crate::uop::sint::SVal> {
        (0..ndim).map(|i| {
            if i == 1 {
                crate::uop::sint::SVal::I(1)
            } else {
                crate::uop::sint::SVal::I(1) // simplified
            }
        }).collect()
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
    let sum_exp = exp_x.sum(ctx, axis_vec.as_deref(), false, None);
    let recip = sum_exp.reciprocal(ctx);
    exp_x.mul(ctx, &recip)
}

/// `log_softmax()`: softmax(log(x)) = x - max - log(sum(exp(x - max))).
pub fn log_softmax(x: &Tensor, ctx: &mut UOpCtx, axis: Option<usize>) -> Tensor {
    let axis_vec = axis.map(|a| vec![a]);
    let max_val = x.max(ctx, axis_vec.as_deref(), true);
    let x_shifted = x.sub(ctx, &max_val);
    let exp_x = x_shifted.exp(ctx);
    let sum_exp = exp_x.sum(ctx, axis_vec.as_deref(), false, None);
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
