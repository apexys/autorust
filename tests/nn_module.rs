//! `Module` parameter discovery (`nn.state.get_state_dict` / `get_parameters`
//! analogue): nested modules flatten to dotted names, a `Vec<Module>` indexes
//! like a Python list, and `into_params` materializes every parameter into a
//! buffer-backed leaf.
use autorust::tensor::nn::{self, BatchNorm, Conv2d, LayerNorm, Linear, Module};
use autorust::tensor::{Ctx, Tensor};
use autorust::uop::ops::Ops;

struct Net {
    c: Conv2d,
    bn: BatchNorm,
    l: Linear,
}

impl Net {
    fn new(ctx: &Ctx) -> Self {
        Net {
            c: Conv2d::new(ctx, 1, 8, &[3, 3], 1, &[(1, 1), (1, 1)], 1, 1, true),
            bn: BatchNorm::new(ctx, 8, 1e-5, true, true, 0.1),
            l: Linear::new(ctx, 8, 1, true),
        }
    }
}

impl Module for Net {
    fn named_params(&self) -> Box<dyn Iterator<Item = (String, Tensor)> + '_> {
        Box::new(
            nn::scoped("c", self.c.named_params())
                .chain(nn::scoped("bn", self.bn.named_params()))
                .chain(nn::scoped("l", self.l.named_params())),
        )
    }

    fn into_params(self) -> Self {
        Net {
            c: self.c.into_params(),
            bn: self.bn.into_params(),
            l: self.l.into_params(),
        }
    }
}

/// A parameter is a buffer-backed leaf: its storage base (movement stripped) is
/// a `PARAM`/`BUFFER` node, not a computed expression.
fn is_leaf(ctx: &Ctx, t: &Tensor) -> bool {
    let c = ctx.raw();
    matches!(c.op_of(&c.base(&t.uop())), Ops::Param | Ops::Buffer)
}

#[test]
fn nested_named_params_are_path_prefixed() {
    let ctx = Ctx::new();
    let net = Net::new(&ctx);
    let names: Vec<String> = net.named_params().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        vec![
            "c.weight",
            "c.bias",
            "bn.weight",
            "bn.bias", // running stats are buffers, not params
            "l.weight",
            "l.bias",
        ]
    );
    assert_eq!(net.params().count(), 6);
}

#[test]
fn vec_module_indexes_like_a_list() {
    let ctx = Ctx::new();
    let layers = vec![Linear::new(&ctx, 4, 5, true), Linear::new(&ctx, 5, 6, true)];
    let names: Vec<String> = layers.named_params().map(|(n, _)| n).collect();
    assert_eq!(names, vec!["0.weight", "0.bias", "1.weight", "1.bias"]);
    assert_eq!(layers.params().count(), 4);
}

#[test]
fn into_params_materializes_every_parameter() {
    let ctx = Ctx::new();
    let net = Net::new(&ctx).into_params();
    for (name, p) in net.named_params() {
        assert!(
            is_leaf(&ctx, &p),
            "param {name} is not a buffer-backed leaf"
        );
        assert!(p.device().is_some(), "param {name} has no device");
    }
}

#[test]
fn vec_into_params_materializes_every_element() {
    let ctx = Ctx::new();
    let layers = vec![Linear::new(&ctx, 4, 5, true), Linear::new(&ctx, 5, 6, true)].into_params();
    for (_name, p) in layers.named_params() {
        assert!(is_leaf(&ctx, &p));
    }
}

/// `LayerNorm.__call__` is `x * weight + bias` (tinygrad nn/__init__.py:258),
/// *not* `x + weight*bias`. The default affine params are `ones`/`zeros` (which
/// make the two indistinguishable), so use non-trivial weight/bias.
/// tinygrad oracle for this exact input/params: each row
/// `[-1.949471, -0.5, 5.898942]`.
#[test]
fn layernorm_applies_weight_then_bias() {
    use autorust::engine::session::Session;
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x =
        Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], None, None).reshape([2, 3]);
    let mut ln = LayerNorm::new(&ctx, vec![3], 1e-5, true);
    ln.weight = Some(Tensor::from_slice(&ctx, &[2.0f32, 3.0, 4.0], None, None));
    ln.bias = Some(Tensor::from_slice(&ctx, &[0.5f32, -0.5, 1.0], None, None));

    let y = ln.forward(&x).to_list::<f32>(&mut sess);
    for (a, b) in y
        .iter()
        .zip([-1.949471f32, -0.5, 5.898942, -1.949471, -0.5, 5.898942])
    {
        assert!(
            (a - b).abs() < 1e-4,
            "got {y:?}, want row [-1.949471, -0.5, 5.898942]"
        );
    }
}
