# NN Module Implementation Status

## Completed
- **Tensor methods added:**
  - `Tensor::dot(w)`: Matrix multiply over last axis
  - `Tensor::linear(weight, bias)`: dot/mul + bias add (used by Linear layer)
  - `Tensor::batchnorm(weight, bias, mean, invstd, axis)`: Batch normalization
  - `Tensor::layernorm(axis, eps)`: Layer normalization
  - `Tensor::pool(kernel_size, stride, dilation)`: Sliding window pool (for conv2d internals)
  - `Tensor::uniform(*shape, low, high)`: Random uniform [low, high)
  - `Tensor::rand(*shape)`: Random uniform [0, 1)
  - `Tensor::tanh()`: Hyperbolic tangent
  - `Tensor::div(other)`: Elementwise division
  - `Tensor::square()`, `Tensor::sqrt()`, `Tensor::abs()`, etc.

- **nn module created** (`src/tensor/nn.rs`):
  - `nn::Linear`: Kaiming uniform init, `forward(x)` = `x.linear(w.T, bias)`
  - `nn::BatchNorm`: Training/inference modes with running stats (stub implementation)
  - `nn::LayerNorm`: Per-sample normalization
  - `nn::GroupNorm`: Group normalization
  - Activations: `relu`, `gelu`, `softmax`, `log_softmax`, `sigmoid`, `tanh`, `sin`, `cos`, `abs`, `sqrt`, `rsqrt`
  - `nn::Conv2d`: Stub implementation (returns bias or zero tensor)

## Compilation Status
- Many borrow checker and move errors remain
- Key issues:
  1. `Tensor` doesn't implement `Copy`, so must clone before multiple uses
  2. Methods like `shape()`, `mean()`, `reshape()` take `self` by value
  3. Activation functions need axis type fixes
  4. Some iterator/type mismatch issues in pool implementation

## Blocked: Conv2d Implementation

The `nn::Conv2d` forward pass requires implementing the full tinygrad conv2d logic:

```python
# Python conv2d (tinygrad/mixin/op.py:1566-1665):
def conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
    # 1. Pad input
    x = self._pad_constant(((0,0),)*(self.ndim-len(HW)) + flat_to_grouped(padding_), 0.0)
    # 2. Pool (im2col extraction)
    x = x._pool(HW, stride, dilation)
    # 3. Reshape for convolution
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW)
    x = x.permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])
    # 4. Multiply with weight
    ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW))
    # 5. Sum over reduction dims
    ret = ret.sum([-1-i for i in range(1+len(oyx))], keepdim=True)
    ret = ret.reshape(bs, cout, *oyx)
    # 6. Add bias
    return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))
```

## Next Steps

### Immediate (Fix Compilation)
1. Fix borrow checker errors in `nn.rs`:
   - Clone `Tensor` values before calling methods that take `self` by value
   - Use `x.clone()` before `x.shape()`, `x.mean()`, etc.
2. Fix activation function axis type issues in `nn.rs`
3. Fix pool implementation type mismatches in `movement.rs`

### After Compilation
4. Implement full Conv2d forward pass (see Python code above)
5. Add tests:
   - `test_nn_linear.py`: Test Linear forward pass vs tinygrad goldens
   - `test_nn_conv2d.py`: Test Conv2d forward pass vs tinygrad goldens
   - `test_nn_batchnorm.py`: Test BatchNorm training/inference
   - `test_nn_optim.py`: Test SGD/Adam optimizers with small networks
6. Implement optimizers (nn/optim.py):
   - `nn::SGD`: Stochastic gradient descent
   - `nn::Adam`: Adam optimizer
7. Implement state management (nn/state.py)
