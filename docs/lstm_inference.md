# LSTM Inference

We follow the notation of [LSTMs on Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory).

## Standard Operations

### Variables

* `d`: dimension of the input vector
* `h`: dimension of the hidden / output vector
* `x_t` in `R^d`: input vector at time `t`
* `h_t` in `R^h`: hidden vector at time `t`
* `W` in `R^{4h x d}`: weight matrix applied to the input vector
* `U` in `R^{4h x h}`: weight matrix applied to the hidden vector
* `b` in `R^{4h}`: bias vector

### Operations

* `sigma_g`: sigmoid function
* `sigma_c`: hyperbolic tangent function

### Formulas

For each layer and timestep, run the matrix-vector multiplications

```
gates_t = W x_t + U h_{t - 1} + b
```

Decompose `gates_t` into four vectors of equal length:

```
f_t, i_t, o_t, c'_t = split(gates_t, 4)
```

> **Note:** The codebase and the .npy storage use the **iofc** gate order. This is 
> in contrast to the **fioc** order mentioned here or the **ifco** order used by 
> PyTorch. Weight matrices have to be brought into the correct form depending on the 
> context that they are used in.
>
> Ref: [ModelDef](../test/cpp/nvlstm_inf/model_def_test.cpp) and [Python Reference](../src/python/streaming_ref_model.py)

Update the cell state with elementwise operations recursively (`*` denotes elementwise multiplication):

```
c_t = sigma_g(f_t) * c_{t-1} + sigma_g(i_t) * sigma_c(c'_t)
```

Compute the hidden / output state as:

```
h_t = sigma_g(o_t) * sigma_c(c_t)
```

## Fast Inference with Precomputed Terms

For each layer, precompute

```
b_t = U h_{t - 1} + b
```

Fast inference procedure (run per layer):

```python

def fast_lstm_inference(W, x_t, b_t, c_t):
    # Matrix vector multiplication
    (f_t, i_t, o_t, c_prime_t) = split(gemv(W, x_t) + b_t, 4)
    c_t = sigma_g(f_t) * c_t + sigma_g(i_t) * sigma_c(c_prime_t)
    h_t = sigma_g(o_t) * sigma_c(c_t)
    return c_t, h_t
```

We iterate over all layers by passing `x_t = h_t` to the next layer.

### FLOPS

Note that matrix `W` has size `4h x d` for the first layer and size `4h x h` for all subsequent layers.

We ignore the contribution of the elementwise to number of FLOPS since the dominant term is the matrix multiplication.  

| Model  | Layers (`l`) | Input size (`d`) | Output size (`h`) | Weights `l == 0` | Weights `l > 0` | MFLOPS |
|--------|--------------|------------------|-------------------|------------------|-----------------|--------| 
| lstm s | 2            | 128              | 96                | 4 * 128 * 96     | 4 * 96 * 96     | 0.172  | 
| lstm m | 3            | 192              | 160               | 4 * 192 * 160    | 4 * 160 * 160   | 0.655  | 
| lstm l | 4            | 512              | 736               | 4 * 512 * 736    | 4 * 736 * 736   | 16.02  | 

Each weight in `W` corresponds to one `FMA` operation in the matrix-vector multiplication. The formula for computing the
total number of FLOPS is

```
2 * (4 * h * d + (l - 1) * 4 * h * h)
```