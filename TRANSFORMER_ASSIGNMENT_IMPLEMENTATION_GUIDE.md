# Transformer 作业实现参考

这份文档按两个视角整理：

- 推荐实现顺序：按依赖关系来写，最省时间
- 测试顺序：按测试文件里的实际执行顺序来对照排查

核心实现入口在 [tests/adapters.py](./tests/adapters.py)。

## 1. 总览

如果你想最稳地推进，建议按下面顺序实现：

1. `run_linear`
2. `run_embedding`
3. `run_silu`
4. `run_softmax`
5. `run_cross_entropy`
6. `run_swiglu`
7. `run_rmsnorm`

8. `run_scaled_dot_product_attention`
9. `run_rope`
10. `run_multihead_self_attention`
11. `run_multihead_self_attention_with_rope`
12. `run_transformer_block`
13. `run_transformer_lm`

14. `run_get_batch`
15. `run_gradient_clipping`
16. `get_adamw_cls`
17. `run_get_lr_cosine_schedule`
18. `run_save_checkpoint`
19. `run_load_checkpoint`

原因很简单：

- 前 7 个是基础算子
- 中间 6 个是 Transformer 本体
- 最后 6 个是训练配套工具

## 2. 推荐实现顺序

### Step 1: `run_linear`

位置：`tests/adapters.py:15`

要做什么：

- 实现线性层前向
- 输入形状是 `(..., d_in)`
- 权重形状是 `(d_out, d_in)`
- 输出形状是 `(..., d_out)`

核心公式：

```python
out = in_features @ weights.T
```

注意：

- 权重要转置
- 不需要 bias

对应测试：

- `tests/test_model.py::test_linear`

---

### Step 2: `run_embedding`

位置：`tests/adapters.py:37`

要做什么：

- 从 embedding table 里按 token id 取向量
- 权重形状是 `(vocab_size, d_model)`
- 输入 token id 形状是 `(...)`
- 输出形状是 `(..., d_model)`

核心写法：

```python
out = weights[token_ids]
```

注意：

- 不需要自己写 one-hot

对应测试：

- `tests/test_model.py::test_embedding`

---

### Step 3: `run_silu`

位置：`tests/adapters.py:386`

要做什么：

- 实现 SiLU 激活函数

核心公式：

```python
out = x * torch.sigmoid(x)
```

对应测试：

- `tests/test_model.py::test_silu_matches_pytorch`

---

### Step 4: `run_softmax`

位置：`tests/adapters.py:423`

要做什么：

- 在指定维度做数值稳定的 softmax

推荐写法：

```python
shifted = in_features - in_features.amax(dim=dim, keepdim=True)
exp_x = shifted.exp()
out = exp_x / exp_x.sum(dim=dim, keepdim=True)
```

注意：

- 先减最大值，避免溢出

对应测试：

- `tests/test_nn_utils.py::test_softmax_matches_pytorch`

---

### Step 5: `run_cross_entropy`

位置：`tests/adapters.py:439`

要做什么：

- 对 logits 和 targets 计算平均交叉熵

推荐公式：

```python
log_denom = inputs.logsumexp(dim=-1)
gold = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
loss = (log_denom - gold).mean()
```

注意：

- 不要先做 softmax 再取 log
- `targets` 是类别 id，不是 one-hot

对应测试：

- `tests/test_nn_utils.py::test_cross_entropy`

---

### Step 6: `run_swiglu`

位置：`tests/adapters.py:59`

要做什么：

- 实现 SwiGLU 前馈层

公式：

```python
w1x = in_features @ w1_weight.T
w3x = in_features @ w3_weight.T
hidden = silu(w1x) * w3x
out = hidden @ w2_weight.T
```

shape：

- `w1_weight`: `(d_ff, d_model)`
- `w2_weight`: `(d_model, d_ff)`
- `w3_weight`: `(d_ff, d_model)`

注意：

- `w2` 也是按 `(out, in)` 存的，所以最后要转置

对应测试：

- `tests/test_model.py::test_swiglu`

---

### Step 7: `run_rmsnorm`

位置：`tests/adapters.py:363`

要做什么：

- 实现 RMSNorm

公式：

```python
rms = torch.sqrt(in_features.pow(2).mean(dim=-1, keepdim=True) + eps)
normed = in_features / rms
out = normed * weights
```

注意：

- RMSNorm 不减均值
- `weights` 形状是 `(d_model,)`，按最后一维广播

对应测试：

- `tests/test_model.py::test_rmsnorm`

---

### Step 8: `run_scaled_dot_product_attention`

位置：`tests/adapters.py:91`

要做什么：

- 实现缩放点积注意力

公式：

```python
scores = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
probs = softmax(scores, dim=-1)
out = probs @ V
```

注意：

- 要支持 3D 和 4D 输入
- `mask` 形状是 `(..., queries, keys)`
- 这份作业里通常把 `True` 当作可见位置，所以填 `-inf` 的是 `~mask`

对应测试：

- `tests/test_model.py::test_scaled_dot_product_attention`
- `tests/test_model.py::test_4d_scaled_dot_product_attention`

---

### Step 9: `run_rope`

位置：`tests/adapters.py:186`

要做什么：

- 对 query 或 key 做 RoPE

实现要点：

1. 把最后一维按偶数位和奇数位拆开
2. 生成每个位置、每个偶数通道对应的角度
3. 用二维旋转公式混合偶数位和奇数位
4. 再交错拼回原维度

参考骨架：

```python
x_even = x[..., 0::2]
x_odd = x[..., 1::2]

inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=x.device, dtype=x.dtype) / d_k))
angles = token_positions[..., None].to(x.dtype) * inv_freq
cos = torch.cos(angles)
sin = torch.sin(angles)

y_even = x_even * cos - x_odd * sin
y_odd = x_even * sin + x_odd * cos
```

注意：

- RoPE 是偶数位和奇数位成对旋转，不是前半段和后半段
- `token_positions` 可能是 `(seq,)`，也可能是 `(batch, seq)`，广播要处理好
- `d_k` 必须是偶数

对应测试：

- `tests/test_model.py::test_rope`

---

### Step 10: `run_multihead_self_attention`

位置：`tests/adapters.py:112`

要做什么：

- 一次性做所有 heads 的 QKV 投影
- 拆 head
- 做 causal self-attention
- 合并 heads
- 过 output projection

推荐顺序：

```python
B, T, _ = in_features.shape
H = num_heads
Dh = d_model // H

q = in_features @ q_proj_weight.T
k = in_features @ k_proj_weight.T
v = in_features @ v_proj_weight.T

q = q.reshape(B, T, H, Dh).transpose(1, 2)
k = k.reshape(B, T, H, Dh).transpose(1, 2)
v = v.reshape(B, T, H, Dh).transpose(1, 2)

causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=in_features.device))
attn_out = run_scaled_dot_product_attention(q, k, v, causal_mask)

attn_out = attn_out.transpose(1, 2).reshape(B, T, d_model)
out = attn_out @ o_proj_weight.T
```

注意：

- 多头拆分之后的形状应是 `(B, H, T, Dh)`
- 这里一般默认 causal mask
- 如果你这里不过，先检查是不是漏了 causal mask

对应测试：

- `tests/test_model.py::test_multihead_self_attention`

---

### Step 11: `run_multihead_self_attention_with_rope`

位置：`tests/adapters.py:146`

要做什么：

- 在上一步基础上，对 `q` 和 `k` 做 RoPE

推荐顺序：

1. 先做 QKV 线性投影
2. reshape 成 `(B, H, T, Dh)`
3. 对 `q`、`k` 调 `run_rope`
4. 做 causal attention
5. concat heads
6. 过 output projection

注意：

- RoPE 的维度是每个 head 的维度 `Dh`
- `token_positions is None` 时，可以自己构造 `0..T-1`

对应测试：

- `tests/test_model.py::test_multihead_self_attention_with_rope`

---

### Step 12: `run_transformer_block`

位置：`tests/adapters.py:208`

要做什么：

- 组装一个 pre-norm Transformer block

推荐结构：

```python
h = in_features + mha_with_rope(rmsnorm1(in_features))
out = h + swiglu(rmsnorm2(h))
```

从 `weights` 里要取：

- `attn.q_proj.weight`
- `attn.k_proj.weight`
- `attn.v_proj.weight`
- `attn.output_proj.weight`
- `ln1.weight`
- `ffn.w1.weight`
- `ffn.w2.weight`
- `ffn.w3.weight`
- `ln2.weight`

注意：

- 这里是 pre-norm，不是 post-norm
- attention 这一层要用带 RoPE 的版本

对应测试：

- `tests/test_model.py::test_transformer_block`

---

### Step 13: `run_transformer_lm`

位置：`tests/adapters.py:281`

要做什么：

- 组装整个语言模型前向

推荐结构：

```python
x = token_embedding(in_indices)
for layer_idx in range(num_layers):
    x = transformer_block(layer_idx, x)
x = final_rmsnorm(x)
logits = x @ lm_head_weight.T
```

要取的关键权重：

- `token_embeddings.weight`
- `layers.{i}.attn.*`
- `layers.{i}.ln1.weight`
- `layers.{i}.ffn.*`
- `layers.{i}.ln2.weight`
- `ln_final.weight`
- `lm_head.weight`

注意：

- `sequence_length` 可能小于 `context_length`
- token 位置只需要按当前输入长度构造

对应测试：

- `tests/test_model.py::test_transformer_lm`
- `tests/test_model.py::test_transformer_lm_truncated_input`

---

### Step 14: `run_get_batch`

位置：`tests/adapters.py:400`

要做什么：

- 从一维 token 序列中随机采样训练样本

目标：

- `x.shape == (batch_size, context_length)`
- `y.shape == (batch_size, context_length)`
- `y = x` 向右平移一位

思路：

1. 随机采样起点 `start`
2. `x = dataset[start : start + context_length]`
3. `y = dataset[start + 1 : start + context_length + 1]`
4. 拼成 batch 并搬到目标 device

注意：

- 起点最大只能到 `len(dataset) - context_length - 1`
- 测试会检查采样是否基本均匀

对应测试：

- `tests/test_data.py::test_get_batch`

---

### Step 15: `run_gradient_clipping`

位置：`tests/adapters.py:457`

要做什么：

- 把所有存在梯度的参数，按总 L2 norm 做裁剪

思路：

```python
grads = [p.grad for p in parameters if p.grad is not None]
total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
clip_coef = max_l2_norm / (total_norm + 1e-6)
if clip_coef < 1:
    for g in grads:
        g.mul_(clip_coef)
```

注意：

- 要跳过 `grad is None` 的参数
- 原地修改梯度

对应测试：

- `tests/test_nn_utils.py::test_gradient_clipping`

---

### Step 16: `get_adamw_cls`

位置：`tests/adapters.py:469`

要做什么：

- 返回一个实现 AdamW 的优化器类

最低要求：

- 继承 `torch.optim.Optimizer`
- 支持 `lr`、`weight_decay`、`betas`、`eps`
- 维护 `step`、`exp_avg`、`exp_avg_sq`

标准更新逻辑：

1. 读参数梯度
2. 更新一阶矩
3. 更新二阶矩
4. 做 bias correction
5. 做 decoupled weight decay
6. 做参数更新

注意：

- 这是 AdamW，不是 Adam
- weight decay 应该和梯度更新解耦

对应测试：

- `tests/test_optimizer.py::test_adamw`
- `tests/test_serialization.py::test_checkpointing`

---

### Step 17: `run_get_lr_cosine_schedule`

位置：`tests/adapters.py:476`

要做什么：

- 实现线性 warmup + cosine decay 学习率调度

推荐逻辑：

```python
if it < warmup_iters:
    return max_learning_rate * it / warmup_iters
if it > cosine_cycle_iters:
    return min_learning_rate

progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
```

注意：

- 这里按测试给出的边界条件来写

对应测试：

- `tests/test_optimizer.py::test_get_lr_cosine_schedule`

---

### Step 18: `run_save_checkpoint`

位置：`tests/adapters.py:504`

要做什么：

- 保存模型状态、优化器状态、iteration

推荐结构：

```python
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    },
    out,
)
```

对应测试：

- `tests/test_serialization.py::test_checkpointing`

---

### Step 19: `run_load_checkpoint`

位置：`tests/adapters.py:523`

要做什么：

- 读回 checkpoint
- 恢复 model 和 optimizer
- 返回 iteration

推荐结构：

```python
ckpt = torch.load(src, map_location="cpu")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
return ckpt["iteration"]
```

对应测试：

- `tests/test_serialization.py::test_checkpointing`

## 3. 按测试顺序对照

如果你想严格按测试文件的执行顺序来实现，对照下面这张表。

### `tests/test_model.py`

1. `test_linear` -> `run_linear`
2. `test_embedding` -> `run_embedding`
3. `test_swiglu` -> `run_swiglu`
4. `test_scaled_dot_product_attention` -> `run_scaled_dot_product_attention`
5. `test_4d_scaled_dot_product_attention` -> `run_scaled_dot_product_attention`
6. `test_multihead_self_attention` -> `run_multihead_self_attention`
7. `test_multihead_self_attention_with_rope` -> `run_multihead_self_attention_with_rope`
8. `test_transformer_lm` -> `run_transformer_lm`
9. `test_transformer_lm_truncated_input` -> `run_transformer_lm`
10. `test_transformer_block` -> `run_transformer_block`
11. `test_rmsnorm` -> `run_rmsnorm`
12. `test_rope` -> `run_rope`
13. `test_silu_matches_pytorch` -> `run_silu`

### `tests/test_nn_utils.py`

1. `test_softmax_matches_pytorch` -> `run_softmax`
2. `test_cross_entropy` -> `run_cross_entropy`
3. `test_gradient_clipping` -> `run_gradient_clipping`

### `tests/test_data.py`

1. `test_get_batch` -> `run_get_batch`

### `tests/test_optimizer.py`

1. `test_adamw` -> `get_adamw_cls`
2. `test_get_lr_cosine_schedule` -> `run_get_lr_cosine_schedule`

### `tests/test_serialization.py`

1. `test_checkpointing` -> `get_adamw_cls`
2. `test_checkpointing` -> `run_save_checkpoint`
3. `test_checkpointing` -> `run_load_checkpoint`

## 4. 最容易写错的点

### 4.1 线性层权重方向

这份作业里基本都存成：

- `weights.shape == (out_dim, in_dim)`

所以前向一般写：

```python
x @ weights.T
```

---

### 4.2 Attention 的 mask 语义

通常这里 `True` 表示这个位置可见。

所以屏蔽时更常见的是：

```python
scores.masked_fill(~mask, float("-inf"))
```

---

### 4.3 Multi-head attention 默认要 causal

虽然 `run_multihead_self_attention` 的参数里没有显式传 mask，但它通常对应语言模型自注意力，所以建议默认加下三角 mask。

---

### 4.4 RoPE 的旋转方式

RoPE 不是：

- 前半段和后半段互转

而是：

- 偶数位和奇数位成对旋转

---

### 4.5 Transformer Block 是 pre-norm

正确思路：

```python
x = x + attn(rmsnorm(x))
x = x + ffn(rmsnorm(x))
```

不是：

```python
x = rmsnorm(x + attn(x))
```

## 5. 推荐调试顺序

建议你每写完一块就只跑相关测试，不要一上来就全跑。

### 第一轮：基础层

```bash
pytest tests/test_model.py -k "linear or embedding or swiglu or rmsnorm or silu"
pytest tests/test_nn_utils.py -k "softmax or cross_entropy"
```

### 第二轮：attention

```bash
pytest tests/test_model.py -k "scaled_dot_product_attention or rope or multihead"
```

### 第三轮：整块模型

```bash
pytest tests/test_model.py -k "transformer_block or transformer_lm"
```

### 第四轮：训练工具

```bash
pytest tests/test_data.py
pytest tests/test_nn_utils.py -k "gradient_clipping"
pytest tests/test_optimizer.py
pytest tests/test_serialization.py
```

### 最后

```bash
pytest
```

## 6. 最短完成路线

如果你只想尽快做完，按下面走：

1. `run_linear`
2. `run_embedding`
3. `run_silu`
4. `run_swiglu`
5. `run_rmsnorm`
6. `run_softmax`
7. `run_cross_entropy`
8. `run_scaled_dot_product_attention`
9. `run_rope`
10. `run_multihead_self_attention`
11. `run_multihead_self_attention_with_rope`
12. `run_transformer_block`
13. `run_transformer_lm`
14. `run_get_batch`
15. `run_gradient_clipping`
16. `get_adamw_cls`
17. `run_get_lr_cosine_schedule`
18. `run_save_checkpoint`
19. `run_load_checkpoint`

这条路线基本符合依赖关系，返工最少。
