# Tokenizer / BPE 工作计划

本文档只覆盖 Assignment 1 里 tokenizer / BPE 这一段工作，依据以下真实约束整理：

- `tests/adapters.py`
  - `run_train_bpe(...)`
  - `get_tokenizer(...)`
- `tests/test_train_bpe.py`
- `tests/test_tokenizer.py`
- `cs336_spring2025_assignment1_basics.pdf` 第 2.4–2.6 节
- `cs336_basics/pretokenization_example.py`

## 推荐构建顺序

1. 先做 `run_train_bpe`，固定训练输出契约。
2. 再做 `get_tokenizer`，能从 `vocab + merges` 构造 tokenizer。
3. 再做 `encode / decode` 基本回环。
4. 再补 special tokens 行为。
5. 最后做 `encode_iterable` 流式编码。
6. 做 tokenizer + BPE 联合验证。

这样可以先锁定训练产物，再消费这些产物构造运行时 tokenizer，最后处理流式和内存约束。

## 阶段 0：确认约束

目标：把 handout 和测试里的硬约束整理成实现清单。

需要确认：

- 初始词表是 byte-level UTF-8 bytes
- 使用 handout 给出的 GPT-2 regex pretokenization
- 不能跨 pre-token 边界 merge
- pair 频次相同时时，选字典序更大的 pair
- special tokens 要进 vocab，但不能跨它们 merge
- `decode` 要按 replacement 语义处理非法 UTF-8
- `encode_iterable` 要保持惰性、低内存

验证：

- 不写代码，先形成一页以内的实现检查表。

## 阶段 1：BPE 训练正确性基线

目标：先让 `run_train_bpe(...)` 在参考语料上输出正确结果。

实现重点：

- 先做正确、可复现的 byte-level BPE 训练流程
- 用 handout 指定的 regex 做 pretokenization
- 从基础 bytes + special tokens 构造初始 vocab
- 按确定顺序生成 merges，直到达到 `vocab_size`
- 返回：
  - `vocab: dict[int, bytes]`
  - `merges: list[tuple[bytes, bytes]]`

TDD 顺序：

1. `uv run pytest tests/test_train_bpe.py -k test_train_bpe`
2. 每次只修下一个失败断言
3. 优先保证 tie-break 和输出格式稳定

阶段完成标准：

- `test_train_bpe` 通过

## 阶段 2：BPE 训练强化

目标：补齐 special token 和性能约束。

实现重点：

- special tokens 必须进入 vocab
- merge 不能跨 special-token 分隔边界
- 按 handout 要求，pretokenization 前先按 special tokens 切分
- 在正确性稳定后，再优化 pair count / merge 更新路径

TDD 顺序：

1. `uv run pytest tests/test_train_bpe.py`
2. 先修 special token 相关失败
3. 最后处理 `speed` 测试

阶段完成标准：

- `test_train_bpe`
- `test_train_bpe_special_tokens`
- `test_train_bpe_speed`

## 阶段 3：Tokenizer 核心能力

目标：让 `get_tokenizer(...)` 返回可用 tokenizer，并通过 encode / decode 主路径测试。

实现重点：

- 从 `vocab`、`merges`、`special_tokens` 构造 tokenizer
- 实现 `encode(text) -> list[int]`
- 实现 `decode(ids) -> str`
- 编码时保持 special tokens 原子化

TDD 顺序：

1. 先过 empty / 单字符 / ASCII / Unicode 回环
2. 再对齐 GPT-2 参考 ID 行为
3. 再补 special token 保留场景

阶段完成标准：

- `tests/test_tokenizer.py` 中核心 roundtrip 测试通过
- 标准 encode/decode 的 GPT-2 对齐测试通过

## 阶段 4：流式编码

目标：实现 `encode_iterable(...)`，在不改变结果的前提下支持大文件低内存编码。

实现重点：

- 从 iterable 惰性地产出 token IDs
- 结果必须与整串 `encode(...)` 一致
- 避免 chunk 边界处的 special token / pre-token 错误
- 参考 `cs336_basics/pretokenization_example.py` 的 chunk boundary 思路

TDD 顺序：

1. 先跑 iterable roundtrip
2. 再跑 iterable 与参考 tokenizer 的 ID 一致性
3. 最后看 memory 相关测试

阶段完成标准：

- `test_encode_iterable_tinystories_sample_roundtrip`
- `test_encode_iterable_tinystories_matches_tiktoken`
- Linux 环境下的 memory test 可通过时通过

## 阶段 5：最终联调

验证顺序：

1. `uv run pytest tests/test_train_bpe.py`
2. `uv run pytest tests/test_tokenizer.py`
3. `uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py`

退出标准：

- BPE 训练结果与参考文件一致
- tokenizer 能正确处理 empty / ASCII / Unicode / fixture 文本
- special tokens 始终保持原子化
- 流式输出与非流式输出一致

## Ultrawork 执行原则

- 每一阶段都要小步推进、测试驱动
- 先保证确定性正确，再做优化
- 每过一个阶段再考虑局部整理
- `adapters.py` 只做胶水，不放实质逻辑
- 严格限制在 tokenizer / BPE 范围内，不扩散到 transformer 任务

## 原子提交策略

建议按阶段切 commit，每个 commit 都对应一个已验证里程碑。

1. `docs: add tokenizer/BPE execution plan`
2. `feat: implement baseline BPE training output`
3. `fix: handle special-token boundaries and BPE speed constraints`
4. `feat: implement tokenizer encode/decode core`
5. `fix: preserve special tokens during tokenization`
6. `feat: add streaming encode_iterable support`
7. `chore: finalize tokenizer/BPE verification cleanup`

提交规则：每个 commit 提交前，至少保证该阶段对应测试已经通过。
