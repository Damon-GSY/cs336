# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Learning Reminders

- **Practice Questions → Declarative Knowledge**: 做练习题来建立概念理解
- **Write Assignments → Procedural Knowledge**: 写作业来建立动手能力
- **Minimum Knowledge Set**: 每个模块先问自己——最小知识集是什么？哪些是不可省略的核心？

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Extension Ideas

完成课程要求的基础实现后，可以探索以下扩展方向：

### Optimizer
- [ ] **Muon Optimizer**: 实现 [Muon](https://arxiv.org/abs/2502.16982) optimizer（momentum + orthogonalization），对比 AdamW 在相同超参下的收敛曲线

### Tokenizer & Distillation
- [ ] **Full-vocab On-policy Distillation**: 尝试修改词表大小，做一个 full-vocabulary 的 on-policy distillation，观察词表变化对模型质量的影响

### Attention Variants
- [ ] **Hybrid Attention**: 混合 full attention 与 sparse attention
- [ ] **Sliding Window Attention**: 实现固定窗口的局部注意力
- [ ] **CSA (Continuous Streaming Attention)**: 实现 streaming 场景下的连续注意力机制
- [ ] **DSA (Dynamic Sparse Attention)**: 实现动态稀疏注意力，根据输入内容决定注意力模式

### Multi-token Prediction
- [ ] **Multi-token Prediction Head**: 在标准 next-token prediction 之外，增加同时预测未来多个 token 的 head，对比单 token 预测的训练动态和生成质量
