# CS336 Assignment1 的学习记录

这是我在学习 CS336 Assignment 1 过程中完成的项目，使用 Python 和 PyTorch 实现了一个小型 Transformer 语言模型，并在 TinyStories 上进行了训练和文本生成。

我想通过这个项目弄清楚：一个模型怎样从文本中学习，又怎样接着一句话继续生成。于是从 tokenizer 开始，逐步完成模型结构、优化器、训练循环和生成过程，最后把这些部分连接起来。

## 我做了什么

| 部分 | 内容 |
| --- | --- |
| Tokenizer | BPE 训练、特殊 token 处理、编码与解码、词表和 merge 规则保存 |
| 数据准备 | 将文本编码为 token ID，保存为二进制文件，通过 memmap 读取 |
| Transformer | Embedding、因果多头注意力、RoPE、RMSNorm、SwiGLU 和残差连接 |
| 训练 | 交叉熵、AdamW、梯度裁剪、warmup、余弦学习率调度和 checkpoint |
| 文本生成 | 加载训练权重，使用 temperature 和 top-p 采样，逐步生成文本 |
| 优化尝试 | 为 BPE 训练引入反向索引和增量计数，减少重复扫描 |

## 从单个模块到完整模型

一开始，单独看每个模块时，感觉它们都能理解，但真正写训练脚本时才发现，知道一个模块的公式和知道它怎样参与训练，是两回事。

例如，我最初会混淆模型、模型参数和 logits，也不清楚反向传播得到的梯度保存在哪里、优化器怎样使用这些梯度。把训练循环组装起来以后，这些概念才逐渐对应到具体的对象和操作上。

我现在理解的训练过程是：

```text
原始文本 → tokenizer → token IDs
                         ↓
                输入片段与下一token标签
                         ↓
                  Transformer预测
                         ↓
                   计算loss
                         ↓
              反向传播 → 梯度裁剪 → 参数更新
                         ↓
                  保存checkpoint
```

在这个过程中，我也逐渐分清了两种“训练”：BPE 训练得到词表和合并规则；语言模型训练则学习模型权重。准备验证文本时，使用已有 tokenizer 编码即可，不需要重新训练一套 tokenizer。

## 训练完成后的实际体验

完成训练并加载保存的 checkpoint 后，我开始输入 prompt，观察模型实际生成的文本。相比训练时不断打印的数字，看到模型接着输入继续写下去，让我更直观地感受到前面实现的各个模块已经连接起来了。

最明显的感受是：它更像一个故事续写模型。我最初会期待类似 ChatGPT 的一问一答，但实际使用后发现，输入一句话时，模型更倾向于把它当作一段文本的开头继续写。

这让我把训练目标和使用体验联系了起来：训练时，模型学习的是根据已有内容预测下一个 token；而 TinyStories 提供的主要是故事文本。因此，完成语言模型训练，并不意味着它就具备了遵循指令、回答问题的聊天能力。

## 训练实践

我先用较短的训练尝试不同学习率，再进行了一轮较长训练，并保存权重用于生成。下面是这轮训练使用的主要配置：

| 配置 | 数值 |
| --- | --- |
| 数据集 | TinyStories |
| 词表大小 | 10,000 |
| 上下文长度 | 256 |
| 层数 | 4 |
| 隐藏维度 | 512 |
| 注意力头数 | 16 |
| 前馈层维度 | 1,344 |
| Batch size | 8 |
| 训练步数 | 19,532 |
| 精度与设备 | float32 / Apple Silicon MPS |

这轮训练累计处理约 4,000 万个 token 位置。早期的实验记录主要保存在控制台中，图表和生成样例还需要进一步整理。

调参时有一次让我印象比较深：短实验和正式训练用了相同的峰值学习率，但跑到相同步数时，结果并不一样。后来结合学习率调度才理解，总训练步数改变后，warmup 和衰减过程也变了，“峰值学习率相同”并不意味着每一步的学习率都相同。

## 算法优化

跑通基本流程后，我也针对 tokenizer 中的重复计算做了一些改进：

- **BPE 训练**：原版每轮 merge 都重新扫描全部 pre-token。优化后通过 `pair → pre-token` 反向索引定位受影响的片段，再减去旧 pair 的贡献、加入新贡献，增量维护计数。
- **文本编码 encode**：提前建立 pair 的合并优先级表，编码时直接查表；同时缓存每个 pre-token 的编码结果，遇到重复片段时直接复用。优先级表和缓存保存在 tokenizer 实例中，避免每次调用 encode 都重新建立。

这两处改动让我更具体地理解了如何通过查表、缓存和增量更新减少重复工作，也需要留意额外的内存占用和状态维护。

## 文件说明

```text
assignment1-basics/
├── cs336_basics/                              # 核心实现
│   ├── tokenizer.py                           # BPE 训练与文本编解码
│   ├── train_tokenizer.py                     # 训练并保存 TinyStories tokenizer
│   ├── linear_embedding_module.py             # Linear 与 Embedding 基础模块
│   ├── normalization.py                       # RMSNorm 等归一化模块
│   ├── transformer_modules.py                 # RoPE、注意力、SwiGLU 与 TransformerLM
│   ├── optimizer.py                           # AdamW、学习率调度与梯度裁剪
│   ├── training_utils.py                      # 交叉熵等训练辅助函数
│   ├── training_loop.py                       # 数据采样与 checkpoint 工具
│   ├── prepare_data.py                        # 将训练文本编码为 token ID
│   ├── prepare_data_valid.py                  # 将验证文本编码为 token ID
│   ├── experiment_for_training_script.py      # 小规模训练与调参实验
│   ├── training_script.py                     # 正式训练入口
│   ├── generation.py                          # temperature 与 top-p 自回归生成
│   ├── generate_text.py                       # 加载 checkpoint 并生成文本
│   ├── pretokenization_example.py             # 预分词示例（课程提供）
│   └── __init__.py                            # Python 包初始化文件（课程提供）
├── models/                                    # 词表、merge 规则与模型配置
├── tests/                                     # 模块测试、测试数据与快照（课程提供）
├── README.md                                  # 项目学习记录
├── README_COURSE.md                           # 原始课程仓库说明（课程提供）
├── cs336_assignment1_basics.pdf               # Assignment 1 作业单（课程提供）
├── make_submission.sh                         # 测试与提交脚本（课程提供）
├── pyproject.toml                             # 项目依赖与工具配置
├── uv.lock                                    # uv 依赖锁定文件
└── LICENSE                                    # 开源许可证（课程提供）
```

准备好数据、tokenizer 和对应配置后，可以在项目根目录运行：

```bash
uv run python -m cs336_basics.training_script
uv run python -m cs336_basics.generate_text
```

生成脚本需要先配置已有 checkpoint 的路径。数据集、编码后的 token ID 文件和模型 checkpoint 不包含在源码中。

## 可以继续尝试的优化

- **BPE 训练**：测量目前各阶段的耗时，再考虑减少整张字典的复制和扫描，或使用堆维护候选 pair。
- **生成速度**：尝试 KV cache，复用历史 token 的 Key 和 Value，减少重复计算。
- **配置管理**：集中保存模型结构、训练参数和 tokenizer 路径，减少训练与生成脚本中重复设置的问题。
- **实验记录**：记录每次实验的配置、耗时和结果，让后续比较更方便。

## 我的学习方式

学习主要围绕视频课程以及官方作业单展开：阅读要求，理解各个模块的作用，再自己手动编写核心实现，逐步调试并把完整流程跑通。

过程中会借助 AI 解释 Python 语法和模型概念、讨论报错、审阅自己写的代码，以及梳理优化思路。代码由自己逐步编写和修改，没有让 AI 直接生成过代码。
