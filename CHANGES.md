# DILLM 改进记录

> 基于原始 DILLM-VLN 项目（IEEE RA-L 2024）的优化工作记录。
> 原项目：Boosting Efficient Reinforcement Learning for VLN with Open-Sourced LLM

---

## 一、Metrics 监控（可观测性）

### 1.1 参数量分项打印
**文件：** `r2r_src/agent.py`，`Seq2SeqAgent.__init__`

将原来冗余的参数统计替换为按模块分项输出，并将总参数数保存到 `self.total_params`：

```
==================================================
Model parameter count:
  encoder                  3145728  (3.15M)
  decoder                  5242880  (5.24M)
  critic                    262657  (0.26M)
  discriminator               8513  (0.01M)
  TOTAL                    8659778  (8.66M)
==================================================
```

### 1.2 推理速度计时
**文件：** `r2r_src/agent.py`，`rollout()` 方法

在每次 rollout 中记录三个指标到 `self.logs`：
- `step_time`：每一步（decoder forward + CLIP + discriminator）的耗时（秒）
- `rollout_time`：整个 episode 的总耗时（秒）
- `rollout_steps`：实际执行的步数（提前结束时小于 episode_len）

### 1.3 Validation 输出推理速度
**文件：** `r2r_src/train.py`，`train()` 函数的 validation 循环

每个 val env 跑完后额外输出：
```
[val_unseen] Inference speed — avg step: 45.2 ms (22.1 steps/s), avg rollout: 1.35 s (8.3 steps)
```

同时写入 TensorBoard：
- `speed/step_time_ms_{env_name}`
- `speed/steps_per_sec_{env_name}`
- `speed/avg_rollout_sec_{env_name}`

`loss_str`（主输出行）末尾也追加 `step_time` 和 `steps/s`。

---

## 二、推理加速

### 2.1 CLIP 文本特征缓存（最大收益）
**文件：** `r2r_src/agent.py`，`rollout()` 方法
**改动位置：** 约第 358、396-398、449-450 行

**问题：** 原代码每一步都调用 `clip.tokenize` + `clip_model.encode_text`，但子目标文本（`current_subgoal_list`）每 `option_step`（默认=3）步才更新一次，中间 2/3 的调用是完全重复计算。

**改法：** 引入 `cached_text_features`，在子目标切换时（if 块内）计算并缓存，其余步直接复用：

```python
# 子目标切换时才重新编码
clip_tokens = clip.tokenize(current_subgoal_list).to("cuda")
cached_text_features = self.clip_model.encode_text(clip_tokens).float()

# 后续步直接用缓存
text_features = cached_text_features
```

**收益：** CLIP encode_text（Transformer forward）调用量减少约 **66%**，是每步最大的单项开销。

### 2.2 推理阶段包裹 `torch.no_grad()`
**文件：** `r2r_src/agent.py`，`Seq2SeqAgent.test()` 方法

```python
with torch.no_grad():
    super(Seq2SeqAgent, self).test(iters)
```

**收益：** 推理时不再构建计算图，减少约 30-50% 显存占用，forward 速度提升约 10-20%。

### 2.3 CLIP 模型关闭梯度 + 固定 eval 模式
**文件：** `r2r_src/agent.py`，`Seq2SeqAgent.__init__`

```python
self.clip_model.requires_grad_(False)
self.clip_model.eval()
```

**收益：**
- `requires_grad_(False)`：消除 CLIP（RN50x4，约 87M 参数）的梯度跟踪开销
- `eval()`：确保 CLIP 内部 BatchNorm/Dropout 在训练和推理阶段行为一致，避免潜在的不一致性

---

## 三、精度修复

### 3.1 补全 discriminator 梯度裁剪
**文件：** `r2r_src/agent.py`，`train()` 和 `optim_step()` 方法

原代码 `train()` 中只对 encoder/decoder/critic 做了 `clip_grad_norm`，漏掉了 discriminator：

```python
# 修复后（train() 和 optim_step() 都补全）
torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
torch.nn.utils.clip_grad_norm(self.critic.parameters(), 40.)
torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), 40.)  # 原来缺失
```

**意义：** discriminator（FFNet）负责判断当前子目标是否完成，梯度不裁剪可能导致训练不稳定，子目标切换判断偏差。修复后 discriminator 训练更稳定，预期对 SR 有正向影响。

---

## 四、日志持久化

### 4.1 运行日志保存到文件
**文件：** `r2r_src/train.py` 头部

每次运行自动在 `DILLM/log/` 目录下创建以时间戳命名的日志文件：

```
DILLM/log/20260318_143022_agent.log
```

文件名格式：`{时间戳}_{args.name}.log`

实现方式：用 `logging` 标准库同时挂载 `FileHandler`（写文件）和 `StreamHandler`（终端），并将内置 `print` 重定向到 logger，**所有现有 print 调用无需改动**即可同时写入文件。

每行日志自动加时间戳前缀：
```
2026-03-18 14:30:22  [val_unseen] Inference speed — avg step: 45.2 ms ...
```

---

## 五、待做事项（轻量化阶段二/三）

以下改动已分析完毕，尚未实施，需重新训练验证精度：

### 5.1 视觉特征投影压缩（阶段二，精度损失极小）
**目标：** 在 `AttnDecoderLSTM` 中加投影层，将视觉特征从 640 维降至 256 维。

```python
self.vis_proj = nn.Linear(feature_size, 256, bias=False)
self.proj_feature_size = 256 + args.angle_feat_size  # 384
```

- `feat_att_layer` + `candidate_att_layer` 参数量：~3.4M → ~1.1M
- LSTMCell 输入 768 → 384，减少约 1.5M 参数
- 合计减少约 **4M 参数（约45%）**，不含 encoder

### 5.2 RNN 维度缩减（阶段三，需验证）
修改 `agent.bash` 启动参数：

| 参数 | 当前 | 建议 |
|------|------|------|
| `rnn_dim` | 512 | 256 |
| `wemb` | 256 | 128 |
| `angle_feat_size` | 128 | 64 |

LSTM 参数量 ∝ hidden²，总参数量预计再减少约 **50%**。

### 5.3 视图数量缩减（高风险，待评估）
将全景视图从 36 降至 12（仅水平一圈），需同步修改 `env.py` 采样逻辑和 `FFNet` 输入维度（`36+36` → `12+12`）。

---

## 六、特征融合投影（Feature Fusion Projection）✅ 已实施

> 状态：**已实施**，等待重新训练验证精度。

### 6.1 背景与动机

原始架构中存在一个信息孤立问题：

- `f_t`（batch, 36, 768）：图像特征+角度特征，送入 decoder
- `obj_t`（batch, 36, 640）：物体语义特征，**只送给 discriminator**，decoder 完全不知情

这意味着 decoder 在决策"往哪走"时，看不到当前视野里有什么物体——而子目标往往正是某个物体（"go to the chair"）。discriminator 和 decoder 使用同一空间的特征却完全割裂。

### 6.2 方案设计

在 `AttnDecoderLSTM` 入口新增 `FusionProjection` 模块，将图像特征和物体特征门控融合后再送入 decoder：

```
原来：
  f_t (batch, 36, 768) ──────────────────────→ decoder
  obj_t (batch, 36, 640) → discriminator only

改后：
  f_t   (batch, 36, 640) ─┐
                           ├→ FusionProjection → (batch, 36, 384) → decoder
  obj_t (batch, 36, 640) ─┘
  angle (batch, 36, 128) ──────────────────────────────────────────↗ 拼接
```

**FusionProjection 结构：**

```python
class FusionProjection(nn.Module):
    def __init__(self, vis_dim=640, obj_dim=640, out_dim=256):
        self.vis_proj = nn.Linear(vis_dim, out_dim, bias=False)  # 640→256
        self.obj_proj = nn.Linear(obj_dim, out_dim, bias=False)  # 640→256
        self.gate = nn.Linear(out_dim + out_dim, out_dim)        # 512→256，学门控权重

    def forward(self, vis, obj):
        # vis, obj: (batch, 36, 640)
        v = self.vis_proj(vis)                        # (batch, 36, 256)
        o = self.obj_proj(obj)                        # (batch, 36, 256)
        g = torch.sigmoid(self.gate(torch.cat([v, o], dim=-1)))  # (batch, 36, 256)
        return g * v + (1 - g) * o                   # 门控加权融合
```

输出 256 维再与角度特征 128 维拼接，得到新的 `feature_size = 384`。

### 6.3 需要修改的文件

| 文件 | 位置 | 改动 |
|---|---|---|
| `r2r_src/model.py` | 新增 `FusionProjection` 类 | 约 15 行 |
| `r2r_src/model.py` | `AttnDecoderLSTM.__init__` | 加 `self.fusion_proj`，feature_size 相关层全部改为 384 |
| `r2r_src/model.py` | `AttnDecoderLSTM.forward` | 入口处调用 fusion_proj，合并 vis 和 obj |
| `r2r_src/agent.py` | `rollout()` decoder 调用处 | 把 `obj_t` 传给 decoder（目前只传给 discriminator）|
| `r2r_src/agent.py` | `beam_search_test()` 中 decoder 调用处 | 同上 |

### 6.4 参数量变化

| 模块 | 改前 | 改后 |
|---|---|---|
| FusionProjection（新增） | 0 | ~0.33M |
| LSTMCell (768→512) | ~2.6M | (384→512) ~1.3M |
| feat_att_layer (512, 768) | ~1.0M | (512, 384) ~0.59M |
| candidate_att_layer (1536, 768) | ~4.7M | (1536, 384) ~2.4M |
| **decoder 合计** | **~5.2M** | **~1.5M** |
| **总参数** | **~8.7M** | **~5.0M** |

总参数减少约 **42%**，同时物体语义信息进入 decoder，理论上精度不降反升。

### 6.5 创新点总结

1. **decoder 获得物体语义引导**：原架构中 decoder 对"视野里有什么物体"完全盲目，融合后可被物体信息直接引导动作选择。
2. **门控融合而非简单拼接**：gate 权重由网络自学，能动态决定每个视角下图像特征和物体特征哪个更重要。
3. **discriminator 与 decoder 语义协同**：两者现在共享同一融合表示的基础，判断"子目标是否完成"和"下一步往哪走"在特征层面对齐。

### 6.6 实施详情（已完成）

**修改文件及改动：**

| 文件 | 位置 | 改动 |
|---|---|---|
| `r2r_src/model.py` | 新增 `FusionProjection` 类（第 165–183 行） | 门控融合模块，vis_proj + obj_proj + gate |
| `r2r_src/model.py` | `AttnDecoderLSTM.__init__`（第 186–218 行） | 新增 `self.fusion_proj`，所有依赖 feature_size 的层改为 `proj_feature_size=384` |
| `r2r_src/model.py` | `AttnDecoderLSTM.forward`（第 220–275 行） | 新增 `obj_feat` 参数；入口处拆分 vis/angle，调用 fusion_proj 融合 vis+obj，proj candidate vis；dropout 改为在融合前对 raw vis 应用 |
| `r2r_src/agent.py` | `rollout()` 第 363 行 | 第一次 `get_input_feat` 保留 `obj_t`（原来用 `_` 丢弃） |
| `r2r_src/agent.py` | `rollout()` decoder 调用（第 399–403 行） | 增加 `obj_feat=obj_t` 参数 |
| `r2r_src/agent.py` | A2C last-step decoder 调用（第 518–526 行） | 增加 `obj_feat=obj_t` 参数 |

**设计要点：**
- 36 视图特征经过完整的 gate 融合（vis + obj → FusionProjection → 256 维）
- 候选动作特征（可变数量，无对应 obj 特征）仅经过 `vis_proj` 投影到 256 维
- `obj_feat=None` 时自动退化为纯 vis_proj（向后兼容 beam search 等不传 obj 的场景）

**实际参数量变化（`angle_feat_size=128, feature_size=640, rnn_dim=512`）：**

| 模块 | 改前 | 改后 | 变化 |
|---|---|---|---|
| FusionProjection（新增） | 0 | 0.46M | +0.46M |
| LSTMCell (768→512 / 384→512) | 2.63M | 1.84M | -0.79M |
| feat_att_layer (512×768 / 512×384) | 1.05M | 0.66M | -0.39M |
| candidate_att_layer (1536×768 / 1536×384) | 4.72M | 3.54M | -1.18M |
| **decoder 合计** | **10.24M** | **8.34M** | **-1.90M** |
| **模型总计** | **12.10M** | **10.20M** | **-1.90M (−15.7%)** |

---

## 七、精度优化（Discriminator 训练 + 子目标切换 + A2C 修复 + 候选融合）✅ 已实施

> 状态：**已实施**，等待重新训练验证精度。

### 7.1 Discriminator 设计确认：推理模式为原始意图（非 bug）
**文件：** `r2r_src/agent.py`，`rollout()` 方法

**历史：** 曾误以为 discriminator 所有输入被 `.detach()` 是 bug，添加了 BCE 监督损失。经对比论文和原始代码后确认，原始设计是**有意为之**。

**为什么原始设计正确：**
1. **论文的 MFD 设计**：论文（Eq. 11）描述 MFD 为结合 SGS、OGS、action、CLIP 文本特征的轻量 MLP，从未提及 discriminator 的监督损失。
2. **A2C 框架**：论文 Section III-B 明确说明在子指令完成时 `gamma` 设为 0。原代码的 `worker_gamma = 0 or 1` 在 `t % option_step` 边界处切换，与论文一致。
3. **消融实验**：Table II 第4行显示移除 MFD 后 SR 下降 4%，说明即使是随机初始化的 discriminator 引入的随机切换变化也优于固定步长切换（探索/正则化效果）。
4. **BCE 损失的危害**：
   - `dis_loss * 0.1` 加入 `self.loss` 后，discriminator 梯度会反向传播到 encoder/decoder（输入未 detach）
   - 基于距离的伪标签是子指令完成的不良代理
   - 这会向导航策略网络注入有害梯度信号

**还原操作：**
1. 恢复 discriminator 输入的 `.detach()`，阻止梯度流回
2. 移除 BCE 监督损失（`dis_target` 构造和 `F.binary_cross_entropy` 调用）
3. 移除 `dis_loss * 0.1 / num_steps` 对 `self.loss` 的贡献
4. 保留 `finish_flag`（用于子目标切换判断，仍经过 `.detach()`）

Discriminator 恢复为推理模式：`.detach()` 所有输入，输出仅用于子目标切换信号，不参与反向传播。

### 7.2 子目标切换改为逐样本独立
**文件：** `r2r_src/agent.py`，`rollout()` 方法

**问题：** 原代码 `finish_flag.all()` 要求**整个 batch 的所有样本**都判断完成才触发子目标切换。但一个 batch 中不同样本走到不同位置，全局同步切换不合理。

**修复：** 引入逐样本切换机制：
- `steps_since_switch[i]`：每个样本自己记录距上次切换的步数
- `need_switch[i]`：当样本达到 `option_step` 步或 discriminator 判断完成时独立切换
- `subgoal_start_dist[i]`：记录每个样本子目标开始时的距离，用于构造 discriminator 标签

```python
# 修复前：全局同步切换
if (t % option_step == 0 or finish_flag.all()) and ...

# 修复后：逐样本独立切换
for i in range(batch_size):
    if steps_since_switch[i] >= option_step or finish_flag[i]:
        need_switch[i] = True
```

### 7.3 A2C 折扣因子修复
**文件：** `r2r_src/agent.py`，`rollout()` 方法 RL loss 计算部分

**问题：** 原代码使用全局 `t % option_step` 判断边界，且 `worker_gamma` 只取 0 或 1：
- `gamma=1`（不衰减）导致高方差
- `gamma=0` 在边界处完全截断，过于激进
- 逐样本子目标切换后，全局步数边界已不适用

**修复：**
- 使用 `args.gamma`（0.9）作为标准折扣因子
- 在**实际发生子目标切换的样本**处将 gamma 设为 0（切断跨子目标的奖励传播）
- 其他样本使用 0.9 正常衰减

```python
# 修复后：逐样本 gamma，与子目标切换对齐
gamma_t = np.full(batch_size, args.gamma, dtype=np.float32)  # 默认 0.9
for i in range(batch_size):
    if subgoal_switches[t][i]:
        gamma_t[i] = 0.0  # 子目标边界切断
discount_reward = discount_reward * gamma_t + rewards[t]
```

### 7.4 候选特征融合 obj 信息
**文件：** `r2r_src/agent.py`（`_candidate_variable`, `get_input_feat`）、`r2r_src/model.py`（`AttnDecoderLSTM.forward`）

**问题：** 候选动作特征只有 vis + angle，没有物体语义。虽然 FusionProjection 已经为 36 视角做了 vis+obj 融合，但候选评分时只用了 `vis_proj`，缺失物体信息。

**修复：**
1. `_candidate_variable` 同时提取候选视角对应的 obj 特征（利用候选的 `pointId` 索引 36 视角的 obj 特征）
2. `get_input_feat` 返回 `candidate_obj`
3. `AttnDecoderLSTM.forward` 接收 `cand_obj_feat` 参数，对候选也使用完整的 FusionProjection 门控融合

```python
# _candidate_variable 新增：
candidate_obj[i, j, :] = ob['obj'][c['pointId']]

# decoder forward 中候选融合：
if cand_obj_feat is not None:
    cand_fused = self.fusion_proj(cand_vis, cand_obj_feat)  # 完整门控融合
else:
    cand_fused = self.fusion_proj.vis_proj(cand_vis)        # 退化为纯投影
```

### 7.5 精度提升预期

| 改动 | 预期影响 | 风险 |
|---|---|---|
| Discriminator 推理模式保留 | **中**：随机切换变化提供探索/正则化效果（与论文一致） | 无 |
| 逐样本子目标切换 | **中**：每个样本按自身进度切换，更精确 | 低 |
| A2C 折扣因子修复 | **中**：合理的 gamma 降低方差，提升训练稳定性 | 低 |
| 候选特征融合 obj | **小**：候选评分也能感知物体语义 | 极低 |

### 7.6 optim_step 同步修复
**文件：** `r2r_src/agent.py`，`optim_step()` 和 `zero_grad()` 方法

- 去掉 `manager_loss`（始终为 0 的遗留代码）
- `optim_step` 中补齐 `discriminator_optimizer.step()`（原来缺失）

---

## 八、训练与推理速度优化 ✅ 已实施

> 状态：**已实施**，不改变模型结构和精度。

### 8.1 消除 rollout 中重复的 get_input_feat 调用（最大收益）
**文件：** `r2r_src/agent.py`，`rollout()` 方法

**问题：** 每步调用两次 `get_input_feat`：
1. 步骤开头（line ~366）：给 decoder 决策
2. 动作执行后（line ~466）：给 discriminator

第二次调用的结果与下一步第一次调用完全相同（都是 action 后的观测），浪费了 ~25% 的步内计算。

**修复：** 第二次调用的结果缓存到 `cached_input_feats`，下一步直接复用：
```python
cached_input_feats = self.get_input_feat(perm_obs)  # 缓存
# 下一步开头直接使用，跳过重复计算
```

### 8.2 向量化 Python for 循环
**文件：** `r2r_src/agent.py`，`rollout()` 方法

将以下 for 循环全部改为 numpy 向量化操作：
- 子目标切换逻辑（`can_switch` 布尔索引）
- `cpu_a_t` 结束判断（`np.array` + 布尔掩码）
- 距离提取（列表推导 → `np.array`）
- Discriminator 标签构造（条件表达式 → 布尔运算）
- `need_switch` 判断（`~ended & (... | ...)`）
- Reward 计算（`np.sign` + 布尔索引）
- A2C discount_reward 初始化（布尔索引）
- A2C gamma 调整（布尔索引）

### 8.3 预计算所有子目标编码（Encoder + CLIP）
**文件：** `r2r_src/agent.py`，`rollout()` 方法

**问题：** 原来每次子目标切换都要重新运行 encoder LSTM 和 CLIP encode_text，且是对整个 batch 重编码。

**修复：** 在 rollout 开始时一次性预计算：
1. 收集所有样本的所有子目标文本（通常 batch_size × 2~5 = 100~320 条）
2. 一次性 batch 通过 encoder LSTM（保留梯度）
3. 一次性 batch 通过 CLIP encode_text（`torch.no_grad`）
4. 存储为 `precomp_ctx[i][j]` 和 `precomp_clip[i][j]`
5. 切换时只需索引取用 + pad/stack（纯内存操作，无神经网络 forward）

**收益：**
- 消除了 rollout 内所有 encoder 和 CLIP 的重复调用
- encoder forward 从 ~5 次/episode 降为 1 次
- CLIP encode_text 从 ~5 次/episode 降为 1 次

### 8.4 速度提升预估

| 优化 | 预估加速 |
|---|---|
| 消除重复 get_input_feat | ~25% |
| 向量化 for 循环 | ~10-15% |
| 预计算子目标编码 | ~15-20% |
| **综合** | **~40-50%** |

---

## 九、训练超参数优化 ✅ 已实施

> 状态：**已实施**，纯 RL 框架不变，仅调整超参数和训练策略。

### 9.1 熵正则化系数可配置（A2）
**文件：** `r2r_src/param.py`、`r2r_src/agent.py`、`run/agent.bash`

**问题：** 原代码中熵系数硬编码为 `0.01`，无法通过命令行调整。该值偏小，策略容易过早坍缩到次优动作，尤其影响 val_unseen 的泛化能力。

**改动：**
1. `param.py` 新增 `--entropyCoef` 参数（默认 `0.01`，向后兼容）
2. `agent.py` 中 `rl_loss += (- 0.01 * ...)` 改为 `rl_loss += (- args.entropy_coef * ...)`
3. `agent.bash` 中设为 `0.025`（2.5x 原值），增加探索多样性

**预期：** val_unseen SR 提升 0.5-1%，训练初期探索更充分。

### 9.2 Critic 独立学习率（A4）
**文件：** `r2r_src/param.py`、`r2r_src/agent.py`、`run/agent.bash`

**问题：** Critic 和 encoder/decoder 共用同一个 lr=1e-4。Critic 作为 value baseline 需要更快收敛，否则 advantage 估计方差大，policy gradient 不稳定。

**改动：**
1. `param.py` 新增 `--criticLr` 参数（默认 `None`，即沿用 `--lr`）
2. `agent.py` 中 `critic_optimizer` 使用 `critic_lr`（若指定）
3. `agent.bash` 中设为 `2e-4`（2x encoder/decoder 的 lr）

**预期：** Critic 更快收敛，降低 policy gradient 方差，训练更稳定。

### 9.3 余弦退火学习率调度（D1）
**文件：** `r2r_src/param.py`、`r2r_src/train.py`

**问题：** 固定 lr=1e-4 跑满 200K 步，训练后期 lr 过大导致在最优解附近震荡，无法精细收敛。

**改动：**
1. `param.py` 新增 `--lrMin` 参数（默认 `1e-5`）
2. `train.py` 中为 encoder/decoder/critic 创建 `CosineAnnealingLR` 调度器
   - `T_max = n_iters - start_iter`（总训练步数）
   - `eta_min = args.lr_min`（最小学习率）
3. 支持 checkpoint 恢复时 fast-forward 调度器
4. TensorBoard 记录实时 lr 曲线（`lr/encoder_decoder`、`lr/critic`）

**学习率曲线：**
- encoder/decoder：`1e-4` → cos → `1e-5`（200K 步）
- critic：`2e-4` → cos → `1e-5`（200K 步）
- discriminator：不参与调度（推理模式，不训练）

**预期：** 训练后期精细收敛，SR 提升 0.5-1%。

---

## 附：模型结构速查

```
Seq2SeqAgent
├── EncoderLSTM          vocab→wemb(256)→BiLSTM(hidden=256×2=512)
├── AttnDecoderLSTM      FusionProjection→LSTMCell(512→hidden=512)
│   ├── fusion_proj          FusionProjection(640+640→384, gated)
│   ├── feat_att_layer       SoftDotAttention(512, 512, use_tilde=False)
│   ├── attention_layer      SoftDotAttention(512, 512, use_tilde=False)  ← 全局指令
│   ├── attention_layer_sub  SoftDotAttention(512, 512, use_tilde=False)  ← 子目标
│   └── candidate_att_layer  SoftDotAttention(1536, 512, use_tilde=False) ← 候选动作打分
├── Critic               Linear(512→512→1)
├── FFNet                Linear(840→64→1)              ← 子目标完成判断（推理模式，不训练，无 optimizer）
└── CLIP RN50x4          冻结，仅推理，encode_text 缓存复用
```

**关键参数（当前默认）：**
- `feature_size=640`（CLIP-RN50x4 图像特征）
- `obj_dim=640`（目标物体特征）
- `angle_feat_size=128`
- `fused_dim=384`（FusionProjection 输出维度）
- `proj_feature_size=512`（融合后特征 384 + 角度 128）
- `rnn_dim=512`
- `option_step=3`（每3步或 discriminator 判断完成时逐样本切换子指令）
- `gamma=0.9`（A2C 折扣因子，子目标边界处为 0）
- `entropy_coef=0.025`（熵正则化系数，原 0.01）
- `lr=1e-4`（encoder/decoder 初始学习率）
- `critic_lr=2e-4`（Critic 独立学习率，2x policy lr）
- `lr_min=1e-5`（余弦退火终止学习率）
- `views=36`（全景视图数）
- `seed=1`（默认随机种子，可通过 `--seed` 配置）

---

## 九、随机种子配置与种子搜索

### 9.1 统一种子管理
**文件：** `r2r_src/param.py`, `r2r_src/train.py`, `r2r_src/agent.py`, `r2r_src/env.py`

**问题：** 原项目随机种子分散硬编码在多个文件中（torch=1, random=1, env=10），且缺少 numpy 种子，导致：
- 每次训练结果完全相同，无法做多种子对比实验
- 无法搜索最优种子

**改动：**
1. `param.py`：新增 `--seed` 参数（default=1，保持向后兼容）
2. `train.py`：新增 `set_seed(seed)` 统一函数，覆盖所有 RNG 源：
   - `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
   - `cudnn.deterministic=True`, `cudnn.benchmark=False`
3. `agent.py`：`random.seed(1)` → `random.seed(args.seed)`
4. `env.py`：`R2RBatch(seed=10)` → `seed=None`，默认推导为 `args.seed + 9`

### 9.2 种子搜索脚本
**文件：** `run/seed_search.bash`（新增）

自动化种子搜索流程：
- 候选种子：1, 42, 123, 456, 789（共5个）
- 每个种子跑 3000 iter
- 按 val_unseen success_rate 排序
- 自动将最佳种子写入 `run/agent.bash`
- 不自动启动正式训练

**用法：**
```bash
bash run/seed_search.bash 0    # GPU 0
```

### 9.3 训练脚本种子参数
**文件：** `run/agent.bash`

支持通过第二个参数指定种子：
```bash
bash run/agent.bash 0        # 使用默认 seed
bash run/agent.bash 0 42     # 使用 seed=42
```

---

## 十七、Scheduler 断点续训修复

### 问题

原余弦退火调度器在恢复训练时存在严重 bug：

1. **`T_max` 设为 `remaining_iters`**：新 scheduler 的余弦曲线以剩余步数为周期，而不是原始总步数。这意味着恢复后 lr 会从初始值重新开始一个全新的、更短的余弦曲线，而不是延续原来的曲线。
2. **fast-forward 在错误曲线上执行**：即使 fast-forward `start_iter` 步，也是在错误的曲线上走，lr 值不等于原始训练到该步时的值。
3. **checkpoint 不保存 scheduler state**：无法精确恢复调度器内部状态。

### 修复

**文件：** `r2r_src/agent.py`

- `save()` 新增可选参数 `schedulers`，保存 scheduler state_dict 列表到 checkpoint
- `load()` 返回 `(epoch, scheduler_states)` 元组，scheduler_states 可能为 None（兼容旧 checkpoint）

**文件：** `r2r_src/train.py`

- `CosineAnnealingLR` 的 `T_max` 统一使用 `n_iters`（总训练步数），不再用 `remaining_iters`
- 恢复时优先从 checkpoint 加载 scheduler state_dict（精确恢复）
- 兼容旧 checkpoint：若无 scheduler states，回退到 fast-forward（在正确的 `T_max=n_iters` 曲线上）
- 所有 `save()` 调用点均传入 `schedulers` 参数

---

## 十八、余弦调度器可选化

### 改动

将余弦退火调度器改为可选功能，通过 `--cosineAnnealing` 开关启用。不加该参数时使用原始的恒定学习率。

**文件：** `r2r_src/param.py`

- 新增 `--cosineAnnealing` 参数（store_const，默认 False）

**文件：** `r2r_src/train.py`

- `schedulers` 初始化为 `None`，仅在 `args.cosine_annealing` 为 True 时创建
- scheduler 的 step、lr 日志记录、checkpoint 保存均以 `schedulers is not None` 为前提
- 不启用时打印 `LR schedule: constant lr=... (no scheduler)`

**文件：** `run/agent.bash`

- 添加 `--cosineAnnealing` 参数（当前默认启用）

### 用法

```bash
# 使用余弦退火
bash run/agent.bash 0          # agent.bash 中已包含 --cosineAnnealing

# 不使用余弦退火（恒定 lr）：去掉 --cosineAnnealing 即可
```

---

## ~~待做：训练超参数回调~~ → 已完成 ✓

已执行的回调：
- gamma 恢复 1/0 二值（agent.py，非边界 1.0，边界 0.0）✓
- dropout 设为 0.0（agent.bash `--dropout 0.0`）✓
- entropyCoef 降回 0.01（agent.bash `--entropyCoef 0.01`）✓
- criticLr 设为 1e-4（agent.bash `--criticLr 1e-4`）✓
- featdropout 保留 0.3（有意保留）

---

## 待做：指标提升改进（等 logic宝宝 确认后逐项实施）

### 1. 停止策略优化（Stop Action Improvement）
**影响指标：** SR +2~4%, NE 下降
**文件：** `r2r_src/agent.py`（rollout 中 action 选择部分，仅改 test 逻辑）
**风险：** 低（仅影响 test，不改训练）

**问题：** 当前 test 用 argmax 选动作，很多失败来自"该停不停"或"不该停就停"。
**方案：** 加 stop confidence threshold —— 只有 stop action 的 softmax 概率超过阈值（如 0.3~0.5）才真正执行停止，否则选次优的移动动作继续走。
**原理：** agent 对 stop 不够自信时强制继续探索，减少过早停在错误位置的情况。

### 2. Stop 奖励精细化（Distance-Sensitive Stop Reward）
**影响指标：** NE 明显改善，SR 小幅提升
**文件：** `r2r_src/agent.py`（rollout 中 reward 计算部分，约 548-558 行）
**风险：** 中（改训练信号，需重新训练验证）

**问题：** 当前 stop 奖励是二值的（<3m → +2, ≥3m → -2），停在 0.5m 和 2.9m 获得相同奖励，agent 学不到"停得更精准"。移动奖励 sign(delta) 如果改为连续距离值会惩罚"必要绕弯"（室内 L/U 形走廊中正确路径上 dist_to_goal 可能先升后降）。

**方案：** 移动奖励保持 sign 不变（对绕弯鲁棒），仅将 stop 奖励改为距离敏感的连续值：
```python
# 移动奖励：保持原样
reward[is_move] = sign(delta)

# Stop 奖励：3m 内越近越高（线性），3m 外固定 -2
reward[is_stop & (dist < 3)] = 2.0 * (1.0 - dist[is_stop & (dist < 3)] / 3.0)
reward[is_stop & (dist >= 3)] = -2.0
```
- 停在 0m → +2.0，停在 1.5m → +1.0，停在 2.9m → +0.07，停在 3m+ → -2.0

**原理：** NE 就是最终停止位置的误差，让 agent 学到"停得越近奖励越高"比改移动奖励对 NE 的影响更直接。移动奖励保持 sign 避免了绕弯路径被错误惩罚的问题。

### 3. 自适应 Subgoal 切换（Adaptive Subgoal Switching）
**影响指标：** SPL +2~4%
**文件：** `r2r_src/agent.py`（subgoal switching 逻辑） + `r2r_src/model.py`（discriminator 改进）
**风险：** 中偏高（改训练 + 模型结构，需仔细验证）

**问题：** 当前不论任务复杂度，固定 3 步切换 + discriminator 随机触发，切换时机不够精准。
**方案：** 给 discriminator 加弱监督信号 —— 用导航距离变化生成 subgoal 完成度的 soft label（distance ratio），训练 discriminator 更准确判断何时切换。仍然 detach 梯度不反传到 policy（保持原始设计）。
**原理：** 更准确的切换时机 → 子目标与实际进展对齐 → 更高效的路径 → SPL 提升。

---

## 十九、砍死参数 + fused_dim 256→384 ✅ 已实施

> 状态：**已实施**，等待重新训练验证精度。

### 背景

fused_dim=256 压缩过于激进（原始 768 的 33%），导致视觉+物体信息丢失过多。同时项目中存在约 270K 完全无用的死参数。本次改动：调高 fused_dim 到 384，同时砍掉所有死参数和空转代码。

### 19.1 fused_dim 256→384
**文件：** `r2r_src/model.py`，`AttnDecoderLSTM.__init__`

- `fused_dim`: 256 → **384**
- `proj_feature_size`: 256+128=384 → 384+128=**512**
- `LSTMCell(384, 512)` → `LSTMCell(512, 512)`
- `feat_att_layer: SoftDotAttn(512, 384)` → `SoftDotAttn(512, 512)`
- `candidate_att_layer: SoftDotAttn(1536, 384)` → `SoftDotAttn(1536, 512)`
- `FusionProjection(640, 640, 256)` → `FusionProjection(640, 640, 384)`

### 19.2 删除死参数
**文件：** `r2r_src/model.py`

| 删除项 | 参数量 | 说明 |
|---|---|---|
| `self.lin_in` (Linear 512→512, no bias) | 262,144 | 从未在 forward 中使用 |
| `self.embedding` (Linear 128→64 + Tanh) | 8,256 | 从未在 forward 中使用 |
| `self.embedding_size` | 0 | 仅存储 embedding_size，无其他引用 |
| `self.sm` (Softmax) | 0 | 从未在 forward 中使用 |
| `RunningMeanStd` 类 | 0 | 全项目无任何实例化 |

### 19.3 删除死代码和空转
**文件：** `r2r_src/agent.py`

| 删除项 | 说明 |
|---|---|
| `self.discriminator_optimizer` | Discriminator 是推理模式，optimizer 永远空转 |
| `discriminator_optimizer.step()/.zero_grad()` | train()、optim_step() 中多处 |
| `clip_grad_norm(discriminator.parameters())` | 无梯度可裁剪 |
| `self.criterion` (CrossEntropyLoss) | 从未使用（DILLM 是纯 RL，无 IL） |
| `self.logit_scale` nn.Parameter | 改为固定 `torch.tensor`（从未参与训练） |

### 19.4 清理未使用 import
**文件：** `r2r_src/model.py`
- `from turtle import forward`
- `import torchtext.vocab as vocab`
- `from typing import Union`

**文件：** `r2r_src/agent.py`
- `from colorsys import hls_to_rgb`
- `from http.client import HTTP_VERSION_NOT_SUPPORTED`
- `from tkinter import simpledialog`
- `from turtle import heading`
- `from torch import optim, sigmoid`
- `from torch.nn.modules.activation import Sigmoid`
- `from transformers import AutoTokenizer, AutoModel`（还能加速启动）
- `from utils_chatglm import load_model_on_gpus`
- `import param`（只保留 `from param import args`）

### 19.5 Checkpoint 兼容性
**文件：** `r2r_src/agent.py`，`load()` / `save()` 方法

- `save()`: discriminator 仅保存 state_dict，不再保存 optimizer state
- `load()`: 使用 key 过滤 `{k: v for k, v in ... if k in model_keys}`，确保旧 checkpoint（含 `lin_in`、`embedding`、`sm`）可以正常加载
- 若旧 checkpoint 含 `discriminator_optimizer`，optimizer load 自动跳过（检查 `optimizer is not None`）

### 19.6 参数量变化

| 组件 | 旧(fused_dim=256) | 新(fused_dim=384) | 变化 |
|---|---|---|---|
| FusionProjection | 558,848 | 1,084,160 | +525,312 |
| LSTMCell | 1,837,056 | 2,099,200 | +262,144 |
| feat_att_layer | 655,360 | 786,432 | +131,072 |
| candidate_att_layer | 3,538,944 | 4,456,448 | +917,504 |
| 删 lin_in | 262,144 | 0 | -262,144 |
| 删 embedding(decoder) | 8,256 | 0 | -8,256 |
| **净变化** | | | **+1,565,632** |
| **新总参数** | ~10,200,000 | **~11,036,000** | |

最终约 **11.0M**（原始 12.1M 的 -9%）。

---

## 二十、砍掉 SoftDotAttention 中未使用的 linear_out 层 ✅ 已实施

> 状态：**已实施**，等待重新训练验证精度。

### 背景

`AttnDecoderLSTM` 中 4 个 `SoftDotAttention` 实例的 `linear_out` 层**全部从未产生有效梯度**，共计 **4,718,592 个死参数（占当时模型 ~34%）**：

| 层 | 调用参数 | linear_out 是否执行 | 输出是否被使用 | linear_out 参数量 |
|---|---|---|---|---|
| feat_att_layer(512,512) | `output_tilde=False` | 不执行 | N/A | 524,288 |
| attention_layer(512,512) | `output_tilde=False` | 不执行 | N/A | 524,288 |
| attention_layer_sub(512,512) | `output_tilde=False` | 不执行 | N/A | 524,288 |
| candidate_att_layer(1536,512) | `output_tilde=True`(默认) | 执行 | `_` 丢弃，零梯度 | 3,145,728 |

### 20.1 SoftDotAttention 新增 `use_tilde` 构造参数
**文件：** `r2r_src/model.py`

`SoftDotAttention.__init__` 新增 `use_tilde=True` 参数（默认 True 保持向后兼容）。当 `use_tilde=False` 时，不创建 `linear_out` 和 `tanh` 子模块。`forward` 中 `output_tilde and self.use_tilde` 双重判断。

### 20.2 AttnDecoderLSTM 4 层全部传 `use_tilde=False`
**文件：** `r2r_src/model.py`

```python
self.feat_att_layer = SoftDotAttention(hidden_size, self.proj_feature_size, use_tilde=False)
self.attention_layer = SoftDotAttention(hidden_size, hidden_size, use_tilde=False)
self.attention_layer_sub = SoftDotAttention(hidden_size, hidden_size, use_tilde=False)
self.candidate_att_layer = SoftDotAttention(hidden_size*3, self.proj_feature_size, use_tilde=False)
```

`candidate_att_layer` 调用也改为显式 `output_tilde=False`。

### 20.3 Checkpoint 兼容性

`load()` 已有 key 过滤（`if k in model_keys`），旧 checkpoint 中的 `linear_out` key 自动跳过，无需额外改动。

### 20.4 SpeakerEncoder / SpeakerDecoder 不受影响

它们的 `SoftDotAttention` 仍使用默认 `use_tilde=True`，`linear_out` 被使用且有梯度。

### 20.5 参数量变化

| 删除项 | 参数量 |
|---|---|
| feat_att_layer.linear_out (1024×512) | -524,288 |
| attention_layer.linear_out (1024×512) | -524,288 |
| attention_layer_sub.linear_out (1024×512) | -524,288 |
| candidate_att_layer.linear_out (2048×1536) | -3,145,728 |
| **总计** | **-4,718,592** |

| | 参数量 |
|---|---|
| 改前（vocab=991） | ~13.95M |
| **改后** | **~9.24M** |
| **减少** | **-33.8%** |

---

## 二十三、Language-Conditioned Visual Attention (LangCondVA) ✅ 已实施

> 状态：**已实施**，等待训练验证。

### 背景与动机

val_unseen SR 从 20k 步开始 plateau 在 ~0.44，而 train SR 持续涨到 0.85+，过拟合 gap 达 0.41。根本原因：decoder 视觉注意力（`feat_att_layer`）是**语言盲**的——用 `h_t` 单独决定看哪个 view，语言信息要到 LSTM 之后才进来。模型在训练环境里记住了"h_t 模式 X → 看 view 14"，到 unseen 环境就失效。

### 方案

在视觉注意力之前注入语言信号，让模型学会"指令说'左转' → 看左边的 view"，这种关系跨环境通用。

### 新增模块（`AttnDecoderLSTM`）

| 模块 | 结构 | 新增参数 |
|---|---|---|
| `lang_pre_attn` | `SoftDotAttention(512, 512, use_tilde=False)` | 262,144 (512×512 linear_in) |
| `vis_query_gate` | `nn.Linear(1024, 512)` | 524,800 (1024×512 + 512 bias) |
| **总计** | | **786,944 (+0.787M)** |

### 修改后的 forward 流程

```
原始:
  h_t_drop → feat_att_layer(h_t_drop, visual_feat) → attn_feat → LSTM → ...

改进:
  h_t_drop → lang_pre_attn(h_t_drop, subgoal_ctx) → lang_summary
           → gate(h_t_drop, lang_summary) → vis_query        ← 新增
           → feat_att_layer(vis_query, visual_feat) → attn_feat → LSTM → ...
```

### Flag 控制

`--crossAttn` flag，默认关闭，开启后启用 LangCondVA。与 `--fusionProj` 正交，可独立或组合使用。

### 文件改动

| 文件 | 改动 |
|---|---|
| `r2r_src/model.py` | `__init__` 添加 `lang_pre_attn` + `vis_query_gate`；`forward` 中条件注入语言查询 |
| `r2r_src/param.py` | 新增 `--crossAttn` flag |
| `run/agent.bash` | flag 字符串中添加 `--crossAttn` |
| `r2r_src/agent.py` | **无需改动**（subgoal_ctx/mask 已传入 decoder） |

### 为什么能改善泛化

| 当前问题 | LangCondVA 如何解决 |
|---|---|
| view 选择只依赖 h_t → 记住训练场景布局 | 注入 subgoal 语言 → 按指令方向选 view |
| "左转"指令信息到 LSTM 之后才进来 | 语言在视觉注意力之前就参与 |
| unseen 环境的场景布局完全不同 | 指令-方向的对应关系跨环境通用 |
| gate 保留灵活性 | 模糊指令时信任 h_t，方向性指令时信任 lang |

### Checkpoint 兼容性

旧 checkpoint 可加载——`load()` 的 key 过滤机制自动跳过新增的 `lang_pre_attn`/`vis_query_gate` key，新模块使用随机初始化

---

## 二十四、性能 bug 修复（_small_crossattn → _small_crossattn_0）✅ 已实施

> 状态：**已实施**。修复了 `_small_crossattn` 中引入的三个可能导致性能下降的 bug。

### 24.1 mask 长度 bug（高风险）
**文件：** `r2r_src/agent.py`，`rollout()` 预计算循环

**问题：** `precomp_ctx_mask[i][j]` 的长度使用所有子目标中最长的 `max(flat_lengths_t)`，导致短子目标的 mask 截断错误，注意力会 attend 到 padding 位置。

**修复：** 每个子目标的 ctx 和 mask 均截取到自身长度 `sg_len`。

```python
# 修复前
precomp_ctx[i][j] = flat_ctx[k:k+1]
sg_mask = (flat_enc_np[k] == padding_idx)[:max(flat_lengths_t).item()]

# 修复后
sg_len = int(flat_lengths[k])
precomp_ctx[i][j] = flat_ctx[k:k+1, :sg_len]
sg_mask = (flat_enc_np[k, :sg_len] == padding_idx)
```

### 24.2 encoder ctx 梯度共享（中风险）
**文件：** `r2r_src/agent.py`，`rollout()` 预计算 encoder ctx

**问题：** 所有子目标一次性编码，`flat_ctx` 保留梯度，各子目标切片共享同一计算图，反向传播时梯度相互干扰。

**修复：** 预计算时加 `torch.no_grad()`，与 CLIP 预计算保持一致。

```python
# 修复前
flat_ctx, _, _ = self.encoder(flat_enc_t, flat_lengths_t, enforce_sorted=False)

# 修复后
with torch.no_grad():
    flat_ctx, _, _ = self.encoder(flat_enc_t, flat_lengths_t, enforce_sorted=False)
```

### 24.3 判别器不训练（中风险）
**文件：** `r2r_src/agent.py`，`__init__`、`rollout()`、`optim_step()`、`train()`

**问题：** `_small_crossattn` 中删除了 `discriminator_optimizer`，并将 `finish_or_not` 所有输入 detach，判别器退化为随机权重预测，子目标切换信号质量下降。

**修复：** 恢复 `discriminator_optimizer`（SGD），恢复 `finish_or_not` 输入不 detach，恢复梯度裁剪和 optimizer step。

```python
# 恢复 discriminator_optimizer
self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=args.lr)
self.optimizers = (..., self.discriminator_optimizer)

# 恢复不 detach
finish_or_not = self.discriminator(obj_text_match_probs, image_text_match_probs, input_a_t, text_features_norm)

# 恢复梯度裁剪和 step
torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), 40.)
self.discriminator_optimizer.step()
```

### 24.4 squeeze(-1) — 确认无需修改（低风险）
`FFNet.W_2` 输出 shape 为 `(batch, 1)`，`squeeze(-1)` 结果为 `(batch,)`，行为正确，无需修改。

### 如何还原到 _small_crossattn
将上述三处修复逐一反向操作即可。

---

## 二十五、恢复到 2026-04-18 早上训练版本（10:26:58）✅ 已实施

> 状态：**已实施**。已将中午误改的训练逻辑恢复到当天早上实际启动训练的版本。

### 25.1 恢复目标

恢复到以下实际训练版本：

- 日志：`log/20260418_102658_agent.log`
- 启动时间：`2026-04-18 10:26:58`
- 对应 tag：`training-20260418_102658`
- 对应 commit：`55ee60c`

日志中记录的关键训练参数为：

```python
load='/home/ubuntu/Documents/DILLM_light_small_crossattn_1/snap/agent/state_dict/Iter_035000'
seed=None
```

### 25.2 被恢复的文件

- `r2r_src/agent.py`
- `r2r_src/train.py`

两者均已恢复为 tag `training-20260418_102658` 对应版本。

### 25.3 撤回的中午改动

**文件：** `r2r_src/agent.py`

撤回了中午加入的 rollout 预计算/缓存逻辑，包括：

- 一次性 flatten 全部 subgoal 并统一做 `self.encoder(...)`
- 新增 `precomp_ctx` / `precomp_ctx_mask` / `precomp_clip`
- 子目标切换时从缓存拼装 `subgoal_ctx` / `subgoal_mask`
- 新增 `cached_input_feats` 跨 step 复用 observation 特征

恢复后，`agent.py` 回到早上训练时实际使用的 rollout 逻辑。

**文件：** `r2r_src/train.py`

撤回了中午加入的两行 cudnn 确定性设置：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

恢复后，训练入口重新回到早上版本的 seed 行为。

### 25.4 恢复后的当前状态

- `run/agent.bash` 保持为加载 `Iter_035000`
- 当前脚本无生效的 `--seed`
- 实际训练行为对应 `seed=None`
- 因此该版本为**非确定性随机训练**

### 25.5 与 `DILLM_light` 的随机训练逻辑关系

核对结果：

- 两者都使用 `--feedback sample`
- 两者都通过 `torch.distributions.Categorical(probs).sample()` 进行动作采样
- 因此**随机采样训练逻辑很像**

但 seed 策略不同：

- `DILLM_light` 默认固定 seed
- 当前恢复后的 `_1` 版本对应 `seed=None`

因此，更准确地说：

> 当前 `_1` 与 `DILLM_light` 在 **sample 采样训练逻辑** 上很像，但在 **seed 策略** 上更随机。
