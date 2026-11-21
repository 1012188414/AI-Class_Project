# PTB-XL AFIB / 心律失常序贯筛查

## 流程一：数据准备

| 文件 | 格式 | 内容 |
|---|---|---|
| `X_{split}.npy` | float32 `[N, 12, 1000]` | 12 导联波形（0.5–40 Hz 零相位滤波 + 逐导联 Z-score） |
| `y_{split}.npy` | int64 `[N, 4]` | One-hot 标签，列顺序 `[NORM, AFIB, MI, STTC]` |
| `meta_{split}.csv` | 3 列 | `ecg_id, age_norm, sex`（0 = 男，1 = 女） |
| `patient_map.csv` | 2 列 | `ecg_id, patient_id` |
| `preprocessing.json` | JSON | 类别顺序、导联顺序、类别先验 `train_class_prior`、年龄标尺、滤波参数 |
- `{split}`：train / val / test = 官方第 1–8 / 9 / 10 折，行数 15259 / 1891 / 1914。

### 哪个流程用哪些文件

| 流程 | 用到的文件 | 用途 |
|---|---|---|
| 二：1D-ResNet + Medformer | `X`、`y`、`meta`（只取 `ecg_id`）、`patient_map.csv`、`preprocessing.json` | `X` 作 Medformer 输入；`patient_map` 做按病人分组的 5 折；json 核对类别顺序 |
| 三：校准 + HMM | `preprocessing.json`、`y_val` | 取 P_train 做校准；在 val 上搜索 ε |
| 四：RL 导联控制 | `meta`、`y_train`、`y_val` | `age_norm`、`sex` 进状态；`y_train` 算奖励；`y_val` 调超参数 |
| 五：评测 | `y_test` | 金标准标签 |

流程三读取 P_train：
```python
import json, numpy as np
stats = json.load(open("1_Output/preprocessing.json", encoding="utf-8"))
P_train = np.array(stats["train_class_prior"], dtype=np.float32)  # [NORM, AFIB, MI, STTC]
```

## 流程二：感知模型

| 文件 | 格式 | 内容 |
|---|---|---|
| `obs_probs_single_{split}.npy` | float32 `[N, 10, 4]` | 单导联（Lead I）ResNet 逐秒概率 |
| `obs_probs_dual_{split}.npy` | float32 `[N, 10, 4]` | 双导联（Lead I + V1，V1 从第 0 秒开启）ResNet 逐秒概率 |
| `preds_g5_medformer_{val,test}.npy` | float32 `[N, 4]` | G5：12 导联 Medformer 记录级概率 |
| `model_lead1.pt`、`model_dual.pt` | 权重 | 用全部训练集训练的最终模型，负责 val / test |
| `fold_models/{single,dual}_fold{1-5}.pt` | 权重 | 5 折模型，负责 train 的 OOF 预测 |
| `model_medformer_12lead.pt` | 权重 | G5 模型 |
| `record_index.csv` | 5 列 | `split, row, ecg_id, patient_id, oof_fold` |
| `stream_preprocessing.json` | JSON | 流式滤波系数、导联均值和标准差 |

- 概率列顺序 `[NORM, AFIB, MI, STTC]`；第 2 维下标 `t-1` 对应第 t 秒。
- train 的概率是按病人分组的 5 折 OOF：每条记录由没见过该病人的折模型预测。

### 哪个流程用哪些文件

| 流程 | 用到的文件 | 用途 |
|---|---|---|
| 三：校准 + HMM | `obs_probs_single`、`obs_probs_dual` | G1 / G2 用 single，G4 用 dual；在 val 上搜索 ε |
| 四：RL 导联控制 | `obs_probs_single`；唤醒后用 `fold_models/`、`model_dual.pt`、`record_index.csv`、`stream_preprocessing.json` | 唤醒前查 single 概率；唤醒后在线重算双导联概率 |
| 五：评测 | `preds_g5_medformer_test.npy` | G5 测试集预测 |

流程四在 V1 唤醒后取概率（`RawECGReplay` 定义在 `2.ipynb` 第 6 节）：
```python
replay = RawECGReplay(DATA_DIR, "2_Output", device)
p = replay.predict(ecg_id, end_step=t, wake_step=w)  # 第 w 秒末唤醒 V1，返回第 t 秒的 4 类概率
```

### 相对 proposal 的方法拓展

| 拓展 | proposal | `2.ipynb` 的做法 |
|---|---|---|
| V1 唤醒过渡的数据增强 | 只说刚唤醒时 V1 左边补零 | 训练双导联分支时，25% 的样本随机只给 1–2 秒 V1 历史，让模型学过真实的过渡窗口 |
| 精确的唤醒回放 | t ≥ t_wake + 3 起直接复用缓存 | V1 从唤醒时刻开始滤波，之后每一秒都重算（`RawECGReplay`）；train 记录用所在折的模型，回放时也保持 OOF |
| 每折单独标准化 | 只要求按病人分组做 5 折 | 每一折只用自己的训练病人统计均值和标准差，被预测的病人不参与标准化 |
| 因果性和泄露验证 | 无 | 改动未来信号或唤醒前的 V1，验证输出不变；回放与保存的概率逐条比对；原始波形重算后与 `X` 核对（`pathway_checks.json`） |
| 固定唤醒对照实验 | 无 | 验证集上比较只用 Lead I、V1 全程开启、第 4 秒后唤醒三种情况（`wake_validation.csv`） |

## 注意事项
- `X` 是离线零相位滤波，要用到整段 10 秒，只给 Medformer。ResNet 从原始波形做单向因果滤波，再用训练集统计的均值和标准差标准化，不用到未来信号。
- `obs_probs_dual` 表示 V1 全程开启。中途唤醒后每一秒都要在线重算，满 3 秒后也不能直接复用缓存；train 记录会自动用所在折的模型。
- 双导联训练时，25% 的样本随机只给 1–2 秒 V1 历史，用来适应刚唤醒的过渡窗口。
