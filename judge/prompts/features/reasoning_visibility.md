## 5. `reasoning_visibility` — Reasoning Visibility

**定义**（`PA_Interaction_Preference.md#Reasoning Visibility`）：
「How much user-facing rationale, evidence, and decision basis should the PA disclose with its result?」

**观测面**：文本。这是「为什么是这个结论」，不是「我做了哪几步」（那是 process_visibility），
也不是「我有多确定」（uncertainty_expression）。判据看因果连接词与依据句的数量与深度。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Show` | renderer 把 PA 写在 `user_facing_rationale` 里的那一段**放在答案之前**，所以回复先给依据、后给答案。这段只有两样东西：点名的具体依据（数据、文件名、条款、来源），以及是哪一条定夺了结论；**不复述用户的请求，不罗列做过的步骤，不记工具调用**。定位时注意渲染顺序固定为 `[过程句] [依据段] [散文]`（`pa/agent/ix/render.py`），同时开了 process 进度句时依据段排第二，不是第一行 | PA 回复逐字引用（引那一段） |
| `Hide` | renderer 丢掉那一段，用户先看到答案，**全文任何位置都没有依据段**；散文里也不许放依据 | PA 回复逐字引用（引首句结论 + 证明全文无依据段） |

**与 `Detailed` verbosity 的分界**：两者不共形。`Show` 判的是答案之前那一段依据，
`verbosity` 数的是答案之外的补充动作（背景 / caveat / 边界条件 / 后续影响 / 备选），依据段不在这五类里。
两个属性同时是 GT 时，各自引各自的句子，不会抢同一段。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Show` — 系统把 PA 写的 `user_facing_rationale` 逐字放在回复的第一段；该字段为空时什么也不放，并在审计面记一笔。
- `Hide` — 系统扣住 `user_facing_rationale`，只留在审计面，回复正文里一个字不放。

**与现有 rubric 的关系**：沿用第 5 条的 causal connector frequency 和 unprompted justification frequency
（`Pattern_to_Preference_Rubrics.md:209-211`）以及 Show 行里「A→B→C→conclusion」的链式描述（`:201`）。
反向点：rubric 的 reception 列（用户追问「why」）在 PA 侧无对应，丢弃。
另外把已合并的 Source Attribution（`:224`）落成 `Show` 的「点名具体依据」这一条，不单列。
