## 3. `emotional_engagement` — Emotional Engagement

**定义**（`PA_Interaction_Preference.md#Emotional Engagement`）：
「How much relational or emotional acknowledgment should accompany task completion?」

**观测面**：文本，且判据是**位置与有无**，不是次数——共情句在任务内容之前出现、还是全文都没有，就是分档，
出现几次不参与判档。
所有分档都必须完成任务（`PA_Interaction_Preference.md#Emotional Engagement`），共情不能替代结果。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Task-focused` | `outbound` 里 0 处共情标记（sorry to hear / that sounds / I understand / must be）；第 1 句即任务内容；对用户表露的情绪不做回应，直接转事实 | PA 回复逐字引用（引第 1 句） |
| `Relationship-focused` | 先回应情绪再办事：开头**至少 1 句**点名对方看起来是什么情绪并作出回应，之后才把任务推下去。不要求共情句的条数、不要求它独立成段、不要求结尾的关系性收束 | PA 回复逐字引用（引开头那一句） |

**不算本属性的情况**：共情铺得再多但没给出任务结果 —— 那是任务失败，判 1，不算 `Relationship-focused`
（`PA_Interaction_Preference.md#Emotional Engagement`）。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Task-focused` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。
- `Relationship-focused` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。

**与现有 rubric 的关系**：沿用第 3 条的 empathy marker 清单和「validation-before-problem-solving」测试
（`Pattern_to_Preference_Rubrics.md:104-106`, `117-120`），后者原本是判用户 reception 的，
这里直接搬成 PA 的**顺序判据**（共情在任务前 = Relationship-focused）。这是三层里唯一可以近乎原样反用的。

**未决点**：如果 probe 场景里没有任何情绪素材，`Relationship-focused` 的「情绪确认在前」无从触发；
这种场景下 5 和 4 可能分不开。scene 设计要保证有情绪素材。


**为什么去掉了条数与结尾收束**：wrapper 的 `Relationship-focused` 描述只有
"open with at least one sentence that names what the person seems to be feeling and responds to it,
and only then carry the task forward"——没有 ≥2 个标记、没有 ≥2 句的独立段落、没有结尾的关系性收束。
按 owner 的定义统一规则，这三条 judge 自造的门限不再进 prompt。
