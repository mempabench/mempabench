## 2. `verbosity` — Verbosity

**定义**（`PA_Interaction_Preference.md#Verbosity`）：
「How much relevant information should the PA include about the current topic?」

**观测面**：纯文本，判据是**补充动作的条数与类别数**，不是字数——task_expansion 高也会把回复撑长。
一个「补充动作」指以下五类之一：背景、caveat、边界条件、后续影响、备选方案；
数的是类别，同一类说两遍仍算一类。只数「关于用户所问那个 topic」的补充。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Terse` | 答案/下一步在第 1 句出现；补充动作 **至多 1 个**（背景、caveat、边界条件、后续影响、备选之中的一个），给完就停；结尾无总结句 | PA 回复逐字引用（引答案句 + 那一处补充，如果有） |
| `Detailed` | 给出答案，再给 **≥2 个分属不同类**的补充动作；没有一处是同义复述——多出来的篇幅必须是多出来的信息 | PA 回复逐字引用（逐类引用） |

**不算本属性的情况**：rubric 允许「单词回答」当 Terse 证据，PA 侧不行——
`PA_Interaction_Preference.md#Verbosity` 明说不可用即任务失败，不算 Terse。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Terse` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。
- `Detailed` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。

**与现有 rubric 的关系**：沿用第 2 条的 avg words/turn 和「unsolicited elaboration frequency」
（`Pattern_to_Preference_Rubrics.md:67-69`, `75-77`），把 cast-relative 的三分位换成绝对句数/词数阈值。
反向点：rubric 用用户的短回合推 Terse；这里是 PA 输出短。另外 rubric 允许「单词回答」当 Terse 证据，
PA 侧不行——`PA_Interaction_Preference.md#Verbosity` 明说不可用即任务失败，不算 Terse。

**阈值的地位**：judge 面没有句数/词数阈值。wrapper 的 setting 描述里没有任何长度门限，
只有「补充动作至多 1 个」与「≥2 个且分属不同类」，judge 面因此只数补充动作。
≤8 句 / ≤150 词 这类交叉参照从未用 baseline run 校过，不同 pa_model 的默认长度差异会冲掉它，
不出现在 prompt 里。
