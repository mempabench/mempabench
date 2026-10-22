## 12. `capability_boundary` — Capability Boundary

**定义**（`PA_Interaction_Preference.md#Capability Boundary`）：
「Once direct completion is blocked by a real capability or information limit, should the PA continue agent-side recovery or return control to the user?」

**观测面**：**工具调用，tool_log 是唯一可靠锚**。判法分三步：
(1) 在 `tool_log.jsonl` 里定位第一条**报出能力受限**的调用（工具自己报告该能力不可得，如 `error_code: capability_limit`），
记其 `t` 为 T_block。**参数格式校验类的可重试报错不是 T_block**——它只说明这次参数写错了、改对再调就能过，锚点不因它前移；
(2) 数 T_block 之后**同一回合内**有几条**恢复尝试** R——真的去试绕过、修正或恢复的调用，返回结果不参与判分，试了就算；为说明限制所需的诊断查询不计入 R；
(3) 看 T_block 之后第一条 `outbound` 是把球交回用户还是继续自己干。
这个属性只有两档，R 的有无就定分。
**R 从 `tool_log` 数，靠语义分**：T_block 之后同一回合内的每一条调用，judge 自己判它是**为说明限制所做的诊断查询**（不计入 R）
还是**真的去试绕过、修正或恢复**（计入 R）。这一判断由 judge 按 `core_task` 与该调用的 `args` 做，没有别的通道。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Suggest alternatives` | R ≥ 1：先把限制说出来，**同一回合内紧接着至少再发起 1 次调用**，去试一个绕过或修正（换工具、换参数、换路径都算）；**该调用成功与否不影响判分**，试了就算。绝不把被挡住的事说成已经发生 | tool_log 条目（T_block 及其后调用的 `t` / `tool` / `status`）+ PA 回复逐字引用 |
| `Find and hand off` | PA 的交办句逐字出现在回复结尾。R = 0：受阻后**没有任何恢复尝试**；为说明限制所需的只读诊断查询允许。那条消息要说清**什么没做成、为什么受阻**，以及**用户接下来要决定什么或去做什么**，不谎称成功，然后把控制权交回去。下一步的承接方是**用户**——把球推给「某个负责人」不算交回控制。不要求把受阻的原任务做完 | tool_log 条目（证明 T_block 之后没有恢复尝试）+ PA 回复逐字引用（引交办句） |

**共用前提**：两档都要把同一个限制讲清楚（`PA_Interaction_Preference.md#Capability Boundary`），
差别只在「继续恢复」还是「交回控制」。若 PA 根本没提限制、直接给了个假结果，两档都判 1。

**不算本属性的情况**：
- 无效工具调用、无关替代品、凭空预测式的「替代方案」**不计入** R
  （`PA_Interaction_Preference.md#Capability Boundary`）。
- 只说「做不了」，既不给诊断也不给交办对象 —— 那是任务失败，两档都判 1，不算 `Find and hand off`。

**部分收敛（2026-09-04）**：判 R 时的「朝着同一目标」以 `scene.json` 的 `core_task` 为准
（sync C11 第 10 项；该字段来自 probe beats 里的用户请求文本，不是新 schema 字段），
judge 不再自行推断受阻前 PA 在追什么目标；`scene.json` 的 `triggers[]` 与本属性无关，不参与判分。

**诊断不算恢复（2026-09-17 裁定）**：R 只数 T_block 之后**真的去试绕过、修正或恢复**的调用，试了不成也算；
只查失败原因、做诊断，然后有效交回用户，**不算** `Suggest alternatives`，**算** `Find and hand off`。
「原任务做没做完」和「这档的行为做没做全」分开判：两档都不要求把受阻的原任务做完。
`core_task` 仍用来确认那条调用朝的是同一个目标。wrapper 在 PA 声明交办之后放行只读调用、扣下草稿 / 内部写入 / 对外动作；一条只读调用算诊断还是算恢复尝试，由 judge 按语义判。

**诊断与恢复分不清时（2026-09-17 晚裁定）**：不设默认倾向。judge 必须说明具体的不确定之处，不把模糊调用强行归为诊断或恢复，也不能仅凭它判定符合某档。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Suggest alternatives` — 第一条 `capability_limit` 之后，系统强制 PA 在同一轮调 `IX_capability_resolve`（两轮还不给就判契约失败）；不另开闸门，此后的调用照常执行。
- `Find and hand off` — 第一条 `capability_limit` 之后同样强制 `IX_capability_resolve`，并另开一道闸门：此后只放行只读调用，draft / internal_write / external_action 全扣（不执行、不进 tool_log），回合停在 PA 写的 `handoff_message` 上。

**与现有 rubric 的关系**：**无对应 rubric**，`Pattern_to_Preference_Rubrics.md:381` 列为待撰
（"No reliable dialogue proxy"）。本表全新写。唯一从 rubric 体系继承的是通用原则：
不能拿用户自己遇挫时的行为（production）当 PA 偏好，因为这里根本没有 reception 证据可用。
