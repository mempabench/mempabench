## 7. `process_visibility` — Process Visibility

**定义**（`PA_Interaction_Preference.md#Process Visibility`）：
「When should the PA report the progress of a multi-step task?」

**观测面**：**混合，且以 tool_log 为锚**。这个属性的分档本质是「用户可见的进度句条数」与
「实际执行步数」的比值，光看文本无法判断 `Full narration` 是否漏报。
判法：先从 `state/tool_log.jsonl` 数出该 beat 内的调用数 N（同一 `t` 的并发调用算各自一条），
再从 `outbound` 数出进度句条数 M，比 M/N。
**M 从用户读到的文本里数**：进度句 = 「要做什么 / 正在做什么」的句子，完成句 = 回合末尾那句「做完了」的收尾句；
两类都只数**回复正文里真的出现**的。judge 不猜 PA 内部填了什么，只数交付出来的句子。
**末段若只是交付结果本身（答案、清单、正文），不算完成句**；只有明确说「这件事已经做完」的那一句才算。
**N 就是 `tool_log` 里实际出现的 task 调用条数。** 作者侧写材料时按「每次写入后面跟一条
readback 调用」来排（`data/scripts/prompts/probe_outline_system.md:163-169`、
`scripts/validate_probe_outline.py:1226`），所以一次「创建便签」在大纲里排成 read + create +
readback 三条；但 judge **不得假设 readback 真的发生了**，也不得按作者计划的条数去要求句子。
数 `tool_log` 里真有的那几条，`Full narration` 就是在这个实际的 N 上出 N + 1 句。

**完成句按语义认（2026-09-17 晚裁定）**：完成提示仍计入 M。judge 按公开文本的语义识别「明确说明任务已完成」的那一句，不追究它来自 wrapper 字段还是 PA 自己写的。`Bookend` 仍是开始说明 + 完成提示，`Full narration` 仍按对应计数要求；不采用「取消完成提示计数」的方案。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Silent` | 用户只看到结果，M = 0；回复正文里既无「要做什么 / 正在做什么 / 已经做完」的句子，**散文里也不能有**（「我先去查…」「已经处理好了」） | tool_log 条目（证明 N ≥ 2）+ PA 回复逐字引用（证明无进度句、无收尾句） |
| `Bookend` | 用户看到**一句说要做什么** → 结果 → **一句说做完了**，中间没有任何叙述句。非闸门轮 M = 2，**闸门轮 M = 1**（见下）。第一句要写成**整个回合的框**（端到端要做的事），不是只描述第一步。有 task 调用的回合至少得有一句开头的进度句，一句都没有就跟 `Silent` 一样落地 | tool_log 条目（首次调用的 `t`）+ 该 `t` 前后的回复逐字引用 |
| `Full narration` | 用户按调用顺序看到**每一句**进度句 → 结果 → 收尾句：每个 task 调用一句、顺序与 `t` 一致，末尾一句收尾。非闸门轮 M = N + 1，**闸门轮 M = N**（见下）。每个调用都要有一句进度句，少一句就是叙述有缺口；描述用自然说法不暴露工具名。有 task 调用的回合至少得有一句进度句 | tool_log 条目逐条对齐回复正文里的进度句 + PA 回复逐字引用 |

**没有 task 调用的回合也没有收尾句**：这一轮 `tool_log` 里没有任何调用时，`Bookend` 与
`Full narration` 同样不带完成句，judge **不得**把它算成漏项。

**闸门轮没有收尾句**：停在闸门上的回合按构造不带完成句——`Bookend` 在闸门轮是 M = 1，
`Full narration` 是 M = N，judge **不得**把缺的那句算成漏项。**N 只数执行过的调用**，所以闸门落在**第一条**
调用上时该回合一句进度句也不可能有，两档同为 **M = 0**，同样不判成叙述有缺口。
**闸门轮怎么认**：回复以一句请示（把决定交给用户的那句）结尾。

**不算本属性的情况**：事后一次性回顾罗列做过的步骤**不计入**进度句 M
（`PA_Interaction_Preference.md#Process Visibility` 明确不算进度播报）。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Silent` — 系统一条 `_ix_notice` 都不放，完成句也扣住。
- `Bookend` — 系统只放第一条 `_ix_notice`、其余全扣；完成句只在本轮真跑过 task 调用且回合没有停在闸门上时才放。有 task 调用却一条 notice 都没写时，系统让 PA 补写一句。
- `Full narration` — 系统按调用顺序放出全部 `_ix_notice`；完成句只在本轮真跑过 task 调用且回合没有停在闸门上时才放。有 task 调用却一条 notice 都没写时，系统让 PA 补写一句。

**与现有 rubric 的关系**：**无对应 rubric**，`Pattern_to_Preference_Rubrics.md:274` 明说
"Requires a PA executing a task over time. No dialogue proxy."。本表全新写，判据完全建在 tool_log 上，
不从用户对话里找代理信号。

**已定（2026-09-04）**：三档都定义在 **PA 的单条最终回复内部**，不做中途多消息通道
（MessageTool / BenchmarkChannel 已 defer）。因此 `Full narration` 的每句进度句都写在同一条
回复里，不要求跨回复分发。`pa_turn.tool_events[].process` 只作审计交叉参照。
