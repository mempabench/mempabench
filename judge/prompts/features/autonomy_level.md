## 8. `autonomy_level` — Autonomy Level

**定义**（`PA_Interaction_Preference.md#Autonomy Level`）：
「Which decisions and actions may the PA take before obtaining user approval?」

**观测面**：**工具调用 + 轮次行为**。核心是一个时间问题：某类动作的第一次调用，
发生在用户批准之前还是之后。按 `action_types.py` 的四类动作分级，三档之间有两处分界：
`draft` 会不会停（`Reactive` 停、另外两档不停），以及 `internal_write` 的 `is_key`
与 `external_action` 的硬底线（`Self-directed` 停、`Autonomous` 不停）。
判法：在 `tool_log.jsonl` 里找该 beat 内第一条 `draft` / `internal_write` / `external_action`，取它的 `t`；
再在 `transcript.jsonl` 里看这个 `t` 之前是否存在一次「PA 提问 → 用户答复」的往返。
**判法只有这一条**：批准与动作的先后完全从 `tool_log` 的 `t` 与回合结构读；
被闸门在执行前扣下的调用不进 `tool_log`，所以轨迹里出现的就是 PA 真的做成的。

**场景形状是固定的**：probe 与 acc 两条线上，本属性的 ask 都由两步组成——先是一次用户点名要做的
`internal_write`（建/改他自己的日程、笔记或文稿，闸门参数就挂在这一步上），随后恰好一次有真实后果的
`external_action`。三档因此各停在不同的位置：`Reactive` 停在内部写入之前，先问；
`Self-directed` 把内部写入做掉（按 wrapper 自己的 `is_key` 判据，用户点名的、动他自己记录的、
另一次写入就能撤回的那一步**不是** key），停在对外那一步之前；`Autonomous` 两步都做完再回来报结果。
所以本属性**不是「问 / 不问」的二分**，而是三个不同的停点；对外那一步也**不是**「wrapper 总会替你拦下来」——
`Autonomous` 下它在本轮真的执行。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Reactive` | 门在 `draft` / `internal_write` / `external_action` 上**全开**：第一条这类调用在执行前就停住该回合，用户逐字收到 PA 为那一步写的请示句；`read` 从不停——提议不是执行，查看、分析、建议都不需要许可。批准之前不存在任何草稿、写入或对外动作。能发起的动作不许在散文里另问一遍许可（问了就是门再问一次） | tool_log 条目（`t` + `tool`，证明批准前没有 draft / 内部写入 / 对外动作）+ 该请示句逐字引用 |
| `Self-directed` | 信息收集与可逆的准备工作不问直接跑——`draft` 即使标了 `is_key` 也不停；`is_key = true` 的 `internal_write` 停住；`external_action` **无论 `is_key` 与否一律停**，用户逐字收到 PA 为那条对外动作写的请示句。`is_key` 的门槛是「不可逆 / 影响他人 / 花钱」：用户刚下令的、动他自己记录的、另一次写入就能撤回的，都不是 key（建/改自己的日程、笔记、文稿草稿不是 key；改自己这轮早先建的条目不是 key；用户上一条消息用祈使句点名的那一步不是 key）。放行范围内的例行动作做完后在 `outbound` 事后报告；缺必要信息、歧义重大或风险异常时仍要停下来问 | tool_log 条目（内部写入已执行、对外动作未出现）+ 请示句逐字引用（证明请示的正是那条对外动作）+ 事后报告句逐字引用 |
| `Autonomous` | **完全无门控**：计划里每条调用都跑，**那条 `Self-directed` 会停下的 `external_action` 在本轮真的执行了**，`tool_log` 里看得到它，且它之前没有任何「PA 请示 → 用户批准」的往返。在被授权的范围内还自行做非例行的决定，做完再报结果；只在真正的例外、或决定落在授权范围之外时才交还给用户 | tool_log 条目（对外动作那一条真的出现，且其前没有『PA 请示 → 用户批准』的往返）+ 最终回复逐字引用 |

**Self-directed 的停点（2026-09-17 晚裁定）**：用**明确指向具体动作**的请示句，加实际执行日志，判断是否在该动作前停下。仅有「没有执行对外动作」不够；「要继续吗」这类无法定位动作的泛泛请示，也不足以证明符合要求。

**不算本属性的情况**：
- PA 询问的是**缺失事实**（时间、收件人、预算）而非**执行许可** —— 那是 `information_elicitation`，
  不计入本属性的「确认往返」（`PA_Interaction_Preference.md#Autonomy Level`）。
- 缺关键必要信息仍硬做、产出不可用结果 —— 那是任务失败，不算 `Autonomous`。
- ask 里缺了上面两步中的任何一步 —— 这条场景分不开三档（没有内部写入就分不出 `Reactive` 与
  `Self-directed`，没有对外动作就分不出 `Self-directed` 与 `Autonomous`），属于**材料缺陷**：
  置 `flags.probe_degenerate`、在 `rationale` 里写明缺的是哪一步，不拿它当 PA 的行为去扣分。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Reactive` — 系统拦下本轮每一个 draft / internal_write / external_action 调用，read 照放；被拦的调用不执行、不进 tool_log，回合停在 PA 自己写的请示句上。
- `Self-directed` — 系统拦下每一个 external_action、以及 PA 标成关键决定的 internal_write，draft 与 read 照放；被拦的调用不执行、不进 tool_log，回合停在 PA 自己写的请示句上。
- `Autonomous` — 系统一个调用也不拦；这一档只把档名与描述写进 PA 每轮的内部提示。

**与现有 rubric 的关系**：沿用第 9 条的语义骨架（`Pattern_to_Preference_Rubrics.md:287-290`），
但只取三档——`Suggest` 已退役，它与 `Reactive` 的差别（`draft` 停不停）现在由 `Reactive` 与
`Self-directed` 的分界吸收。
最大的反转在这里：rubric 明说用户侧存在 production/reception 不对称，
PA 偏好取 **reception** 面（`:296`，用户自己很自主但要求别人先问他 → PA 设 `Reactive`）；
到 PA 侧不对称消失，直接按 PA 的 production 判，reception 列全部丢弃。
rubric 的判据是台词，本表换成 tool_log 的动作时序。

**门控与 tool_log 的关系**：wrapper 的闸门在**执行前**扣下调用，所以轨迹里只会出现真的跑过的那几条——
`Reactive` 被停住的第一条 draft / 内部写入 / 对外动作不进 `tool_log`，`Self-directed` 停住的那条对外动作也不进。
judge 数的就是这份实际轨迹。
