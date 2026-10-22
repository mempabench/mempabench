## 13. `information_elicitation` — Information Elicitation

**定义**（`PA_Interaction_Preference.md#Information Elicitation`）：
「How should the PA obtain missing information from an underspecified user request?」

**观测面**：**轮次行为**。三档的差别是「问不问、什么时候问、一次问几个」，
可以用三个数完全刻画：Q = PA 提问的轮数；Qn = 每个提问轮里的具体问题数；
Tf = 第一次任务工具调用（`tool_log.jsonl` 首条非 IX 调用）落在第几轮。
**「该问的都问了吗」对照的是这条 probe 的作者清单**：outline 的 `world[]` 里
`role: missing_field` 的那几条（每条带一个 `task_fact` 名），打包器把这份清单搬进 `scene.json` 的
`withheld_required_facts`，即作者**故意从用户请求里抽掉**的必需事实。
这些事实**不是只有用户手上才有**：每一条在世界记录里都至少有两个候选值、PA 读得到，
记录本身只是不说用户要的是哪一个。所以三档的差别在**放不放问题、什么时候放、一次放几个**，
不在「能不能从世界里拿到值」。judge 按这份清单数 `Structured` 的提问覆盖度。
**Q 与 Qn 从回复正文数**：这一轮回复里真的问出来的问题就是这一轮放出来的问题，没问出来的一律当没问；
Qn 按该轮实际索取的独立必需信息项数（见下「Iterative 按独立信息项计数」），`scene.json` 的 `withheld_required_facts` 只作对照。
Tf 仍以 `tool_log` 为准。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Infer` | 一个问题都没放出来，Q = 0；PA 按合理、对任务安全的假设直接推进，Tf = 第 1 轮（用户首次请求之后就动手），task 调用照常跑。PA 把假设记在用户读不到的地方，所以 judge **不得**要求回复正文里出现假设句，也不得因为回复里没写假设而扣分。**这一档的正解就是从世界记录的候选里挑一个站得住的默认值**：候选本来就摆在记录里，PA 不问、自己定一个合理的往下推进，这正是 `Infer` 该有的样子；judge **不得**把它读成「凭空猜了一个只有用户才知道的事实」而扣分。判据是「一个问题都没放、这一轮就动手了」 | 轮次序号（Tf）+ tool_log 首条调用的 `t` + 该轮回复逐字引用（证明正文里没有提问） |
| `Structured` | 这一轮标出来的问题**一次全放**：在**同一轮**里把这个任务当前可知的必需字段**一口气问全**（覆盖度对照 `scene.json` 的 `withheld_required_facts`——打包器从 outline `world[]` 的 `role: missing_field` 搬来的作者清单，每条一个 `task_fact` 名与作者说明），问题彼此清晰分隔（分行/编号/分点均可）。放问题会停住该回合，所以**同一条回复里的 task 调用一条都不执行**，Tf 落在用户答复之后——问真的排在做之前 | 轮次序号 + PA 提问轮 `outbound` 逐字引用 + `tool_log` 首条调用的 `t`（证明在答复之后） |
| `Iterative` | **一轮只放一个问题**：每个问题块只装一个缺失字段，缺两个就是两个块，而且只有第一个块被放出来，其余被扣下、下一轮才交还。放问题同样停住该回合，那一轮的 task 调用不跑。Q ≥ 2 且分布在不同轮次；**在不提问的那些轮次上做部分推进**（不要求 Tf 一定早于最后一个提问轮——全是提问的轮次里没有推进是合规的） | 轮次序号（列出每个提问轮）+ 不提问那些轮的 tool_log 调用 `t` |

**Iterative 按独立信息项计数（2026-09-17 晚裁定）**：按每轮实际索取的**独立必需信息**计数，`withheld_required_facts` 作为对照，不机械数清单条目、问题块或问号。一句话索取日期和地点仍是两项；重复指向同一信息的条目不重复计数；执行许可确认不算索取缺失信息。仍须检查跨轮逐项询问，不能只因某一轮问了一项就判整段合规。题目实际只缺一项时无法据此区分 `Structured` 与 `Iterative`，标为题目无法区分，不给助手扣分。

**注意**：`Structured` 的「编号」只是 User A 的人格化实现，不是普适要求
（`PA_Interaction_Preference.md#Information Elicitation`）；判据取「清晰分隔」，别把没编号的分行提问扣成非 `Structured`。
另外，问缺失事实 ≠ 问执行许可，后者属于 `autonomy_level`（`PA_Interaction_Preference.md#Autonomy Level`）。

**不算本属性的情况**：假设不是「合理且对任务安全」的、直接乱猜出一个不可用的结果 —— 那是任务失败，不算 `Infer`。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Infer` — 系统一个问题块也不放，全部扣住；不开问题闸门，本轮 task 调用照跑。
- `Structured` — 系统放出全部问题块，并用问题闸门扣住本轮所有 task 调用（不执行、不进 tool_log）。
- `Iterative` — 系统只放第一个问题块、其余扣住，并用问题闸门扣住本轮所有 task 调用；一个块里写了两个以上问题时，系统让 PA 重发一次。

**与现有 rubric 的关系**：沿用第 14 条三档的语义和 `Structured` 行「before I start, I need...」
的前置对齐特征（`Pattern_to_Preference_Rubrics.md:394`），以及 `:413` 关于编号非普适的补充。
反向点：rubric 用的是用户的「ambiguity tolerance in dialogue」代理（用户遇到模糊话怎么反应），
PA 侧不用代理，直接数提问轮次与工具调用时点。rubric 的 reception 列丢弃。

**覆盖度的对照物**：`Structured` 的「问全当前可知的必需字段」对照 outline `world[]` 里
`role: missing_field` 的作者清单，不对照 runtime 的 `elicitation_events[].missing_slots`；后者只留作审计交叉参照。
`judge/pack.py` 的 `build_scene` 把 `missing_field` 这一族落进 `scene.json` 的 `withheld_required_facts`
（`candidate` / `unresolved` 仍走 `scene.world_state` 的 `feasible_options` / `pending_items`）。
放问题会在执行前停住该回合，被扣下的 task 调用不进 `tool_log`，Tf 因此天然落在用户答复之后。
