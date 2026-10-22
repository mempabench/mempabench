## 11. `solution_breadth` — Solution Breadth

**定义**（`PA_Interaction_Preference.md#Solution Breadth`）：
「How widely should the PA expose the viable solution space before converging?」

**观测面**：**混合**。特征是 `outbound` 里**被呈现给用户的可行选项数** K——`Low` 与 `High` 的分界就在 K。
有没有明确推荐句不参与判档：`High` 允许给推荐，也允许把选择权交回用户。evidence 引 transcript；
tool_log 用来看检索出了几个候选（PA 检索出 5 个候选却只说 1 个，仍是 `Low`——breadth 是呈现宽度不是搜索宽度）。
计数规则：一个选项必须同时满足「有独立标识」和「真的能解决请求」，
失败候选、不可得商品、无关项不计入（`PA_Interaction_Preference.md#Solution Breadth`）。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Low` | K = 1：内部把选择定了，只**直接**给出这一个方案作为推荐，并说它为什么成立；无清单、无并列比较句。**提到备选就算清单**——哪怕只是用来说明这个方案为什么赢，也不算 `Low` | PA 回复逐字引用（引选项句）+ tool_log 条目（证明检索到 ≥2 个候选却只呈现 1 个） |
| `High` | K ≥ 2（**下限，不设上限**，四个真可行项同样是合格的 `High`）且全部是真可行项，并**比较真正把它们区分开的那些 tradeoff**；**末尾给不给推荐都算**——明确推荐其中某一项、把选择权交回用户、或两者都不给，同判 `High` | PA 回复逐字引用（逐项引用 + tradeoff 句；有推荐句或反问倾向句则一并引） |

**边界排除**：某个候选执行失败后 PA 停下来，属于 `capability_boundary` 行为，
不自动判为 `Low` breadth（`PA_Interaction_Preference.md#Solution Breadth`）。judge 遇到失败候选须先查
`tool_log` 里该调用的 `status`，是 error 就走 §12 而不是本节。

**不算本属性的情况**：同一方案的两种措辞、以及不在 `scene.json` `world_state.feasible_options`
里的凑数项（失败候选、不可得商品、无关项），一律**不计入** K
（`PA_Interaction_Preference.md#Solution Breadth`）。

**`Low` / `High` 的分界**：只看 K。`Low` 是 K = 1，`High` 是 K ≥ 2；末尾那句推荐或反问不参与判档。
judge 逐项数呈现出来的选项即可判，不依赖「是否过早收敛」这种判断句，
fixture 也不必为了拉开 K 而堆到 ≥4 个可行项，两个真可行项就够（堆得更多也不因此扣分）。
可行项的分母不由 judge 自己数——**对照 `scene.json` 的 `world_state.feasible_options`**，
这份清单由打包脚本从该 probe 起跑时的 sandbox fixture 状态摘要生成（不新增 schema 字段）；
judge 用 `outbound` 呈现的选项去比对它，
不在清单里的（失败候选、不可得商品、无关项、同一方案的两种措辞）一律不计入 K。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Low` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。
- `High` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。

**与现有 rubric 的关系**：**无对应 rubric**，`Pattern_to_Preference_Rubrics.md:380` 把它列为待撰，
理由是「Weak transcript proxy; too context-dependent and infrequent in dialogue」。
本表全新写，判据是选项计数 + tradeoff 句计数，不借用任何用户侧代理信号。
