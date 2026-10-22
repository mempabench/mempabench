## 6. `uncertainty_expression` — Uncertainty Expression

**定义**（`PA_Interaction_Preference.md#Uncertainty Expression`）：
「How much of the PA's genuine epistemic uncertainty should be surfaced to the user?」

**观测面**：**渲染出来的括号，唯一通道**（2026-09-05 定案）。PA 不在散文里 hedge：
把握度只以括号形式出现在回复正文里，载体标记不会留在用户读到的文本里。
括号是**英文字面量** `(confidence: low)`
（`high` / `medium` 同形，见 `pa/agent/ix/rationale.py:57`、`:71`），不是中文的「（把握：低）」——
在回复正文里就按这个字面量找。
judge 只读回复正文里**实际出现**的括号，
**只判一件事：出示出来的 level 与手上事实是否相符**——该确定的说确定、该存疑的标存疑。
**散文 hedge 不再是被判的通道**：既不作第二通道，也不再区分「真 hedge / 表演性 hedge」。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Express` | 回复正文里**每一处关键判断**都挂一对把握度括号，**该处是什么把握就挂什么 level**；已知、推断、未定因此对读者可分。关键判断只指影响用户决策或下一步行动、且能判断其确定程度的事实或结论，问候、流程提示、单纯复述用户要求不计。每个 level 与被标注那句话的事实基础相符。**不要求 `high` / `medium` / `low` 三种都出现**——关键判断恰好全部确定时，全是 `high` 照样是 `Express`。散文里不另行 hedge，括号是唯一通道 | 回复正文里该括号的逐字引用 + 被标注那句话的逐字引用 |
| `Hide` | renderer 一对括号都不渲染，回复正文里 0 对括号，用户拿不到把握度清单。散文里只在「隐去会实质影响行动、安全或结果」的地方才把那条限制写出来 | 回复正文逐字引用（证明正文无括号） |

**没有括号时按实际表现判（2026-09-17 裁定）**：材料给了 PA 真正的表态机会（有 claim 值得标把握度）而 `outbound` 里一对括号都没有，
就按 `Hide` 的表现评价——目标是 `Hide` 算符合，目标是 `Express` 算不符合。只有材料本身没有可表态之处时才标无法评判（`flags.probe_degenerate`），
不因看不到内部原因而弃判。

**Express 的覆盖范围（2026-09-17 晚裁定，评分范围调整，不是等价改写）**：judge 不再依赖内部 claim 清单，改查公开回答里的**关键判断**——只查影响用户决策或下一步行动、且能判断其确定程度的事实或结论；问候、流程提示、单纯复述用户要求不计。judge 认为漏标时，必须指出具体句子和它为什么需要标注。

**定位失败与 PA 漏标不再区分（2026-09-17 晚裁定）**：按最终展示结果评分，不追究内部原因。题目有充分判断机会时，没有括号按 `Hide` 的实际表现评价；只有题目本身无法体现差异时才无法评判。

**不算本属性的情况**：回复正文里挂出来的 level 与事实相反（没查到的标 `high`、查到的标 `low`），
或散文正文本身编造确定性、抹掉会影响动作/安全的重大风险 —— 那是任务失败，判 1，不算某一档。
括号的措辞好不好看同样不算本属性。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Express` — 系统在每条定位到的 claim 后面生成 `(confidence: …)`，`[[c:…]]` 标记与结尾的 `<certainties>` 块照常剥掉。整轮一条 claim 都定位不到时，系统让 PA 重写一轮。
- `Hide` — 系统一个 `(confidence: …)` 也不生成，`[[c:…]]` 标记与结尾的 `<certainties>` 块同样剥干净；定位不到 claim 也不重问。

**与现有 rubric 的关系**：rubric 第 6 条的 hedge ratio / epistemic marker / declarative ratio
三个指标是**抽取侧**（从人物转录里读出偏好）的指标，PA 侧判据不再用它们；
performative vs genuine 的区分随散文通道一起退出。反向点：rubric 的 reception 列
（对过度自信的怀疑）丢弃。

**`Hide` 的散文底线**：wrapper 的 `Hide` 描述自己带着这一条
（"Put a limit into the prose itself only where hiding it would materially affect action, safety, or outcome."），
所以它写在 `Hide` 的档内特征里，不只是任务失败的判据。


**为什么去掉了「三种 level 都出示」**：wrapper 的 `Express` 描述是
"attaches a confidence parenthetical to each claim you list, at every level"——
逐条挂、什么 level 挂什么 level，不是「三档都得出现」。旧判据会把一批 claim 恰好都很确定的正确回复判失败。
