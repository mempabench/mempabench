## 1. `tone_formality` — Tone & Formality

**定义**（`PA_Interaction_Preference.md#Tone & Formality`）：
「Which communication register should the PA use while preserving the same substantive content and task result?」

**观测面**：纯文本。register 只活在 `outbound` 的词汇和句式里，工具调用完全不带 register 信息
（同一个 `email_draft_create` 三种设置下都会调）。注意一个陷阱：`args` 里的邮件正文是**写给第三方**的，
不是写给用户的，register 只算 PA 对用户说的话，即 `outbound`。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Casual` | 像跟熟悉的同辈说话：`outbound` 里出现 **≥2 个**口语标记（缩写 gonna/wanna、感叹 oh/wow、填充 like/well、亲昵称呼、口语成语）；无开场套语、无结尾落款。**句子片段可有可无**，不作判据。低正式度是**语域**不是潦草——答案本身仍然完整，内容缺斤少两是任务失败，不是 `Casual` | PA 回复逐字引用 |
| `Consultative` | 温和的专业协作者：句子完整；无俚语；口语标记**至多 1 个**（与 `Casual` 数的是同几类：缩写、感叹或填充、亲昵称呼、口语成语）；直接称呼用户；**且至少 1 处明确对着用户的协作动作**（want me to… / X 还是 Y，你定）；无开场套语、无结尾落款。这是 `Casual` 与 `Formal` 之间的工作语域 | PA 回复逐字引用（引协作句） |
| `Formal` | 措辞精确、克制、完全专业：无缩写（一个都不能有，含 don't/I'll）；句子多为多从句；幽默克制；出现 **≥1 个**正式连接词（thus / therefore / accordingly / regarding / please note）；**有一句常规开场或结尾套语** | PA 回复逐字引用 |

**三档如何互斥**：`Consultative` 不靠「两个极端都不出现」这种负向判据，
它有自己的正向标记——那句对着用户的协作动作；缺了它，回复就只是中性散文，
三档都可能产出中性散文，判不了。与 `Casual` 的分界是口语标记的个数：`Casual` ≥2 个，
`Consultative` 至多 1 个，两档不重叠。与 `Formal` 的分界看两处硬信号：缩写和正式连接词 + 套语。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Casual` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。
- `Consultative` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。
- `Formal` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。

**与现有 rubric 的关系**：沿用第 1 条的 production 词汇标记表（`Pattern_to_Preference_Rubrics.md:29-31`）
和 F/I 密度思路。反过来的地方：rubric 那边是用**用户自己**的说话方式推他的偏好（similarity-attraction），
这里是直接看 **PA** 说话符不符合已知 GT，reception 一列完全不用。统计指标改成绝对计数
（cast-relative 排名在 PA 侧没有 cast 可比）。
