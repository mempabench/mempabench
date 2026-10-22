## 4. `guidance_level` — Guidance Level

**定义**（`PA_Interaction_Preference.md#Guidance Level`）：
「How much topic knowledge and procedural familiarity should the PA assume the user already has?」

**观测面**：文本，判据是**术语是否被解释**，不是长度。

| Setting | 可观测特征 | evidence 类型 |
|---|---|---|
| `Assumed` | 假定用户懂这个领域、它的词汇和它的工具：领域术语一律**裸用**，对其中任何一个都**不写定义句**（「简单说就是」「也就是」），也不给编号走查；按专家密度作答 | PA 回复逐字引用（引术语所在句） |
| `Guided` | 给脚手架：对本回复来说是新的术语，在**首次出现处**给出解释，**用户在自己那条消息里示弱的那个术语至少要在内**；步骤**按顺序排开**，并**点明第一步做什么** | PA 回复逐字引用 + 用户消息里示弱那句的逐字引用 |

**「用户示弱的那个术语」从用户自己的消息里读，而且可能根本没有**：canon 只要求用户那条消息带着
「这些说法对她是新的」这个信号，**具体解释哪个词由回复自己挑**。judge 在 `transcript.simulator_turn.message`
里找用户点名的那个词（「这是什么意思」「我不太懂 X」之类）：找得到，那个词就是 `Guided` 必须落解释的地方；
找不到，只判「对本回复来说新的术语有没有在首次出现处被解释」，不因为用户没点名就判 `Guided` 不成立。
`Assumed` 不依赖这个定位——它的判据是**一句定义都没有**，用户有没有示弱都一样判。
`scene.json` 的 `familiarity_signals`（`known_term` / `unknown_term`）在这一属性的场景块里给出那对术语。它只辅助判断，以实际对话为准——用户消息里没示弱的术语，标注也不能替它示弱（主人 2026-09-17）。

## 备注（不进 judge prompt）

**系统动作**（IX 面事实，不参与判分）：这一属性各档系统实际放行 / 扣住 / 强制什么。

- `Assumed` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。
- `Guided` — 系统不放不扣：只把这一档的档名与描述写进 PA 每轮的内部提示，差别全在 PA 自己的话里。

**与现有 rubric 的关系**：沿用第 4 条的「technical term usage without definition」指标
（`Pattern_to_Preference_Rubrics.md:157`）。反向最彻底的一条：rubric 明说
用户侧「giving guidance ≠ wanting guidance」，PA 偏好取 reception 面（`:162`）；到了 PA 侧这个不对称消失，
PA 就是在 give guidance，直接按 production 面数。rubric 的 question ratio / hedge ratio 两个指标
在 PA 侧不适用（PA 提问属于 information_elicitation），丢弃。

`Detailed + Assumed` 是合法组合（`Pattern_to_Preference_Rubrics.md:94`），所以不能用词数代判。

**未决点**：术语计数——「什么算领域术语」还需要一份该 scene 的术语表才能机器数，
是否给每条 probe 预置 term list 未定。
