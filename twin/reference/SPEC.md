# SPEC.md — 數位孿生 Agent 系統規格

| 項目 | 值 |
|---|---|
| 版本 | 0.4 |
| 日期 | 2026-08-27 |
| 狀態 | 部分裁決。§11 列出的項目尚未裁決，**不構成規格** |
| 驗收文件 | `EVAL.md` |
| 採集文件 | `INTERVIEW.md` |
| 本版變更 | 同步 `INTERVIEW.md` §8 四項裁決：§8 移除同意流程、新增 D34–D36、剩餘開放項收斂為兩項 |

---

## 0. 文件規則

本文件只收錄**已裁決**的決策。判定一項決策「已裁決」需同時滿足：

1. 有明確的取捨對象（我們選了 A 而不是 B）
2. 有理由，且理由可被推翻（寫在 §10 決策紀錄）
3. 有可觀測的後果（若做錯，會在 `EVAL.md` 的哪個指標上顯現）

不滿足者一律進 §11，標記為阻塞項，**實作不得依賴 §11 的任何內容**。

規範語彙：**MUST**（違反即不合規）、**MUST NOT**、**SHOULD**（可偏離但需記錄理由）、**MAY**。

---

## 1. 目標與非目標

### 1.1 目標

打造一個框架，使**任一個人**餵入自己的歷史資料後，能得到一個具備以下性質的 Agent：

- **G1 人格保真**：在本人未曾表態過的新情境下，其判斷與偏好接近本人。
- **G2 工具行為**：具備使用外部工具的能力，且**工具的使用風格**（何時查、查幾次、何時放棄）像本人。
- **G3 主動性**：能在無人呼叫的情況下判斷「此刻本人會不會有動作」，包含正確地**不動作**。
- **G4 可泛用**：新使用者接入不需修改程式碼，只需走 onboarding 流程。
- **G5 零成本建置**：建置階段（資料工廠、訓練、評測、儲存）全程在免費層內完成。

### 1.2 非目標

- **NG1 零成本常駐服務**。24/7 低延遲推論無免費解，v1 採 scale-to-zero，接受冷啟動。
- **NG2 端到端多模態微調**。影音圖像一律先降維為文字，不進權重。詳見 §4.1。
- **NG3 知識正確性**。孿生的目標是像本人，包含**像本人一樣記錯**。事實正確性不是驗收項。
- **NG4 全自動代理發言**。v1 一律 human-in-the-loop，詳見 §6.4。

---

## 2. 名詞定義

本節為規範性。文件中出現的下列詞彙 MUST 依此處定義解讀。凡定義含「不是」條款者，該條款與定義本身同等具約束力。

### 2.1 主體

| 詞彙 | 定義 | 不是 |
|---|---|---|
| **本人 (Principal)** | 被建模的真實人類。系統中所有資料以 `principal_id` 為隔離單位。 | 不是「使用者」。使用孿生的人可能是第三方。 |
| **孿生 (Twin)** | 一組 `(LoRA adapter, memory store, eval set)` 的三元組，繫於單一 principal。 | 不是模型。base model 為全體共用，不屬於任何孿生。 |
| **第三方 (Third Party)** | 出現在本人資料中、但未同意被建模的其他自然人。 | 不是 principal。第三方資料的處置見 §11-C。 |

### 2.2 資料單位

| 詞彙 | 定義 | 不是 |
|---|---|---|
| **碎片 (Fragment)** | 最小記憶單位。一則已降維為文字、且帶 `event_time` 的內容。 | 不是原始檔案。原始媒體不進記憶層。 |
| **片段 (Episode)** | 時間鄰近且實體重疊的碎片群集。檢索的中間層級。 | 不是主題聚類。時間是主軸，主題是次要條件。 |
| **時期 (Period)** | 片段的上層聚合，對應月／季／人生階段。檢索的入口層級。 | — |
| **軌跡 (Trajectory)** | 一段含「觀察 → 決策 → 動作或不動作」的完整序列。訓練樣本的單位。 | 不是對話紀錄。對話紀錄是軌跡的來源之一，非軌跡本身。 |
| **自陳資料 (Self-Report)** | 本人主動針對自己的陳述，含訪談逐字稿與結構化問卷。 | 不是歷史資料。它是本人在當下對過去的重述，具備歷史資料沒有的自我詮釋。 |

### 2.3 事件類型（本規格中最常被誤解的一組）

| 詞彙 | 定義 | 不是 |
|---|---|---|
| **動作 (Action)** | 本人對外產生可觀測輸出的事件。 | — |
| **不動作 (No-Action)** | 本人**在具備行動機會的情況下**選擇不行動。是一種顯式的、可被訓練的動作類型。 | 不是資料缺失。不是「那段時間沒紀錄」。 |
| **曝光 (Exposure)** | 本人接觸到某項刺激的事件，無論其後是否行動。 | 不是動作。曝光是動作與不動作的**共同前置條件**。 |
| **硬負例 (Hard Negative)** | 一筆 `曝光 → 不動作` 的軌跡，且其情境接近某筆 `曝光 → 動作` 軌跡。 | 不是「無活動時段」。無活動時段沒有曝光，不構成負例。 |

三者的關係：**沒有曝光紀錄，就無法區分「不想做」與「沒看到」，也就無法產生硬負例。** 這是 §4 的核心設計約束。

### 2.4 處理角色

| 詞彙 | 定義 | 不是 |
|---|---|---|
| **Teacher** | 用於將原始資料轉為碎片與軌跡的外部強模型。 | 不是被蒸餾的對象。我們不複製 teacher 的能力，只借用它的標註勞力。 |
| **Judge** | 評測時用於判定語意等價的外部模型。 | 不得與 teacher 共用 prompt 模板（見 `EVAL.md` §6.1）。 |
| **反省 (Reflection)** | 軌跡中一種不對外輸出的步驟，內容為第一人稱的動機陳述。 | 不是 chain-of-thought。它不用於提升正確性，用於塑形傾向（見 §5.4）。 |

### 2.5 切分與衛生

| 詞彙 | 定義 |
|---|---|
| **時間切分 (Time-based Split)** | 訓練集所有樣本的 `context_time` 早於測試集全部樣本。 |
| **保留時段 (Held-out Window)** | 於 ingest 階段即標記、永不進入訓練的時間區間。eval set 由此產生。 |
| **封存分割 (Sealed Split)** | 保留時段中再切出的一部分，僅於最終驗收開封（見 `EVAL.md` §9）。 |

---

## 3. 架構分層

四層，職責邊界為硬約束。

```
┌─ L4 執行層  Agent Runtime + MCP 工具 + Surface(LINE/Web)
├─ L3 權重層  Base Model + Twin LoRA Adapter
├─ L2 記憶層  Memory Store (碎片 / 片段 / 時期)
└─ L1 資料層  原始資料 → 碎片 → 軌跡
```

### 3.1 邊界約束（本規格的核心）

- **C1**：記憶的**內容** MUST 存在 L2；記憶的**回想方式** MUST 學在 L3。
  - 違反的徵狀：孿生開始編造具體人名、日期、地點。
- **C2**：工具的 **schema 與清單** MUST 由 L4 於推論時注入 context；MUST NOT 進入 L3 權重。
  - 理由：否則每新增一個 plugin 就需重訓，與 G4 直接衝突。
- **C3**：L3 學的是**政策**（選擇準則、放棄門檻、語氣、動作/不動作傾向），不是知識，也不是特定工具名。
- **C4**：`recall()` MUST 實作為一個普通工具，與其他 plugin 走同一條介面。
  - 理由：「有意識」與「會用工具」因此是同一個機制，不需兩套。

---

## 4. L1/L2 — 資料與記憶規格

本節依 `EVAL.md` 的量測需求反向設計。每一條規則後標註它服務的評測項；無法對應到任何評測項的規則不應存在於此。

### 4.1 資料來源分類

| 類別 | 例 | 去處 | 服務 |
|---|---|---|---|
| **自陳資料** | 訪談逐字稿、結構化問卷 | 記憶層（時期層級）+ 可原文注入 | S1、S1-B2 baseline |
| **行為資料** | 對話、工具呼叫、貼文、沉默 | 軌跡資料集 | S2、S3 |
| **曝光資料** | 讀取紀錄、瀏覽紀錄、未讀已讀狀態 | 軌跡資料集（作為前置條件） | S3 硬負例 |
| **知識資料** | 文章、PDF、筆記 | RAG index | 無（不參與訓練） |

- 知識資料 MUST NOT 進入 LoRA 訓練集。
- 自陳資料 MUST 可被原文取出並注入 context。理由：`EVAL.md` §3.4 的 B2 baseline 需要它，而 B2 是本專案的 kill switch。若自陳資料被切碎後無法還原，kill switch 失效。

### 4.2 多模態降維

所有非文字輸入 MUST 於 ingest 階段降維為文字碎片。原始媒體檔 MUST NOT 進入訓練資料集，MUST NOT 上傳至跨雲儲存（見 §7.2）。

| 來源 | 抽取內容 | 時間來源 |
|---|---|---|
| 影像 | 說明文字、可辨識實體、場景 | EXIF |
| 影片 | 轉錄稿、分段摘要 | 檔案時間 + 相對時間軸 |
| 文件 | Markdown 正文 | 檔案/內容時間 |
| 對話 | 逐則訊息 | 訊息時間戳 |
| 語音訪談 | 逐字稿 | 訪談日期 + 相對時間軸 |

### 4.3 曝光事件的採集（本次新增，優先級最高）

**規則**：ingest 管線 MUST 記錄曝光事件，而非僅記錄動作事件。

理由（可觀測）：`EVAL.md` §5.2 要求 S3 的負例為「情境接近真實觸發、但本人未行動」的樣本。若系統只記錄「本人發了文」，則無法區分「他看了那則新聞但沒發文」與「他那天根本沒上網」。前者是硬負例，後者是 trivial negative。**沒有曝光紀錄，S3 就只能用 trivial negative，分數會虛高而實際部署失敗。**

最低可行採集面：

| Surface | 曝光訊號 | 可得性 |
|---|---|---|
| LINE | 已讀狀態、開啟聊天室 | 平台限制，SHOULD 盡力採集 |
| 新聞/RSS | 閱讀器已讀、點擊紀錄 | 高 |
| 瀏覽器 | 歷史紀錄 | 高 |

- 若某 surface 的曝光訊號不可得，該 surface MUST NOT 用於 S3 評測，並 MUST 於評測報告中聲明。
- 曝光事件 MUST 記錄時間戳與內容摘要，MUST NOT 僅記錄計數。

**歷史資料的曝光通常已遺失。** 因此曝光採集 SHOULD 在 §11-A 裁決後立即上線，先於大規模歷史 ingest；歷史時段的 S3 負例只能以較低信心使用，並於報告中標示。

### 4.4 碎片 Schema（規範）

```jsonc
{
  "fragment_id": "str",            // MUST, 全域唯一
  "principal_id": "str",           // MUST
  "source_class": "self_report|behavior|exposure|knowledge",  // MUST, 見 §4.1
  "modality": "text|image|video|doc|message|audio",
  "content": "str",                // MUST, 已降維的文字
  "event_time": {                  // MUST
    "value": "2024-06",            // ISO-8601, 可為部分精度
    "precision": "year|month|day|hour|minute",
    "confidence": 0.0              // 0..1
  },
  "ingest_time": "2026-08-27T10:00:00Z",  // MUST
  "split": "train|heldout|sealed",         // MUST, 見 §4.8
  "entities": { "people": [], "places": [], "topics": [] },
  "third_party_spans": [           // MUST, 可為空陣列; 見 §4.9
    { "start": 0, "end": 0, "party_ref": "str" }
  ],
  "conflicts_with": ["fragment_id"],       // 見 §4.7
  "source_uri": "r2://...",        // 指標，非內容
  "salience": 0.0                  // 0..1, 見 §4.6
}
```

- 無 `event_time` 的碎片 MUST NOT 進入記憶層。它只能作為知識資料使用。
- `precision` MUST 顯式表示。孿生 SHOULD 能輸出「大概去年夏天」這種模糊度；虛假的精確度比模糊更不像本人，且會在 `EVAL.md` §3.5 的模糊度匹配項失分。

### 4.5 三層粒度與檢索

- Fragment → Episode：以**時間鄰近 × 實體重疊**聚類。
- Episode → Period：以月／季／事件邊界聚合。
- 自陳資料 MUST 掛在 Period 層，作為該時期的敘事骨架。
- 檢索 MUST 為 coarse-to-fine：先定位時期，再下拉碎片。這是「拼湊」的實作機制，也是抑制幻覺的手段（觀測點：`EVAL.md` §3.5 內容正確性）。

### 4.6 顯著性與遺忘

```
salience = f(提及次數, 情緒強度, 近期性, 自陳提及)
```

- **自陳提及 MUST 為 salience 的輸入項之一。** 本人在訪談中主動講起的事，依定義即為對本人重要的事。這是自陳資料相對歷史資料的獨有訊號，成本為零。
- 檢索時 salience 作為權重衰減項。

不實作遺忘的後果：孿生記得過於清楚，於 `EVAL.md` S4 盲測中被辨識出來，且辨識理由會直接指向此處。這是可觀測的失敗模式，非美學考量。

### 4.7 記憶衝突 MUST NOT 被消解

當同一事件存在不一致的碎片（本人在不同時間對同一件事有不同說法），系統 MUST 兩者皆保留，並以 `conflicts_with` 互相標記。

- MUST NOT 自動選擇「正確」版本。
- MUST NOT 合併為單一版本。
- 檢索時 SHOULD 依 `event_time` 與 salience 決定優先呈現哪一版。

理由：見 §1.2 NG3。孿生的目標是像本人，**包含像本人一樣記錯**。消解衝突會製造一個比本人更準確、因而更不像本人的東西。

### 4.8 切分標記於 ingest 階段決定（本次新增）

`split` 欄位 MUST 於 ingest 時寫入，MUST NOT 於訓練時才決定。

- 切分 MUST 為時間切分（見 §2.5）。
- `heldout` 時段用於產生 eval set；`sealed` 為其中約 20%，僅於最終驗收開封。
- 訓練管線 MUST 於載入時硬性過濾 `split != "train"` 的樣本，且此過濾 MUST 有測試覆蓋。

理由：時間洩漏是無法事後偵測的失敗。若切分在訓練時才做，任何一次資料重整都可能靜默地污染 eval，而所有指標仍會看起來正常。

### 4.9 第三方標記（為 §11-C 預留）

`third_party_spans` MUST 於 ingest 階段標註，即使第三方政策尚未裁決。

- 標註內容為「哪一段文字屬於第三方」，不含處置方式。
- 處置（保留／匿名化／移除）MUST 為 ingest 之後的可重跑步驟。

理由：§11-C 為法遵阻塞項，裁決可能任一方向。若標記未在 ingest 時完成，任何處置都需重跑整條管線；若標記完成，切換政策只需重跑最後一步。**這是為一個尚未做出的決定預留低成本的翻轉能力。**

此標記存在不等於允許 ingest。§11-C 裁決前，含第三方內容的 surface MUST NOT 開始 ingest。

### 4.10 軌跡 Schema（規範）

```jsonc
{
  "trajectory_id": "str",
  "principal_id": "str",
  "context_time": "2026-03-14T09:12:00Z",   // MUST, 時間切分依據
  "split": "train|heldout|sealed",           // MUST
  "exposure": {                              // MUST, 見 §4.3
    "occurred": true,
    "stimulus": "str",                       // 本人接觸到什麼
    "evidence": "read_receipt|history|inferred|absent"
  },
  "observation": "str",
  "available_tools": ["recall", "web_search", "..."],
  "steps": [
    { "type": "tool_call", "tool": "...", "args": {}, "result_digest": "..." },
    { "type": "reflection", "content": "..." },      // 見 §5.4
    { "type": "action", "surface": "line", "content": "..." },
    { "type": "no_action", "reason": "..." }         // MUST 為顯式型別
  ],
  "negative_class": "none|hard|trivial",     // MUST, 見 §4.11
  "ground_truth_source": "observed|principal_annotated|teacher_synthesized"
}
```

- `exposure.evidence = "absent"` 的 `no_action` 軌跡 MUST 被標為 `negative_class: "trivial"`。
- `ground_truth_source = "teacher_synthesized"` 的軌跡 MUST NOT 用於 `EVAL.md` 的任何 suite。合成資料可訓練，不可評測。

### 4.11 負例（本規格中最容易做錯、後果最嚴重的一項）

- 資料集 MUST 包含 `no_action` 軌跡。
- 訓練集中 `negative_class: "hard"` 的樣本 MUST 存在，且 SHOULD 不少於全部負例的一半。
- `negative_class: "trivial"` 的樣本 MAY 少量保留作為對照，MUST NOT 作為負例的主體。
- 目標比例：`no_action` 樣本占比 SHOULD 接近本人在該 surface 上的真實不回應率，並由 `EVAL.md` §5.5 的 `silence_rate_delta` 驗證。

不遵守的後果：模型天生極度傾向回應，最終產出一個話多、亂發文的孿生。這是本專案第一號失敗模式，且它**只在資料層被隱藏，在 S3 分數上看不出來**——這正是 §4.3 曝光採集必須先行的原因。

---

## 5. L3 — 訓練規格

### 5.1 底模

- Base model MUST 為 open-weight、permissive license（Apache-2.0 或同等）。
- 全體孿生 MUST 共用同一 base model；個體差異 MUST 僅存在於 LoRA adapter。
- **語言需求（已裁決）**：MUST 同時支援繁體中文（台灣用語為主）與英文；SHOULD 支援簡體中文。
- **尺寸（已裁決）**：v1 為 8B 級別。降級路徑：若 Modal credits 成為迭代瓶頸，MAY 降至 4B，但降級 MUST 以 `EVAL.md` S1 與 S4 驗證，MUST NOT 僅憑成本理由靜默降級。
- 選型 MUST 驗證繁簡一致性：同一問題以繁／簡輸入，語意判斷 SHOULD 一致；輸出字體 MUST 依本人真實使用習慣（見 `EVAL.md` §8.4）。

### 5.2 Teacher

- Teacher MUST 透過 `teacher.py` 介面存取，實作可替換。
- v1 綁定：Gemini Flash 系列免費層（1,500 RPD / 15 RPM / 1M TPM，多模態輸入、function calling、JSON mode 均在免費範圍）。
- **MUST 使用專屬 GCP 專案，且永不啟用帳單。** 該專案一旦啟用 billing，免費層即消失，所有呼叫從第一個 token 起計費。此為不可逆的成本事故。
- 呼叫策略 MUST 為「少次、大批」：免費層瓶頸是 RPD 而非 TPM，故 SHOULD 一次注入整個 Episode（利用 1M context）並一次產出多條軌跡。
- Pro 系列已於 2026-04 移出免費層，v1 MUST NOT 依賴之。

### 5.3 訓練方法

- v1 MUST 為 LoRA/QLoRA SFT。
- v1 MUST NOT 使用 RLHF/DPO。人在環（§6.4）產生的拒絕樣本回流後，才於 v2 考慮偏好學習。
- 工具名稱 MUST 於訓練時做隨機置換或遮蔽，強迫模型學習選擇準則而非特定工具名。這是 C2/C3 的落實手段。

### 5.4 反省訓練（Counterfactual Reflection）

依據 Anthropic *Verbalizable Representations Form a Global Workspace in Language Models* (2026-07-06)：訓練模型「若被中途打斷並要求反省時會說出什麼」，可改變其在**未被打斷**情境下的行為。

規格：

- 軌跡 MUST 可選擇性插入 `reflection` step，內容為第一人稱的「我現在在想什麼／為什麼選這個工具／為什麼不想回」。
- 推論時 MUST NOT 輸出 reflection 內容（除非明確要求 introspect）。
- reflection 樣本占比 SHOULD 為 15–30%，於 §11-B 門檻確立後以 ablation 驗證。

明確排除：J-lens 本身。計算每層 d_model × d_model 的平均 Jacobian 並於千條 prompt 上平均，成本不在免費層範圍。若需觀測內部狀態，SHOULD 使用 logit lens 作為近似。

---

## 6. L4 — 執行層規格

### 6.1 工具介面

- 所有工具 MUST 透過 MCP 暴露。
- 新增 plugin MUST NOT 需要重訓、重啟或改動 `train.py`。
- 工具清單於 runtime 注入 system context。

### 6.2 執行模型（本版新增，依 §11-A 裁決）

Runtime MUST 為 **tick loop**，MUST NOT 為 request-response。

- 每個 tick 向孿生提供當前 context：新訊息、時間、環境事件。
- 孿生於該 tick 選擇呼叫零個或多個工具。
- **呼叫零個工具即為 `no_action`**，與其他選擇走完全相同的路徑，無特殊分支。

理由：裁決「回訊息也是 tool call（突然想回）」。此設計使主動性不再是獨立子系統——「要不要回」與「要不要查」成為同一決策空間中的並列選項。`EVAL.md` S3 的量測點即為「此 tick 是否呼叫 `reply`」。

已知副作用：S2 與 S3 的界線因此模糊，兩者共用同一 runtime。EVAL 中仍 MUST 分開計分（見 `EVAL.md` §2）。

### 6.3 內建工具（v1）

| 工具 | 用途 |
|---|---|
| `recall(query, time_hint)` | 記憶檢索，coarse-to-fine |
| `web_search(query)` | 外部查找 |
| `reply(surface, content)` | 對外回覆 |

- `reply` MUST 與其他工具位於同一層級，MUST NOT 有特殊呼叫路徑或特殊 prompt 位置。
- 送出與否由 runtime 的閘門決定（§6.5），**不由工具定義決定**。

### 6.4 Surface

v1 裁決為 **LINE**。

Surface MUST 為可插拔配接器，核心 runtime MUST NOT 認識任何特定平台。LINE 的曝光訊號（已讀狀態、聊天室開啟）MUST 依 §4.3 採集；若平台限制導致不可得，該 surface MUST 於評測報告中聲明，且其 S3 結果 MUST 標為低信心。

### 6.5 送出閘門（取代原 §6.4）

`reply` 的實際送出由 runtime 攔截，分三級。**攔截發生在 runtime 層，工具定義與權重不變；升降級不需重訓。**

| 級別 | 行為 | 進入條件 |
|---|---|---|
| **L0 草稿** | 全部攔截，轉為草稿待本人確認 | v1 預設 |
| **L1 白名單** | 對指定聯絡人自動送出，其餘攔截 | 見 `EVAL.md` §7.2 |
| **L2 全自動** | 全部自動送出 | 見 `EVAL.md` §7.2 |

- v1 MUST 由 L0 起始。
- 每一次本人的「不送出」MUST 被記錄為硬負例並回流資料集（`negative_class: "hard"`）。
- **降級 MUST 為自動**：任一輪 eval 未達當前級別門檻，runtime MUST 立即降回下一級，MUST NOT 需人工介入。
- 升級 MUST 為手動，且 MUST 有連續兩輪達標的紀錄。

理由：`EVAL.md` §7.1 的 T1 標準允許 False Alarm 0.30，即每十次主動回覆約三次是本人不會回的。此數字在 L0 無害（本人會看到並否決，且該否決即為訓練資料），在 L2 則不可接受。**因此「孿生堪用」與「可自動送出」是兩個不同門檻**，見 `EVAL.md` §7。

---

## 7. 基礎設施規格

### 7.1 免費層綁定與抽象

程式碼 MUST 只在兩處耦合外部供應商：

- `launch/*.sh` — 雲端啟動（拉 image、掛憑證、呼叫 `train.py`）
- `teacher.py` — Teacher 供應商

`train.py` MUST NOT 認識任何雲。MUST NOT 出現 SageMaker Estimator、Vertex CustomJob SDK 或任何寫死路徑。

理由：免費層半年內已變動兩次（Gemini Pro 於 2026-04 移出免費層；AWS 免費層於 2026-07-30 停止新註冊）。抽象層是「免費」這個命題能持續成立的前提。

### 7.2 儲存

- 所有路徑 MUST 為 URI（`r2://`、`s3://`、`file://`），經 fsspec 統一。程式碼中 MUST NOT 出現本機絕對路徑。
- 跨雲中樞 MUST 選用零 egress 費用之物件儲存（v1：Cloudflare R2，10GB 免費層）。
- 原始媒體 MUST NOT 進入跨雲中樞。僅碎片、軌跡、adapter 上傳。

### 7.3 訓練算力

| 用途 | 平台 | 額度性質 |
|---|---|---|
| 主要訓練 | Modal Starter | $30/月循環 credits，閒置不計費，1 TiB/月 volume |
| 長時 run | Kaggle | T4×2，約 30 小時/週 |
| 備援 | Lightning AI | 約 15 credits/月 |

MUST 優先使用 spot/preemptible。

### 7.4 Checkpoint 契約（硬約束）

`train.py --resume auto` MUST 能自遠端最新 checkpoint 續跑。Checkpoint MUST 包含：

1. adapter weights
2. optimizer state
3. lr scheduler state
4. **RNG state**
5. **global_step**
6. **dataloader sample cursor**

缺少 4–6 的後果：續跑時重看資料，等同偷偷多訓，產生假的收斂曲線。

驗收方式：訓練中途 `kill -9`，重啟，loss 曲線 MUST 連續。此為 CI 項目，非人工檢查。

Checkpoint 上傳頻率 SHOULD 為 10–15 分鐘。adapter + optimizer state 僅數百 MB，被 spot 回收最多損失一個間隔。

### 7.5 可重現性

一個 `run_id` MUST 綁定：seed、資料集版本 hash、config hash，並寫入 checkpoint metadata。跨平台的 run 若無此三者一致，MUST NOT 進行比較。

### 7.6 不自建 trainer

MUST 使用既有訓練框架（TRL + Accelerate 或同級）。其 `resume_from_checkpoint` 已處理 §7.4 的 4–6 項。自建 trainer MUST NOT 出現在 v1。

---

## 8. 多使用者與隱私

- 每個 principal 對應：一個 adapter、一個 memory store、一組 eval set。三者 MUST 以 `principal_id` 隔離。
- Adapter 為個資。它可反推行為特徵，MUST 加密儲存，MUST NOT 跨 principal 共用或合併。
- **不實作同意流程**（`INTERVIEW.md` §8 I-D）。本專案為個人自用之開源工具，principal 即操作者，形式同意無實質意義。責任歸屬由 README 聲明。
- **第三方資料政策（已裁決，§11-C）**：本專案為個人自用之開源工具，不提供代管服務，故不實作匿名化或去識別化管線。改採三項技術護欄，MUST 全數實作：
  1. `third_party_spans` 標記保留（§4.9）。成本已付，且它是日後任何政策的前提。
  2. Repo MUST 有 `.gitignore` 與 pre-commit hook，硬性阻擋 `data/`、`adapters/`、`transcripts/`、`eval/` 進入版控。
  3. README MUST 聲明：使用者自行負責其資料中第三方內容的合法性；本專案不代為處理。
- 開源不降低第三方風險，只轉移責任歸屬。實際最高風險路徑不是模型，是 repo 誤 commit 與 issue 中貼出的 log；護欄 2 針對此設計。
- §11-C 裁決後，LINE 資料的 ingest 封鎖解除。

---

## 9. 驗收

驗收標準全數定義於 `EVAL.md`。本規格 MUST NOT 自行定義通過條件。

四個 suite：S1 人格保真、S2 工具使用、S3 主動性、S4 盲測。

**任何未通過 `EVAL.md` S1–S4 的孿生 MUST NOT 開放非草稿模式。**

---

## 10. 決策紀錄（已裁決）

| # | 決策 | 取捨對象 | 理由 | 失敗徵狀（觀測點） |
|---|---|---|---|---|
| D1 | 記憶內容進 store，回想方式進權重 | 全部進權重 | episodic 細節壓進 LoRA 會退化為統計傾向；新記憶需重訓 | 編造人名/日期（S1） |
| D2 | 工具 schema 不進權重 | 烤進權重 | 與 G4 衝突 | 新 plugin 無法使用（S2） |
| D3 | `recall()` 為普通工具 | 獨立記憶子系統 | 單一機制，減少表面積 | — |
| D4 | 多模態先降維為文字 | 端到端 VLM 微調 | 顯存不在免費層範圍，且行為載體是決策非像素 | G5 破功 |
| D5 | 一底模 + 每人一 adapter | 每人一完整模型 | 儲存與服務成本線性 vs 常數 | G5 破功 |
| D6 | `no_action` 為顯式型別 | 以資料缺失表示 | 缺失無法訓練；模型天生偏向回應 | False Alarm 偏高（S3） |
| D7 | 負例須為 hard negative | 使用無活動時段 | trivial negative 使模型學到「有事就回」 | S3 分數虛高但實用失敗 |
| D8 | Teacher 專用 GCP 專案且不綁 billing | 共用主專案 | 啟用 billing 會使免費層消失 | 突發帳單 |
| D9 | Teacher 呼叫少次大批 | 一碎片一呼叫 | 免費層瓶頸是 RPD 非 TPM | RPD 耗盡，資料工廠停擺 |
| D10 | 零 egress 儲存為跨雲中樞 | S3 | 跨平台搬運的隱形成本 | G5 破功 |
| D11 | Checkpoint 含 RNG/step/cursor | 僅存權重 | 否則續跑重看資料 | 假收斂曲線 |
| D12 | 供應商耦合僅限 launch/ 與 teacher.py | 直接呼叫 SDK | 免費層變動頻繁 | 遷移成本爆炸 |
| D13 | v1 純 SFT | 直接上 DPO | 無偏好資料；人在環尚未產生拒絕樣本 | — |
| D14 | v1 全程 human-in-the-loop | 自動送出 | 文獻 proactivity F1 約 0.66 | 對外事故 |
| D15 | 不實作 J-lens | 複製論文方法 | 計算成本不在免費層 | G5 破功 |
| D16 | 訓練時遮蔽工具名 | 保留真實名稱 | 迫使學習選擇準則 | 新工具不遷移（S2） |
| D17 | 不自建 trainer | 自寫訓練迴圈 | resume 正確性極易寫錯 | §7.4 CI 失敗 |
| D18 | 記憶檢索 coarse-to-fine | 純向量 top-k | 記憶是 episodic，時間是主索引 | 拼湊出時序錯亂的回憶（S1） |
| D19 | Onboarding MUST 同時採集訪談與結構化問卷 | 只做其中一項 | 2411.10109 v3：訪談型 83%、問卷型 82%、**兩者合併 86%**，合併顯著較佳；兩者皆優於 demographic/persona 型 | S1 偏低（見 `INTERVIEW.md`） |
| D20 | 採集曝光事件，非僅動作事件 | 只記錄本人做了什麼 | 無曝光紀錄則無法區分「不想做」與「沒看到」，硬負例無從產生 | S3 分數虛高但部署失敗 |
| D21 | `split` 於 ingest 階段寫入 | 訓練時才切分 | 時間洩漏無法事後偵測，且指標仍會看起來正常 | 靜默污染，S1/S3 全面失真 |
| D22 | 記憶衝突保留雙版本，不消解 | 選出正確版本 | 消解會製造比本人更準確、因而更不像本人的東西 | S4 辨識率上升 |
| D23 | 第三方標記於 ingest 完成，處置延後 | 等政策裁決後再標 | 為 §11-C 兩種可能結果都預留低成本翻轉 | 政策一變即需重跑全管線 |
| D24 | 自陳提及計入 salience | 僅用頻率與近期性 | 本人主動講起 = 對本人重要，是自陳資料獨有且零成本的訊號 | S1 回想題失分 |
| D25 | 合成軌跡可訓練、不可評測 | 合成資料同時用於兩者 | 以 teacher 產物評測等同自我確認 | 指標與盲測結果背離 |
| D26 | 自陳資料須可原文還原 | 僅存切碎後的碎片 | `EVAL.md` §3.4 的 B2 baseline 需原文注入，B2 是本專案 kill switch | kill switch 失效，無法判斷 LoRA 是否該存在 |
| D27 | v1 surface 為 LINE，單一 | 同時做多個場景 | 三個 dial 互相拉扯，同時做則歸因不可能 | 指標改善但不知為何 |
| D28 | Runtime 為 tick loop，回覆亦為 tool call | 主動性做成獨立子系統 | 使「要不要回」與「要不要查」成為同一決策空間 | 兩套機制行為不一致 |
| D29 | 送出攔截在 runtime 層，非工具層 | 用 `draft()` 與 `reply()` 兩個工具 | 升降級不需重訓、不需改工具定義 | 每次放寬都要重訓 |
| D30 | 「堪用」與「可自動送出」為兩個門檻 | 單一門檻 | T1 允許的 False Alarm 在草稿模式無害、在自動送出不可接受 | 對外事故 |
| D31 | 閘門降級自動、升級手動 | 兩者皆手動 | 降級是安全動作，不應等人；升級是風險動作，應要求證據 | 指標退化後仍在自動送出 |
| D32 | 不實作第三方匿名化，改採三項技術護欄 | 建匿名化管線 | 個人自用開源工具，不代管他人資料；最高風險路徑是 repo 誤 commit 而非模型 | 資料外洩經版控 |
| D33 | Judge 為 Claude Code，非計量 API | Gemini Flash 免費層 | 評測不耗 token 預算；且 teacher(Gemini)／judge(Claude) 天然跨供應商，自我確認風險自動消除 | 評測瓶頸由額度轉為人的時間 |
| D34 | 訪談為單一連續場次 102 分鐘 | 分 2–3 次進行 | 區塊 A 敘事與區塊 B 事例互相召回，跨場次失效 | S1 偏低（`INTERVIEW.md` §3.1） |
| D35 | 不補訪，品質檢核改為標記低信心 | 未達標即補採集 | 接受較低信心優於重來；使訪談員即時檢核成為單點 | 低信心輪次無法用於閘門升級 |
| D36 | 語音品質於資料處理階段解決，不事前試錄 | 先試錄再決定管線 | 不因轉錄品質阻塞採集 | ASR 錯誤污染自陳資料，緩解僅靠後處理與音檔保留 |
| D37 | 自陳資料（逐字稿、問卷）的 `split` 一律 `train`，於 ingest 決定（2026-08-30） | 依 §4.8 時間規則（會落入 sealed） | §2.2 自陳資料「不是歷史資料」，時間規則對它無意義；D19 要它進訓練，`EVAL.md` §3.4 要 B2 讀得到它——兩者需同一份資訊 | 本選項自身的代價：訪談內容可重述 heldout 時段的事件，而 S1 題庫正由該時段生成——T 對 B0/B1 的優勢會混入「記憶」（`EVAL.md` §3.1、§12-7），故 **S1 的 T 結果只以 T vs B2 解讀**。反向錯誤：依時間切分 → B2 讀不到、T 學不到，kill switch 無法裁決 |
| D38 | 訪談員 v1 為文字版（終端逐題、Teacher 追問；未達成的必達點於區塊末補問一輪後即進入下一區塊），語音版延後（2026-08-30） | 等語音管線再訪談；或無限補問直到達成 | Wave 2 前需有 B1/B2；語音管線無規格可依；D35 不補訪，補問上限是同一取捨 | `INTERVIEW.md` §7 Q1–Q4 大概率不過 → 該輪 S1 低信心，S3 在曝光採集上線前不得用於閘門 |
| D39 | **效果優先**：在 kill switch 有結果、孿生效果確認足夠之前，自陳資料（逐字稿、persona）以**完整、未去名**的形式進入 B2 推論與 T 訓練，走既有雲端路徑（Modal 函式輸入；訓練樣本與 LINE 軌跡同樣以明文上 R2 `data/`）；隱私縮減措施（去名過境、本機推論、加密訓練樣本）延後至效果確認後再裁決（2026-08-30） | 先做隱私縮減再看效果；或放棄 B2／不讓 T 學自陳 | 沒有 B2 就沒有 kill switch；不學自陳資料 T 預期輸給 B2（D19）；去名會讓 B2 與 T 的比較失真，量不到真實上限 | 自陳內容經雲端函式輸入與 R2 明文暴露（§8 D32 路徑）——與 LINE 軌跡上 R2 的既有暴露同類、對象更敏感；`INTERVIEW.md` §6.3 對「逐字稿檔」的本機保留仍成立，本條放寬的是由它衍生的訓練樣本與推論 context |

---

## 11. 裁決紀錄（原阻塞清單，已解除）

2026-08-27 全數裁決。本節保留原題目與裁決結果，供日後回溯。

### A. v1 Surface 範圍 → **已裁決：LINE，且回覆為 tool call**

裁決內容：v1 只做 LINE 自動回覆；回覆行為透過 tool call 實現（「突然想回」即為孿生選擇呼叫 `reply`）。

衍生規格：§6.2 執行模型改為 tick loop、§6.3 `reply` 工具、§6.5 送出閘門。

### B. 通過門檻數值 → **已裁決：分階段，T1 為中間標準**

裁決內容：門檻先設中間值，隨迭代調高。具體數字見 `EVAL.md` §7。

**裁決的延伸（需知悉）**：本規格將門檻拆為兩組。T1/T2 為「孿生堪用」的階段性標準，可以放中間並逐步調高；**自動送出閘門（§6.5 L1/L2）另設較高門檻，不適用「先放中間」**。理由見 §6.5。若此延伸不符原意，需重新裁決。

### C. 第三方資料政策 → **已裁決：不實作匿名化，改採技術護欄**

裁決內容：本專案為個人自用之開源工具，非產品，不建匿名化管線。

衍生規格：§8 三項護欄。LINE ingest 封鎖解除。

### D. 底模與尺寸 → **已裁決：多語言需求確定；尺寸取 8B**

裁決內容：MUST 支援繁體中文（台灣用語為主）與英文，SHOULD 支援簡體中文。

尺寸未在裁決中指定，本規格取 **8B** 為預設並記錄降級路徑（§5.1）。此為規格代決，MAY 被推翻。

衍生規格：`EVAL.md` §8.4 新增語碼轉換量測——中英混用比例與切換時機是強烈的人格訊號，且台灣使用者普遍存在。

### E. Judge 一致性下限 → **已裁決：judge 改為 Claude Code，門檻維持 0.80**

裁決內容：LLM-as-judge 使用既有的 Claude Code 訂閱，評測不消耗計量 API token。

衍生規格：`EVAL.md` §6 全面改寫，新增 §6.4 重跑紀律（互動式 judge 帶來 p-hacking 風險），§10 成本模型改寫。

Judge agreement 門檻維持 0.80，且 MUST NOT 隨 §7 的階段性標準下修。它是量測儀器的校準值，不是產品品質目標。

### 剩餘開放項

| # | 項目 | 阻塞範圍 | 狀態 |
|---|---|---|---|
| G | 8B 為規格代決 | §5.1 | 可隨時推翻 |
| H | LINE 曝光訊號的實際可得性 | §4.3、§6.4 | 需技術驗證，非決策 |

原 F（訪談語音管線）已於 `INTERVIEW.md` §8 I-A 裁決，事前試錄取消。**所有決策項均已關閉，剩餘兩項為代決與技術驗證，不阻塞開工。**

---

## 12. 明確排除（v1 不做）

- 端到端多模態微調
- J-lens 實作
- RLHF / DPO
- 24/7 常駐低延遲服務
- 事實正確性保證
- 自建 trainer
- 原始媒體上雲
- 第三方去識別化管線（改採 §8 護欄）
- LINE 以外的 surface
- 計量付費的評測（judge 為既有訂閱）

註：「自動送出對外訊息」已自本清單移除。它不再是 v1 的排除項，而是受 §6.5 閘門管制的能力——預設關閉，達標後開啟。

---

## 附錄：參考文獻

| 引用點 | 文獻 |
|---|---|
| §5.4 反省訓練 | Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*, 2026-07-06 |
| D19 訪談＋問卷合併最佳；S1 正規化方法；`INTERVIEW.md` 全文 | Park et al., arXiv:2411.10109（v1 題為 *Generative Agent Simulations of 1,000 People*；v3 更名為 *LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals*，本規格採 v3 數據） |
| S2 pass^k | Yao et al., *τ-bench*, arXiv:2406.12045 |
| D6/D7 False Alarm；§6.4 F1 現況 | Lu et al., *Proactive Agent*, arXiv:2410.12361 |
| D7 hard negative 選取、time-based split | *ProAgentBench*, arXiv:2602.04482 |
