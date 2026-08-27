---
name: eval-harness
description: 以 Claude Code 作為 judge 執行 EVAL.md 的四個 suite，且不讓評測樣本污染主 session 的 context。使用者說「跑 eval」「跑一輪評測」「judge 一下」「算分數」「這輪結果如何」「對齊 judge」「rubric 改了」時使用；設計或修改評測腳本、切 shard、寫報告格式時也使用。核心規則：樣本只進 subagent，分數只由腳本算，主 session 只讀報告。
---

# eval-harness — context 友善的評測執行

## 為什麼「context 友善」是正確性問題，不只是省錢問題

`EVAL.md` §6 裁決 judge 為 Claude Code。這解決了成本，但引入四個新問題，而它們都長得像「context 太長」：

| 表面症狀 | 實際的正確性問題 |
|---|---|
| 樣本塞滿 context | **逐筆判定不再獨立**。judge 看過前 50 筆才判第 51 筆，判準會漂移。這不是效能問題，是量測儀器被污染。 |
| 對話裡累積前幾輪結果 | **p-hacking 的入口**。看得到上一輪分數，就會不自覺往那個方向判（`EVAL.md` §6.4 明列此風險）。 |
| 主 agent 轉述 rubric 給 subagent | **rubric hash 失去意義**。每次轉述都不同，`EVAL.md` §6.4 要求的 rubric hash 記錄變成裝飾品。 |
| 同一 session 連跑兩次量差異 | **量到假的低差異**。`EVAL.md` §6.4 要量的是跨 session 非確定性；同 session 有記憶，差異被人為壓低。 |

所以下面的規則同時服務兩件事：context 不爆，以及數字可信。**兩者衝突時以數字可信為準。**

---

## 核心原則：資料的可見範圍

```
| 資料                | 允許出現在哪裡                          |
|---------------------|----------------------------------------|
| 原始評測樣本         | 只在 judge subagent，一次一 shard        |
| rubric 全文          | 只在 judge subagent（讀檔，非轉述）      |
| 逐筆判定結果         | 只在磁碟                                |
| 判定理由全文         | 只在磁碟                                |
| 聚合指標             | 主 session（透過 report 檔）             |
| 理由的分類統計       | 主 session（S4 需要，見 EVAL.md §8.2）   |
```

**主 session 的職責只有編排**：產生 run_id、切 shard、派 subagent、跑聚合腳本、讀報告。

主 session MUST NOT 讀取任何 `eval/in/` 或 `eval/out/` 的檔案內容。要確認格式時用 `wc -l`、`head -c 500`、`jq 'keys'`，不用 `cat`。

---

## 目錄契約

```
eval/
  rubric/<suite>.md              # judge 的唯一 prompt 來源，版本化
  runs/<run_id>.json             # manifest，一旦寫入即 immutable
  in/<run_id>/shard-000.jsonl    # 已剝除來源標籤的樣本
  out/<run_id>/shard-000.jsonl   # 逐筆判定，append-only
  report/<run_id>.md             # 唯一允許進主 session 的產物
```

`eval/` 全目錄 MUST 在 `.gitignore` 中（`SPEC.md` §8 護欄 2）。

---

## 執行流程

### 步驟 0：檢查是否該跑

讀 `eval/runs/` 下的既有 manifest。若已存在相同 `(dataset_hash, rubric_hash, split)` 的 run：

- **MUST 拒絕執行**，回報既有的 `run_id`。
- 唯一的例外：使用者明示 rubric 已變更。此時依 `EVAL.md` §6.4，MUST 重跑**全部歷史 run**，否則跨輪比較失效。**MUST 在動手前指出這件事的代價，而不是默默只跑新的一輪。**
- 「上次分數不好想再跑一次」**不是**例外。這是 `EVAL.md` §12 反模式第 10 條。

### 步驟 1：產生 run manifest

在跑任何 judge 之前寫入，內容至少涵蓋 `EVAL.md` §11 報告格式所要求的識別欄位（run_id、base_model、adapter_hash、dataset_hash、eval_set_version、rubric_hash、日期、split、shard 清單）。

- MUST 先寫 manifest 再跑 judge。反過來的話，跑完才決定要不要記錄，就是選擇性回報。
- manifest 寫入後 MUST NOT 修改。要改代表要開新 run_id。

### 步驟 2：切 shard（腳本做，不是 Claude 做）

- **來源標籤 MUST 在切 shard 的腳本中物理剝除**，MUST NOT 靠 prompt 叫 judge「不要看標籤」。標籤只要在檔案裡，它就在 judge 的 context 裡。這是 `EVAL.md` §6.1 盲測要求的唯一可靠實作。
- **樣本 id MUST 由內容 hash 派生，MUST NOT 用序號。** 序號在重排後對不上，`EVAL.md` §6.4 的跨 session 逐筆比對就做不了。
- 每 shard 的筆數 MUST 寫在 config，MUST NOT 由「context 還剩多少」決定。理由：shard 大小是 rubric 的一部分（它決定 judge 一次看到多少上下文），因此 MUST 進 rubric_hash。**大小會飄，rubric_hash 就不代表任何東西。**
- 過長樣本的截斷／摘要 MUST 在腳本層完成，規則 MUST 進 rubric_hash。

### 步驟 3：派 judge subagent（一 shard 一個，全新無狀態）

用 `eval-judge` subagent。每個 shard 開一個新的，**MUST NOT 讓同一個 subagent 連跑多個 shard**。

派工時只給三樣東西：rubric 檔路徑、shard 檔路徑、輸出檔路徑。

- MUST NOT 在 prompt 中複述 rubric 內容——subagent 自己讀檔。
- MUST NOT 在 prompt 中提及本輪目標、前幾輪分數、或這批樣本可能來自哪裡。
- subagent 的回傳值 MUST 只有統計：處理筆數、無法判定筆數、輸出檔路徑。**MUST NOT 回傳任何判定內容或樣本文字**——那等於把樣本搬進主 context，前面的隔離全部作廢。

### 步驟 4：程式驗證的部分不進 judge

依 `EVAL.md` §4.2，S2a 的任務完成判定 MUST 可程式驗證，MUST NOT 由 LLM 判斷。

harness MUST 有兩條互不重疊的路徑：

- **programmatic verifier 路徑** — S2a 的終局狀態比對（檔案內容／DB state／檢索結果集合）。完全不經 judge。
- **judge 路徑** — 需要語意等價判定的部分。

把 S2a 丟給 judge 是雙重錯誤：既浪費 context，又違反 `EVAL.md` §4.2。

### 步驟 5：聚合（腳本算，Claude 不算）

依 `EVAL.md` §6.2 第 4 點，分數 MUST 由腳本計算。

- Claude MUST NOT 口述、心算、估計或「大概是」任何指標。
- 聚合腳本 MUST 拒絕產生跨 suite 加權總分（`EVAL.md` §1.4）。這 MUST 是腳本層的斷言，不是一句叮嚀。
- 報告欄位依 `EVAL.md` §11，逐字對齊。**MUST NOT 自行增減欄位或改名。**

### 步驟 6：閘門檢查（在輸出報告之前）

依 `EVAL.md` §6.3，judge 對齊未達門檻時，judge 結論 MUST NOT 被採信。

harness 的實作 MUST 是**中止並拒絕輸出報告**，MUST NOT 是「輸出報告 + 附上警語」。

理由：附警語的報告一定會被當成報告用。警語會被讀一次，數字會被引用十次。

同理，`EVAL.md` §6.4 標記為低信心的輪次，harness MUST 在報告中把 `confidence: low` 放在**第一行**，並 MUST 拒絕輸出任何閘門升級建議。

---

## 非確定性的量測（最容易做假的一步）

`EVAL.md` §6.4 要求對 sealed split 連跑兩次並回報差異。

- 兩次 MUST 為**兩個獨立的 session**，不同的 run_id。
- MUST NOT 在同一次對話中連跑兩次。同一 session 有記憶，量到的差異必然偏低，而偏低的差異會讓低信心輪次被誤判為高信心，進而被用於閘門升級——這正是 `EVAL.md` §12 反模式第 13 條。
- 差異的計算 MUST 為逐筆比對（靠步驟 2 的內容 hash id），MUST NOT 只比總分。總分相同而逐筆判定不同，是最壞的情況，且只有逐筆比對看得見。

---

## Judge 對齊（每次 rubric 變更都要重做）

依 `EVAL.md` §6.3：隨機抽樣、本人親自標註、計算 agreement。

harness 的責任：

- 抽樣 MUST 由腳本用固定 seed 執行，seed 寫入 manifest。**Claude MUST NOT 挑樣本**——挑出來的樣本會偏向好判的。
- 本人的標註與 judge 的判定 MUST 在**不同檔案**產生，比對由腳本做。若本人標註時看得到 judge 的答案，agreement 就沒有意義。
- agreement 門檻取自 `EVAL.md` §6.3。**MUST NOT 因為孿生分數不好而下修它**（`EVAL.md` §12 反模式第 12 條）。它是儀器校準值，不是產品目標。

---

## 一輪評測要跑哪些

依 `EVAL.md` §10.1 的頻率建議。此處只強調兩條：

- **MUST NOT 只跑 S1 就宣告改善。** S1 提升常以 S3 惡化為代價（`EVAL.md` §10.1、§12 反模式第 8 條）。
- 需要人工的部分（盲測、judge 對齊）MUST 在報告中明確標示「本輪未執行」，MUST NOT 沿用上一輪的數字填空。

---

## 反模式（本 skill 專屬）

1. `cat` 樣本檔進主 context 檢查格式（改用 `wc -l` / `head -c` / `jq`）
2. 主 agent 轉述 rubric 給 subagent（rubric_hash 失效）
3. 同一 subagent 連跑多個 shard（判準跨 shard 漂移）
4. subagent 回傳判定內容而非只回統計（樣本進主 context）
5. 同一 session 連跑兩次量非確定性（量到假的低差異）
6. 靠 prompt 而非腳本剝除來源標籤
7. shard 大小依 context 剩餘量動態決定（rubric_hash 不再穩定）
8. 樣本 id 用序號（跨 run 逐筆比對做不了）
9. 讓 judge 給總分或百分比（`EVAL.md` §6.2）
10. agreement 未達標仍輸出報告加警語
11. 把 S2a 的完成判定交給 judge
12. 分數不理想重跑同一批樣本
13. 產生跨 suite 加權總分
14. 只跑 S1 就宣告改善
