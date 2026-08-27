---
name: data-contract
description: 維持碎片、軌跡、切分標記、時間欄位在整條管線上的一致性。使用者要新增或修改資料欄位、寫 ingest／載入／檢索／回流程式碼、debug「資料看起來對但下游行為怪」、或處理 schema 版本問題時使用。核心規則：SPEC 的 schema 是唯一事實來源，改欄位的順序不可反，ingest 決定的欄位下游不可改。
---

# data-contract — 資料流一致性

## 這個 skill 解決的問題

資料流不一致的特徵是：**每一段程式碼單獨看都對，串起來靜默地錯**，而且測試全綠。

`SPEC.md` 已經標出了本專案最貴的兩個實例：

- 時間洩漏（D21）：「無法事後偵測，且指標仍會看起來正常」
- 硬負例缺失（§4.11）：「只在資料層被隱藏，在 S3 分數上看不出來」

兩者的共同結構是：**錯誤發生在寫入端，症狀出現在指標端，中間沒有任何一處會報錯。** 下面的規則全部是為了把錯誤往前推到寫入的那一刻。

---

## 規則 1：Schema 是規範的，且只有一份

`SPEC.md` §4.4（碎片）與 §4.10（軌跡）具約束力。

- Repo 中 MUST 有機器可讀的 schema（例如 `schemas/fragment.schema.json`、`schemas/trajectory.schema.json`），且 MUST 有一個測試斷言它與 `SPEC.md` 的欄位集合一致。schema 與 SPEC 分歧時，**測試要紅**。
- 程式碼中 MUST NOT 出現手寫 dict literal 構造碎片或軌跡。一律走同一個 constructor + validator。
- 理由：dict literal 會漏欄位，而漏掉的通常是 `split`、`negative_class`、`third_party_spans` 這種「可以空著」的欄位——它們正是最不能空的。

**判斷方法**：在 repo 裡搜 `"fragment_id"` 或 `"trajectory_id"` 這種字面量。除了 schema 檔與 constructor 之外的每一處命中，都是候選 bug。

---

## 規則 2：改欄位的順序不可反

```
SPEC.md（§4.4／§4.10）  →  schema 檔  →  程式碼  →  遷移腳本
```

- MUST 從左到右。反過來做的結果是：程式碼裡有一個 SPEC 沒有的欄位，沒人知道它算不算規範，下一個人不敢刪也不敢用。
- 新增欄位 MUST 同時回答：舊資料這個欄位是什麼？沒有遷移答案的新欄位 MUST NOT 加。
- 刪除欄位 MUST 先確認沒有任何 eval 報告依賴它（`EVAL.md` §11 的欄位是硬契約）。

---

## 規則 3：Ingest 決定的欄位，下游不可改

下列欄位於 ingest 階段寫入後 **MUST NOT 被任何下游修改**：

| 欄位 | 依據 | 被下游改動的後果 |
|---|---|---|
| `split` | `SPEC.md` §4.8、D21 | 時間洩漏。無法事後偵測，指標仍正常 |
| `event_time` | `SPEC.md` §4.4 | 時序錯亂的回憶（S1 失分） |
| `negative_class` | `SPEC.md` §4.11 | S3 分數虛高、部署後亂發文 |
| `ground_truth_source` | `SPEC.md` §4.10 | 合成資料混入評測（D25） |
| `third_party_spans` | `SPEC.md` §4.9 | 政策一變即需重跑全管線 |

實作要求：

- 這些欄位 MUST 在 constructor 中設定，之後為唯讀。若語言層面做不到唯讀，MUST 有測試覆蓋「下游不寫這些欄位」。
- 訓練載入層 MUST 硬性過濾 `split != "train"`，且此過濾 **MUST 有測試覆蓋**（`SPEC.md` §4.8 明文要求）。
- **這條測試 MUST NOT 被 skip、MUST NOT 被標 xfail。** 它是整個專案唯一能擋住時間洩漏的東西。

---

## 規則 4：時間是結構，不是字串

`SPEC.md` §4.4 的 `event_time` 是 `{value, precision, confidence}` 三元組。

- 任何把它當 ISO 字串直接 parse 的程式碼即為 bug。它會靜默丟掉 `precision`，而 `precision` 正是 `EVAL.md` §3.5「模糊度匹配」的量測對象。
- 比較兩個 `event_time` MUST 考慮 precision。`"2024-06"` 與 `"2024-06-15"` 不是「其中一個資料不全」，是**兩種不同的記憶精度**，兩者都對。
- 排序時 MUST 定義部分精度的排序規則並集中在一處，MUST NOT 每個呼叫點各自處理。

**判斷方法**：搜 `fromisoformat`、`strptime`、`parse_date`。每一處命中都要確認它拿到的是 `event_time["value"]` 而不是 `event_time`，而且 precision 有被一起帶走。

---

## 規則 5：資料流單向，回流走管線

依 `SPEC.md` §3 的分層，依賴方向 MUST 為 `L1 → L2 → L3 → L4`。

唯一的反向資料移動是送出閘門的否決回流（`SPEC.md` §6.5：每一次本人的「不送出」MUST 被記錄為硬負例並回流資料集）。它的實作 MUST 是：

- 產生**新的**軌跡樣本，走一般 ingest 管線，由 ingest 決定其 `split` 與 `negative_class`。
- **MUST NOT 由 L4 直接改寫既有樣本**，MUST NOT 由 L4 自行決定 `split`。
- 理由：若 runtime 能寫 `split`，那麼「今天跑過的 runtime」就會影響「明天的訓練/測試邊界」，評測不可重現，而且看不出來。

---

## 規則 6：曝光是前置條件，不是可選欄位

`SPEC.md` §2.3 與 §4.3：沒有曝光紀錄，就無法區分「不想做」與「沒看到」。

- 產生 `no_action` 軌跡的程式碼 MUST 同時處理 `exposure`。缺少曝光證據的 `no_action` MUST 被標為 trivial（`SPEC.md` §4.10 明文規定），MUST NOT 標為 hard，MUST NOT 留空。
- 標成 hard 的成本是：S3 分數虛高、部署後孿生整天發廢文，且**分數上看不出來**。這是本專案第一號失敗模式。
- 若某 surface 的曝光訊號不可得，依 `SPEC.md` §4.3 該 surface MUST NOT 用於 S3 評測，且 MUST 於報告中聲明。這個聲明 MUST 由程式產生，MUST NOT 靠人記得寫。

---

## 規則 7：路徑是 URI，不是本機路徑

`SPEC.md` §7.2：所有路徑 MUST 為 URI，經 fsspec 統一；程式碼中 MUST NOT 出現本機絕對路徑。

**判斷方法**：搜 `open(`、`Path(`、`os.path.join`、`/home/`、`C:\`、`./data`。每一處命中要確認它走 fsspec 而非直接檔案系統。

---

## 規則 8：Hash 的計算方式是契約

`SPEC.md` §7.5：一個 `run_id` MUST 綁定 seed、資料集版本 hash、config hash。

- `dataset_hash` 的計算方式 MUST 固定並集中在一處（例如：排序後的 fragment_id + content 的 hash）。
- 若計算方式改變，**所有歷史 run 的可比較性即刻失效**。改動它 MUST 明說這件事，並依 `EVAL.md` §6.4 的邏輯處理歷史 run。
- hash MUST NOT 包含時間戳、檔案 mtime、字典順序等非決定性輸入。這類輸入會讓「同一份資料」每次算出不同 hash，於是 `SPEC.md` §7.5 的比較前提永遠不成立，而沒有人會發現——只會覺得「怎麼每次都是新的 run」。

---

## 快速稽核清單

改動涉及 L1/L2 時逐條走一遍：

- [ ] 新／改欄位在 `SPEC.md` 有對應，且順序是 SPEC → schema → 程式碼
- [ ] 沒有新的 dict literal 構造碎片／軌跡
- [ ] 沒有下游寫入 `split` / `event_time` / `negative_class` / `ground_truth_source`
- [ ] `split != "train"` 的過濾測試仍存在、未被 skip
- [ ] `event_time` 的 precision 沒有在任何轉換中被丟掉
- [ ] `no_action` 樣本都帶著 exposure 證據，證據缺失者標 trivial
- [ ] 沒有新的本機絕對路徑
- [ ] `dataset_hash` 的計算方式未變；若變了，已回報歷史 run 失效
