# PLAN.md — twin 子系統實現計畫

| 項目 | 值 |
|---|---|
| 版本 | 0.5 |
| 日期 | 2026-08-27（Phase 0、Phase 1 完成；L2/L3/L4/harness interface-first 補建完成，見 §3.8；Phase 4 訓練垂直切片程式部分完成，見下方與 Phase 4 章節） |
| 依據 | SPEC.md v0.4、EVAL.md v0.2、INTERVIEW.md v0.2（見附錄） |
| 狀態 | Phase 0（護欄/套件骨架）、Phase 1（Fragment schema／split／teacher.py／最小 ingest）程式部分皆已落地；兩者各自的人工步驟（GCP/雲端帳號、使用者真實資料）仍待辦，見各自章節。§3.8 記錄了一輪跨 phase 的 interface-first 補建（L2/L3/L4/harness 的介面與 core 的支撐型別），刻意不算某個特定 phase 完成，細節仍待鎖定區塊時補上。**Phase 4（最小 L2 + 精簡軌跡集 + 第一版 LoRA）程式部分已完成**（底模定案 Qwen3-8B，`train/{formatting,model,checkpoint,reproducibility,run}.py`、根目錄 `train.py`、`launch/*`，SPEC §7.4 的 kill-9/resume CI 測試已通過，見該章節「狀態」）；LoRA rank 硬體 probe、真實資料訓練跑仍待辦。Phase 2、3、5 起仍為草稿。路線圖依 SPEC/EVAL 的既有裁決推導，未包含任何本文件自行代決的規範性內容 |

---

## 0. 這份文件的定位

`twin/reference/` 是一份已全數裁決、但完全沒有程式碼落地的規格。這份文件把它轉成一條**可驗收、有順序**的建置路線圖，加上一套**具體的套件結構**。它不重述 SPEC.md 的規則本身——每一條都引用條號，SPEC.md 永遠是唯一的規範來源；這份文件錯了可以改，SPEC.md 的規則錯了要走它自己的裁決流程。

讀法建議：先讀 §1（關鍵路徑），那是全案唯一的硬約束，決定了最早能做完什麼。再依序讀 §2 的各 phase。§3 是等真的要動手寫程式碼時才需要細讀的套件結構。

---

## 1. 關鍵路徑（先讀這節）

```
Phase 0（護欄）
  → Phase 1（最小 ingest，產出真實 held-out 時段）
  → Phase 2 完成 = Wave 1 作答
  → 〔14 天不可壓縮日曆等待 — EVAL.md §1.2 / §3.2〕
      期間平行進行（不在關鍵路徑上）：
      Phase 3（訪談 + 後處理）
      Phase 4（最小 L2 + 精簡 LoRA）
      Phase 5（baseline + judge harness）
  → Phase 6 = Wave 2 作答 + judge 對齊
  → Phase 7 = kill switch 裁決
```

這 14 天是整個專案早期唯一的硬約束（EVAL.md §1.2：「自我一致性是 S1 的分母……第一輪作答為專案第 0 天工作項，先於任何訓練」），而且長度剛好足以吸收 Wave 2 之前幾乎全部的建置工作（訪談、最小 L2、精簡 LoRA、harness）——前提是這些工作從第 0 天就平行開始。**核心排程建議：Phase 3、4、5 從第 0 天開始，不是等 Wave 2 結束才依序動工。**

下游還有兩個較小、不在最早關鍵路徑上但同樣真實的不可壓縮項目：
- **S4 盲測排程**（Phase 12）：協調本人 + 2–3 位熟識者，數天量級。
- **30 天 L1 實跑**（Phase 14，延伸項）：遠長於 14 天等待，但依 SPEC.md §9 對「完成」的定義（T1 全過即可進入 L0 草稿模式），並非必要條件。

---

## 2. 分階段路線圖

### Phase 0 — 護欄與帳號（阻塞項）

**狀態：程式部分已完成（2026-08-27）。人工帳號部分：GCP、Modal、Cloudflare R2、Kaggle 已完成（2026-08-28）；Lightning AI 仍待辦，見下方「仍待辦」。**

- [x] 修改 repo 根目錄 `.gitignore`：加入 `twin/data/`、`twin/adapters/`、`twin/transcripts/`、`twin/eval/`（**目前缺失，本次規劃已核實**）。實作時改用 `twin/` 前綴而非原文的裸露路徑——`twin/.gitignore`（巢狀）已完整涵蓋這四個目錄，根目錄這份純屬 defense-in-depth，加前綴可避免未來 repo 其他地方出現同名目錄時被誤傷；不加任何測試斷言這份根目錄副本的存在，真正的防線是 `twin/.gitignore` 與 pre-commit hook。
- [x] 新增 `twin/.gitignore`（同樣四個路徑，裸露寫法，範圍限定 `twin/` 之下，git 支援巢狀 `.gitignore`），外加 twin 自己的 build 產物樣式（`__pycache__/`、`.venv/`、`dist/` 等，見 §3.7）。
- [x] 新增 repo 根目錄 `.pre-commit-config.yaml`（**repo 目前完全沒有任何 pre-commit 設定，已核實**）：local hook，`language: fail`（pre-commit 內建、專為「這個路徑永遠不准進」設計，不需 shell/subprocess，避開 CLAUDE.md 提過的 Windows shell-quoting 陷阱），`files` 限定 `^twin/(data|adapters|transcripts|eval)/`，只在 staged 路徑落在這四個子目錄時觸發，對 ai-studio 貢獻者零影響。已跑過真實 end-to-end 驗證：`pre-commit install` 後對 `twin/data/_verify.json` 執行 `git add -f` + `git commit`，commit 被擋下，訊息引用 SPEC.md §8 護欄 2；驗證後已清除測試檔案，未留下任何 commit。根目錄 `pyproject.toml` dev 依賴新增 `pre-commit`。
- [x] `twin/README.md` 補上責任聲明：第三方內容的合法性由使用者自行負責，本專案不代為處理（SPEC.md §8 護欄 3 原文精神）。
- [x] 建立 Teacher 專用 GCP 專案，確認**未**綁定 billing。**已完成（2026-08-28）**——使用者已在 GCP Console 的 Billing 頁面確認未連結任何付款方式；API key 已產生，存入 `twin/.env`（gitignored，`git check-ignore -v twin/.env` 驗證過）。
- [ ] 開通 Modal、Cloudflare R2、Kaggle/Lightning AI 帳號（核准時程不可控，及早申請）。
  - **Modal 已完成（2026-08-28）**：帳號註冊、`uv run modal setup` 完成 CLI 認證（workspace `jieyao-milestonehub`，token 存於 `~/.modal.toml`），以 `modal profile current`／`modal app list` 驗證過。`modal` 補進 twin 依賴（`launch/modal_app.py` 原本 `import modal` 但套件從未被安裝，此輪修正）；同時修正 `launch/modal_app.py` 的 `add_local_dir` 少了 `ignore=`：`copy=True` 會把整個本機目錄烤進 image layer，這是 SPEC.md §8 護欄 2（`data/adapters/transcripts/eval` 不可離開本機）以外的獨立管道——git 的 `.gitignore`／pre-commit hook 完全管不到它——已加 `ignore=[".env", ".git", ".venv", "__pycache__", "data", "adapters", "transcripts", "eval"]`。
  - **Cloudflare R2 已完成（2026-08-28）**：bucket `twin-checkpoints` 已建立（原本帳號裡沒有任何 bucket，用使用者提供的 access key/secret 透過 `checkpoint.py` 的 `_fs_and_path()` 直接建立）。**已核實而非猜測**：`_fs_and_path()` 是裸 `fsspec.core.url_to_fs(uri)`，沒有帶 `endpoint_url` 之類的額外設定，實測確認純靠 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_ENDPOINT_URL_S3`/`AWS_DEFAULT_REGION=auto` 四個環境變數即可正確路由到 R2 endpoint，**不需要改 `checkpoint.py` 本身**。但發現這四個變數必須真正進到 process 的 `os.environ`——`twin.config.settings.Settings` 讀 `.env`只餵給 pydantic 內部欄位，從不寫回 `os.environ`——所以在 `train.py` 加了一行 `load_dotenv()`（新增 `python-dotenv` 依賴），已用「完全比照 train.py 實際行為、shell 不手動 export」的方式重新驗證過一次，確認可行。`TWIN_CHECKPOINT_STORE_URI=s3://twin-checkpoints` 寫入 `twin/.env`。
  - **Kaggle 已完成（2026-08-28）**：帳號註冊。**過程中發現一個文件陷阱**：先前用網路搜尋查到的「kaggle.json 存 username+key」是官方文件標示的 **Legacy** 認證法；實際裝進來的 `kaggle==2.2.4` CLI 要求的是單一 bearer token，存在 `~/.kaggle/access_token`（或 `KAGGLE_API_TOKEN` 環境變數）——已用真實 `kaggle kernels list --mine` 呼叫驗證通過（列出使用者既有的 kernel）。Token 存於機器層級的 `~/.kaggle/access_token`，不進 `twin/.env`（`launch/kaggle.sh` 直接呼叫 CLI，不經過 `train.py` 的 `load_dotenv()`），性質上跟 Modal 的 `~/.modal.toml` 一樣是本機認證檔案。
  - **Lightning AI 仍待辦**（理由同上，人工帳號步驟；優先序原本就排在 Modal/Kaggle 之後）。
- [x] `twin/` 套件骨架：`twin/pyproject.toml`、`twin/src/twin/`、`twin/uv.lock`（獨立於 root 的 import-linter 契約，見 §3.2）。9 層（`twin.cli`/`harness`/`agent`/`memory`/`train`/`ingest`/`teacher`/`config`/`core`）皆為空殼 package（僅 module docstring，無邏輯——Fragment/Trajectory/teacher.py 的 Gemini 綁定等留給 Phase 1）；`uv run lint-imports` 6 條契約全數通過。§3.2 原文的 `forbidden_modules = ["google.genai", "google.generativeai"]` 在實作時發現 import-linter 不支援 external package 的子模組層級封鎖，改為封鎖整個 `google` namespace（效果等同、範圍更嚴格，因為 twin 目前沒有理由 import 任何其他 `google.*`）；並補上 `include_external_packages = true`（契約引用外部套件如 `modal`/`kaggle`/`google` 時的必要設定，§3.2 原文未列出，屬本輪實作補完）。CLI 端點 `twin` 目前僅一個 `version` 子命令（其餘 noun 隨各自 phase 落地，見 §3.4）；空的 `typer.Typer()` 無任何 command 時無法被呼叫，因此加了一個空的 `@app.callback()` 以維持多子命令模式，供未來 `ingest`/`interview`/... 使用同一個 group 型 CLI。

**依據**：SPEC.md §5.2/D8、§7.1、§7.2、§7.3、§8 護欄 2/3。
**驗收**：對 `data/` 底下的測試檔案 commit 會被 hook 擋下（**已用真實 commit 嘗試驗證，見上**）；GCP console 確認無 billing 帳號（**已完成 2026-08-28**）；README 已有聲明（**已完成**）。
**類型**：程式（已完成）+ 人工（GCP／Modal／R2／Kaggle 已完成 2026-08-28；Lightning 仍待辦）。

**仍待辦（人工）**：
1. ~~建立 Teacher 專用 GCP 專案並確認未綁 billing。~~ **已完成 2026-08-28**。
2. ~~開通 Cloudflare R2、Kaggle/Lightning AI 帳號。~~ **Modal／R2／Kaggle 已完成 2026-08-28，見上；Lightning AI 仍待辦**。
（1）的專案憑證是 `teacher.py` 實際發出第一次真實 Gemini 呼叫的前提——**該呼叫已於 2026-08-28 成功執行，見 Phase 1「仍待辦」第 2 項**；Phase 4 的訓練算力需要（2）——**R2 的真實 bucket 已就緒（見上），Modal 也已認證，Phase 4「仍待辦」清單裡「真實訓練跑」的帳號面前提已排除，剩下的是硬體 probe 與精簡軌跡集本身**。實作發現：介面本身、`GeminiTeacher` 綁定、D9 的 RPD ledger 皆不需要真實憑證即可寫出並測試（google-genai SDK 的呼叫用 dependency-injected client 驗證，見 Phase 1）。

### Phase 1 — L1 骨架 + 最小可用 ingest（產出第一個真實 held-out 時段）

**狀態：程式部分已完成（2026-08-27）。以使用者真實資料跑一次正式 ingest 仍待辦（見下方「仍待辦」）。**

- [x] Fragment schema（§4.4）：所有 MUST 欄位落地（`core/fragment.py`：`Fragment`、`EventTime`、`Entities`、`ThirdPartySpan`），含 `split`、`third_party_spans`（預設空陣列）。模型設為 `frozen=True`（data-contract skill 規則3：split/event_time 寫入後唯讀，語言層面直接強制，不只是靠慣例）。`event_time`／`precision` 皆為必填，不給預設值——遵照 §4.4「precision MUST 顯式表示」，避免虛假預設信心。新增 `tests/unit/test_schema_matches_spec.py`：直接解析 SPEC.md §4.4 的 `jsonc` 區塊（剝除註解後即為合法 JSON，含巢狀 `event_time`/`entities`/`third_party_spans[]`）比對 `Fragment.model_fields`，SPEC 與程式碼分歧時測試會紅，而非維護一份手工同步的 schema 檔（見 PLAN §3.6 的取捨理由）；同檔案另有一支 grep 測試斷言 repo 中沒有 `core/fragment.py` 建構子以外的 `"fragment_id"` 字面量。
- [x] ingest 時決定 `split`（§4.8）：`ingest/split.py` 的 `decide_split()`（純函式，train/heldout/sealed 三段時間切分，`sealed_cutoff < train_cutoff` 時明確拋錯）與 `sealed_cutoff_for()`（依 EVAL.md §9 的 20% 保留分割計算 sealed 邊界，取 heldout 視窗中最晚的一段）。測試覆蓋於 `tests/unit/test_split.py`（10 個測試，含邊界值、跨 awareness 比較會拋 `TypeError` 而非靜默錯誤）；未加任何可跳過的 marker。此邏輯之後永遠不可搬到訓練期——`twin.train` 尚未存在，但 import-linter 的 C1 契約已預先擋死 `train ⊬ memory`，`ingest.split` 本身則因為在分層中位於 `train` 之下、且 `train` 尚無任何程式碼引用它，暫時只靠「這條路徑還沒被寫出來」保護，等 Phase 4 寫 `train/data.py` 時要對照 §3.5 的設計補上真正的過濾測試。
- [x] `teacher.py` 介面（可替換）：`Teacher` Protocol（單一 `generate(prompt, *, response_schema) -> T` 方法，刻意通用，Phase 2/3/9 共用同一呼叫形狀）+ `GeminiTeacher`（v1 綁定，client 以 dependency injection 注入，測試用假 client，不需真實憑證）+ `TeacherCallLedger`（比照 `ai_studio.runtime.budget.SpendLedger` 的 day-scoped JSON ledger 形狀，只是量測請求數而非金額，落實 D9「少次、大批」——RPD 耗盡時 `_refuse_if_exhausted()` 主動拒絕而非讓呼叫悄悄失敗）。**已核實**（非猜測）：直接 `uv add google-genai` 並用 `inspect` 內省已安裝的 `google-genai==2.20.0` 套件，確認 `GenerateContentConfig.response_schema` 接受 pydantic model class、`GenerateContentResponse.parsed` 會回傳已解析物件，而非憑記憶假設 SDK 形狀。`config/settings.py` 新增（原規劃未列在 Phase 1，但 teacher.py 讀取憑證/模型名稱是必要支撐）：`gemini_model` 刻意不給預設值——SPEC.md §5.2 只裁決「Gemini Flash 系列」這個家族，沒有釘死單一 model ID，且免費層資格是逐模型判定的，猜一個字串進去正是 D8 想擋的「突發帳單」失敗徵狀的來源。**尚未對真實帳號打過任何一次真的請求**（Phase 0 的 GCP 專案仍待辦）。
- [x] 對一個真實、純文字的資料來源做最小 ingest：選擇 LINE 聊天記錄純文字匯出（`ingest/sources/line.py` 解析器 + `ingest/fragment.py` 組裝成 `Fragment`），對應 SPEC.md §6.4/D27 已裁決的 v1 Surface，是「訊息匯出」這個建議最貼合規格的選擇。`ingest/store.py` 以 fsspec 寫入/讀回 JSONL（§7.2「所有路徑 MUST 為 URI」的要求，即使目前只會解析到 `file://`）。**驗收标准已用貼近真實形狀的資料端到端跑過一次**（非 pytest，手動腳本，見下方「仍待辦」）：6 筆訊息、100% 涵蓋 MUST 欄位、零筆缺 `event_time`、`heldout`/`sealed` 時間確實晚於 `train`。**尚未對使用者真正的 LINE 匯出檔跑過**——匯出格式的確切樣式（分行符號、日期表示法可能因語系/App 版本而異）只在拿到真實檔案時才能最終確認，`ingest/sources/line.py` 的 docstring 已註明這點，解析器對無法辨識的行會直接拋錯而非靜默跳過。
- [x] **`third_party_spans` 實際標註（§4.9、§8 護欄1）**——`spec-auditor` 初次審查判為 BLOCK 後修正：原實作只讓欄位「可為空陣列」，但一個雙人 LINE 對話裡，另一方的每一則訊息本身就是全文皆屬第三方內容，留空不符合「MUST 於 ingest 階段標註...成本已付」。修正為 `fragments_from_line_export()` 新增必填參數 `principal_display_name`（無預設值——不知道本人在這份匯出裡的顯示名稱，就無法判斷誰是第三方，沒有安全的預設可退）；非本人 sender 的訊息，整段 `content` 標記一個涵蓋全文的 `ThirdPartySpan`。**明確不含**：本人自己訊息中「提及」第三方的偵測（屬 §4.9 更細緻的實體抽取，`ingest/entities.py`，留待後續 phase）。同一輪順手修正 `spec-auditor` 另外點出的 SHOULD 偏離：`_format_event_time_value()` 讓 `event_time.value` 依 `precision` 正確截斷（原本不論 precision 一律輸出分鐘級字串，LINE 恰好都是分鐘級所以沒炸過，但下一個非分鐘精度的來源重用這支共用函式時會產生「虛假精確度」）。複審後 `spec-auditor` 判定 PASS。

**依據**：SPEC.md §4.4、§4.8/D21、§4.9/D23、§5.2、§7.1/D12。
**驗收**：ingest 產出的碎片 100% 涵蓋 MUST 欄位（**已用測試與手動端到端腳本驗證，見上**）；腳本檢查零筆缺 `event_time`（**已驗證**）；`heldout` 時段的時間確實晚於 `train` 時段（**已驗證**）；`third_party_spans` 對非本人發言者的訊息確實有標註（**已驗證，見上**）。
**類型**：程式（已完成）+ 人工（**仍待辦**：提供使用者真實匯出檔案，對其跑一次正式 ingest）。
**審查**：`spec-auditor`（初次 BLOCK → 修正 → 複審 PASS）、`data-hygiene`（PASS，時間洩漏/切分污染/frozen-model 防寫入三項未見違規；記錄兩項非阻塞觀察：`write_fragments_jsonl` 的全覆寫語意在未來多來源 ingest 前需先定使用慣例，`train/data.py` 的 split 過濾測試仍待 Phase 4 補上）。

**仍待辦**：
1. 使用者提供一份真實的 LINE 聊天記錄（或其他純文字）匯出檔案，對其實際執行 `fragments_from_line_export()` 並用 `write_fragments_jsonl()` 落地到 `twin/data/`（gitignored）。
2. ~~Phase 0 的 GCP 專案就緒後，對 `GeminiTeacher` 打第一次真實請求，確認 RPD ledger 與 D8 的「未啟用 billing」防線在真實流量下仍然成立。~~ **已完成 2026-08-28**：`GeminiTeacher.from_settings()` 對 `gemini-3.5-flash-lite` 打了一次真實請求（一次性驗證腳本，非落地測試），成功解析 `PingResponse`，ledger 從 0 累加到 1，確認記帳邏輯在真實流量下正確。`TWIN_GEMINI_MODEL` 依 AI Studio 的免費層儀表板選定，而非猜測值。

### Phase 2 — S1 題庫 + Wave 1 作答（**專案第 0 天**）

- 一次批次 Teacher 呼叫，依 §3.2 的題型比例（30/25/25/20：價值取捨/偏好/反應傾向/回想），從 Phase 1 的 held-out 碎片產生 60–80 題情境題。
- 最簡單的作答與計時工具（試算表即可）。

**依據**：EVAL.md §1.2、§3.2、§3.3。
**驗收**：`R1` 已記錄，題庫已凍結／雜湊，時間戳存在——此時間戳即為專案第 0 天。
**類型**：程式（少量）+ 人工（作答本身）——**全計畫最重要的一個驗收點**。

### Phase 3 — AI 訪談員、訪談本身、後處理（與 Phase 1/2 平行，須在 14 天內完成）

- AI 訪談員（INTERVIEW.md §4/§6）：語音對語音、Teacher 驅動、追蹤大綱與必達點（A1–A4、B1–B8、C1–C3、D1–D2）、即時追問、單一連續 102–120 分鐘場次（D34，不得分段）、即時自我檢核發言占比（≥70%，Q5）。
- 可重跑的後處理管線（§6.2）：專有名詞校正（需聯絡人/訊息詞表）、中英混用還原、口語保留、`[unclear]` 標記。音檔保留至 QC 通過。
- 品質檢核（§7，Q1–Q9）：Q8（`third_party_spans` 已標註）是唯一硬阻擋；其餘未過僅記錄為對應 suite 的低信心，不阻擋流程。
- 結構化問卷（§5），訪談後才施測，題庫與 S1 題庫互斥。
- **特別護欄**：逐字稿與原始音檔永遠留在 `file://`，絕不進 `r2://`（INTERVIEW.md §6.3、§8 I-D——這不是待裁決事項，是匿名化被否決後僅存的防線之一，D32）。只有衍生出的 Period 層級碎片可走一般跨雲同步路徑（§7.2）。

**依據**：INTERVIEW.md §3–§8；SPEC.md §4.1、§4.6/D24、D19、D26、D34–D36。
**驗收**：一份連續逐字稿 ≥5,500 字，Q8 已過；逐字稿標記 `source_class: self_report`，可原文還原（D26）。
**類型**：程式（訪談員 + 後處理管線）+ 人工（102–120 分鐘場次本身，須預留緩衝，不能卡在第 13 天才做）。

### Phase 4 — 最小 L2 + 精簡軌跡集 + 第一版（精簡）LoRA（與 14 天等待平行）

**狀態：訓練垂直切片（`train/model.py`、`checkpoint.py`、`run.py`、`reproducibility.py`、新增 `train/formatting.py`、根目錄 `train.py`、`launch/*`）程式部分已完成（2026-08-27）。最小 `recall()` 已在 §3.8 建好；精簡軌跡集本身（取自真實行為資料）與 LoRA rank 的硬體 probe、真實一輪訓練仍待辦，見下方「仍待辦」。**

- 最小 `recall(query, time_hint)`，以一般工具（C4）形式包在 Phase 1 的碎片之上——刻意簡陋（關鍵字 + 時間窗過濾），尚非完整的 Episode/Period 分層。（已於 §3.8 落地）
- 精簡軌跡集（§4.10），取自容易觀測、有明確 ground truth 的行為資料——刻意跳過嚴謹的硬負例篩選（§4.3 明確允許在大規模歷史 ingest 之前，先以較低信心或無曝光門檻的負例上路，因為 S3 不在 kill switch 的關鍵路徑上）。**仍待辦**：這批真實精簡軌跡集本身尚未產生，`train/formatting.py` 已備妥把 `Trajectory` 轉成 SFTTrainer 訓練樣本的管線（含 §5.3/D16 的工具名稱遮蔽），等軌跡資料就緒即可餵入。
- 選定底模：**Qwen3-8B**（dense、Apache-2.0，2025-05 發布）——8B 級、open-weight、permissive license（§5.1）。與 Qwen3.5-9B（Unsloth 官方文件明講不建議 QLoRA 4-bit，量化品質劣化，bf16-LoRA 需 22GB 超出 T4 預算）、Breeze-7B-Instruct-v1.0（唯一過授權關的繁中專用選項，但生態小、agentic 能力弱、已停止更新）、Llama-3-Taiwan／TAIDE／Gemma 2-3（皆因授權條款不符「Apache-2.0 同等」被排除）比較後定案。繁中/台灣用語預設偏簡體的已知代價，計畫靠訓練資料本身（精簡軌跡集）矯正，而非 prompt。**抽測繁簡一致性仍待辦**（需真實訓練跑完成後才能做）。
- `train.py` 走 TRL + Accelerate（§7.6 禁止自建 trainer）；checkpoint 契約（§7.4：adapter、optimizer state、LR schedule、RNG state、`global_step`、dataloader cursor）——**已用真實 `kill -9` + `--resume auto` 驗證**（`tests/unit/test_train_checkpoint_kill_resume.py`：全本地、無網路依賴的玩具 Qwen3 模型，真子行程 `SIGKILL`，比對中斷續跑與不中斷對照組的逐步 loss 曲線，容忍度依實測校準）；`run_id` 綁定 seed/dataset_hash/config_hash（§7.5，`train/reproducibility.py`）；載入時做工具名稱遮蔽（§5.3/D16，接在 `train/formatting.py`）；Modal/Kaggle/Lightning 的 `launch/*`（D12）。
- **Adapter 權重（checkpoint 與最終產出）皆加密儲存**（§8：「Adapter 為個資...MUST 加密儲存」）——`core/encryption.py`（Fernet 對稱加密）、`train/checkpoint.py` 把每次 checkpoint／最終 adapter 打包成單一加密封存檔上傳，`config/settings.py` 新增 `TWIN_ADAPTER_ENCRYPTION_KEY`（無預設值，理由同 `gemini_model`），`examples/generate_adapter_encryption_key.py` 供操作者產生金鑰。範圍明確界定於 adapter 權重本身，不含 `AdapterManifest` JSON（後者是 run_id/雜湊/時間戳等 metadata，本身無法反推行為特徵）。
- **`reply` 在訓練資料中確實表示成跟其他工具同層級的 tool call**（§11-A/D28/D29：「回覆為 tool call」）——`train/formatting.py` 把 `ActionStep` 先轉成合成的 `ToolCallStep(tool="reply", args={surface, content})` 再進 §5.3/D16 的工具名稱遮蔽，讓 `reply` 進入跟 `recall`／`web_search` 同一個遮蔽詞彙池，而不是被模型學成一個永遠不變、不遮蔽的特殊字面量。

**這一輪的 `spec-auditor` 審查**：初次審查判 BLOCK，四項發現——(1) adapter 加密儲存缺失（§8，已修正，見上）、(2) `reply` 未表示成 tool call（§11-A/D28/D29，已修正，見上）、(3) `launch/*` 非純 shell 對 §7.1 字面解讀的不確定性（已記錄於下方「已知偏離」第 1 項，維持現狀，需人裁決）、(4) §7.3 spot/preemptible 未记录（已查證並記錄於下方「已知偏離」第 3 項）。修正 (1)(2) 過程中同時發現並修正一個獨立的正確性 bug：`trainer.model.save_pretrained(final_adapter_uri)` 直接把 fsspec URI 字串傳給不認得 URI scheme 的 `save_pretrained`，會悄悄寫到錯誤的本機路徑而非真正落地——現在改為先存本機再透過 `checkpoint.upload_adapter()` 走加密上傳，並在 kill/resume 測試中新增「下載回最終 adapter 且內容正確」的斷言防止回歸。

**已知偏離，記錄於此而非悄悄發生**：
1. `launch/` 實際上不是純 shell——Modal 的 SDK 是 Python-first（`@app.function(gpu=...)` decorator），CLI 無法純用 flag 定義自訂 GPU job（2026-08-27 對照 Modal 現行文件核實），Kaggle 的 `kernels push` 同樣需要一支程式檔 + `kernel-metadata.json`。新增 `launch/modal_app.py`、`launch/kaggle_kernel.py`、`launch/kaggle_kernel-metadata.json` 作為基礎設施接線（絕不 import `twin.train`，只透過 subprocess 呼叫 `train.py`），放在 `launch/` 底下但在 `src/twin/` 之外，結構上不受 import-linter `root_package="twin"` 掃描，不違反既有契約，但這是對本節開頭樹狀圖「只放 shell」字面意思的偏離。`launch/kaggle_kernel.py` 另有一個真正開放的問題（如何把 `twin` 原始碼本身搬上 Kaggle kernel，目前假設走 git clone + Kaggle Secret，未驗證）；`launch/lightning.sh` 的確切 CLI 動詞集也未對照現行文件核實，三者優先序皆為 Modal > Kaggle > Lightning（同 §7.3 算力表順序），照此排序處理未驗證風險。
2. `requires-python` 由 `>=3.11` 提高為 `>=3.12`（連動 `[tool.mypy] python_version`、`[tool.ruff] target-version`）——`numpy>=2.3`（被 torch/transformers 遞移依賴）在 PyPI 上本身即宣告 `requires-python>=3.12`（已即時查證，非臆測），`>=3.11` 已不是誠實的宣告，在真正的 3.11 直譯器上 `uv sync` 會直接解析失敗。
3. §7.3「MUST 優先使用 spot/preemptible」在三個平台上核實後的現況（2026-08-27 查證）：**Modal 不需額外設定**——其官方文件明講「All Modal Functions are subject to preemption by default」，且 `nonpreemptible` 參數明文不支援 GPU Function，所以 GPU 一律強制可搶佔，`launch/modal_app.py` 的裸 `@app.function(gpu="T4", ...)` 已經滿足這條 MUST，不是遺漏。**Kaggle 這條 MUST 不適用**——其免費層是每週時數配額（30h/週）model，不是 spot 定價/搶佔機制，平台上找不到對應開關。**Lightning AI 未核實**，與該平台 CLI 動詞集的不確定性一併留待實作時查證。

**依據**：SPEC.md §3.1/C4、§4.10、§5.1、§5.2、§5.3/D16、§7.3–§7.6、§7.1/D12、§8、§11-A/D28/D29。
**驗收（CI，非人工）**：`train.py --resume auto` 撐過訓練中途的 `kill -9`，loss 曲線連續（§7.4 明文列為 CI 項目，**已通過**）；產出一個有標籤的 LoRA artifact `T`（**機制已驗證，真實資料上的產出仍待辦**）。
**類型**：程式。

**仍待辦**：
1. `examples/probe_lora_rank.py` 在實際目標硬體（T4 或同級）上跑過，把選定的 LoRA rank（TRL 現行參考設定 r=256 起，含 §11 項目 G 要求的降級留痕）記錄進真實的 `TrainingConfig`。
2. 精簡軌跡集本身（真實行為資料，非玩具/合成）。
3. Phase 0 待辦的 GCP/Modal/R2/Kaggle/Lightning 帳號就緒後，第一次真實訓練跑；抽測繁簡一致性。

### Phase 5 — Baseline + Judge harness（平行，等待期間完成）

- `B0`（底模+空白）、`B1`（底模+persona 段落）、`B2`（底模+Phase 3 逐字稿注入 context）的推論 harness。
- 依 EVAL.md §6.2 腳本化的 Claude Code judge：樣本剝除來源標籤後寫入 `eval/in/<run_id>.jsonl`，judge 僅逐筆語意判定，分數由腳本計算，judge 不給總分。使用 `eval-harness` skill。

**依據**：SPEC.md §4.1/D26；EVAL.md §3.4、§6.1、§6.2。
**驗收**：harness 對合成/假資料能跑通全流程，產出格式正確的 `eval/out/<run_id>.jsonl`。
**類型**：程式。

### Phase 6 — Wave 2 作答 + judge 對齊 + baseline 計分（**第 14 天**）

- Wave 2：本人重新作答同一批（凍結的）題庫 → `R2`。
- judge 對齊（EVAL.md §6.3，任何 judge 結論前必做）：隨機抽 30 筆本人親自標註，計算 agreement；< 0.80 則 rubric 須改寫重測，judge 結論不可採信。
- `B0`/`B1`/`B2` 對 `R2` 計分；`self_consistency = agreement(R1, R2)` 首次可算——這是 S1 的分母。

**依據**：EVAL.md §1.1–§1.2、§3.2、§3.3、§6.3。
**驗收**：`self_consistency`、`B0`/`B1`/`B2` 的 `normalized_accuracy`、`judge_agreement ≥ 0.80` 三者皆已記錄成數字——這是「`T` 必須打敗的門檻」，與 `T` 是否已存在無關。
**類型**：人工（作答、30 筆標註）+ 程式（計分）。

### Phase 7 — KILL SWITCH 裁決點（不是一個建置階段）

- 用 Phase 4 的精簡 `T` 對同一批 S1 題目、對 `R2` 計分。
- 依 EVAL.md §3.4 比較 `T` 與 `B2`：**若 `T` 未顯著優於 `B2`，LoRA 不應存在**。
- 這是一個裁決點，不是交付物。兩種結果：
  - **過**：進入 Phase 8–12（此時投入完整 L1/L2/L4 才有正當性）。
  - **不過**：不圍繞這個 LoRA 擴大投資。依 §10 實際出現的失敗徵狀對症下藥（例如編造人名/日期 → D1 → 先補強 coarse-to-fine 檢索再重訓），或退回偏向 `B2` 型的 context 注入產品型態，在重測前不追加 LoRA 投資。

**依據**：EVAL.md §3.4、§12 反模式第 4 條（只跟 `B0` 比不跟 `B2` 比）。
**驗收**：有 `run_id` 標記、可重現的比較結果，裁決明確過/不過，不是憑印象判斷。
**類型**：程式（比較）+ 人工裁決。

### Phase 8 — 曝光採集

- 各 surface 的曝光採集（§4.3）：RSS/閱讀器已讀＋點擊（高可得性）、瀏覽器歷史（高可得性）、LINE 已讀狀態／聊天室開啟（平台限制，屬待驗證的開放項 H，見 §4）。
- 每筆事件記錄時間戳＋內容摘要，不可只記計數。
- 對不可得訊號的 surface 明確聲明「不可用於 S3」（§4.3），聲明由程式產生。
- **必須先於 Phase 9 的大規模 ingest**（§4.3 明文要求的順序）。

**依據**：SPEC.md §4.3/D20、§11 項目 H。
**驗收**：至少一個 surface 產出真實的時間戳＋摘要曝光紀錄；不可用的 surface 已有書面聲明。
**類型**：程式 + 少量人工（開通帳號/匯出權限）。

### Phase 9 — 全量歷史 ingest + 完整 L2（episode/period/salience/conflict）

- 大規模 ingest 各真實歷史來源，含多模態降維（§4.2）。
- Episode 分群（時間鄰近 × 實體重疊）、Period 聚合、自陳資料掛在 Period 層（§4.5）。
- salience 計分，含自陳提及加權（§4.6/D24）。
- 衝突保留：矛盾碎片兩者皆存、互相標記 `conflicts_with`，絕不自動消解（§4.7/D22）。
- `recall()` 從 Phase 4 的簡陋過濾升級為真正的 coarse-to-fine 檢索——優先處理 Phase 7 實際出現的失敗徵狀。

**依據**：SPEC.md §3.1/C1、§4.2、§4.5/D18、§4.6/D24、§4.7/D22。
**驗收**：`recall()` 能在真實資料上示範 period→episode→fragment 逐層下鑽；刻意製造的矛盾碎片對維持不消解（抽測）；自陳加權後的 salience 可量測地高於純頻率版本（ablation）。
**類型**：程式。

### Phase 10 — 完整軌跡集：硬負例、反省比例

- `no_action` 軌跡正確標 `negative_class`（`hard` 須有真實曝光證據；`exposure.evidence = absent` 強制標 `trivial`，§4.10）。
- 目標：`hard` 佔全部負例至少一半（SHOULD）；`no_action` 佔比對照真實不回應率追蹤。
- 反省步驟（§5.4）比例 15–30%，依 §11 項目 B 確立的門檻做 ablation 驗證。
- 用這份完整資料集重新訓練 `T`。

**依據**：SPEC.md §4.11/D6、D7、D20；§5.4。
**驗收**：腳本稽核確認硬負例比例達標；零筆事後改動 `split`（回歸測試）；零筆 `teacher_synthesized` 軌跡混入任何 eval 檔案（D25）。
**類型**：程式。

### Phase 11 — L4 tick loop、工具、L0 送出閘門、S2/S3 harness

- tick loop（§6.2/D28）：每個 tick 零或多個工具呼叫；呼叫零個工具即為 `no_action`，無特殊分支。
- 內建工具：`recall`、`web_search`、`reply`（§6.3），皆以 MCP 暴露，schema 僅於 runtime 注入（C2）。
- L0 送出閘門（草稿模式，§6.5）——本人每次不送出都記為硬負例，回流進 Phase 10 的資料集。
- LINE surface 配接器（§6.4），可插拔，核心 runtime 不認識任何特定平台。
- S2 harness：20–30 個有可程式驗證終局狀態的 held-out 任務（非 LLM 判斷完成與否）、`pass^1`/`pass^4`、保留 1–2 個訓練時未出現的工具做遷移測試（§4.4）。
- S3 harness：time-based split、負例僅取硬負例（絕不用無活動時段）、以 confusion matrix 為基礎，**False Alarm 為主看板指標**、`silence_rate_delta`。

**依據**：SPEC.md §6.1–§6.5；EVAL.md §4、§5。
**驗收**：一次真實 tick loop 執行中同時出現至少一筆真的 `reply` 草稿與至少一筆真的 `no_action`（同一決策空間）；S2/S3 harness 首次對真實（非合成）資料產出計分結果。
**類型**：程式。

### Phase 12 — 迭代至 T1、S4 盲測、L0 正式上線

- 迭代訓練/檢索，直到 EVAL.md §7.1 的 T1 全部指標**同時**達標（S1、S2a/S2b、S3、S4、judge agreement）——不是只看 S1（§12 反模式第 8 條）。
- S4 盲測（§8）：每類 N≥20 筆真實／孿生樣本，本人＋2–3 位熟識者盲測評分，理由需記錄而非只記數字。

**依據**：EVAL.md §7.1、§8、§11 報告格式。
**驗收**：所有 T1 各項在同一個 `run_id` 下全數達標；孿生進入日常 L0 草稿模式使用；報告中記錄 `gate_level: L0`。
**類型**：程式 + 人工（S4 需排開本人＋2–3 位熟識者，每輪約 60–90 分鐘）。

### Phase 13（延伸）— 升級至 L1 白名單自動送出

手動觸發，須連續兩輪達到更嚴格門檻（§7.2：False Alarm ≤ 0.15、`silence_rate_delta` ≤ 0.10、S1 ≥ 0.78）。

**依據**：SPEC.md §6.5/D31；EVAL.md §7.2、§7.3。
**驗收**：連續兩個 `run_id` 皆過 L1 門檻；閘門手動切換，報告中引用這兩輪。
**類型**：程式（評測）+ 人工裁決。

### Phase 14（延伸）— 升級至 L2 全自動

需要真實 30 天 L1 實跑，期間本人否決率 ≤ 0.10 且 S3 False Alarm ≤ 0.10——第二個、更大的不可壓縮日曆等待，且非「孿生堪用」的必要條件。

**依據**：EVAL.md §7.2。
**驗收**：30 個日曆天在 L1 下實跑完成，否決率與 False Alarm 皆達標。
**類型**：人工（純日曆等待＋真實使用監控）。

---

## 3. 套件結構與工程慣例

### 3.1 目錄樹（src-layout）

選擇 `twin/src/twin/...`（而非扁平的 `twin/twin/...`），原因不只是跟 ai-studio 一致：twin 的訓練端依賴（TRL/Accelerate/peft/torch，跑在遠端 GPU 機器上）與互動/serving 端（typer、MCP 工具客戶端）是完全不同量級的安裝需求。src-layout + 獨立的 `pyproject.toml`/`uv.lock`，能確保 `launch/*.sh` 在遠端機器上 `uv sync` 後執行 `python train.py` 時，不會意外撈到一個扁平佈局下同名的雜散模組——這正是 src-layout 存在的理由，而且對 twin 的重要性高於 ai-studio，因為 twin 真的會被搬上臨時的遠端機器跑。

```
twin/
├── reference/                  # 既有 — SPEC.md, EVAL.md, INTERVIEW.md, 論文筆記。唯讀。
├── CLAUDE.md, README.md, LICENSE, docs/     # 既有
├── .gitignore                  # 新增 — twin 範圍，git 支援巢狀 .gitignore
├── pyproject.toml              # 新增 — 獨立，自己的 [tool.importlinter]，不進 uv workspace
├── uv.lock                     # 新增 — 獨立鎖檔
├── train.py                    # 新增 — 薄殼層：`python train.py --resume auto`（§7.1/§7.4 明文指定的進入點）
├── launch/                     # 新增 — 只放 shell。modal.sh, kaggle.sh, lightning.sh
│                                  （§7.1 允許耦合供應商 SDK 的另一個檔案）
├── examples/                   # 新增 — 每個重要能力一支可執行腳本
│   ├── build_fragment.py       #   Fragment/Trajectory constructor，示範 typed API
│   ├── retrieve_memory.py      #   在 agent loop 之外示範 coarse-to-fine recall()
│   └── tick_dry_run.py         #   對 fixture 記憶庫跑一次 tick
├── src/twin/
│   ├── core/                   # L0 — 不 import 任何內部套件。資料模型與共用葉節點。
│   │   ├── fragment.py         #   Fragment pydantic model — 唯一的 constructor（data-contract 規則 1）
│   │   ├── trajectory.py       #   Trajectory pydantic model — 唯一的 constructor
│   │   ├── enums.py            #   SourceClass, Modality, Split, NegativeClass, GroundTruthSource...
│   │   ├── ids.py, errors.py
│   │   ├── hashing.py          #   dataset_hash/config_hash — 唯一算法所在（data-contract 規則 8、§7.5）
│   │   └── adapter.py          #   AdapterManifest/ModelSpec — train（寫）與 agent（讀）共用的依賴反轉產物，見 §3.3
│   ├── schemas/                # L0 — 純資料，無邏輯
│   │   ├── fragment.schema.json
│   │   └── trajectory.schema.json
│   ├── config/                 # L0 — pydantic_settings Settings、get_settings(refresh=)
│   ├── teacher.py              # 葉節點，位於 config/core 之上 — §7.1 允許耦合供應商 SDK 的兩個檔案之一
│   ├── ingest/                 # L1 — SPEC §4
│   │   ├── sources/{line,rss,browser_history,interview_transcript,questionnaire}.py
│   │   ├── reduce/{image,video,audio,doc}.py     # 多模態降維（§4.2）
│   │   ├── exposure.py         #   曝光事件採集（§4.3）— 必須與 no_action 產出同步存在
│   │   ├── entities.py         #   實體抽取 + third_party_spans 標註（§4.9）
│   │   ├── split.py            #   time-based split 判定 — split 唯一被決定的地方（§4.8/D21）
│   │   ├── fragment.py, trajectory.py, dedupe.py, store.py
│   ├── memory/                 # L2 — SPEC §4.5–4.7
│   │   ├── cluster.py, period.py, salience.py, conflicts.py, retrieve.py, store.py
│   ├── train/                  # L3 — SPEC §5, §7.3–7.6
│   │   ├── data.py             #   讀 L1 軌跡，硬性過濾 split!="train"（§4.8 的 CI 測試即針對此函式）
│   │   ├── masking.py          #   工具名稱隨機置換/遮蔽（§5.3/D16）— 只處理已存字串，不 import agent 的工具 schema
│   │   ├── reflection.py, model.py, checkpoint.py, run.py, reproducibility.py
│   ├── agent/                  # L4 — SPEC §6
│   │   ├── tick.py             #   tick loop（§6.2）— 零工具呼叫即 no_action，無特殊分支
│   │   ├── tools/{base,recall,web_search,reply}.py    # C4：recall 與其他工具同介面
│   │   ├── context.py          #   組裝 system context — C2 的邊界所在（schema 注入處）
│   │   ├── gate.py             #   L0/L1/L2 送出閘門（§6.5）— 降級自動、升級手動（D31）
│   │   ├── reflow.py           #   否決 → 硬負例。只能呼叫 ingest.trajectory/store，不可原地改寫（data-contract 規則 5）
│   │   └── surface/{base,line}.py
│   ├── harness/                 # 跨層，位於最上方 — eval-harness skill 的落地
│   │   ├── manifest.py, shard.py, aggregate.py, gate_check.py, report.py
│   │   └── suites/{s1,s2,s3,s4}.py   # s2 明確拆開程式驗證路徑與 judge 路徑（EVAL §4.2）
│   └── cli/main.py              # typer app — composition root，見 §3.4
├── tests/unit/                  # 扁平，無 conftest.py，各檔案自帶 tmp_path fixture，marker 寫在 pyproject
└── data/ adapters/ transcripts/ eval/     # 不由骨架建立 — 被 gitignore，見 §3.7
```

repo 根目錄新增（在 `twin/` 之外，理由見 §3.7）：`/home/docker_admin/develop/ai-studio/.pre-commit-config.yaml`。

### 3.2 分層契約（import-linter，具體規則）

```toml
[tool.importlinter]
root_package = "twin"

[[tool.importlinter.contracts]]
name = "Layered spine"
type = "layers"
layers = [
    "twin.cli",
    "twin.harness",
    "twin.agent",     # L4
    "twin.memory",    # L2
    "twin.train",     # L3
    "twin.ingest",    # L1
    "twin.teacher",
    "twin.config",
    "twin.core",
]

# C1：訓練不可讀取「活的」記憶內容 — spine 順序已隱含此規則，這裡明講以防日後改動 layers 清單
[[tool.importlinter.contracts]]
name = "C1 — training never reads live memory content"
type = "forbidden"
source_modules = ["twin.train"]
forbidden_modules = ["twin.memory"]

# C2/C3：工具 schema 是 L4、推論期的事，不可進 L3 權重。spine 已擋 train→agent，
# 這條擋的是反方向 —— 純 layers 型契約不會自動擋住 agent import train
[[tool.importlinter.contracts]]
name = "C2/C3 — serving does not depend on trainer internals"
type = "forbidden"
source_modules = ["twin.agent"]
forbidden_modules = ["twin.train"]

# D12/§7.1：供應商 SDK 耦合僅限 teacher.py（另一個允許的檔案 launch/*.sh 是 shell，不在 import-linter 管轄範圍）
[[tool.importlinter.contracts]]
name = "Vendor SDK coupling confined to teacher.py"
type = "forbidden"
source_modules = ["twin.core", "twin.config", "twin.ingest", "twin.memory", "twin.train", "twin.agent", "twin.harness", "twin.cli"]
forbidden_modules = ["google.genai", "google.generativeai"]

# §7.1：train.py 不可認識任何特定雲端
[[tool.importlinter.contracts]]
name = "train.py stays cloud-agnostic"
type = "forbidden"
source_modules = ["twin.train"]
forbidden_modules = ["modal", "kaggle", "lightning_sdk", "boto3", "runpod"]

# harness 是葉節點，跟 ai-studio 的 api/bots 同理：正式產出路徑不可依賴評測管線
[[tool.importlinter.contracts]]
name = "Eval harness stays a leaf"
type = "forbidden"
source_modules = ["twin.core", "twin.config", "twin.teacher", "twin.ingest", "twin.memory", "twin.train", "twin.agent"]
forbidden_modules = ["twin.harness"]
```

「L3 讀取 L2 碎片算不算違反邊界」的釐清：在這個套件切法下，碎片與軌跡是在 `twin.ingest`（L1）建構的，`twin.train.data` 合法 import `twin.ingest`（訓練資料本來就要從這裡讀），這是預期行為，不是違規。`twin.memory`（L2）是另一件事：分群、salience、衝突標記、以及檢索機制本身。C1 真正在管的是「train 不可呼叫 memory 的檢索機制」（不能在準備資料時呼叫 `memory.retrieve()` 把超出軌跡 `result_digest` 已記錄範圍的即時情節細節內嵌進去）——這正是 `train ⊬ memory` 這條契約在擋的。

import-linter 結構上查不到的兩件事，改由既有工具把關：(a) 回流紀律（data-contract 規則 5）— `agent.reflow` import `twin.ingest` 在 spine 下是合法的（L4 可以 import L1），真正的規則是「怎麼用」（走 constructor 產生新記錄，不可原地改寫既有記錄），這是 `data-hygiene` agent／測試該管的事；(b) C4 的介面一致性（`recall` 以一般 Tool 實作，不是 `tick.py` 裡的特殊分支）— 這是 `architecture-reviewer`／`spec-auditor` 該查的（`spec-auditor.md` 現有的必查清單已經有「有沒有反向依賴（L4 import L1、L3 認識工具名稱等）」一項，剛好對應這個分層方案）。

### 3.3 沿用 ai-studio 慣例的取捨表

| ai-studio 的做法 | 判定 | 為什麼 |
|---|---|---|
| Provider `Protocol` + capabilities 放在介面下一層 | **沿用（重新實作，不共用程式碼）** | 剛好對到 SPEC 自己說的兩處：`teacher.py`「實作可替換」（§5.2）與 Surface「MUST 為可插拔配接器」（§6.4）。`SurfaceCapabilities`（這個 surface 有沒有曝光訊號？§4.3/§6.4）該放在 `twin.core`，位於 `agent.surface` 下一層，讓 `harness` 與 `agent.gate` 不需要 import LINE 配接器就能判斷曝光訊號是否可得。 |
| SQLite atomic-claim job queue | **只沿用一半** | `agent` 的 LINE 入站事件佇列跟 ai-studio 已解決的問題幾乎一樣（同平台、同樣的重送行為）— `UNIQUE(event_id)`、原子 claim、tick loop 重啟時的 `release_running()` 全部沿用。ingest 只拿 idempotency key 那一半（去重已 ingest 過的來源記錄），不需要原子 claim — ingest 依 D9「少次、大批」設計是循序批次工作，不是多個併發 claimer。訓練完全不沿用，見下一列。 |
| Checkpoint-resume 契約（§7.4） | **不對應 provider-protocol 的 job-handle 想法 — 是完全不同的形狀** | provider protocol 存在是因為背後有多個可替換的 backend。訓練 backend 只有一個（§7.6 禁止自建 trainer 之外的選擇），沒有「介面」要設計 — 這個「契約」是一份目錄/狀態格式，只有一種實作，不是一個呼叫介面。依賴反轉的直覺仍然適用，但用法不同：`twin.core.adapter.AdapterManifest` 是 train（寫入端）與 agent（讀取端）共用的產物，讓 agent 不需要 import train 的訓練迴圈內部就能載入權重做推論。 |
| Budget guard（ledger + `refuse_if_broke` + `throttle`） | **沿用，用在兩處** | `teacher.py` 的 Gemini RPD/RPM 上限（§5.2、D8/D9）與 Modal credit 的額度（§7.3）都是「JSON ledger、依悲觀最壞情況拒絕、throttle 只會縮短不會延長」同一形狀，只是天花板不同。 |
| 分層走 import-linter + `lint-imports` CLI，不進 pytest | **原樣沿用** | 沒有理由不同 — twin 有自己的 `[tool.importlinter]`，`root_package = "twin"`，在 twin 自己的 CI 裡當 lint 步驟跑，不寫進 `tests/`。 |
| Config：`pydantic_settings.BaseSettings`、每欄位 `alias=`、`SecretStr`、`get_settings(refresh=)` | **原樣沿用** | 用 `TWIN_` 前綴的 alias；singleton/refresh 的形狀完全一致，理由也一樣（測試要 monkeypatch 環境變數後強制重讀）。 |
| CLI：一個根 `Typer()` + 每個名詞一個 `add_typer()` | **沿用做法，名詞不同，兩個刻意的例外** | 見 §3.4 — `train.py` 與 `launch/*.sh` 刻意留在 typer app 之外。 |
| 測試：扁平 `tests/unit/`、無 `conftest.py`、marker 寫在 pyproject | **原樣沿用** | 有一點要說明但不算偏離慣例本身：checkpoint kill/resume 測試（§3.5）比一般 unit test 重，但 SPEC 明文禁止它可被跳過/選擇性執行，所以它不掛 marker、留在預設跑的範圍內，而不是掛個 `slow` marker 把它排除在外 — 這其實更貼合 marker 慣例背後的規則（marker 是用來標記環境需求，不是用來標記「要不要跑」）。 |
| Tooling：獨立 `pyproject.toml` + `uv.lock`，不進 workspace | **原樣沿用（已確認）** | 已核實 root `pyproject.toml` 完全沒有 `[tool.uv.workspace]`，`twin/` 有自己獨立的 `uv.lock`，與上面的 src-layout 決定互相呼應。 |
| `docs/architecture.md` 的文件模板 | **原樣沿用** | `twin/docs/architecture.md`（目前是空的）：把 C1–C4 當成開頭陳述的不變量 → 四層 ASCII 圖 → 設計取捨段落（`core.adapter` 的依賴反轉、reflow 的例外）→ artifact 樹狀圖（`data/`、`adapters/`、`eval/` 的佈局）→ 每個模組的現況表。 |
| `examples/` — 每個能力一支可執行、有 docstring 的腳本 | **原樣沿用** | 見上面目錄樹：`build_fragment.py`、`retrieve_memory.py`、`tick_dry_run.py`。 |

### 3.4 CLI 形狀

根 typer app（`twin/src/twin/cli/main.py`），每個名詞一個 `add_typer()`：`ingest`（依來源 run/status）、`interview`（transcribe/questionnaire/ingest/quality-check，對應 INTERVIEW.md §6–7）、`memory`（inspect/retrieve/conflicts/salience — 除錯用的檢視介面）、`eval`（manifest/shard/aggregate/gate-check/report — eval-harness skill 中「可腳本化」的那一半；judge 本身的派工是 Claude Code 編排，不在這個 CLI 裡，見該 skill）、`agent`（tick/serve/gate status）、`principal`（init，對應 G4 的 onboarding），外加一個扁平的 `doctor`。

`train.py` 與 `launch/*.sh` 刻意都留在 typer app 之外 — 不是漏做。SPEC 直接點名這兩個檔名（「`launch/*.sh` — 呼叫 `train.py`」；「`train.py --resume auto`」），因為 `launch/*.sh` 是 Python 之前的開機層（拉 image、掛憑證、裝 uv），要在一台遠端機器上呼叫一個固定、無歧義的腳本路徑，而那台機器上完整 CLI 的依賴集不見得都相關；把 `train.py` 包進 typer 子命令只會讓這個呼叫變得間接，沒有好處。`train.py` 是呼叫 `twin.train.run.main()` 的薄殼層；可以額外提供 `twin train run --resume auto` 作為本機開發的別名，但對 launch script 而言，正式的進入點就是那支裸腳本。

### 3.5 兩個 CI 關鍵測試

**split 過濾測試（§4.8）**：`tests/unit/test_train_data_split_filter.py` 用真正的 constructor（不是 dict literal）建構涵蓋三種 split 值的 Trajectory，呼叫真正的 `twin.train.data.load_training_examples()`（不是重新實作一份邏輯去驗證自己），斷言輸出集合在數量與身份上都精確等於 `split=="train"` 的子集。這個測試不掛任何 marker（不像 `ffmpeg`/`runpod` 那種可選退出的 marker），所以永遠不會被悄悄跳過；再加一個 meta-test 掃過整個測試樹，確認沒有人在這個測試的 node id 上掛 skip/xfail 裝飾器 — 把「不可跳過」變成機制，而不是註解提醒。

**Kill/resume 連續性測試（§7.4）**：一次真實（但用玩具規模模型）的訓練跑，以子行程啟動好讓 `SIGKILL` 是真的訊號而非模擬 — 輪詢直到出現 checkpoint 標記、`os.kill(pid, signal.SIGKILL)`、斷言 checkpoint 目錄含六項必要產物齊全，然後對同一個 checkpoint 目錄重跑 `train.py --resume auto`。「連續」具體操作化成兩個斷言：續跑後第一筆記錄的 `global_step` 精確等於「上次完成的 step + 1」（不多不少 — 這正是抓出「dataloader cursor 沒存」導致 D11 假收斂曲線的地方），以及一次相同種子、不中斷、跑完同樣總步數的對照組，其最終 loss 要落在跟「被中斷後續跑」版本的極小誤差範圍內（這才是真正需要 RNG state 有被正確還原的地方，而不只是「沒丟例外」）。留在扁平的 `tests/unit/` 裡，用玩具模型讓成本可控，不掛可選退出的 marker。

### 3.6 Schema 與 SPEC 一致性測試

`twin.core.fragment`/`trajectory` 裡的 pydantic model 同時是「唯一的 constructor」（data-contract 規則 1）也是 schema 的來源；`twin/src/twin/schemas/*.schema.json` 是進版控、由 CI 重新產生並 diff（而非手動維護）的 package data。真正扛住風險的測試不是「schema 檔 == model」這種自我一致性檢查，而是 `tests/unit/test_schema_matches_spec.py`：解析 SPEC.md §4.4、§4.10 底下 ```jsonc``` 圍起來的區塊（一個容忍註解的簡單 key 抽取器，往下遞迴一層處理 `event_time{value,precision,confidence}` 與 `third_party_spans[]{start,end,party_ref}`），斷言抽出來的欄位集合精確等於 `Fragment.model_fields` / `Trajectory.model_fields`（pydantic v2 的內省），任何一邊改了另一邊沒跟上就會讓 CI 紅燈。這樣比對的對象是 SPEC.md 本身，而不是另一份手動維護的 schema 說明文件，符合 data-contract skill「SPEC 的 schema 是唯一事實來源」的規則。搭配一個 grep 型測試（沿用 `test_gitignore.py` 那種「grep 即測試」的寫法），斷言除了 `core/fragment.py`、`core/trajectory.py` 與 schema 檔之外，repo 中不存在 `"fragment_id"`／`"trajectory_id"` 這種字面量。

### 3.7 護欄落地位置

**`.gitignore`**：`twin/.gitignore`（git 的巢狀 `.gitignore` 機制，範圍限定在 `twin/` 之下），列出 `data/`、`adapters/`、`transcripts/`、`eval/` 加上 twin 自己的 build 產物樣式 — 完全包在「獨立套件」這個邊界內，搭配一支從 ai-studio 現有、已驗證過的 `tests/unit/test_gitignore.py` 直接移植（parse-and-assert-patterns-present 的寫法）的 `twin/tests/unit/test_gitignore.py`。

**Pre-commit hook**：這是「獨立套件」這個原則唯一要讓步的地方 — git hook 是整個 repo 共用一份，`.git/hooks/pre-commit` 只有一個，`pre-commit install` 會往上找到 repo 根目錄的 `.pre-commit-config.yaml`，不會找子目錄的設定。已確認 ai-studio 目前完全沒有 pre-commit 基礎設施（repo 中不存在任何 `.pre-commit-config.yaml`；它自己的等價護欄只有 `.gitignore` + 一支存在性測試），所以 twin 會是這個 monorepo 第一個引入 pre-commit 的地方。它必須放在 repo 根目錄（`/home/docker_admin/develop/ai-studio/.pre-commit-config.yaml`），寫成一個 local hook，`files` 限定為 `^twin/(data|adapters|transcripts|eval)/`，只在 staged 路徑落在這四個 twin 子目錄下時才觸發 — 對 ai-studio 貢獻者零影響，存在的目的就是接住 `git add -f` 硬繞過 `.gitignore` 的情況。因為這份 hook 設定本來就不在 `twin/` 自己的測試範圍內，它的存在與規則涵蓋範圍應該由 root 的 `tests/unit/` 斷言（在既有的 `test_gitignore.py` 旁邊新增一支），因為護欄 2 的驗證本來就是 repo 層級的事，跟檔案本身該放哪裡是一致的。

### 3.8 跨 phase 的 interface-first 補建（2026-08-27）

Phase 1 完成後，實作方向改變：不再逐 phase 蓋完整垂直切片，改為先把 L2/L3/L4/harness 的 interface 一次建好，細節留到鎖定特定區塊時再補。**每個 interface 都必須能對應到 SPEC.md/EVAL.md 的具體條號，或一個此刻就存在的跨層依賴**，不可只因為 §3.1 的樹狀圖列了檔名就建。

觸發原因兩個：（1）本人尚未確定 Teacher 要不要真的用 Gemini（SPEC.md §5.2/D9 已裁決 v1 綁 Gemini，但「本人是否願意把資料送給它」是分開的、尚未解決的疑慮）；（2）同樣的可替換性也要用在 Agent 實際依賴哪個 LLM 做推論決策、以及 Agent 的工具設計上（本人直接確認，且 SPEC.md C2/C4 本來就要求如此）。

**已建（real，非 stub）**：
- `core/trajectory.py`（§4.10 Trajectory schema，frozen，含 §4.11/D20 的 evidence=absent→trivial 驗證）＋ `core/enums.py` 新增 `NegativeClass`/`GroundTruthSource`/`ExposureEvidence`/`GateLevel`
- `core/hashing.py`（§7.5 的 dataset_hash/config_hash/adapter_hash）
- `core/adapter.py`（`ModelSpec`/`AdapterManifest` — train 寫、agent 讀的依賴反轉產物，PLAN §3.3 點名但原先沒寫出形狀）
- `core/gate_metrics.py`（`GateMetrics` + `JUDGE_AGREEMENT_FLOOR`）— **本輪發現的一個 PLAN 原樹狀圖沒預料到的分層衝突**：`agent/gate.py` 需要這輪的評測數字才能判斷 L0/L1/L2 是否該變動，但完整報告在 `harness/report.py`，`agent` 直接 import `harness` 會違反既有的「Eval harness stays a leaf」契約。解法是在 `core` 放一個小的、frozen 的投影型別，`harness.report` 同時產出完整報告與這個投影，`agent.gate` 只依賴 `core.gate_metrics`，與 `core.adapter.AdapterManifest` 的依賴反轉手法完全一致。`JUDGE_AGREEMENT_FLOOR`（EVAL §6.3，0.80，MUST NOT 下修）也放在這裡，因為 `agent.gate`（送出閘門）與 `harness.gate_check`（堪用閘門）都需要同一個數字，而兩者不能互相 import。
- `core/capabilities.py`（`SurfaceCapabilities`）— **一個對 Plan 子代理原始建議的修正**：不提前放一個 `LINE_CAPABILITIES` 常數猜測 `exposure_signal_available=False`。§11 項目 H 說 LINE 曝光訊號可得性是「需技術驗證」，不是「已裁決為否」；提前寫死 False 等於搶先回答了一個明確還沒驗證的技術問題。型別本輪建，LINE 專屬的實例留到 Phase 8/11 真正驗證後才寫。
- `memory/retrieve.py`（naive 版 `recall()`，§3.1/C4）
- `train/data.py`（§4.8/D21 的 `split!=train` 硬過濾，PLAN §3.5 原本就指名的測試現在補上）＋ `train/masking.py`（§5.3/D16 的工具名稱置換）
- `agent/tools/{base,recall,reply}.py`（`Tool` protocol — 本人直接問到的「工具設計要可替換」）、`agent/gate.py`（送出閘門，EVAL §7.2 全數字面數字）、`agent/reflow.py`（§6.5 否決回流硬負例）、`agent/context.py`、`agent/surface/base.py`（`Surface` protocol）、`agent/tick.py`（`TickResult`/**`Decider` protocol — 本人問到的「Agent 依賴哪個 LLM 要可抽換」，答案就是這個 protocol：不管背後是本地 HF+PEFT、vLLM endpoint 還是別的，tick loop 只認 `decide()`**）
- `harness/{schema,manifest,shard,aggregate,gate_check,report}.py`（EVAL §11 報告格式逐欄位、§7.1 T1/T2 全表、§1.4 跨 suite 總分斷言 — eval-harness skill 明講這些是純腳本邏輯，沒有懸而未決的設計）

**`teacher.py` → `teacher/` 套件**：拆成 `teacher/__init__.py`（只 re-export `Teacher`/`TeacherError`/`TeacherRateExhausted`/`TeacherCallLedger`，刻意不含 `GeminiTeacher`）、`teacher/base.py`（介面本體）、`teacher/gemini.py`（`GeminiTeacher`，原樣搬移）。不刪除 `GeminiTeacher`——SPEC.md §5.2/D9 已經裁決 v1 綁 Gemini，這是已關閉的規格決定，跟本人「要不要真的對真實資料跑它」是兩件事。做法是讓隔離變成 import graph 的性質而非註解：要用 `GeminiTeacher` 必須寫出 `from twin.teacher.gemini import GeminiTeacher`，不再能從 `twin.teacher` 直接拿到。新增回歸測試斷言 `"GeminiTeacher" not in dir(twin.teacher)`。

**刻意不建（沒有此刻的消費者，或屬於「全有全無」的例外）**：`memory/{cluster,period,salience,conflicts,store}.py`（§4.5/§4.6 的分群門檻與權重本身未裁決，且本輪沒有任何程式碼引用 `Episode`/`Period`）；`train/{model,checkpoint,run,reproducibility}.py` 與根目錄 `train.py`（§7.4 的驗收標準是二元的——真的撐過 `kill -9` 或不存在，沒有介於兩者之間的 interface 可寫，這是「interface-first」原則本身承認的例外，留給 Phase 4 整段一起做）；`agent/surface/line.py`（等 Phase 11 與項目 H 的驗證）；`harness/suites/{s1,s2,s3,s4}.py` 的樣本建構本體（thin，`NotImplementedError`，各自等自己的 phase）。

**審查**：`spec-auditor`、`data-hygiene`（本輪動到 `core/trajectory.py` 與 `train/data.py`，兩個 agent 皆適用），皆判定 PASS，各自留下一項處理如下：

- `data-hygiene` 發現 `core/trajectory.py` 的 validator 只查了「`evidence=absent` 的 `no_action` 必須是 `trivial`」這個方向，沒查反方向：§2.3 把 hard／trivial 負例都**定義**為「曝光→不動作」軌跡，代表 `negative_class ∈ {hard, trivial}` 結構上就該要求存在一個 `NoActionStep`。目前無任何呼叫路徑會踩到這個漏洞（`agent/reflow.py` 是唯一產生 `hard` 的地方，且必定帶一個 `NoActionStep`），但這是後續任何人工標註修正／回填腳本可能無聲踩到的地雷。**已修正**：validator 新增這個方向的檢查，新增 4 個測試覆蓋。`data-hygiene` 另記錄一項留待日後：`agent/reflow.py` 的 `reflow_veto()` 目前寫死 `exposure.evidence=READ_RECEIPT`、沒有參數可覆寫，若日後有別的呼叫路徑（例如逾時自動視為否決、或從紀錄回填歷史）誤用同一個函式，會無聲捏造一筆不存在的曝光證據——`reflow_veto` 目前零呼叫者（尚未接進 `agent/gate.py`／`agent/tick.py`），視為介面先行階段的已知風險留待接線時處理，不在本輪修正範圍。
- `spec-auditor` 指出一項需要人裁決的分類問題：`memory/retrieve.py` 本輪標為「real」（程式碼完整、非 stub），但它做的是全域關鍵字子字串比對，不是 §4.5 規定的「MUST 為 coarse-to-fine：先定位時期，再下拉碎片」——且它已經被 `agent/tools/recall.py` 接上，是 tick loop 未來會真的呼叫的路徑。**裁決**：維持現狀，不在本輪補建 coarse-to-fine（那正是 Phase 9 的工作，Episode/Period 分群門檻本身尚未裁決，提前做等於搶先決定一個懸而未決的設計）。`retrieve()` 對「real」的定義僅指「這份程式碼完整、可執行、不是 NotImplementedError」，不代表它宣稱滿足 §4.5 的 MUST——docstring 已明講 Phase 9 會取代這個後端。記錄於此，作為明確的已知、暫時接受的落差，而非被忽略的缺口：**任何以 `agent.tools.RecallTool` 產出的檢索結果餵給 S1 harness 之前，MUST 先確認 Phase 9 的 coarse-to-fine 檢索已經取代這個 naive 後端**，否則 EVAL §3.5 內容正確性的量測會失真。

---

## 4. 剩餘開放項如何處理

SPEC.md §11 所有字母項目（A–E）已於 2026-08-27 全數裁決；只剩兩項不阻塞開工的項目（§11 原文：「不阻塞開工」）：

- **項目 G — 8B 底模為規格代決，MAY 被推翻。** 影響 Phase 4（底模選型）。處理方式：預設採 8B；任何降級 MUST 以 S1/S4 驗證（§5.1），不可只憑成本理由靜默降級 — 所以 Phase 4 與 Phase 10 都不該在別處寫死 8B 的假設。
- **項目 H — LINE 曝光訊號的可得性需技術驗證，不是裁決。** 影響 Phase 8（曝光採集）與 §6.4（Surface）。處理方式：Phase 8 應包含一個及早的技術驗證，確認 LINE Messaging API 是否真的能拿到已讀狀態／聊天室開啟訊號；若不行，LINE MUST NOT 用於 S3（§4.3、§6.4），且此事實必須寫進評測報告 — 在這個驗證完成前，LINE surface 的 S3 信心維持保留狀態。

本計畫沒有任何一個階段依賴 G/H 以外的東西，符合 SPEC.md §0「實作不得依賴 §11 的任何內容」在 A–E 已裁決後的現況。

---

## 5. 驗收：如何確認每個 phase 真的完成

每個 phase 的「驗收」欄本身就是可觀測、非主觀的判準（見 §2）；額外三條跨階段的驗收紀律，直接繼承自 twin/reference/ 與既有 `.claude` 設定，不是新規則：

1. **每個 phase 收尾前跑 `spec-auditor` agent**，針對該 phase 的 diff 檢查是否違反 SPEC.md 的 MUST/MUST NOT 與決策紀錄，以及 EVAL.md §12 反模式清單。
2. **任何動到 L1/L2 的 phase（1、3、8、9、10）收尾後跑 `data-hygiene` agent**，稽核時間洩漏、切分污染、負例品質、曝光採集這四類「指標正常但實際已壞」的問題。
3. **動手前用 `spec-trace` skill** 把要做的事對到 SPEC.md 的 §-條號與 EVAL.md 的觀測點；填不出來就停下，見該 skill 的三種結果處理。

---

## 附錄：本計畫的依據

- `twin/reference/SPEC.md`（v0.4）
- `twin/reference/EVAL.md`（v0.2）
- `twin/reference/INTERVIEW.md`（v0.2）
- `twin/reference/2411.10109 實作筆記.md`
- `twin/CLAUDE.md`
- `.claude/skills/data-contract/SKILL.md`、`.claude/skills/eval-harness/SKILL.md`、`.claude/skills/spec-trace/SKILL.md`
- `.claude/agents/data-hygiene.md`、`eval-judge.md`、`spec-auditor.md`
- 對照參考：`pyproject.toml`、`docs/architecture.md`、`src/ai_studio/providers/base.py`、`src/ai_studio/core/provider_spec.py`、`src/ai_studio/pipeline/queue.py`、`src/ai_studio/runtime/budget.py`、`src/ai_studio/config/settings.py`、`src/ai_studio/cli/main.py`、`tests/unit/test_gitignore.py`
