# PLAN.md — twin 子系統實現計畫

| 項目 | 值 |
|---|---|
| 版本 | 0.7 |
| 日期 | 2026-08-30（第一次真實 LoRA 訓練跑完成；Phase 3-B 以最小文字版訪談員取代；Phase 5 真實推論路徑落地；自陳資料 split 裁決為 train——見各自章節）。前版 2026-08-28（Phase 5 全部、Phase 3-A（訪談後處理＋逐字稿/問卷 ingest）程式部分落地，見各自章節；Phase 3-B（真人語音訪談員本身）仍為未解決的開放設計問題，刻意未動工，見 Phase 3 章節） |
| 依據 | SPEC.md v0.4、EVAL.md v0.2、INTERVIEW.md v0.2（見附錄） |
| 狀態 | Phase 0（護欄/套件骨架）、Phase 1（Fragment schema／split／teacher.py／最小 ingest）程式部分皆已落地；兩者各自的人工步驟（GCP/雲端帳號、使用者真實資料）仍待辦，見各自章節。§3.8 記錄了一輪跨 phase 的 interface-first 補建（L2/L3/L4/harness 的介面與 core 的支撐型別），刻意不算某個特定 phase 完成，細節仍待鎖定區塊時補上。**Phase 4（最小 L2 + 精簡軌跡集 + 第一版 LoRA）程式部分已完成**（底模定案 Qwen3-8B，`train/{formatting,model,checkpoint,reproducibility,run}.py`、根目錄 `train.py`、`launch/*`，SPEC §7.4 的 kill-9/resume CI 測試已通過，見該章節「狀態」）；LoRA rank 硬體 probe、真實資料訓練跑仍待辦。**Phase 5（Baseline B0/B1/B2 + judge harness）程式部分已完成**（`harness/{baseline,s1_run,eval_io}.py`、`eval/rubric/s1.md`、三支 `examples/*_s1_eval_round.py` 驅動腳本），`spec-auditor` 兩輪皆 PASS，詳見該章節。**Phase 3 拆成兩半**：後處理管線＋逐字稿/問卷 ingest（「Phase 3-A」）程式部分已完成，`spec-auditor`／`data-hygiene` 皆 PASS；真人語音訪談員本身（「Phase 3-B」）INTERVIEW.md 完全未指定實作機制（無 STT/TTS、無連續 session 架構），本輪刻意不解決，留待下一輪與使用者一起裁決。**Phase 2 已完成（2026-08-29）：真實 ingest 96,750 筆、題庫 70 題凍結、Wave 1 答畢，專案第 0 天 = 2026-08-29T03:58Z，Wave 2 於 2026-09-12 開放**。路線圖依 SPEC/EVAL 的既有裁決推導，未包含任何本文件自行代決的規範性內容 |

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

**狀態：程式部分已完成（2026-08-27）。人工帳號部分：GCP、Modal、Cloudflare R2、Kaggle、Lightning AI 帳號註冊皆已完成（2026-08-28）；Lightning AI 的 CLI/SSH 串接（`launch/lightning.sh` 的實際動詞集）仍待驗證，見下方「仍待辦」。**

- [x] 修改 repo 根目錄 `.gitignore`：加入 `twin/data/`、`twin/adapters/`、`twin/transcripts/`、`twin/eval/`（**目前缺失，本次規劃已核實**）。實作時改用 `twin/` 前綴而非原文的裸露路徑——`twin/.gitignore`（巢狀）已完整涵蓋這四個目錄，根目錄這份純屬 defense-in-depth，加前綴可避免未來 repo 其他地方出現同名目錄時被誤傷；不加任何測試斷言這份根目錄副本的存在，真正的防線是 `twin/.gitignore` 與 pre-commit hook。
- [x] 新增 `twin/.gitignore`（同樣四個路徑，裸露寫法，範圍限定 `twin/` 之下，git 支援巢狀 `.gitignore`），外加 twin 自己的 build 產物樣式（`__pycache__/`、`.venv/`、`dist/` 等，見 §3.7）。
- [x] 新增 repo 根目錄 `.pre-commit-config.yaml`（**repo 目前完全沒有任何 pre-commit 設定，已核實**）：local hook，`language: fail`（pre-commit 內建、專為「這個路徑永遠不准進」設計，不需 shell/subprocess，避開 CLAUDE.md 提過的 Windows shell-quoting 陷阱），`files` 限定 `^twin/(data|adapters|transcripts|eval)/`，只在 staged 路徑落在這四個子目錄時觸發，對 ai-studio 貢獻者零影響。已跑過真實 end-to-end 驗證：`pre-commit install` 後對 `twin/data/_verify.json` 執行 `git add -f` + `git commit`，commit 被擋下，訊息引用 SPEC.md §8 護欄 2；驗證後已清除測試檔案，未留下任何 commit。根目錄 `pyproject.toml` dev 依賴新增 `pre-commit`。
- [x] `twin/README.md` 補上責任聲明：第三方內容的合法性由使用者自行負責，本專案不代為處理（SPEC.md §8 護欄 3 原文精神）。
- [x] 建立 Teacher 專用 GCP 專案，確認**未**綁定 billing。**已完成（2026-08-28）**——使用者已在 GCP Console 的 Billing 頁面確認未連結任何付款方式；API key 已產生，存入 `twin/.env`（gitignored，`git check-ignore -v twin/.env` 驗證過）。
- [ ] 開通 Modal、Cloudflare R2、Kaggle/Lightning AI 帳號（核准時程不可控，及早申請）。
  - **Modal 已完成（2026-08-28）**：帳號註冊、`uv run modal setup` 完成 CLI 認證（workspace `jieyao-milestonehub`，token 存於 `~/.modal.toml`），以 `modal profile current`／`modal app list` 驗證過。`modal` 補進 twin 依賴（`launch/modal_app.py` 原本 `import modal` 但套件從未被安裝，此輪修正）；同時修正 `launch/modal_app.py` 的 `add_local_dir` 少了 `ignore=`：`copy=True` 會把整個本機目錄烤進 image layer，這是 SPEC.md §8 護欄 2（`data/adapters/transcripts/eval` 不可離開本機）以外的獨立管道——git 的 `.gitignore`／pre-commit hook 完全管不到它——已加 `ignore=[".env", ".git", ".venv", "__pycache__", "data", "adapters", "transcripts", "eval"]`。
  - **Cloudflare R2 已完成（2026-08-28）**：bucket `twin-checkpoints` 已建立（原本帳號裡沒有任何 bucket，用使用者提供的 access key/secret 透過 `checkpoint.py` 的 `_fs_and_path()` 直接建立）。**已核實而非猜測**：`_fs_and_path()` 是裸 `fsspec.core.url_to_fs(uri)`，沒有帶 `endpoint_url` 之類的額外設定，實測確認純靠 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_ENDPOINT_URL_S3`/`AWS_DEFAULT_REGION=auto` 四個環境變數即可正確路由到 R2 endpoint，**不需要改 `checkpoint.py` 本身**。但發現這四個變數必須真正進到 process 的 `os.environ`——`twin.config.settings.Settings` 讀 `.env`只餵給 pydantic 內部欄位，從不寫回 `os.environ`——所以在 `train.py` 加了一行 `load_dotenv()`（新增 `python-dotenv` 依賴），已用「完全比照 train.py 實際行為、shell 不手動 export」的方式重新驗證過一次，確認可行。`TWIN_CHECKPOINT_STORE_URI=s3://twin-checkpoints` 寫入 `twin/.env`。
  - **Kaggle 已完成（2026-08-28）**：帳號註冊。**過程中發現一個文件陷阱**：先前用網路搜尋查到的「kaggle.json 存 username+key」是官方文件標示的 **Legacy** 認證法；實際裝進來的 `kaggle==2.2.4` CLI 要求的是單一 bearer token，存在 `~/.kaggle/access_token`（或 `KAGGLE_API_TOKEN` 環境變數）——已用真實 `kaggle kernels list --mine` 呼叫驗證通過（列出使用者既有的 kernel）。Token 存於機器層級的 `~/.kaggle/access_token`，不進 `twin/.env`（`launch/kaggle.sh` 直接呼叫 CLI，不經過 `train.py` 的 `load_dotenv()`），性質上跟 Modal 的 `~/.modal.toml` 一樣是本機認證檔案。
  - **Lightning AI 帳號已完成（2026-08-28）**：Free plan（up to 30 credits/月）。建立流程本身提供了兩個先前「已知偏離」清單裡標成未核實的資訊，這次是從實際 signup 畫面讀到的，不是猜的：(a) 免費層有 T4（36 free hrs，遠多於 L40S/A100/H100/H200 的 2-5 hrs）——選 T4，跟 Modal/Kaggle 的算力等級一致，也是 `probe_lora_rank.py` 校準的目標硬體；(b) 「80% off on interruptible (spot)」字樣證實 Lightning 的 spot/preemptible 是**選配**，不像 Modal 是無條件套用——這代表 SPEC.md §7.3「MUST 優先使用 spot/preemptible」在 Lightning 上必須是主動勾選的設定，跑真正的訓練前要記得選 interruptible，不是預設就滿足。**`launch/lightning.sh` 的實際 CLI 動詞集本身仍未核實**——這次只完成帳號註冊＋方案選擇（Advanced／Local IDE／T4），還沒有 SSH 憑證，串接 CLI 屬於獨立的、之後才需要做的工作。
- [x] `twin/` 套件骨架：`twin/pyproject.toml`、`twin/src/twin/`、`twin/uv.lock`（獨立於 root 的 import-linter 契約，見 §3.2）。9 層（`twin.cli`/`harness`/`agent`/`memory`/`train`/`ingest`/`teacher`/`config`/`core`）皆為空殼 package（僅 module docstring，無邏輯——Fragment/Trajectory/teacher.py 的 Gemini 綁定等留給 Phase 1）；`uv run lint-imports` 6 條契約全數通過。§3.2 原文的 `forbidden_modules = ["google.genai", "google.generativeai"]` 在實作時發現 import-linter 不支援 external package 的子模組層級封鎖，改為封鎖整個 `google` namespace（效果等同、範圍更嚴格，因為 twin 目前沒有理由 import 任何其他 `google.*`）；並補上 `include_external_packages = true`（契約引用外部套件如 `modal`/`kaggle`/`google` 時的必要設定，§3.2 原文未列出，屬本輪實作補完）。CLI 端點 `twin` 目前僅一個 `version` 子命令（其餘 noun 隨各自 phase 落地，見 §3.4）；空的 `typer.Typer()` 無任何 command 時無法被呼叫，因此加了一個空的 `@app.callback()` 以維持多子命令模式，供未來 `ingest`/`interview`/... 使用同一個 group 型 CLI。

**依據**：SPEC.md §5.2/D8、§7.1、§7.2、§7.3、§8 護欄 2/3。
**驗收**：對 `data/` 底下的測試檔案 commit 會被 hook 擋下（**已用真實 commit 嘗試驗證，見上**）；GCP console 確認無 billing 帳號（**已完成 2026-08-28**）；README 已有聲明（**已完成**）。
**類型**：程式（已完成）+ 人工（GCP／Modal／R2／Kaggle／Lightning 帳號註冊皆已完成 2026-08-28；Lightning 的 CLI/SSH 串接仍待辦）。

**仍待辦（人工）**：
1. ~~建立 Teacher 專用 GCP 專案並確認未綁 billing。~~ **已完成 2026-08-28**。
2. ~~開通 Modal、Cloudflare R2、Kaggle/Lightning AI 帳號。~~ **五個帳號皆已完成 2026-08-28，見上**。Lightning AI 目前只完成帳號＋方案選擇，還沒有 SSH 憑證，`launch/lightning.sh` 的 CLI 動詞集也還沒對照真實帳號核實過——留給真的要用 Lightning 跑訓練時再做。
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
1. 使用者提供一份真實的 LINE 聊天記錄（或其他純文字）匯出檔案，對其實際執行 `fragments_from_line_export()` 並用 `write_fragments_jsonl()` 落地。**驅動腳本已落地（2026-08-29）**：`ingest/line_ingest.py::ingest_line_export()`（可測的邏輯層：`sealed_cutoff_for` → `fragments_from_line_export` → 拒絕覆寫既有 store、拒絕 0 筆 held-out、heldout 早於 train 即 `raise` → `write_fragments_jsonl`）＋ `examples/ingest_line_export.py`（argv 驅動：`--export`、`--principal-display-name`、`--known-sender`（重複）、`--train-cutoff`、`--now`、`--sealed-fraction`、`--overwrite`），測試 `tests/unit/test_ingest_line_ingest.py`（5 個，虛構資料）。**資料存放位置改為 checkout 之外**：2026-08-29 發現先前已生成的一份真實題庫（約 69 題）因為只存在 gitignored 的 checkout 相對路徑裡而遺失，故 `twin/.env` 現在把 `TWIN_FRAGMENT_STORE_URI`／`TWIN_TRAJECTORY_STORE_URI`／新增的 `TWIN_S1_EVAL_ROOT_URI`／`TWIN_TEACHER_LEDGER_PATH` 全指向 `~/twin-data/`（不在任何 repo/worktree 內；原始匯出檔放 `~/twin-data/raw/`）。**真實 ingest 已完成（2026-08-29）**：9 個 LINE 1:1 聊天室匯出（`~/twin-data/raw/`，以 `manifest.json` 描述每檔發言者；一個檔名因檔案系統去掉 `<>` 而與內文顯示名稱不同，manifest 已修正），`ingest_line_exports()` 合併為單一 store：**96,750 筆**，`train_cutoff=2026-03-01`、`now=2026-08-29T23:59` → `sealed_cutoff=2026-07-24T14:23`；train 78,473 / heldout 15,106 / sealed 3,171；47,794 筆（49.4%）帶第三方 span；零筆缺 `event_time`；與原檔 `HH:MM` 行數總和完全相等。`data-hygiene` PASS（split 逐筆重算 0 不一致、第三方 span 幾何 0 不合格、`fragment_id` 0 重複）。兩項要寫進日後報告的觀察：各聊天室的 split 比例因對話生命週期而極不均（一室幾乎全 train、一室幾乎全 heldout+sealed），S1 題材與訓練集偏向不同對象；1,844 筆同分鐘同短訊的合法重複（`fragment_id` 皆不同，勿以 (event_time, content) 去重）。此 store 全為 `behavior` 類、無曝光訊號，依 SPEC §4.3 不得作為 S3 來源。
2. ~~Phase 0 的 GCP 專案就緒後，對 `GeminiTeacher` 打第一次真實請求，確認 RPD ledger 與 D8 的「未啟用 billing」防線在真實流量下仍然成立。~~ **已完成 2026-08-28**：`GeminiTeacher.from_settings()` 對 `gemini-3.5-flash-lite` 打了一次真實請求（一次性驗證腳本，非落地測試），成功解析 `PingResponse`，ledger 從 0 累加到 1，確認記帳邏輯在真實流量下正確。`TWIN_GEMINI_MODEL` 依 AI Studio 的免費層儀表板選定，而非猜測值。

### Phase 2 — S1 題庫 + Wave 1 作答（**專案第 0 天**）

**狀態：程式部分已完成（2026-08-28）。真實一批題庫生成＋真實 Wave 1 作答仍待辦——見下方「仍待辦」，這是全計畫唯一會真正啟動 14 天時鐘的動作，故意留給使用者主動觸發，不是這輪程式化實作的一部分。**

- [x] `harness/suites/s1.py::build_item_bank()`：一次批次 Teacher 呼叫（D9 少次、大批，函式內最多呼叫一次，無重試邏輯），依 §3.2 的題型比例（30/25/25/20：價值取捨/偏好/反應傾向/回想，SHOULD、非強制精確；prompt 內以目標總數 70 題換算成建議題數 21/18/17/14）從 Phase 1 的 `Split.HELDOUT` 碎片產生情境題；拒絕引用碎片集合以外的 `source_fragment_ids`（`TeacherError`）、拒絕非 held-out 來源碎片（`HarnessError`，引用 §3.1/§9/反模式 #7）、拒絕總題數落在 60–80 之外、拒絕重複題目。`item_id` 為內容雜湊（比照 `harness/shard.py::sample_id` 的做法，對 item_type/prompt/options/source_fragment_ids 取 sha256），使題庫層級的雜湊天然對內容敏感。
- [x] `harness/item_bank.py`（新模組）：`S1BankManifest`／`S1WaveManifest`、`bank_hash()`（複用 `core.hashing.dataset_hash`）、`write_item_bank_once()`／`read_and_verify_item_bank()`（凍結後任何竄改皆可偵測）、`write_wave_manifest_once()`——`S1WaveManifest.completed_at` 就是「專案第 0 天」的存證產物，寫在 `twin/eval/s1/`（既有 SPEC §8 護欄 2 gitignored 目錄）。
- [x] `harness/schema.py` 新增 `S1Answer`（`item_id`/`wave`/`answer`/`answered_at`，`answer` 恆為選項原文，非自由輸入，供 Phase 6 的 R1/R2 精確比對）。
- [x] `examples/build_s1_item_bank.py`：一次性產生＋凍結題庫的驅動腳本，印出題型分布供人眼比對 SHOULD 比例，明確 `y/N` 確認後才凍結（凍結前找不到 fragment store 會給出可行動的錯誤訊息，不是裸 traceback）。
- [x] `examples/collect_s1_answers.py`：`--wave {1,2}` 互動式作答蒐集，作答前先驗證題庫雜湊未被竄改，逐題以選項編號記錄（非自由輸入）、每題即寫檔（可斷點續答），Wave 1 全部答完的瞬間寫入 `r1_manifest.json`——那個時間戳即為第 0 天。
- [x] 測試：`test_harness_suites_s1.py`（11 個測試，含拒絕規則、`item_id` 決定性、恰好一次 Teacher 呼叫）、`test_harness_item_bank.py`（6 個測試，含凍結後竄改偵測、fail-fast 不留半寫狀態）、`test_harness_schema.py` 補 `S1Answer` 往返測試。移除 `test_harness_suites_are_thin.py` 裡 S1 的 pin 測試（S2/S3/S4 維持不動）。`uv run pytest`／`lint-imports`／`ruff`／`mypy` 全過。

**依據**：EVAL.md §1.2、§3.1、§3.2、§3.3、§9、§12 反模式 #7。
**驗收**：`R1` 已記錄，題庫已凍結／雜湊，時間戳存在——此時間戳即為專案第 0 天。**已具備程式機制**（write-once + hash 驗證），**真實一輪尚未執行**。
**類型**：程式（已完成）+ 人工（作答本身，仍待辦）——**全計畫最重要的一個驗收點**。

**仍待辦（人工＋一次性操作）**：
1. Phase 1 的真實 LINE 匯出 ingest（仍是本項目最上游的缺口——`~/twin-data/data/` 目前沒有任何 `fragments.jsonl`，兩支腳本都會在這裡卡住並給出明確訊息）。**注意（2026-08-29）**：先前曾生成並凍結過一份約 69 題的真實題庫，但存在 checkout 相對的 gitignored 路徑下，未被保留，Wave 1 也未曾答完——視為從未發生，重來。為此 `build_s1_item_bank.py`／`collect_s1_answers.py`／`prepare_s1_eval_round.py`／`score_s1_eval_round.py` 不再硬編碼 `file://./eval/s1`，改讀 `Settings.s1_eval_root_uri`（`TWIN_S1_EVAL_ROOT_URI`，預設仍為 `file://./eval/s1`，真實跑時設為 `~/twin-data/eval/s1`）。
2. ~~上述完成後，跑 `uv run python examples/build_s1_item_bank.py`~~ **題庫已生成並凍結（2026-08-29T03:37Z，`~/twin-data/eval/s1/`）：70 題，21/18/17/14 恰為 30/25/25/20 目標；`teacher_model=gemini-3.5-flash-lite`，當日 Teacher 呼叫 2 次。** 兩處為此落地的修正：(a) 真實 held-out 有 15,106 筆、約 170 萬字元，遠超 `_MAX_PROMPT_CHARS`，新增 `harness/suites/s1.py::sample_held_out_windows()`（決定性、seed 固定、時間分層的連續視窗抽樣：40 窗 × 12 筆 = 480 筆，五個 held-out 月份皆有覆蓋；題庫的 `source_fragment_dataset_hash` 對這 480 筆計算）；(b) 第一次真實呼叫時模型引用了一個 store 裡完全不存在、格式卻正確的 32 位 hex `fragment_id`（憑空捏造），prompt 改以短標籤 `F001…` 呈現片段、`source_refs` 由程式映射回真實 id，未知標籤仍 `TeacherError`。**Wave 1 已完成——專案第 0 天 = `2026-08-29T03:58:47Z`**（`~/twin-data/eval/s1/answers/r1_manifest.json`，70/70 題，`bank_hash` 與題庫 manifest 一致）。**Wave 2 最早可在 `2026-09-12T03:58Z`（台北 09-12 11:58）之後作答**：`uv run python examples/collect_s1_answers.py --wave 2`（Phase 6），不得提前。

### Phase 3 — AI 訪談員、訪談本身、後處理（與 Phase 1/2 平行，須在 14 天內完成）

**狀態：拆成 Phase 3-A（後處理管線＋逐字稿/問卷 ingest）與 Phase 3-B（真人語音訪談員本身）。Phase 3-A 程式部分已完成（2026-08-28），`spec-auditor`／`data-hygiene` 皆 PASS（見下方「審查」）。Phase 3-B 完全未動工——見下方獨立說明。**

- ~~AI 訪談員~~ → **Phase 3-B，未動工**（INTERVIEW.md §4/§6）：語音對語音、Teacher 驅動、追蹤大綱與必達點（A1–A4、B1–B8、C1–C3、D1–D2）、即時追問、單一連續 102–120 分鐘場次（D34，不得分段）、即時自我檢核發言占比（≥70%，Q5）。
- [x] 可重跑的後處理管線（§6.2）：`ingest/postprocess.py`——`apply_correction_glossary()`（統一處理步驟 1 專有名詞校正＋步驟 2 中英混用還原，皆為詞表替換）、`mark_unclear_spans()`（步驟 4）、`run_postprocessing_pipeline()`（組合，對同一份原始逐字稿+更新後詞表可重跑，純函式）。步驟 3「口語保留」刻意沒有對應函式——它是一條 MUST NOT（不得順稿），用「這支管線只碰詞表命中的片段，從不碰語氣詞/重複」這個事實本身滿足，而非額外寫一個「不做事」的函式。
- [x] 品質檢核（§7，Q1–Q9）：`ingest/quality_check.py`。Q3/Q4/Q5/Q7/Q9 純腳本；Q6（時間表述保留）為對比原始/校正後文字的啟發式代理，非保證；Q1+Q2（必達點涵蓋＋B1/B2/B6 具體事例數）合併成一次 Teacher 呼叫（`check_coverage_and_instances`，D9 少次大批）。**Q8（third_party_spans 已標註）刻意沒有對應檢查函式**——它不是後驗旗標，而是在 `ingest/sources/interview_transcript.py` 的 ingest 路徑本身結構性阻擋（見下）。
- [x] 逐字稿 ingest（§4、§6：「逐字稿...掛於記憶層 Period 級」）：`ingest/sources/interview_transcript.py::fragments_from_interview_transcript()`，每個必達區塊（A/B/C/D）一個 Fragment，`event_time` 依 §4 建議的區塊分鐘數（42/36/16/8）累加估算——**confidence 刻意標為 0.5、不是 1.0**（§4 本文明講訪談員 MAY 調整區塊時間，對一個推算值給滿信心正是 SPEC §4.4 警告的「虛假精確度」）。**Q8 的結構性阻擋**：`extract_third_party_spans()`（新模組 `ingest/entities.py`，rule-based v1，非 Teacher 驅動——理由是 Q8 是進記憶層前的硬阻擋，不該綁一個會因 RPD 耗盡而失敗的即時網路呼叫）對每個存在的區塊無條件呼叫，沒有任何跳過路徑；`blocks` 字典裡出現非 A/B/C/D 的 key、或 `known_parties` 為空，皆直接 `raise`（前者是「fail loudly」——打錯 label 不該讓整塊內容悄悄消失；後者是「沒有安全預設值」——INTERVIEW §6.3 明講訪談必然涉及第三方，known_parties 為空卻放行等於保證漏標）。
- [x] 結構化問卷（§5）：`ingest/sources/questionnaire.py::fragments_from_questionnaire()`，訪談後才施測（呼叫端責任），答案 MUST 為該題 `scale_labels` 之一（仿 `S1Answer.answer` 的驗證紀律，同時也是「不呼叫 third-party 抽取」這個決定成立的前提——沒有自由文字，第三方就無處可藏）；題庫與 S1 題庫互斥由新模組 `harness/questionnaire_guard.py::assert_disjoint_from_s1_item_bank()` 檢查（放在 `twin.harness` 而非 `twin.ingest`——判斷需要同時看到兩邊的題庫型別，但 import-linter 的 layer spine 禁止 `ingest` 反向 import `harness`）。
- [x] **特別護欄，本輪發現並修正一處落差**：原始規劃（本節先前版本）認為「逐字稿與原始音檔永遠留在 `file://`...只有衍生出的 Period 層級碎片可走一般跨雲同步路徑（§7.2）」，區分「原始逐字稿檔案」與「衍生碎片」。**實際落地 `fragments_from_interview_transcript()` 後這個區分站不住**：為滿足 D26「自陳資料 MUST 可原文還原」，每個區塊 Fragment 的 `content` 就是逐字稿原文本身，不是摘要或衍生物——讓它照一般碎片同步路徑走，等同讓逐字稿原文直接經 `TWIN_FRAGMENT_STORE_URI`（`twin/CLAUDE.md` 記載的正式生產設定即為 R2）離開本機，實質架空 INTERVIEW.md §6.3/§8 I-D「MUST NOT 進入跨雲儲存」。兩輪獨立的 `spec-auditor` 審查各自讀 INTERVIEW.md 原文後都抓到這個落差（其中一輪判 BLOCK）。**修正**：`ingest/store.py::write_fragments_jsonl()` 新增結構性檢查——目標 URI 非 `file://` 時，批次中任何 `source_class == SELF_REPORT` 的 Fragment 一律 `raise`（寫入任何一個 byte 之前，整批檢查，fail-fast），範圍刻意不narrow 到「只有逐字稿來源」——`Fragment` 目前無欄位可分辨自陳內容的來源，寧可保守地讓所有自陳內容（含問卷答案）暫時只能留在本機。新增的 `config.settings.Settings.transcript_store_uri`（file://-only）本身管的是原始音檔/後處理逐字稿「檔案」該存哪，不是這條 Fragment 層防線——兩者職責分開，文件已更新說明。**遺留的已知風險（供後續排查，非本輪落地範圍）**：`write_trajectories_jsonl()` 沒有對稱的檢查（`Trajectory` schema 目前無 `source_class` 欄位，且本輪沒有任何程式碼把自陳文字寫進 `Trajectory`，故此刻不構成真實破口）；另外 `agent/tools/recall.py`（既有程式碼，非本輪新增）完全不依 `split` 過濾，若日後訪談資料的四個區塊因時間相近而全部落在同一個 split（`data-hygiene` 審查記錄的觀察），`split` 標記對它目前沒有實際存取管制效果。

**依據**：INTERVIEW.md §3–§8；SPEC.md §4.1、§4.6/D24、D19、D26、D34–D36、§4.9/D23。
**驗收（Phase 3-A，程式部分）**：新增 6 個模組（`ingest/{entities,postprocess,quality_check}.py`、`ingest/sources/{interview_transcript,questionnaire}.py`、`harness/questionnaire_guard.py`），另修改 `ingest/store.py`、`config/settings.py`，+ 對應測試，`uv run pytest`／`ruff`／`mypy`／`lint-imports` 全過。逐字稿 Fragment 可原文還原（D26，測試覆蓋）；third_party_spans 對逐字稿每個區塊無條件標註（測試以 monkeypatch 計數驗證）。**真正的驗收線（一份連續逐字稿 ≥5,500 字、Q8 已過的真實訪談）仍待 Phase 3-B 完成後才可能發生。**
**審查**：`spec-auditor`（初輪 BLOCK——settings 新欄位未接線、未知區塊標籤靜默丟棄、`known_parties` 空值無防呆、`confidence` 假精確；修正後複審 PASS）、`data-hygiene`（PASS，記錄兩項非阻擋觀察：`event_time.confidence` 對訪談區塊只反映「第幾區塊」而非分鐘級精度、以及上一段記錄的 split/recall() 疊加風險）。
**類型**：Phase 3-A 為程式（已完成）；Phase 3-B 為程式（訪談員本身，完全未動工，見下）+ 人工（102–120 分鐘場次本身，須預留緩衝，不能卡在第 13 天才做）。

**Phase 3-B — 真人語音訪談員 → 已於 2026-08-30 裁決：以最小文字版訪談員取代（使用者裁決）**

INTERVIEW.md §6 把訪談員釘死為語音對語音、Teacher 驅動、單一連續 102–120 分鐘場次，但完全沒有指定實作機制（無 STT/TTS、無 turn-taking、無長 session 架構）。為了讓 B1/B2 在 Wave 2（2026-09-12）前存在、kill switch（Phase 7）能合規裁決，使用者於 2026-08-30 裁決：**先以文字版訪談取代**。落地：`ingest/interview_schedule.py`（§4 四區塊、17 個必達點 A1–D2 與開放式首問，純資料）、`ingest/interviewer.py`（`TextInterviewer`：依大綱逐點提問；`required_instances > 0` 的必達點由 Teacher 依 §6.1 規則產生追問——開放式、無選項、不誘導、形容詞不算事例——每點最多 2 次；單一場次真實計時；雙方發言逐字保留為 `InterviewTranscript`；Teacher 配額耗盡時記入 `notes` 而非中斷）、`ingest/interview_ingest.py`（逐字稿 → §6.2 詞表後處理 → `fragments_from_interview_transcript`（Q8 結構性阻擋）→ 併入 fragment store（先備份、拒絕同一份逐字稿重複 ingest）→ §7 Q1–Q9 報告）、`examples/run_text_interview.py`／`examples/ingest_interview_transcript.py`。測試：`test_ingest_interviewer.py`、`test_ingest_interview_ingest.py`。

**明示的偏離與代價**（INTERVIEW.md §7 將這些列為「標記低信心」而非阻擋，僅 Q8 為硬阻擋）：(1) 形式非語音（§6 形式列）；(2) 時長極可能不到 102 分鐘（Q3）、字數極可能不到 5,500（Q4）→ **整輪 S1 標低信心**，依 EVAL.md §6.4 不得用於閘門升級；(3) §5 結構化問卷未施測（SHOULD，記為待辦，`ingest_interview_transcript.py` 的報告會明列）；(4) 文字輸入無 ASR 不確定性，§6.2 步驟 4 的 `[unclear]` 標記為空集合——這是文字版唯一誠實的優勢；§6.2 步驟 2「中英混用還原」在文字版等同於詞表替換（無 ASR 近音誤寫），無條文可對，記為解讀。(5) §6.1「每區塊結束前 MUST 自我檢核必達點，未達成者 MUST 於同場次內補問」：已實作為區塊末對每個未達成點**一輪**補問（`TextInterviewer` 的 block-end probe），之後無論是否達成即進入下一區塊（D35 不補訪）——補問次數有上限是偏離，未達成點記入 `notes`，ingest 報告的 Q1/Q2 會如實反映。(6) §7 Q5「訪談員 MUST 於每區塊中段自我檢核發言占比」：已實作為區塊中點的占比檢核，低於 70% 只記 `notes`、不改變提問（文字版訪談員發言天然極短，實際失守機率低）。(7) **結構化問卷是 MUST（INTERVIEW.md §3.2、SPEC D19「同時採集」），不是 SHOULD**——本輪未施測，`ingest_interview_transcript.py` 的報告會將 Q7 記為未施測 → S1 低信心；問卷 MUST 於訪談之後另行施測（`ingest/sources/questionnaire.py` 已就位，缺題庫與驅動腳本），列為 Phase 3 待辦。(8) D9「少次、大批」：互動式追問每次一個 Teacher 呼叫（每場上限約 17 點 × 3 次 ≈ 51 次 + ingest 的 Q1/Q2 1 次），遠在 1,500 RPD 內；即時追問無法批次，記為可接受。(9) INTERVIEW.md §4「D2 的回答 MUST 被記錄為孿生的負向約束」：目前只作為區塊 D 內容存入，尚無結構化的負向約束紀錄——消費端是 S4（Phase 12），列為待辦。語音版仍是 INTERVIEW.md 的規範；本節是有紀錄的過渡，不是改寫規格。SPEC.md §10 已補 D37（自陳 split）、D38（文字版訪談員）兩條紀錄。

**真實訪談已完成（2026-08-30，`~/twin-data/transcripts/interview-20260830T083433Z.json`）**：使用者嫌 `run_text_interview.py` 的制式提問追問不足，改由 **Claude Code 本 session 直接擔任訪談員**（依 §4 大綱逐點提問、即時追問到事例、每輪把雙方原文寫入逐字稿）——訪談員從 Gemini Teacher 換成 Claude 是 §6「訪談員 = Teacher 模型」的再一次偏離，記於逐字稿 `notes` 與 D38。結果：63 回合、本人 3,078 字、75 分鐘。ingest（`ingest_interview_transcript.py`，known parties = LINE 對象 + 葉秉鈞/Ray/建富/女友 + 關係詞）：4 個區塊碎片進 store（96,750 → 96,754），Q1 **17/17 涵蓋**、Q2 B1=3／B2=1／B6=2（B2、B6 未達三例——B2 本人明確表示無探索價值，B6 補問兩輪後仍兩例）、Q3 FAIL（75 < 102 分）、Q4 FAIL（3,078 < 5,500）、Q5 FAIL（訪談員發言占比 >30%，文字版追問句偏長）、Q6/Q9 PASS、Q7 問卷未施測。**依 §7：本輪 S1 標低信心、S3 在曝光採集上線前不得用於閘門**。修正：`quality_check.check_coverage_and_instances` 的 Teacher schema 改為 list 型（Gemini Developer API 拒絕 dict/additionalProperties，首次真實呼叫才發現），`ingest_interview_transcript` 新增 `report_only`。

**自陳資料的 split（2026-08-30 裁決：`train`）**：依 §4.8 現行時間規則，訪談 `event_time`（session 時鐘，2026-08-30）晚於 LINE ingest 的 `sealed_cutoff`（2026-07-24），會機械地落入 SEALED——B2 依 EVAL.md §9 讀不到 sealed、LoRA 也永遠學不到訪談內容，與 D19「自陳資料是人格保真的主要來源」直接矛盾。SPEC.md §2.2 明定自陳資料「不是歷史資料」，時間規則對它沒有意義。裁決：自陳資料（逐字稿、問卷）一律 `Split.TRAIN`，仍在 ingest 時一次決定（`ingest/split.py::decide_self_report_split`、`ingest/fragment.py::self_report_fragment`），不是訓練期覆寫；B2 與 T 因此拿到同一份資訊，正是 EVAL.md §3.4 kill switch 要的比較。`harness/baseline.py::load_self_report_transcript` 的 B2 過濾同步改為 TRAIN。選項與取捨已記錄於本輪 spec-trace 表。

### Phase 4 — 最小 L2 + 精簡軌跡集 + 第一版（精簡）LoRA（與 14 天等待平行）

**狀態：訓練垂直切片（`train/model.py`、`checkpoint.py`、`run.py`、`reproducibility.py`、新增 `train/formatting.py`、根目錄 `train.py`、`launch/*`）程式部分已完成（2026-08-27）。最小 `recall()` 已在 §3.8 建好；精簡軌跡集本身（取自真實行為資料）與 LoRA rank 的硬體 probe、真實一輪訓練仍待辦，見下方「仍待辦」。**

- 最小 `recall(query, time_hint)`，以一般工具（C4）形式包在 Phase 1 的碎片之上——刻意簡陋（關鍵字 + 時間窗過濾），尚非完整的 Episode/Period 分層。（已於 §3.8 落地）
- 精簡軌跡集（§4.10），取自容易觀測、有明確 ground truth 的行為資料——刻意跳過嚴謹的硬負例篩選（§4.3 明確允許在大規模歷史 ingest 之前，先以較低信心或無曝光門檻的負例上路，因為 S3 不在 kill switch 的關鍵路徑上）。**已產生（2026-08-29）**：`ingest/trajectories.py::trajectories_from_line_messages()` + `examples/build_trajectories.py`，從同一批 LINE 匯出（同一 manifest、同一組 cutoffs，寫入 `<store>.manifest.json` 供比對）建出 **19,976 筆**（train 15,673 / heldout 3,531 / sealed 772）至 `~/twin-data/data/trajectories.jsonl`。建構規則：對方訊息 burst（同發話者間隔 ≤5 min；實測 p90=2 min）為刺激；相鄰對方 burst 若間隔 ≤ reply_window 且中間無本人發言則合併為一個刺激（`data-hygiene` 指出 22% 的「硬負例」其實是對方連發、本人一併回覆）；本人在 **120 min**（實測回覆延遲 p90）內回覆 → `ActionStep(surface=line)`，否則 `NoActionStep`。曝光證據：LINE 匯出無已讀（§11-H），本人在該室 **24 h 內**再有活動 → `inferred`/`hard`（§4.3 歷史資料低信心，manifest `exposure_note` 明載 **MUST NOT 作 S3 評測來源**），否則 `absent`/`trivial`。晚於 120 min 的回覆在該 tick 記為 `no_action`（`--late-reply skip` 可改為略過）。結果：train 不回應率 14.9%，hard 佔負例 88%（§4.11 SHOULD ≥ 50%；若低於一半腳本直接拒寫，§4.11 MUST NOT）。`train/formatting.build_sft_dataset` 對真實 store 產出 15,673 筆訓練列（中位 203 字元）。`spec-auditor` PASS（三項需人裁決：曝光時點上界、晚回覆歸類、trivial 過半——前兩項已做成參數並取上述預設，第三項改為拒寫）；`data-hygiene` PASS（split 逐筆 0 不符、負例不集中睡眠時段、train 內 0 筆 cutoff 後日期）。**兩個預設（24 h 曝光上界、晚回覆=no_action）是本輪代決，可用 `--exposure-horizon-h`／`--late-reply` 重建推翻。**
- 選定底模：**Qwen3-8B**（dense、Apache-2.0，2025-05 發布）——8B 級、open-weight、permissive license（§5.1）。與 Qwen3.5-9B（Unsloth 官方文件明講不建議 QLoRA 4-bit，量化品質劣化，bf16-LoRA 需 22GB 超出 T4 預算）、Breeze-7B-Instruct-v1.0（唯一過授權關的繁中專用選項，但生態小、agentic 能力弱、已停止更新）、Llama-3-Taiwan／TAIDE／Gemma 2-3（皆因授權條款不符「Apache-2.0 同等」被排除）比較後定案。繁中/台灣用語預設偏簡體的已知代價，計畫靠訓練資料本身（精簡軌跡集）矯正，而非 prompt。**抽測繁簡一致性仍待辦**（需真實訓練跑完成後才能做）。
- `train.py` 走 TRL + Accelerate（§7.6 禁止自建 trainer）；checkpoint 契約（§7.4：adapter、optimizer state、LR schedule、RNG state、`global_step`、dataloader cursor）——**已用真實 `kill -9` + `--resume auto` 驗證**（`tests/unit/test_train_checkpoint_kill_resume.py`：全本地、無網路依賴的玩具 Qwen3 模型，真子行程 `SIGKILL`，比對中斷續跑與不中斷對照組的逐步 loss 曲線，容忍度依實測校準）；`run_id` 綁定 seed/dataset_hash/config_hash（§7.5，`train/reproducibility.py`）；載入時做工具名稱遮蔽（§5.3/D16，接在 `train/formatting.py`）；Modal/Kaggle/Lightning 的 `launch/*`（D12）。
- **Adapter 權重（checkpoint 與最終產出）皆加密儲存**（§8：「Adapter 為個資...MUST 加密儲存」）——`core/encryption.py`（Fernet 對稱加密）、`train/checkpoint.py` 把每次 checkpoint／最終 adapter 打包成單一加密封存檔上傳，`config/settings.py` 新增 `TWIN_ADAPTER_ENCRYPTION_KEY`（無預設值，理由同 `gemini_model`），`examples/generate_adapter_encryption_key.py` 供操作者產生金鑰。範圍明確界定於 adapter 權重本身，不含 `AdapterManifest` JSON（後者是 run_id/雜湊/時間戳等 metadata，本身無法反推行為特徵）。
- **`reply` 在訓練資料中確實表示成跟其他工具同層級的 tool call**（§11-A/D28/D29：「回覆為 tool call」）——`train/formatting.py` 把 `ActionStep` 先轉成合成的 `ToolCallStep(tool="reply", args={surface, content})` 再進 §5.3/D16 的工具名稱遮蔽，讓 `reply` 進入跟 `recall`／`web_search` 同一個遮蔽詞彙池，而不是被模型學成一個永遠不變、不遮蔽的特殊字面量。
- **補強：assistant-only loss masking**（`train/loss_mask.py`、`run.py` 的 `SFTConfig(assistant_only_loss=True)`）——先前 `run.py` 完全未設定此項，等於信任 TRL 對 messages 格式資料集的預設遮蔽行為；非 SPEC.md 明文規定的缺口（無章節直接規範 chat-template 遮蔽解析），但與本節其餘欄位一貫的「顯式記錄、不悄悄信任預設值」工程紀律不一致（比照 `LORA_RANK_FALLBACK_LADDER`、`TrainingConfig` 無預設值欄位的做法）。詳細說明、與參考碼 `twin/reference/llm-twin` 的比對取捨，見 `twin/docs/llm-twin-reference-notes.md`。

**這一輪的 `spec-auditor` 審查**：初次審查判 BLOCK，四項發現——(1) adapter 加密儲存缺失（§8，已修正，見上）、(2) `reply` 未表示成 tool call（§11-A/D28/D29，已修正，見上）、(3) `launch/*` 非純 shell 對 §7.1 字面解讀的不確定性（已記錄於下方「已知偏離」第 1 項，維持現狀，需人裁決）、(4) §7.3 spot/preemptible 未记录（已查證並記錄於下方「已知偏離」第 3 項）。修正 (1)(2) 過程中同時發現並修正一個獨立的正確性 bug：`trainer.model.save_pretrained(final_adapter_uri)` 直接把 fsspec URI 字串傳給不認得 URI scheme 的 `save_pretrained`，會悄悄寫到錯誤的本機路徑而非真正落地——現在改為先存本機再透過 `checkpoint.upload_adapter()` 走加密上傳，並在 kill/resume 測試中新增「下載回最終 adapter 且內容正確」的斷言防止回歸。

**已知偏離，記錄於此而非悄悄發生**：
1. `launch/` 實際上不是純 shell——Modal 的 SDK 是 Python-first（`@app.function(gpu=...)` decorator），CLI 無法純用 flag 定義自訂 GPU job（2026-08-27 對照 Modal 現行文件核實），Kaggle 的 `kernels push` 同樣需要一支程式檔 + `kernel-metadata.json`。新增 `launch/modal_app.py`、`launch/kaggle_kernel.py`、`launch/kaggle_kernel-metadata.json` 作為基礎設施接線（絕不 import `twin.train`，只透過 subprocess 呼叫 `train.py`），放在 `launch/` 底下但在 `src/twin/` 之外，結構上不受 import-linter `root_package="twin"` 掃描，不違反既有契約，但這是對本節開頭樹狀圖「只放 shell」字面意思的偏離。`launch/kaggle_kernel.py` 另有一個真正開放的問題（如何把 `twin` 原始碼本身搬上 Kaggle kernel，目前假設走 git clone + Kaggle Secret，未驗證）；`launch/lightning.sh` 的確切 CLI 動詞集也未對照現行文件核實，三者優先序皆為 Modal > Kaggle > Lightning（同 §7.3 算力表順序），照此排序處理未驗證風險。
2. `requires-python` 由 `>=3.11` 提高為 `>=3.12`（連動 `[tool.mypy] python_version`、`[tool.ruff] target-version`）——`numpy>=2.3`（被 torch/transformers 遞移依賴）在 PyPI 上本身即宣告 `requires-python>=3.12`（已即時查證，非臆測），`>=3.11` 已不是誠實的宣告，在真正的 3.11 直譯器上 `uv sync` 會直接解析失敗。
3. §7.3「MUST 優先使用 spot/preemptible」在三個平台上核實後的現況（2026-08-27 查證）：**Modal 不需額外設定**——其官方文件明講「All Modal Functions are subject to preemption by default」，且 `nonpreemptible` 參數明文不支援 GPU Function，所以 GPU 一律強制可搶佔，`launch/modal_app.py` 的裸 `@app.function(gpu="T4", ...)` 已經滿足這條 MUST，不是遺漏。**Kaggle 這條 MUST 不適用**——其免費層是每週時數配額（30h/週）model，不是 spot 定價/搶佔機制，平台上找不到對應開關。**Lightning AI 未核實**，與該平台 CLI 動詞集的不確定性一併留待實作時查證。

**依據**：SPEC.md §3.1/C4、§4.10、§5.1、§5.2、§5.3/D16、§7.3–§7.6、§7.1/D12、§8、§11-A/D28/D29。
**驗收（CI，非人工）**：`train.py --resume auto` 撐過訓練中途的 `kill -9`，loss 曲線連續（§7.4 明文列為 CI 項目，**已通過**）；產出一個有標籤的 LoRA artifact `T`（**機制已驗證，真實資料上的產出仍待辦**）。
**類型**：程式。

**仍待辦**：
1. ~~`examples/probe_lora_rank.py` 在實際目標硬體上跑過~~ **已完成（2026-08-29，Modal Tesla T4 14.6 GiB，Qwen3-8B 4-bit + all-linear LoRA + 真實 AdamW step，1×512 tokens）**：r=256 **OOM**（峰值 14.40 GiB）、r=128 OK（12.39）、r=64 OK（9.16）、r=32 OK（7.51）。**§11-G 裁決：`lora_rank=64`**——r=128 只剩約 2 GiB 餘裕，真實資料序列長尾（最長 11k 字元）加上 batch 2 會踩線；r=64 留下約 5 GiB。記錄於 `launch/configs/qwen3-8b-t4-r64-v1.json`（lr 1e-4、batch 2×8=16、max_steps 1000 ≈ 一個 epoch）。第一次真實 probe 曾把四個 rung 全報成 ~14.3 GiB OOM——except 路徑沒釋放前一個模型，且未計 optimizer state；已修正並補上 `run.py` 的 gradient checkpointing（記憶體用、非 config_hash 欄位）。
2. ~~精簡軌跡集本身（真實行為資料，非玩具/合成）。~~ 已完成 2026-08-29，見上。
4. **T v2 前的訓練資料處置（2026-08-30 由 T v1 煙霧測試發現）**：(a) `train/formatting.py` 的 `ensure_ascii` 已修；(b) 本人回覆若只是 LINE 媒體佈位符（「圖片」「貼圖」「影片」），目前照原文進 `ActionStep.content`（實測 260/19,976 筆回覆為純媒體佈位符：貼圖 162、圖片 60、多行純媒體 35、影片 3，僅 1.3%，所以 T 反覆回「圖片」主要是退化生成而非資料占比）——**已裁決（使用者，2026-08-30）：改為明確標記**。標記取 `[貼圖]`/`[圖片]`/`[影片]`（Qwen3 tokenizer 各 4/3/3 token；`[media:sticker]` 要 6 個且混英文；與 parser 既有的 `[已收回訊息]` 方括號慣例一致），`ingest/trajectories.py::MEDIA_PLACEHOLDERS` 同時作用於 observation 與本人回覆；只在軌跡層做、不重建 fragment store（凍結題庫綁著 fragment id）。軌跡集以原參數重建（19,976 筆、split 分佈完全相同）＋30 筆訪談軌跡 = 20,006，已上傳 R2。**自陳樣本上採樣（使用者 2026-08-30 要求先處理再訓練）**：30 筆對 15,673 筆是 0.2%，一個 epoch 不到 2 個 optimizer step 的訊號，D19 的「主要來源」會被淹沒；新增 `TrainingConfig.self_report_upsample`（進 config_hash；`build_sft_dataset` 對 `interview` surface 的軌跡重複 N 次、id 一併重複使 dataset_hash 反映），**使用者隨即否決純重複（「不可重複相同的 QA，會背誦」），改為改寫增強**：`ingest/interview_augment.py::augment_interview_trajectories`——每筆觀察到的訪談軌跡請 Teacher（Gemini）一次產出 K 個改寫版本（問題與回答都改寫；硬規則：保留所有事實／人名／時間／數字、不新增資訊、語氣強度與原文一致、不加原文沒有的流行語或俚語、全繁體），輸出再以 zhconv 強制繁體；變體標 `ground_truth_source=teacher_synthesized`（§4.10／D25：只可訓練、`harness.schema.reject_synthesized_for_eval` 擋住進評測），原始 30 筆保留 `observed`。`examples/augment_self_report_trajectories.py --variants 14`：30 次 Teacher 呼叫（D9 一筆一呼叫、K 個一批；首跑撞到免費層 **15 RPM**，`GeminiTeacher` 新增 429 退避重試），dry-run 抽查後修正 prompt（首版 Teacher 會加料成「撩落去」「衝一波」等本人不用的俚語、偶見簡體字）。**再改：改寫者由 Gemini 換成 Claude Code 本身（使用者 2026-08-30 提議）**——擔任訪談員的 session 已握有本人完整口吻，改寫品質與一致性優於 Gemini，也不受 15 RPM 限制；以三個繼承 context 的 fork 各寫 10 題、每題 9 個版本（原答 <40 字者 4 個）到 `~/twin-data/transcripts/variants-*.jsonl`，`examples/import_self_report_variants.py` 以同一套規則（`interview_augment.variant_trajectory`：拒空白／拒與原文相同／繁體正規化／`teacher_synthesized`）併入 store 並上傳；manifest `teacher_model` 記 `claude-code (interviewer session, file-based)`。這是 SPEC §5.2「Teacher 實作可替換」的檔案形式實作，與 EVAL §6 judge 用 Claude Code 的裁決同構；Gemini 路徑（`augment_self_report_trajectories.py`）保留備用，其 14 版本正式跑在寫入前中止。`self_report_upsample` 欄位保留但 v2 config 設 1。**結果（2026-08-30）**：240 個變體（26 題 ×9、4 題短答 ×4）全數通過匯入檢查（0 空白／相同／重複；0 俚語命中；抽查人生敘事題 9 版中 7 版保留全部關鍵名詞、另 2 版以「政大」等同義稍縮），store 20,006 → 20,246，自陳樣本 270 筆 = train 的 1.7%，已上傳 R2。**T v2 以此資料集啟動**（v2 config，Modal `ap-SnfezUKysksMDsa9EJW74K`，~12 s/step）。**訓練前預檢閘（2026-08-30，T v1 浪費 6 小時 GPU 才發現轉義缺陷的直接後果）**：`train/preflight.py::assert_dataset_trainable` 在 `run.py` 建好 SFT dataset 後、任何 GPU 秒之前執行——拒絕 `\u` 轉義的目標、空回覆、自陳佔比 <1%（D19）或 >25%；`train.py --allow-no-self-report` 只給 LINE-only 實驗與 kill/resume 玩具測試。對 T v2 的實際資料離線跑過：0 轉義、0 空白、270 自陳（1.7%），並以真 tokenizer 渲染確認 chat template 不會把中文重新轉義。以 v1 config 啟動的 `ap-U7FGaJkJmqVKgPGHI77Yim` 與純重複 N=20 的 `ap-TSf2ivZWP9w8mk9CMJwNIc` 皆在開跑前停止；(c) 訪談軌跡（`ingest/interview_trajectories.py`）納入訓練集。
3. ~~第一次真實訓練跑~~ **已完成（2026-08-29 19:43Z）**：`run_e6a366ee73958e69`，Qwen3-8B@b968826，r=64/alpha=16 all-linear，seed 42，1000 步，dataset_hash `936f0da6…`（19,976 筆軌跡），config `launch/configs/qwen3-8b-t4-r64-v1.json`。過程：Modal T4 於 step 180 被搶佔 → 等 T4 容量 7 小時 → Modal spend limit 觸頂需手動調高 → 以 `--resume auto` 續跑完成（GPU fallback 清單 T4→L4→A10G 於此加入）。最終 adapter 加密存於 `s3://twin-checkpoints/default/run_e6a366ee73958e69/final`（466 MB；2026-08-30 解密驗證：349 MB fp16 safetensors、504 tensors、無 NaN）。**已知偏離**：完成段的 loss 曲線只在 Modal app log 裡、未落地——`run.py` 現已在 manifest 旁寫 `log_history.json`（2026-08-30；T v1 無此檔，T v2 起有）。繁簡一致性抽測：`examples/generate_s1_candidates.py --consistency-probe`（見 Phase 5），結果由人讀 `candidates/consistency-T.jsonl` 後記錄於此。

### Phase 5 — Baseline + Judge harness（平行，等待期間完成）

**狀態：程式部分已完成（2026-08-28）。`spec-auditor` 兩輪皆 PASS。2026-08-30 真實推論路徑落地：GPU 步驟與 judge 步驟解耦——`examples/generate_s1_candidates.py`（GPU：每個系統對 70 題作答，寫 `harness.s1_run.S1Candidate` JSONL 到 `<s1_root>/candidates/<label>.jsonl`；B0/B1/B2 走 `harness.baseline`，T = 底模 + 從 R2 下載並在容器記憶體解密的 adapter，**尚未接 recall()**，`model` 欄位明寫 `(no recall)`）由 `launch/modal_app.py::s1_candidates`（local entrypoint）→ `s1_candidates_fn`（GPU T4/L4/A10G）驅動；`examples/prepare_s1_eval_round.py`（CPU）改為讀候選檔配 R2 切 shard，所有路徑自 `TWIN_S1_EVAL_ROOT_URI` 推導（eval root = 其上層），不再有 checkout-relative 路徑。`examples/run_baseline_inference.py::HFBaselineBackend` 改為 4-bit（同訓練的 `build_quantization_config`，fp16 權重 16.4 GB 放不進 T4）+ chat template（`enable_thinking=False`；T 是經同一 template SFT 的，餵原始文字量到的是格式混亂而非人格）+ 可選 PEFT adapter。真實一輪 judge 對齊（EVAL.md §6.3）仍待 Wave 2。**

**2026-08-30 發現並修正：`eval/rubric/s1.md` 遺失**——`eval/` 整目錄被 SPEC §8 護欄 2 gitignore，rubric 從未進版控，隨 checkout 消失。rubric 不是個資，且 EVAL.md §6.4 要求對它做 hash，MUST 版控：重建於 `src/twin/harness/rubric/s1.md`（`match`/`no_match`/`unjudgeable` 三值，對齊 `aggregate_s1` 合約），由 `harness.eval_io.rubric_uri("s1")` 解析。這與 `eval-harness` skill 的 `eval/rubric/` 樹狀圖不一致，以本節為準。

**已知偏離（記錄，非悄悄發生）**：(a) B2 的逐字稿與 B1 的 persona 以 `.remote()` 函式參數送進 Modal——Modal 會序列化函式輸入並在其後端暫存（可由 call ID 回取一段時間），再進 GPU 容器的記憶體與 ephemeral `/tmp`；不進 Volume、不進 R2。這比「只在容器記憶體」寬：自陳內容確實短暫經過 Modal 的基礎設施。INTERVIEW.md §6.3 禁止的是進入「跨雲**儲存**」；本機 Jetson 無法跑 8B 推論（torch cu130 不支援 Orin CC 8.7），這是讓 B2 存在的最小暴露。使用者 2026-08-30 裁決 **效果優先**（SPEC §10 D39）：kill switch 有結果、效果確認足夠之前，自陳資料完整、未去名地進 B2 推論與 T v2 訓練，隱私縮減（去名過境、本機推論、加密樣本）延後裁決。(b) **T 的兩個限制**：第一，`label="T"` 的候選答案目前是「adapter，無 recall()」——EVAL.md §3.4 定義 T = LoRA + memory store，20% 回想題（14 題）必然失分；依 §3.5 回想題本來就分開計分，Phase 7 讀分數時 MUST 把回想題與非回想題分開看，`S1Candidate.model` 欄位明寫 `(no recall)`，且 `prepare_s1_eval_round.py` **拒絕**含 `(no recall)` 的 T 候選檔、拒絕有 T 無 B2 的 round（EVAL §12-4）——PLAN 的約束有對應的程式閘。第二，`run_e6a366ee73958e69` 訓於訪談之前的 19,976 筆 LINE 軌跡，`twin.train` 目前根本不讀 fragment store——D37 裁決理由「B2 與 T 拿到同一份資訊」在程式碼上尚不成立。**Phase 7 kill switch 前 MUST 以含自陳資料的訓練集重訓一版 T（T v2），否則 T vs B2 的比較不對等**（T v1 只能當 B0 的對照）。**已落地（2026-08-30）**：`ingest/interview_trajectories.py::trajectories_from_interview`——每個被回答的訪談員提問成為一筆軌跡（observation = 同區塊前文 + 提問、stimulus = 提問、`ActionStep(surface="interview")` = 本人回答原文、exposure `read_receipt`、split=train、無負例；空白回答直接略過，不記 `no_action`），走與 LINE 回覆完全相同的 `trajectory_to_messages` 路徑（回答變成遮蔽後的 `reply` tool call）。`examples/build_self_report_trajectories.py` 併入軌跡 store（備份、拒絕重複、manifest 記 `self_report`）並可 `--upload-to` R2（D39：明文，同既有軌跡）；之後以 v1 config 重訓即為 T v2（dataset_hash 變 → 新 run_id）。測試 `test_ingest_interview_trajectories.py`。(c) `RunManifest.adapter_hash` 對 T 填 `core.hashing.adapter_hash(adapter_model.safetensors)`（真 hash，於 GPU 容器解密後計算，隨候選檔回傳），baseline-only 輪填 `"none"`。

- [x] `B0`（底模+空白）、`B1`（底模+persona 段落）、`B2`（底模+Phase 3 逐字稿注入 context）的推論 harness：`harness/baseline.py`——`InferenceBackend` Protocol（`complete(prompt) -> str`，介面先行，真實 HF/transformers 綁定留給 §「仍待辦」）、`render_b0/b1/b2_prompt()`、`generate_baseline_samples()`。每個樣本 `source_label` 固定填 `"twin"`——只是一個路由標籤，不是「這是孿生 T」的宣稱（B0/B1/B2 皆不碰 LoRA adapter，依 SPEC §2.1 定義本來就不是「孿生」）；`harness.shard.strip_source_label` 在進 `eval/in/` 前就把整個欄位砍掉，judge 從未看到它，S1 真正的 T/B0/B1/B2 歸屬由下面獨立的 side-channel 承擔。
- [x] `harness/s1_run.py`（新模組）：`compute_self_consistency()`（R1 vs R2 純字串比對，`S1Answer.answer` 依合約恆為選項原文，不需要 judge）、`S1SampleIndexEntry` + `build_s1_raw_samples()`（每個 (item, baseline) 組合一則綁定 R2 參考答案+候選答案的樣本，供 judge 逐筆比對）、`regroup_judged_items_by_baseline()`（judge 逐筆判定回來後，靠這個 judge 看不到的 side-channel 精確地掛回哪個 baseline，任何 sample_id 對不上就 `raise`，而非靜默 `.get()`）。
- [x] 依 EVAL.md §6.2 腳本化的 Claude Code judge 產物層：`harness/eval_io.py`（新模組——`shard.py` 先前只做記憶體內切分，這裡才真正把 `eval/in/<run_id>/shard-NNN.jsonl` 寫到磁碟，跟隨 `eval-harness` skill 的巢狀目錄慣例而非 EVAL.md §6.2 字面較粗略的單檔寫法）、`compute_rubric_hash()`（rubric 檔位元組 + shard 大小一起雜湊，因為 shard 大小本身是「rubric 的一部分」）、`read_judged_shard()`——**對照 `.claude/agents/eval-judge.md` 的真實輸出格式核實過**：judge 實際寫的欄位是 `reason`，不是 `harness.schema.JudgedItem` 原有的 `rationale`，這裡是明確的欄位對應層，而非兩邊剛好同名。`eval/rubric/s1.md`（新增，`match`/`no_match`/`unjudgeable` 三值，對齊 `aggregate_s1` 既有的 `agree_verdicts` 預設值）。
- [x] `examples/run_baseline_inference.py`（真實 HF backend，GPU-only，本輪未跑）、`examples/prepare_s1_eval_round.py`（`eval-harness` skill 步驟 0–2：檢查是否重跑、產生 B0/B1/B2 候選答案、剝除標籤、寫 shard；B2 的逐字稿注入只取 `Split.HELDOUT` 的自陳碎片，不含 `SEALED`——EVAL §9 只在最終驗收才開封 sealed，這是首輪 `spec-auditor` 抓到的落差，已修正）、`examples/score_s1_eval_round.py`（步驟 4–6：聚合＋門檻檢查＋寫報告）。
- [x] **`score_s1_eval_round.py` 的門檻設計，回應初輪 `spec-auditor` 的阻擋級發現**：第一版直接把 judge 逐筆判定聚合成真實 `S1Metrics` 並印出/寫入一份平行的 `_s1_metrics.json`，而 EVAL.md §6.3 的 30 筆對齊驗證根本還不存在（不是還沒過門檻，是從未執行）——等同繞過既有的 `harness.gate_check.check_judge_agreement_floor()`，也繞過 `eval-harness` skill 唯一認可的 `eval/report/<run_id>.md` 產出管道。**修正**：腳本現在要求 `--judge-agreement`／`--confidence` 兩個必填 CLI 參數（明講「這是真實 Phase 6 對齊結果，不是本腳本算出來的」），在記憶體中組出完整 `EvalReport`（`s1` 為真實聚合值，`s2/s3/s4` 留 `None`，不捏造），呼叫既有、未改動的 `check_judge_agreement_floor()`——低於 0.80 直接 `raise`，此時任何分數都還沒印出或寫入——通過後才透過 `write_report()` 寫進 `eval/report/`。（judge shard 本身的機械健康度統計——unjudgeable/低信心/flag 計數——不受此限制，先印出：那是「judge 有沒有正常解析」的問題，不是「孿生像不像本人」的結論。）

**依據**：SPEC.md §4.1/D26；EVAL.md §3.4、§6.1、§6.2、§6.3。
**驗收**：harness 對合成/假資料能跑通全流程（**已驗證**，24 個新測試涵蓋 `baseline.py`/`s1_run.py`/`eval_io.py`），產出格式正確的 `eval/out/<run_id>/shard-NNN.jsonl`（**機制已驗證**）。真實 B0/B1/B2 推論、真實 judge 對齊仍待辦，見下。
**審查**：`spec-auditor`（初輪 BLOCK——`score_s1_eval_round.py` 繞過對齊門檻、B2 逐字稿注入未濾 split，皆已修正；`source_label="twin"` 文件不夠精確，已補強；複審 PASS）。此軌不觸及 L1/L2，`data-hygiene` 依其自身觸發條件不適用。
**類型**：程式（已完成）。

**仍待辦**：
1. ~~`examples/run_baseline_inference.py` 在真實 GPU 上跑~~ → 2026-08-30 以 `modal run launch/modal_app.py::s1_candidates --labels B0,T --consistency-probe 8` 首跑；結果記於下方「真實推論紀錄」。B1/B2 待文字版訪談 ingest 與 `persona.txt` 就位後補跑。
2. EVAL.md §6.3 的 30 筆真實對齊驗證（Phase 6，須等 Wave 2／14 天等待完成後才有真實 judge 輸出可對齊）——`score_s1_eval_round.py` 的 `--judge-agreement` 參數就是這一步的消費端。
3. 文字版訪談（Phase 3-B 的替代）ingest 後，B2 的 `transcript_text` 由 `harness.baseline.load_self_report_transcript`（SELF_REPORT ∧ TRAIN）取得；B1 的 persona 段落由本人寫於 `<fragment store 同目錄>/persona.txt`（`harness.baseline.default_persona_uri`）。

**真實推論紀錄（2026-08-30，Modal app `ap-nVEWWMVd9LkaBjmJSprjNQ`，`--labels B0,T --consistency-probe 8`；第一次嘗試因 Modal builder 下載 torch 逾時而失敗，image 加 `UV_HTTP_TIMEOUT=600` 後重跑成功）**：
- **B0**：70/70 有答案，中位 364 字，59/70 的回答逐字含某個選項（judge 可判）。繁簡探針：繁體輸入 8/8 繁體輸出；**簡體輸入 0/8 繁體輸出**（底模跟著輸入字體走）。檔案 `candidates/B0.jsonl`。
- **T v1（adapter，無 recall，直接餵 S1 題目）**：**不可用於評測輪**——34/70 輸出 `<tool_call>` JSON（訓練時 reply 即 tool call 的格式，工具名為遮蔽後的隨機名 `line`/`lineNotify`/`surface`…），6/70 空白，多筆退化重複（`去去去去…`），僅 8/70 含選項原文。兩個原因：(a) 推論 prompt 形狀與訓練不同——訓練樣本是 `system: available tools: …` + 對方訊息刺激 + assistant tool_call，S1 題目以裸 user message 進去量到的是格式錯位不是人格；T 的 S1 作答 MUST 經 L4 形狀的包裝（注入工具清單、把題目當刺激、解析 tool_call 取 `content`）——**已落地（2026-08-30 下午）**：`agent/decode.py::decode_tool_calls/reply_content`（依參數形狀取 `content`，不信任被遮蔽過的工具名；容忍截斷與字串型 arguments），`generate_s1_candidates.py` 的 T 路徑改為 `system: available tools: recall, web_search, reply`（C2：真名於推論時注入）+ 題目為 user 刺激，另存 `T.raw.jsonl` 供檢視 no_action 比例。(b) **訓練資料缺陷**：`train/formatting.py` 的 `json.dumps(step.args)` 預設 `ensure_ascii=True`，所有中文回覆在訓練標的裡都是 `\uXXXX`，T 忠實學會了——每個中文字 6 個 ASCII token，浪費容量且不可讀。已改為 `ensure_ascii=False`（改變 dataset 表示，下一次訓練的 `dataset_hash` 會變，T v1 不受影響）。繁簡探針：繁體輸入 8/8 繁體、**簡體輸入 6/8 繁體**——LoRA 確實把輸出字體拉向本人習慣（SPEC §5.1 的「輸出字體 MUST 依本人真實使用習慣」正在被學到），這是 T v1 唯一的正向訊號。檔案改名為 `candidates/T-v1-smoke-rawformat.jsonl`、`consistency-T-v1-smoke.jsonl`，讓 `prepare_s1_eval_round.py` 不會把它當正式 T 讀入。
- GPU：第一輪未印；第二輪（下）為 NVIDIA L4。
- **T v1 第二輪（L4 形狀，Modal，2026-08-30 下午）**：70/70 都正確發出 `reply` tool call（工具名為真名 `reply`、`surface: line`）——L4 包裝有效，T 學到的是「收到訊息就用 reply 工具回」這個政策形狀。但 41/70 因 `\uXXXX` 轉義把 JSON 撐長被 256 token 截斷（`agent/decode.py` 已加截斷救援，重解碼後 0 題空白）。**內容品質**：回覆中位 17 字、LINE 口語（「對呀」「可以先…嗎」）、1/70 提到選項、10/70 只有「圖片／對呀／可以啊」——T v1 學到的是**LINE 回覆的風格與長度**，完全沒有學到「對情境題做選擇」。這不是 bug，是資料：訓練集裡沒有任何「被問一個兩難、給出立場」的樣本（訪談軌跡正是補這一塊，D19）。另一個資料缺陷：本人的媒體訊息在匯出裡是純文字「圖片」，成了訓練標的，T 會回「圖片\n圖片」——T v2 前 MUST 決定媒體佈位符回覆的處置（排除、或改為明確的 media marker），見 Phase 4 待辦 4。檔案：`candidates/T.jsonl`（已重解碼）、`T.raw.jsonl`。

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
