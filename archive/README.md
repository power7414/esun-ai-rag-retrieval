# archive/ — historical code, not part of the results

**English.** This folder keeps an early demo version of the project for history only.
**The scores in this code are hard-coded; it does not perform real retrieval and does not represent any
actual result.** When run on the official 150-question example set it scores **0/150 (0%)**
(measured 2026-11-07). Nothing in the main README depends on it.

| Path | What it is |
| :--- | :--- |
| `src/` | Modular skeleton (`pipeline.py`, `preprocessor/`, `retriever/`). `FAQ_Q2Q_Retriever.search()` always returns the first candidate with a fixed score of 0.96; `LexiconBM25Retriever.search()` returns a fixed 0.88. |
| `scripts/run_competition.py` | Entry point that calls `src/`; every prediction comes out as `"default_fallback"`. |
| `experiments/q2a_direct_embedding.py`, `experiments/bilstm_dssm_test.py` | Only `print` fixed numbers (12%, 0%); they are not experiments. |

---

**繁體中文。** 這裡是舊示範版，**分數寫死，不代表實際結果，僅保留歷史**。
`src/` 的檢索器回傳固定分數、`run_competition.py` 的輸出全是 `"default_fallback"`，在官方 150 題範例上實測
**0/150（0%）**；`experiments/` 兩支腳本只會印出寫死的 12%、0%，不是實驗。主 README 的任何結果都不依賴這個資料夾。
