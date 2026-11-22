# v0.6 聯盟平均打者基準

本棒打者的 history_exit、history_angle、history_direction、history_n 一律使用凍結模型 profiles.batter_prior：[136.93923236117308, 10.389694183075223, 1.6208534027931416, 0]。這是訓練期的平均打者特徵，並非把所有局面設成同一個得分常數，也不是逐一平均所有球員的模型輸出。

球數、壘況、出局數、球位、球速、投手與下一棒保持原情境效果；E0、E1 都使用相同平均打者，避免混用個人與聯盟基準。E2 的安打種類權重沿用這個共同基準。只換本棒身分時基準相同。

測試：在 repo 根目錄執行 node tests/league-batter-baseline.cjs（需 python3）。150 個打者 ID（含未知球員）×12 球數，共 1800 案例驗證 before/after 完全相同與機率總和為 1。Python trace_model.py 比較王順和／朱育賢通過。

這是推論／評分政策變更，沒有重訓；既有 metrics.json 對應舊個人特徵設定，不能當成本次改版的準確度。新政策的時間切分 Log Loss、Brier、校準與 runs 效能待驗證。
