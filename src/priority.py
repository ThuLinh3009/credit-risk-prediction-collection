"""Sprint 2-3 — Công thức xếp hạng ưu tiên thu hồi nợ.

priority_score = risk_score x muc_do_tre_han

- risk_score: xac suat vo no (0-1) do mo hinh tot nhat du bao.
- muc_do_tre_han: max(PAY_1..PAY_6, 0). Cac gia tri PAY_n <= 0
  (tra dung han / dung tin dung xoay vong / khong phat sinh du no)
  duoc coi la khong tre han (=0), chi cac gia tri duong (so thang
  tre han thuc te) moi tinh vao muc do tre han.
"""

from __future__ import annotations

import pandas as pd

from src.config import PAY_COLS


def compute_delay_severity(df: pd.DataFrame) -> pd.Series:
    return df[PAY_COLS].clip(lower=0).max(axis=1)


def build_priority_ranking(df_raw: pd.DataFrame, ids: pd.Series, risk_scores) -> pd.DataFrame:
    """df_raw: DataFrame chứa các cột PAY_1..PAY_6 tương ứng với ids/risk_scores
    (cùng thứ tự hàng).
    """
    delay_severity = compute_delay_severity(df_raw).to_numpy()
    priority_score = risk_scores * delay_severity

    ranking = pd.DataFrame(
        {
            "ID": ids.to_numpy(),
            "risk_score": risk_scores,
            "delay_severity": delay_severity,
            "priority_score": priority_score,
        }
    )
    ranking = ranking.sort_values("priority_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "priority_rank", ranking.index + 1)
    return ranking
