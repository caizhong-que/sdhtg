import pandas as pd
from sdhtg.data.split import temporal_session_split


def test_temporal_split_has_no_session_leakage():
    rows = []
    for s in range(10):
        for event in range(2):
            rows.append({"session_id": f"s{s}",
                         "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(s, "h"),
                         "source_event_id": s * 2 + event})
    result = temporal_session_split(pd.DataFrame(rows))
    assert result.groupby("session_id").split.nunique().max() == 1
    assert set(result.split) == {"train", "validation", "test"}
