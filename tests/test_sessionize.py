import pandas as pd
from sdhtg.data.config import DataConfig
from sdhtg.data.sessionize import assign_sessions


def test_idle_gap_is_per_entity():
    cfg = DataConfig(name="x", adapter="bgl", raw_dir=None, processed_dir=None,
                     files=(), log_format="<Content>", entity_fields=("Node",),
                     sessionization="idle_gap", idle_gap_seconds=60)
    frame = pd.DataFrame({
        "entity": ["a", "a", "a", "b"],
        "timestamp": pd.to_datetime([0, 10, 100, 50], unit="s", utc=True),
        "source_event_id": [0, 1, 2, 3], "event_label": [0, 0, 1, 0]
    })
    out = assign_sessions(frame, cfg)
    assert out[out.entity == "a"].session_id.nunique() == 2
    assert out.groupby("session_id").session_label.max().max() == 1
