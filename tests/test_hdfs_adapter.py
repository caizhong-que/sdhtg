from pathlib import Path
import yaml
from sdhtg.data.adapters import create_adapter
from sdhtg.data.config import load_config


def test_hdfs_block_duplication_and_official_labels(tmp_path: Path):
    (tmp_path / "HDFS.log").write_text(
        "081109 203518 143 INFO dfs.DataNode$PacketResponder: Received blk_-1 and blk_2\n",
        encoding="utf-8")
    (tmp_path / "labels.csv").write_text(
        "BlockId,Label\nblk_-1,Normal\nblk_2,Anomaly\n", encoding="utf-8")
    config = {
        "name":"hdfs", "adapter":"hdfs", "raw_dir":str(tmp_path),
        "processed_dir":str(tmp_path/"out"),
        "files":[{"path":"HDFS.log","role":"log"},{"path":"labels.csv","role":"labels"}],
        "log_format":"<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "timestamp_fields":["Date","Time"], "timestamp_formats":["%y%m%d %H%M%S"],
        "entity_fields":[], "content_field":"Content", "label_field":"Label",
        "label_file_key":"BlockId", "normal_labels":["Normal"], "anomaly_labels":["Anomaly"],
        "sessionization":"native"
    }
    path=tmp_path/"cfg.yaml"; path.write_text(yaml.safe_dump(config),encoding="utf-8")
    frame=create_adapter(load_config(path)).normalize()
    assert len(frame)==2
    assert set(frame.native_session_id)=={"blk_-1","blk_2"}
    assert frame.groupby("native_session_id").event_label.first().to_dict()=={"blk_-1":0,"blk_2":1}
