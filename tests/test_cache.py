import pandas as pd
from sdhtg.data.cache import build_vocabs, materialize_sessions


def test_vocab_fits_train_only_and_unknown_maps_to_unk():
    frame=pd.DataFrame({
      "split":["train","validation","test"], "template":["a","new","a"],
      "entity_sem":["e","e","e"], "action_sem":["x","x","x"],
      "status_sem":["s","s","s"], "session_id":["1","2","3"],
      "event_label":[0,0,1], "session_label":[0,0,1],
      "timestamp":pd.to_datetime([1,2,3],unit="s",utc=True),
      "source_event_id":[1,2,3]
    })
    vocab=build_vocabs(frame)
    sessions=materialize_sessions(frame,vocab,10)
    validation=sessions[sessions.split=="validation"].iloc[0]
    assert validation.template_ids==[1]
