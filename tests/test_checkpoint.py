import torch
from sdhtg.training.checkpoint import CheckpointManager


def test_checkpoint_round_trip(tmp_path):
    model=torch.nn.Linear(3,2); optimizer=torch.optim.AdamW(model.parameters()); manager=CheckpointManager(tmp_path)
    original={k:v.clone() for k,v in model.state_dict().items()}
    path=manager.save("last",model=model,optimizer=optimizer,scheduler=None,scaler=None,epoch=3,global_step=7,best_metric=.8,patience_count=2,metadata={"x":1})
    with torch.no_grad():
        for value in model.parameters(): value.add_(10)
    payload=manager.load(path,model=model,optimizer=optimizer)
    assert payload["epoch"]==3 and payload["global_step"]==7
    assert all(torch.equal(model.state_dict()[k],v) for k,v in original.items())
