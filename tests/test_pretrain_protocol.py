from sdhtg.training.pretrain import LabelFilteredDataset


class Dataset:
    def __init__(self):
        self.rows = [{"label": 0}, {"label": 1}, {"label": 0}]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]


def test_normal_only_protocol():
    dataset = LabelFilteredDataset(Dataset(), "normal_only")
    assert len(dataset) == 2
    assert all(dataset[i]["label"] == 0 for i in range(len(dataset)))


def test_all_train_protocol():
    dataset = LabelFilteredDataset(Dataset(), "all_train")
    assert len(dataset) == 3
