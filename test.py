from torchmetrics import F1Score, ConfusionMatrix
import torch

def test_model(model, device, test_loader):
    counter = 0
    for inputs, labels in test_loader:
        data = inputs.to(device, non_blocking=True)
        target = labels.to(device, non_blocking=True)
        output = model(data)
        _, pred = torch.max(output, 1)
        f1 = F1Score(task = 'multiclass', num_classes=101, average='macro')
        f1_score = f1(pred, target)
        cm = ConfusionMatrix(task = 'multiclass', num_classes=101)
        conf_matrix = cm(pred, target)
        return f1_score, conf_matrix