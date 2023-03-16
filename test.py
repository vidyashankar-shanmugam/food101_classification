from torchmetrics import F1Score, Confusion_Matrix

def test_model(model, test_loader):
    for data, target in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        f1 = F1Score(task = 'multiclass', num_classes=101, average='macro')
        f1_score = f1(pred, target)
        cm = Confusion_Matrix(task = 'multiclass', num_classes=101)
        conf_matrix = cm(pred, target)
        return f1_score, conf_matrix