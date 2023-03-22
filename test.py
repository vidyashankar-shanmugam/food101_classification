from torchmetrics import F1Score, ConfusionMatrix
import torch

def test_model(model, device, test_loader):
    """Args:
        model: model to test
        device: device to use
        test_loader: test data loader"""

    """F1 score is a commonly used performance metric in classification tasks that combines precision and recall into a single value.
    Precision is the fraction of true positives (TP) out of all predicted positives (TP + false positives (FP)), 
    while recall is the fraction of true positives (TP) out of all actual positives (TP + false negatives (FN)).
    The F1 score is the harmonic mean of precision and recall, and is calculated as:
             F1 score = 2 * (precision * recall) / (precision + recall)"""

    """A confusion matrix is typically organized into rows and columns, with each row representing the actual class of the data, 
    and each column representing the predicted class of the data. The four possible outcomes of a binary classification problem are:
    True Positive (TP): The model predicted a positive outcome, and the actual outcome was positive.
    False Positive (FP): The model predicted a positive outcome, but the actual outcome was negative.
    True Negative (TN): The model predicted a negative outcome, and the actual outcome was negative.
    False Negative (FN): The model predicted a negative outcome, but the actual outcome was positive."""

    f1 = F1Score(task='multiclass', num_classes=101, average='macro').to(device)
    cm = ConfusionMatrix(task='multiclass', num_classes=101).to(device)
    model.eval()
    for inputs, labels in test_loader:
        data = inputs.to(device, non_blocking=True)
        target = labels.to(device, non_blocking=True)
        output = model(data)
        _, pred = torch.max(output, 1)
    f1_score =f1.compute()
    conf_matrix = cm.compute()
    return f1_score, conf_matrix