import torch
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import device, batch_size, encoder_params, data_params, ensemble_models

from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset

from utils.submit import save_predictions, save_predictions_with_probabilities

print('Loading dataset...')
preprocessing = Preprocessing()

train_data, valid_data, test_data = load_data(**data_params)
x_column, y_column = data_params['x_column'], data_params['y_column']

train_set = ClassificationDataset(train_data[:, x_column], train_data[:, y_column], preprocessing=preprocessing.process_text)
valid_set = ClassificationDataset(valid_data[:, x_column], valid_data[:, y_column], preprocessing=preprocessing.process_text)
test_set = ClassificationDataset(test_data[:, x_column], test_data[:, y_column], preprocessing=preprocessing.process_text)

train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
test_loader = DataLoader(test_set, batch_size, collate_fn=collate_fn_cf)

print('Creating model...')
weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

trainer = ClassificationTrainer(None, criterion, optimizer, device)

print('Evaluate...')

gold_labels = data.test_set.labels.astype(int)
predictions = []
losses = []

for model_name in ensemble_models:
    trainer.model = torch.load('checkpoints/' + model_name)

    test_loss, predicted, model_predictions, labels = trainer.evaluate_model(test_loader)
    predictions.append(model_predictions)
    losses.append(test_loss)
    save_predictions(name='submissions/' + model_name, predictions=predicted, original_data=data.test_data)
    save_predictions_with_probabilities(name='submissions/' + model_name + '_full', predictions=predicted, original_data=data.test_data, labels=gold_labels, probabilities=model_predictions)

print('Sum ensemble')

sum_predictions = np.stack(predictions).sum(axis=0)
predicted = np.argmax(sum_predictions, 1)
save_predictions(name='submissions/ensemble', predictions=predicted, original_data=data.test_data)
save_predictions_with_probabilities(name='submissions/ensemble_full', predictions=predicted, original_data=data.test_data, labels=gold_labels, probabilities=sum_predictions)

print('----------------------------------------------------Test results----------------------------------------------------')
print('| Loss: {} | Acc: {}% |'.format(loss, accuracy_score(labels, predicted)))
print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro'), f1_score(labels, predicted)))
print('--------------------------------------------------------------------------------------------------------------------')

print('Voting ensemble')

leaderboard = np.zeros(predictions[0].shape)
for index, prediction in enumerate(predictions):
    for j in range(prediction.shape[0]):
        leaderboard[j][prediction[j].argmax()] += 1

predicted = np.argmax(leaderboard, axis=1)
save_predictions(name='submissions/ensemble_voting', predictions=predicted, original_data=data.test_data)
save_predictions_with_probabilities(name='submissions/ensemble_voting_full', predictions=predicted, original_data=data.test_data, labels=gold_labels, probabilities=sum_predictions)

print('----------------------------------------------------Test results----------------------------------------------------')
print('| Loss: {} | Acc: {}% |'.format(loss, accuracy_score(labels, predicted)))
print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro'), f1_score(labels, predicted)))
print('--------------------------------------------------------------------------------------------------------------------')
