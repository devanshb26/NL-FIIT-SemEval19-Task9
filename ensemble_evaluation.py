import torch
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix as cm
from modules.layers.embeddings.elmo import ELMo
from config import device, ensemble_models,batch_size, model_params, embed_params, encoder_params, transformer_encoder_params, data_params, training_params, paths,save_csv,save_csv_B
from models.rnn_classifier import RNNClassifier
from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset

from utils.submit import save_predictions, save_predictions_with_probabilities

print('Loading dataset...')
preprocessing = Preprocessing()

train_data, valid_data,valid_B, test_data,test_data_B = load_data(**data_params)
x_column, y_column = data_params['x_column'], data_params['y_column']

train_set = ClassificationDataset(train_data[:, x_column], train_data[:, y_column], preprocessing=preprocessing.process_text)
valid_set = ClassificationDataset(valid_data[:, x_column], valid_data[:, y_column], preprocessing=preprocessing.process_text)
valid_set_B = ClassificationDataset(valid_B[:, x_column], valid_B[:, y_column], preprocessing=preprocessing.process_text)
test_set = ClassificationDataset(test_data[:, x_column], test_data[:, y_column], preprocessing=preprocessing.process_text)
test_set_B = ClassificationDataset(test_data_B[:, x_column], test_data_B[:, y_column], preprocessing=preprocessing.process_text)

train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
valid_loader = DataLoader(valid_set, batch_size, collate_fn=collate_fn_cf)
test_loader = DataLoader(test_set, batch_size, collate_fn=collate_fn_cf)
test_loader_B = DataLoader(test_set_B, batch_size, collate_fn=collate_fn_cf)
valid_loader_B = DataLoader(valid_set_B, batch_size, collate_fn=collate_fn_cf)

print('Creating model...')
weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
embeddings = ELMo(**embed_params)
model = RNNClassifier(embeddings, encoder_params, **model_params).to(device)

optimizer = torch.optim.Adam(model.parameters())

trainer = ClassificationTrainer(None, criterion, optimizer, device)

print('Evaluate...')
gold_labels = test_set.labels.astype(int)
predictions = []
losses = []
import pandas as pd
for model_name in ensemble_models:
    trainer.model = torch.load('checkpoints/' + model_name)

    test_loss, predicted, model_predictions, labels = trainer.evaluate_model(test_loader)
    predictions.append(model_predictions)
    losses.append(test_loss)
#     save_predictions(name='submissions/' + model_name, predictions=predicted, original_data=test_data)
#     save_predictions_with_probabilities(name='submissions/' + model_name + '_full', predictions=predicted, original_data=test_data, labels=gold_labels, probabilities=model_predictions)

print('Sum ensemble')

sum_predictions = np.stack(predictions).sum(axis=0)
predicted = np.argmax(sum_predictions, 1)
# save_predictions(name='submissions/ensemble', predictions=predicted, original_data=test_data)
# save_predictions_with_probabilities(name='submissions/ensemble_full', predictions=predicted, original_data=test_data, labels=gold_labels, probabilities=sum_predictions)
print(cm(gold_labels,predicted))
df=pd.DataFrame({'reviews':test_data[ : ,1],'predictions':predicted,'labels':gold_labels}) 
df.to_csv(save_csv[0])
print('----------------------------------------------------Test results/SubtaskA----------------------------------------------------')
print('| Loss: {} | Acc: {}% |'.format(test_loss, accuracy_score(labels, predicted)))
print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='binary'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='binary'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro'), f1_score(gold_labels, predicted)))
print('--------------------------------------------------------------------------------------------------------------------')

print('Voting ensemble')

leaderboard = np.zeros(predictions[0].shape)
for index, prediction in enumerate(predictions):
    for j in range(prediction.shape[0]):
        leaderboard[j][prediction[j].argmax()] += 1

predicted = np.argmax(leaderboard, axis=1)
# save_predictions(name='submissions/ensemble_voting', predictions=predicted, original_data=test_data)
# save_predictions_with_probabilities(name='submissions/ensemble_voting_full', predictions=predicted, original_data=test_data, labels=gold_labels, probabilities=sum_predictions)
print(cm(gold_labels,predicted))
df=pd.DataFrame({'reviews':test_data[ : ,1],'predictions':predicted,'labels':gold_labels}) 
df.to_csv(save_csv[1])
print('----------------------------------------------------Test results/SubtaskA----------------------------------------------------')
print('| Loss: {} | Acc: {}% |'.format(test_loss, accuracy_score(labels, predicted)))
print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='binary'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='binary'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro'), f1_score(gold_labels, predicted)))
print('--------------------------------------------------------------------------------------------------------------------')


gold_labels = test_set_B.labels.astype(int)
predictions = []
losses = []

for model_name in ensemble_models:
    trainer.model = torch.load('checkpoints/' + model_name)

    test_loss, predicted, model_predictions, labels = trainer.evaluate_model(test_loader_B)
    predictions.append(model_predictions)
    losses.append(test_loss)
#     save_predictions(name='submissions/' + model_name, predictions=predicted, original_data=test_data)
#     save_predictions_with_probabilities(name='submissions/' + model_name + '_full', predictions=predicted, original_data=test_data, labels=gold_labels, probabilities=model_predictions)
print('Sum ensemble')

sum_predictions = np.stack(predictions).sum(axis=0)
predicted = np.argmax(sum_predictions, 1)
# save_predictions(name='submissions/ensemble', predictions=predicted, original_data=test_data)
# save_predictions_with_probabilities(name='submissions/ensemble_full', predictions=predicted, original_data=test_data, labels=gold_labels, probabilities=sum_predictions)
print(cm(gold_labels,predicted))
df=pd.DataFrame({'reviews':test_data_B[ : ,1],'predictions':predicted,'labels':gold_labels}) 
df.to_csv(save_csv_B[0])
print('----------------------------------------------------Test results/SubtaskB----------------------------------------------------')
print('| Loss: {} | Acc: {}% |'.format(test_loss, accuracy_score(labels, predicted)))
print('| Macro Precision: {} | Precision: {} |'.format(precision_score(gold_labels, predicted, average='binary'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Recall: {} |'.format(recall_score(gold_labels, predicted, average='binary'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro'), f1_score(gold_labels, predicted)))
print('--------------------------------------------------------------------------------------------------------------------')

print('Voting ensemble')

leaderboard = np.zeros(predictions[0].shape)
for index, prediction in enumerate(predictions):
    for j in range(prediction.shape[0]):
        leaderboard[j][prediction[j].argmax()] += 1

predicted = np.argmax(leaderboard, axis=1)
# save_predictions(name='submissions/ensemble_voting', predictions=predicted, original_data=test_data)
# save_predictions_with_probabilities(name='submissions/ensemble_voting_full', predictions=predicted, original_data=test_data, labels=gold_labels, probabilities=sum_predictions)
print(cm(gold_labels,predicted))
df=pd.DataFrame({'reviews':test_data_B[ : ,1],'predictions':predicted,'labels':gold_labels}) 
df.to_csv(save_csv_B[1])
print('----------------------------------------------------Test results/SubtaskB----------------------------------------------------')
print('| Loss: {} | Acc: {}% |'.format(test_loss, accuracy_score(labels, predicted)))
print('| Macro Precision: {} | Precision: {} |'.format(precision_score(gold_labels, predicted, average='binary'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Recall: {} |'.format(recall_score(gold_labels, predicted, average='binary'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro'), f1_score(gold_labels, predicted)))
print('--------------------------------------------------------------------------------------------------------------------')

# gold_labels_B = test_set_B.labels.astype(int)
# print(gold_labels_B)
# i=0
# for model_name in ensemble_models:
#     trainer.model = torch.load('checkpoints/' + model_name)

#     test_loss_B, predicted_B, model_predictions_B, labels_B = trainer.evaluate_model(test_loader_B)
# #     df=pd.DataFrame({'predictions':predicted,'labels':labels})
#     df=pd.DataFrame({'reviews':test_data_B[ : ,1],'predictions':predicted_B,'labels':labels_B}) 
#     df.to_csv(save_csv_B[i])
#     i=i+1
#     print(f1_score(gold_labels_B, predicted_B))
#     print(cm(labels_B,predicted_B))
#     print('----------------------------------------------------Test results/SubtaskB----------------------------------------------------')
#     print(model_name)
#     print('| Loss: {} | Acc: {}% |'.format(test_loss_B, accuracy_score(labels_B, predicted_B)))
#     print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels_B, predicted_B, average='macro'), precision_score(gold_labels_B, predicted_B, average='micro')))
#     print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels_B, predicted_B, average='macro'), recall_score(gold_labels_B, predicted_B, average='micro')))
#     print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels_B, predicted_B, average='macro'), f1_score(gold_labels_B, predicted_B, average='micro'), f1_score(gold_labels_B, predicted_B)))
#     print('--------------------------------------------------------------------------------------------------------------------')
#     #save_predictions(name='submissions/' + model_name + 'predictions', predictions=predicted_B)
#     save_predictions(name='submissions/' + model_name +'_B', predictions=predicted_B, original_data=test_data_B)
#     save_predictions_with_probabilities(name='submissions/' + model_name + '_full_B', predictions=predicted_B, original_data=test_data_B, labels=labels_B, probabilities=model_predictions_B)
