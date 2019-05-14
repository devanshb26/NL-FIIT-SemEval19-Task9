import torch
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix as cm
from modules.layers.embeddings.elmo import ELMo
from config import device, ensemble_models,batch_size, model_params, embed_params, encoder_params, transformer_encoder_params, data_params, training_params, paths
from models.rnn_classifier import RNNClassifier
from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset

from utils.submit import save_predictions, save_predictions_with_probabilities

print('Loading dataset...')
preprocessing = Preprocessing()

train_data, valid_data, test_data,test_data_B = load_data(**data_params)
x_column, y_column = data_params['x_column'], data_params['y_column']

train_set = ClassificationDataset(train_data[:, x_column], train_data[:, y_column], preprocessing=preprocessing.process_text)
valid_set = ClassificationDataset(valid_data[:, x_column], valid_data[:, y_column], preprocessing=preprocessing.process_text)
test_set = ClassificationDataset(test_data[:, x_column], test_data[:, y_column], preprocessing=preprocessing.process_text)
test_set_B = ClassificationDataset(test_data_B[:, x_column], test_data_B[:, y_column], preprocessing=preprocessing.process_text)

train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
valid_loader = DataLoader(valid_set, batch_size, collate_fn=collate_fn_cf)
test_loader = DataLoader(test_set, batch_size, collate_fn=collate_fn_cf)
test_loader_B = DataLoader(test_set_B, batch_size, collate_fn=collate_fn_cf)

print('Creating model...')
weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
embeddings = ELMo(**embed_params)
model = RNNClassifier(embeddings, encoder_params, **model_params).to(device)

optimizer = torch.optim.Adam(model.parameters())

trainer = ClassificationTrainer(None, criterion, optimizer, device)

print('Evaluate...')
gold_labels = test_set.labels.astype(int)
print(gold_labels)
import pandas as pd
for model_name in ensemble_models:
    trainer.model = torch.load('checkpoints/' + model_name)
   
    test_loss, predicted, model_predictions, labels = trainer.evaluate_model(test_loader)
    df=pd.DataFrame({'predictions':predicted,'labels':labels}) 
    df.to_csv('abc.csv',index=False)
    print(f1_score(labels, predicted))
    print(cm(labels,predicted))
    print('----------------------------------------------------Test results/SubtaskA----------------------------------------------------')
    print(model_name)
    print('| Loss: {} | Acc: {}% |'.format(test_loss, accuracy_score(labels, predicted)))
    print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
    print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
    print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro'), f1_score(labels, predicted)))
    print('--------------------------------------------------------------------------------------------------------------------')
    #save_predictions(name='submissions/' + model_name + 'predictions', predictions=predicted)
    
    
    save_predictions(name='submissions/' + model_name, predictions=predicted, original_data=test_data)
    save_predictions_with_probabilities(name='submissions/' + model_name + '_full', predictions=predicted, original_data=test_data, labels=gold_labels, probabilities=model_predictions)
    
gold_labels_B = test_set_B.labels.astype(int)
print(gold_labels_B)

for model_name in ensemble_models:
    trainer.model = torch.load('checkpoints/' + model_name)

    test_loss_B, predicted_B, model_predictions_B, labels_B = trainer.evaluate_model(test_loader_B)
#     df=pd.DataFrame({'predictions':predicted,'labels':labels})
    print(f1_score(labels_B, predicted_B))
    print(cm(labels_B,predicted_B))
    print('----------------------------------------------------Test results/SubtaskB----------------------------------------------------')
    print(model_name)
    print('| Loss: {} | Acc: {}% |'.format(test_loss_B, accuracy_score(labels_B, predicted_B)))
    print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels_B, predicted_B, average='macro'), precision_score(gold_labels_B, predicted_B, average='micro')))
    print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels_B, predicted_B, average='macro'), recall_score(gold_labels_B, predicted_B, average='micro')))
    print('| Macro F1: {} | Micro F1: {} | Binary F1: {} |'.format(f1_score(gold_labels_B, predicted_B, average='macro'), f1_score(gold_labels_B, predicted_B, average='micro'), f1_score(labels_B, predicted_B)))
    print('--------------------------------------------------------------------------------------------------------------------')
    #save_predictions(name='submissions/' + model_name + 'predictions', predictions=predicted_B)
    save_predictions(name='submissions/' + model_name +'_B', predictions=predicted_B, original_data=test_data_B)
    save_predictions_with_probabilities(name='submissions/' + model_name + '_full_B', predictions=predicted_B, original_data=test_data_B, labels=gold_labels_B, probabilities=model_predictions_B)
    
