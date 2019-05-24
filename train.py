import torch
from torch.utils.data import DataLoader

from config import device, batch_size,save_models, model_params, embed_params, encoder_params, transformer_encoder_params, data_params, training_params, paths,ensemble_models

from sklearn.metrics import accuracy_score, f1_score

from models.rnn_classifier import RNNClassifier

from modules.layers.embeddings.elmo import ELMo
from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset


# def make_weights_for_balanced_classes(images, nclasses):                        
#     count = [0] * nclasses                                                      
#     for item in images:                                                         
#         count[item[1]] += 1                                                     
#     weight_per_class = [0.] * nclasses                                      
#     N = float(sum(count))                                                   
#     for i in range(nclasses):                                                   
#         weight_per_class[i] = N/float(count[i])                                 
#     weight = [0] * len(images)                                              
#     for idx, val in enumerate(images):                                          
#         weight[idx] = weight_per_class[val[1]]                                  
#     return weight         



print('Loading dataset...')
preprocessing = Preprocessing()

train_data, valid_data,valid_data_B,test_data,test_data_B = load_data(**data_params)
x_column, y_column = data_params['x_column'], data_params['y_column']

train_set = ClassificationDataset(train_data[:, x_column], train_data[:, y_column], preprocessing=preprocessing.process_text)
valid_set = ClassificationDataset(valid_data[:, x_column], valid_data[:, y_column], preprocessing=preprocessing.process_text)
test_set = ClassificationDataset(test_data[:, x_column], test_data[:, y_column], preprocessing=preprocessing.process_text)
test_set_B = ClassificationDataset(test_data_B[:, x_column], test_data_B[:, y_column], preprocessing=preprocessing.process_text)
valid_set_B = ClassificationDataset(valid_data_B[:, x_column], valid_data_B[:, y_column], preprocessing=preprocessing.process_text)

# weights = make_weights_for_balanced_classes(train_set, 2)
# weights = torch.DoubleTensor(weights)                                       
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
import pandas as pd
import csv
print(" ".(test_set.data))
df_train=pd.DataFrame({'data':train_set.data,'labels':train_set.labels})
df_test=pd.DataFrame({'data':test_set.data,'labels':test_set.labels})
df_valid=pd.DataFrame({'data':valid_set.data,'labels':valid_set.labels})
print(df_train.head())
df_train.to_csv('checkpoints/train.csv')
df_test.to_csv('checkpoints/test.csv')
df_valid.to_csv('checkpoints/valid.csv')

#df_train=pd.DataFrame({'data':train_set.data,'labels':train_set.labels})
df_test_B=pd.DataFrame({'data':test_set_B.data,'labels':test_set_B.labels})
df_valid=pd.DataFrame({'data':valid_set_B.data,'labels':valid_set_B.labels})
#print(df_train.head())
#df_train.to_csv('checkpoints/train.csv')
df_test.to_csv('checkpoints/test_B.csv')
df_valid.to_csv('checkpoints/valid_B.csv')


train_loader = DataLoader(train_set,batch_size,shuffle=True, collate_fn=collate_fn_cf)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
test_loader = DataLoader(test_set, batch_size, collate_fn=collate_fn_cf)
test_loader_B = DataLoader(test_set_B, batch_size, collate_fn=collate_fn_cf)
valid_loader_B = DataLoader(valid_set_B, batch_size, collate_fn=collate_fn_cf)
print('Creating model...')

embeddings = ELMo(**embed_params)
model = RNNClassifier(embeddings,encoder_params, **model_params).to(device)

optimizer = torch.optim.Adam(model.parameters())

weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.NLLLoss(weight=weights)

trainer = ClassificationTrainer(model, criterion, optimizer, device)

print('Training...')
best_binary_f1 = None
gold_labels = test_set.labels.astype(int)
i=0
for epoch in range(training_params['n_epochs']):

    train_loss = trainer.train_model(train_loader)

    valid_loss, predicted, model_predictions, labels = trainer.evaluate_model(valid_loader)

    print('| Epoch: {} | Train Loss: {:2.5f} | Val. Loss: {:2.5f} | Val. Acc: {:2.5f} | Val. Macro F1: {:2.5f} | Val. Micro F1: {:2.5f} | Val. Binary F1: {:2.5f} |'
          .format(epoch + 1, train_loss, valid_loss, accuracy_score(labels, predicted),
                  f1_score(labels, predicted, average='macro'), f1_score(labels, predicted, average='micro'), f1_score(labels, predicted)))

    macro_f1, binary_f1 = f1_score(labels, predicted, average='macro'), f1_score(labels, predicted)
     
#     if not best_binary_f1 or binary_f1 > best_binary_f1:
#         print('saving binary')
#         best_binary_f1 = binary_f1
#         torch.save(model, paths['f1_score']['model_path'])
#     else:
    torch.save(model,save_models[i])
    i=i+1
    print(i)

