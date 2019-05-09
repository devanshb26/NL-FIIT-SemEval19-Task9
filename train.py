import torch
from torch.utils.data import DataLoader

from config import device, batch_size, model_params, embed_params, encoder_params, transformer_encoder_params, data_params, training_params, paths

from sklearn.metrics import accuracy_score, f1_score

from models.rnn_classifier import RNNClassifier

from modules.layers.embeddings.elmo import ELMo
from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset

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

embeddings = ELMo(**embed_params)
model = RNNClassifier(embeddings, encoder_params, **model_params).to(device)

optimizer = torch.optim.Adam(model.parameters())

weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.NLLLoss(weight=weights)

trainer = ClassificationTrainer(model, criterion, optimizer, device)

print('Training...')
best_binary_f1 = None
gold_labels = test_set.labels.astype(int)

for epoch in range(training_params['n_epochs']):

    train_loss = trainer.train_model(train_loader)

    valid_loss, predicted, model_predictions, labels = trainer.evaluate_model(valid_loader)

    print('| Epoch: {} | Train Loss: {:2.5f} | Val. Loss: {:2.5f} | Val. Acc: {:2.5f} | Val. Macro F1: {:2.5f} | Val. Micro F1: {:2.5f} | Val. Binary F1: {:2.5f} |'
          .format(epoch + 1, train_loss, valid_loss, accuracy_score(labels, predicted),
                  f1_score(labels, predicted, average='macro'), f1_score(labels, predicted, average='micro'), f1_score(labels, predicted)))

    macro_f1, binary_f1 = f1_score(labels, predicted, average='macro'), f1_score(labels, predicted)

    if not best_binary_f1 or binary_f1 > best_binary_f1:
        print('saving binary')
        best_binary_f1 = binary_f1
        torch.save(model, paths['f1_score']['model_path'])
