import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using {} device".format(device))

data_params = {
    'train_file': 'data/subtask-a/V1.4_Training.csv',
    'validation_file': 'data/subtask-a/SubtaskA_Trial_Test_Labeled.csv',
    'test_file': 'data/subtask-a/SubtaskA_EvaluationData_labeled.csv',
    'test_file_B': 'data/subtask-a/SubtaskB_EvaluationData_labeled.csv',
    # 'validation_file': 'data/subtask-b/SubtaskB_Trial_Test_Labeled.csv',
    # 'test_file': 'data/subtask-b/SubtaskB_EvaluationData_labeled.csv',
    'x_column': 1,
    'y_column': 2,
    'header': None
}

batch_size = 32

embed_params = {
    'embedding_dropout': 0.5
}

encoder_params = {
    'hidden_size': 1024,
    'num_layers': 2,
    'bidirectional': True,
    'dropout': 0.3,
    'batch_size': batch_size
}

transformer_encoder_params = {
    'hidden_size': 2048,
    'num_layers': 3,
    'num_heads': 8,
    'dropout': 0.1
}

model_params = {
    'output_dim': 2,
    'dropout': 0.3
}

training_params = {
    'n_epochs': 20
}

paths = {
    'f1_score': {
        'model_path': 'checkpoints/best_valid_f1_model',
        'submission': 'submissions/f1_score'
    }

}

ensemble_models = [
    'best_valid_f1_model'
#     'run_1',
#     'run_2',
#     'run_3',
#     'run_4',
#     'run_5',
#     'run_6',
#     'run_7',
#     'run_8'
#     'b_run_1',
#     'b_run_2',
#     'b_run_3'
]
