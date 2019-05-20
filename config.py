import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using {} device".format(device))

data_params = {
    'train_file': 'data/subtask-a/V1.4_Training.csv',
    'validation_file': 'data/subtask-a/SubtaskA_Trial_Test_Labeled.csv',
    'validation_file_B': 'data/subtask-a/SubtaskB_Trial_Test_Labeled.csv',
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
save_models = [
    'checkpoints/run_1',
    'checkpoints/run_2',
    'checkpoints/run_3',
    'checkpoints/run_4',
    'checkpoints/run_5',
    'checkpoints/run_6',
    'checkpoints/run_7',
    'checkpoints/run_8',
    'checkpoints/run_9',
    'checkpoints/run_10',
    'checkpoints/run_11',
    'checkpoints/run_12',
    'checkpoints/run_13',
    'checkpoints/run_14',
    'checkpoints/run_15',
    'checkpoints/run_16',
    'checkpoints/run_17',
    'checkpoints/run_18',
    'checkpoints/run_19',
    'checkpoints/run_20'
]

save_csv = [
    'checkpoints/ensemble_A.csv',
#     'run_1',
#     'run_2',
#     'run_3',
#     'run_4',
#     'run_5',
#     'run_6',
#     'run_7',
#     'run_8',
#     'run_9',
#     'run_10',
    'checkpoints/voting_A.csv'
#     'run_12',
#     'checkpoints/run_13lp.csv',
#     'checkpoints/run_14lp.csv'
    
]

save_csv_B = [
    'checkpoints/ensemble_B.csv',
#     'run_1',
#     'run_2',
#     'run_3',
#     'run_4',
#     'run_5',
#     'run_6',
#     'run_7',
#     'run_8',
#     'run_9',
#     'run_10',
#     'checkpoints/run_11lp_B.csv',
#     'run_12',
#     'checkpoints/run_13lp_B.csv',
    'checkpoints/voting_B.csv'
    
]

ensemble_models = [
#     'run_11',
#     'run_10',
#    'run_15',
#     'run_18',
    'run_5',
    'run_6',
#     'run_7',
    'run_8'
#     'run_9',
#     'run_10',
#     'run_11',
#     'run_12',
#     'run_13',
#     'run_15'
#     'run_14'
    
]
