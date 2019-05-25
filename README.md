# Overview

The repository contains source code of the models submitted by NL-FIIT team as a part of SemEval 2019 Task 9 submission.

## Citation
```
@inproceedings {pecar2019semeval,
  author={Pecar, Samuel and Simko, Marian and Bielikova, Maria},
  booktitle={Proceedings of the 13th International Workshop on Semantic Evaluation (SemEval-2019)},
  title={NL-FIIT at SemEval-2019 Task 9: Neural model ensemble for suggestion mining},
  year={2019}
}
```

## Prerequisites

- csv datafiles in `data` directory (see `config.py` file)

# Documentation


# Running commands
Install allennlp before running the train command
! pip3 install --ignore-installed allennlp
## Train model
```
python3 train.py
```

## Evaluate model
```
python3 evaluate.py
python3 evaluate_ensemble.py
```

## Docker container
Commands for running docker container.
```
nvidia-docker run -dt --runtime=nvidia --mount type=bind,source=/home/pecars/semeval,target=/workspace/src --name pytorch pytorch/pytorch:latest
docker exec -it pytorch /bin/bash
```

