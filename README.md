# Overview

The repository contains source code of the models submitted by NL-FIIT team.

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

- elmo wight and options file in `elmo` directory
- csv datafiles in `data` directory (see `parameters.py` file)
- create `submissions` and `checkpoints` directories

# Documentation


# Running commands

## Train model
```
python3 main.py
```

## Evaluate model
```
python3 evaluate.py
python3 evaluate_ensemble.py
```

## Docker container
Commands for running docker container.
```
nvidia-docker run -dt --runtime=nvidia --mount type=bind,source=/home/pecars/semeval,target=/workspace/src --name pytorch pytorch/pytorch:nightly-runtime-cuda9.2-cudnn7
docker exec -it pytorch /bin/bash
```

