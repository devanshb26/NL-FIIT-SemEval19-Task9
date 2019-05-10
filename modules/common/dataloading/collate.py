import numpy as np
import torch


def collate_fn_cf(data):
    samples, labels, lengths = zip(*data)
    print(list(set([type(i) for i in labels])))
    for ix , i in enumerate(labels):
        try:
            a = int(i)
        except ValueError:
            labels = list(labels)
            print("LOOK HERE")
            print(type(i))
            print("|",i,"|")
            print(type(labels))
            labels[ix] = 0
    labels = [int(str(i).strip()) for i in labels]
    print("labels", labels)
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)

    if isinstance(samples[0], np.ndarray):
        samples = torch.LongTensor(samples)
    elif isinstance(samples[0], tuple):
        samples = samples

    return samples, labels, lengths
