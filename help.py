import pyterrier as pt
from pyterrier_dr import NumpyIndex, TctColBert
import torch
from pathlib import Path
import ir_datasets
import pandas as pd

pt.init()


dataset = pt.get_dataset('irds:msmarco-passage')

print(dataset)

print(dataset.get_corpus())
