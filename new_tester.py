import pandas as pd
import numpy as np
import mxnet as mx
import datetime as dt
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from random import choice
from string import ascii_uppercase
import os
from scipy.stats import zscore

from gluonts.dataset.util import to_pandas
from gluonts.mx.distribution import StudentTOutput
from gluonts.mx.distribution import GaussianOutput
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
import scipy.stats as st
import pickle
import argparse
import datetime
from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset)
import seaborn as sns
from fixed_helper_financial_metrics import *
sns.set(style="darkgrid")


########################################################################
##############   PARAMETERS & HYPERPARAMETERS  #########################
########################################################################
parser = argparse.ArgumentParser(description='Gradient Attack')
parser.add_argument('--company', default='CHTR', type=str, help='Company')
parser.add_argument('--train_length', default=60, type=int, help='Length of training set')
parser.add_argument('--validation_length', default=70, type=int, help='Length of validation set')
parser.add_argument('--test_length', default=120, type=int, help='Length of testing set')
parser.add_argument('--prediction_length', default=5, type=int, help='Prediction Length')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--nbpe', default=30, type=int, help='Number of batches per epoch')
parser.add_argument('--student_t', default=True, type=bool, help='Student-T distribution or guassian')
parser.add_argument('--twitter', default=True, type=bool, help='Using twitter as dynamic features or not')
parser.add_argument('--save_figs', default=True, type = bool, help = 'decides if the figures and txt files should be saved or not')
parser.add_argument('--show_figs', default=False, type=bool, help='decides if the figures should be shown to the screen')
parser.add_argument('--seed', default=6, type=int, help='seed')
parser.add_argument('--adv_dir', default = +1, type= int, help ="Direction of parameter to modify (-1 or +1)")
parser.add_argument('--epsilon', default=0.5, type=float, help='Percent change in dataset at each iteration')
parser.add_argument('--max_iter', default=4, type=int, help='number of iterations on the adv dataset algorithm')
parser.add_argument('--parameter', default='mu', type=str, help='parameter we want to change. its mu sigma nu for student-t',)
parser.add_argument('--adv_example_type', default='complex', type=str, help='parameter we want to change. its mu sigma nu for student-t',)
parser.add_argument('--bit', default='', type=str, help='p',)