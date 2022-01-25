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

args = parser.parse_args()
plottype ="log_diff"
adv_example_type = args.adv_example_type
company = args.company
train_length = args.train_length
validation_length = args.validation_length
test_length = args.test_length
prediction_length = args.prediction_length
step_size = prediction_length
epochs = args.epochs
batch_size = args.batch_size
num_batches_per_epoch = args.nbpe
student_t = args.student_t
twitter = args.twitter
#adv example generation:
direction = args.adv_dir
epsilon = args.epsilon#in percentage
parameter = args.parameter
max_iter = args.max_iter
print('wtf', args.save_figs)

seed = args.seed
mx.random.seed(seed)
np.random.seed(seed)

if not os.path.isdir(f'plots{args.bit}'):
    os.mkdir(f'plots{args.bit}')
plot_path = f"./plots{args.bit}/"

if not os.path.isdir(f'metrics{args.bit}'):
    os.mkdir(f'metrics{args.bit}')
metric_path = f"./metrics{args.bit}/"

if not os.path.isdir(f'plots{args.bit}/{company}'):
    os.mkdir(f'plots{args.bit}/{company}')
dir_path = f"./plots{args.bit}/{company}/"

if not os.path.isdir(f'metrics{args.bit}/{company}'):
    os.mkdir(f'metrics{args.bit}/{company}')
metric_dir_path = f"./metrics{args.bit}/{company}/"


file_head = f"{company}_DeepAR_Daily_"
file_body = f"_prediction_length={prediction_length}_BS={batch_size}_NBpE={num_batches_per_epoch}_epochs={epochs}" \
               f"_parameter={parameter}_dir={direction}_eps={epsilon}_max_iter={max_iter}_adv_example_type={adv_example_type}"


########################################################################
########################   LOADING DATA  ###############################
########################################################################
#df is the main df, df_target is the target we want to predict, feat_dynamic = covars, plotting_feat_dynamic = covars + prediction length of time.

#filename = f"./Daily_data/4-daily-final_data/{company}_1d.csv"
filename = f"/Users/gabriel/Desktop/Fixed_Stock_Project/input_data/CHTR_1d.csv"

columns = ["Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score","positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count", "neutral_count","General_score"]
# = ["log_diff","Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score","positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count", "neutral_count","General_score"]
df = pd.read_csv(filename, index_col=1)
df = df.drop(['Unnamed: 0'],axis = 1)
df['diff'] = df['Close']-df['Open']
df['log_diff'] = np.log((df['Close']/df['Open']))
df_target = df[plottype]
df_close = df['Close']
df_open = df['Open']


########################################################################################
######################## Z-score normalization of the covariates #######################
########################################################################################
df[columns] = df[columns].apply(zscore)
max_values = df.max()
min_values = df.min()

start_date  =[df_target.index[0] for _ in range(1)]
target_values = np.zeros((1, train_length))
target_values[0,:] = df_target[:train_length].to_numpy()

#dynamic features for training
feat_dynamic = np.zeros((len(columns), train_length))
for i,col in enumerate(columns):
    feat_dynamic[i,:] = df[col][:train_length].to_numpy()
feat_dynamic = [feat_dynamic]

training_data = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: FDR}
            for (target, start,FDR) in zip(target_values, start_date, feat_dynamic)], freq ="1B")

########################################################################################
###### Plotting version of Training Set (Length of "train_length + pred_length")########
########################################################################################

plotting_feat_dynamic = np.zeros((len(columns),train_length+prediction_length))
for i,col in enumerate(columns):
    plotting_feat_dynamic[i,:] = df[col][:train_length+prediction_length].to_numpy()
plotting_feat_dynamic = [plotting_feat_dynamic]

plotting_training_data = ListDataset([{
    FieldName.TARGET: target,
    FieldName.START: start,
    FieldName.FEAT_DYNAMIC_REAL: FDR}
        for (target, start, FDR) in zip(target_values, start_date, plotting_feat_dynamic)], freq ="1B")

########################################################################################
################# Validation Set (Length of "validation_length") #######################
########################################################################################
validation_target_values = np.zeros((1, validation_length))
validation_target_values[0,:] = df_target[:validation_length].to_numpy()

feat_dynamic_val = np.zeros((len(columns), validation_length))
for i,col in enumerate(columns):
    feat_dynamic_val[i,:] = df[col][:validation_length].to_numpy()
feat_dynamic_val = [feat_dynamic_val]

validation_data = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: FDR}
            for (target, start,FDR) in zip(validation_target_values, start_date, feat_dynamic_val)], freq ="1B")

########################################################################################
#################### Testing Set (Length of "testing_length") ##########################
########################################################################################

testing_target_values = np.zeros((1, test_length))
testing_target_values[0,:] = df_target[:test_length].to_numpy()

plotting_test_feat_dynamic =np.zeros((len(columns),test_length+prediction_length))
for i,col in enumerate(columns):
    plotting_test_feat_dynamic[i,:] = df[col][:test_length+prediction_length].to_numpy()
plotting_test_feat_dynamic = [plotting_test_feat_dynamic]

testing_data = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: FDR}
    for (target, start, FDR) in zip(testing_target_values, start_date,plotting_test_feat_dynamic)], freq ="1B")

#Test set:
adversarial_fdr =np.zeros((len(columns),test_length+prediction_length))
for i,col in enumerate(columns):
    adversarial_fdr[i,:] = df[col][:test_length+prediction_length].to_numpy()
adversarial_fdr = [adversarial_fdr]

########################################################################################
################################ Initializing Model ####################################
########################################################################################

estimator = DeepAREstimator(
        freq="1B",
        scaling=True,
        prediction_length=prediction_length,
        batch_size=batch_size,
        distr_output= StudentTOutput() if student_t == True else GaussianOutput(),
        use_feat_dynamic_real= True if twitter == True else False,
        trainer=Trainer(
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            learning_rate=1e-3,),)

########################################################################################
################################# Training Model #######################################
########################################################################################

grad_str = ''.join(choice(ascii_uppercase) for i in range(200))
grad_str = f"./gradient_dir/" +grad_str
predictor = estimator.train(training_data=training_data,validation_data = validation_data, num_workers = 0, grad_str = grad_str)

########################################################################################
###################### Training Rolling Dataset for Plotting ###########################
########################################################################################

dataset_rolled = generate_rolling_dataset(
        dataset=plotting_training_data,
        start_time=pd.Timestamp(df_target.index[0], freq="1B"),
        end_time=pd.Timestamp(df_target.index[train_length+prediction_length], freq = "1B"),
        strategy=StepStrategy(prediction_length=prediction_length, step_size = step_size))

########################################################################################
############################ Plotting Training Forecasts ###############################
########################################################################################

prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()

for i,train_dict in enumerate(dataset_rolled):
    #step 1 create the start
    start_ = [train_dict['start']]
    #step 2, create the target
    interim_target = train_dict['target'].reshape((1,-1))
    #step 3, create the feat dynamic real
    fdr =  [plotting_feat_dynamic[0][:,:train_length + prediction_length - (i*(step_size))]]
    #print(f"shape makes sense ? -> target  = {interim_target.shape} fdr = {fdr.shape}, orig fdr = {plotting_feat_dynamic.shape}")
    train_data = ListDataset([{
            FieldName.TARGET: target,
            FieldName.START: start,
            FieldName.FEAT_DYNAMIC_REAL: FDR}
                for (target, start, FDR) in zip(interim_target, start_, fdr)], freq ="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data, predictor.predict(train_data)):
        to_pandas(test_entry).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7),label = 'Ground Truth')
        if i !=0:
            forecast.plot(color='m', prediction_intervals=prediction_intervals, zorder=10)

plt.title(f"Forecasts on Training Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth",'_nolegend_', "Median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Time (business days)")
plt.legend(legend, loc="upper left")

filename = dir_path + file_head + "1_training" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()
mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
        dataset_rolled, predictor=predictor, num_samples=100)
mx.random.seed(seed)
np.random.seed(seed)
training_agg_metrics, _ = Evaluator(num_workers = 0)(ts_it, forecast_it)

mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
        dataset_rolled, predictor=predictor, num_samples=100)
mx.random.seed(seed)
np.random.seed(seed)
forecasts = list(forecast_it) #this is the pd series
tss = list(ts_it) #this is the dataframe


train_list_samples = np.array([[]])
training_target_vals = np.array([])
training_pred_vals = np.array([])
training_dates = np.array([], dtype = np.datetime64)
for i,(target, forecast) in enumerate(zip(tss, forecasts)):
    #first get the dates. they are added in opposite order so thats why accuracy was wrong before.
    training_dates = np.concatenate((target.index[-step_size:],training_dates))
    #Now get the target values
    training_target_vals = np.concatenate((target.values[-step_size:].reshape((-1,)),training_target_vals))
    #Lastly, get the predictions
    training_pred_vals = np.concatenate((np.mean(forecast.samples, axis =0).reshape((-1,)), training_pred_vals))
    if i != 0:
        train_list_samples = np.concatenate([forecast.samples.reshape((100,prediction_length)),train_list_samples],axis =1)
    else:
        train_list_samples = forecast.samples.reshape((100,prediction_length))


training_target_vals = training_target_vals[prediction_length:]
training_pred_vals = training_pred_vals[prediction_length:]
training_dates = training_dates[prediction_length:]
training_forecasted_samples = train_list_samples[:,prediction_length:]

########################################################################
######################  MAKING TEST SET   ##############################
########################################################################

dataset_rolled = generate_rolling_dataset(
        dataset=testing_data,
        start_time=pd.Timestamp(df_target.index[validation_length], freq="1B"),
        end_time=pd.Timestamp(df_target.index[test_length+prediction_length], freq = "1B"),
        strategy=StepStrategy(prediction_length=prediction_length, step_size = step_size))
fig, ax = plt.subplots()
prediction_intervals=[50.0, 90.0]
for i,(train_dict) in enumerate(dataset_rolled):
    start_ = [train_dict['start']]
    interim_target = train_dict['target'].reshape((1,-1))
    fdr =  [plotting_test_feat_dynamic[0][:,:test_length + prediction_length - (i*(step_size))]]

    train_data = ListDataset([{FieldName.TARGET: target,FieldName.START: start,FieldName.FEAT_DYNAMIC_REAL: FDR} for (target, start, FDR)
                              in zip(interim_target, start_, fdr)], freq ="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data, predictor.predict(train_data)):
        to_pandas(test_entry)[validation_length:].plot(linewidth=1, color='b', zorder=0, figsize=(13, 7),label= 'Ground Truth')
        if i!=0:
            forecast.plot(color='g', prediction_intervals=prediction_intervals)

plt.title(f"Forecasts on Test Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth",'_nolegend_', "Median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Time (business days)")
plt.legend(legend, loc="upper left")

filename = dir_path + file_head + "2_testing" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()

column_counter  = {}
for col in columns:
    column_counter[col] = 0

#Simple or complex, limited or unlimited, what distr parameter, what direction, full gradient or not (and if full gradient, what trading algo)
perturbations = {'Type of Algorithm': f"{adv_example_type}_{direction}", 'perturbation size': epsilon,
                'features': column_counter}


########################################################################
######################  MAKING ADV SET   ##############################
########################################################################
for iter in range(0,max_iter): #number of times to iterate over testing set
    print(f"On iter {iter}")
    #Creating Adversarial Rolling set
    adv_list_dataset = ListDataset([{FieldName.TARGET: target,
                                 FieldName.START: start,
                                 FieldName.FEAT_DYNAMIC_REAL: FDR}
                                for (target, start, FDR) in zip(testing_target_values, start_date, adversarial_fdr)], freq="1B")
    adv_rolling_set = generate_rolling_dataset(
                                dataset=adv_list_dataset,
                                start_time=pd.Timestamp(df_target.index[train_length], freq="1B"),
                                end_time=pd.Timestamp(df_target.index[test_length + prediction_length], freq="1B"),
                                strategy=StepStrategy(prediction_length=prediction_length, step_size=step_size))
    for i, adv_dict in enumerate(adv_rolling_set): #iterating over the rolling dataset
        start_adv = [adv_dict['start']]
        target_adv = adv_dict['target'].reshape((1,-1))
        fdr_adv =  [adversarial_fdr[0][:,:test_length + prediction_length - (i*(step_size))]]
        interim_adv_set = ListDataset([{
            FieldName.TARGET: target, FieldName.START: start, FieldName.FEAT_DYNAMIC_REAL: FDR}
                                for (target, start, FDR) in zip(target_adv, start_adv, fdr_adv)], freq ="1B")
        mx.random.seed(seed)
        np.random.seed(seed)
        for test_entry, forecast in zip(interim_adv_set, predictor.predict(interim_adv_set)):
            for t in range(0,prediction_length):
                grad_filename = grad_str + f"_{t}.p"
                with open(grad_filename, 'rb') as f:
                    gradients = pickle.load(f)
                grad = gradients[parameter][t, 4:]
                if adv_example_type =='complex':
                    sorted_grads = np.argsort(np.abs(grad))
                if adv_example_type =='simple':
                    sorted_grads = np.argsort(grad)
                sorted_grads = sorted_grads[::-1]
                time_idx = test_length - (i * (step_size)) + t
                for tries in range(0,len(columns)):
                    idx = sorted_grads[tries]
                    before = adversarial_fdr[0][idx,time_idx]
                    if adv_example_type =='complex':
                        after = adversarial_fdr[0][idx,time_idx] * (1+ np.sign(grad[idx]) * direction * epsilon)
                    if adv_example_type =='simple':
                        after = adversarial_fdr[0][idx,time_idx] * (1+ direction * epsilon)
                    #print(f'before ={before} after = {after}, change = {(1+ np.sign(grad[idx]) * direction * epsilon)}')
                    if after >= max_values[columns[idx]] or after <= min_values[columns[idx]]:
                        if tries == len(columns) -1:
                            break
                        else:
                            continue
                            #print(f'went over or under at trie ={tries}, (after ={round(after,2)}, max = {round(max_values[columns[i]])} min = {round(min_values[columns[i]])})')
                    if after < max_values[columns[idx]] and after > min_values[columns[idx]]:
                        #print(f'going to update at tries = {tries}!')
                        adversarial_fdr[0][idx, time_idx] = after
                        perturbations['features'][columns[idx]] +=1
                        #The actual change occured, lets record it
                        break


#After that the gradient attack is done, create the final rolling dataset
adv_data = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: FDR}
    for (target, start, FDR) in zip(testing_target_values, start_date,adversarial_fdr)], freq ="1B")

dataset_rolled_adv = generate_rolling_dataset(
        dataset=adv_data,
        start_time=pd.Timestamp(df_target.index[validation_length], freq="1B"),
        end_time=pd.Timestamp(df_target.index[test_length+prediction_length], freq = "1B"),
        strategy=StepStrategy(prediction_length=prediction_length, step_size = step_size))

########################################################################
#####################  PLOTTING ADV TEST DATA  #############################
########################################################################
fig, ax = plt.subplots()
prediction_intervals=[50.0, 90.0]
for i,(train_dict,adv_train_dict) in enumerate(zip(dataset_rolled,dataset_rolled_adv)):
    start_ = [train_dict['start']]
    adv_start_ = [adv_train_dict['start']]

    interim_target = train_dict['target'].reshape((1,-1))
    interim_target_adv = adv_train_dict['target'].reshape((1,-1))

    fdr =  [plotting_test_feat_dynamic[0][:,:test_length + prediction_length - (i*(step_size))]]
    adv_fdr =  [adversarial_fdr[0][:,:test_length + prediction_length - (i*(step_size))]]

    train_data = ListDataset([{FieldName.TARGET: target,FieldName.START: start,FieldName.FEAT_DYNAMIC_REAL: FDR} for (target, start, FDR)
                              in zip(interim_target, start_, fdr)], freq ="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data, predictor.predict(train_data)):
        to_pandas(test_entry)[validation_length:].plot(linewidth=1, color='b', zorder=0, figsize=(13, 7),label='Ground Truth')
        if i!=0:
            forecast.plot(color='g', prediction_intervals=prediction_intervals)

    #plotting the adversarial one
    train_data_adv = ListDataset([{FieldName.TARGET: target, FieldName.START: start, FieldName.FEAT_DYNAMIC_REAL: FDR} for (target, start, FDR)
                                in zip(interim_target_adv, adv_start_, adv_fdr)], freq="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data_adv, predictor.predict(train_data_adv)):
        if i!=0:
            forecast.plot(color='r', prediction_intervals=prediction_intervals)


plt.title(f"Forecasts on Training Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth", '_nolegend_',"Regular Median prediction"] + [f"Regular {k}% prediction interval" for k in prediction_intervals][::-1]
legend_adv =["Adversarial Median prediction"] + [f" Adversarial {k}% prediction interval" for k in prediction_intervals][::-1]
legend += legend_adv
plt.xlabel("Time (business days)")
plt.legend(legend, loc="upper left")

filename = dir_path + file_head + "3_adv_testing" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()

########################################################################
#####################  PLOTTING TEST DATA (No CI)  #####################
########################################################################
fig, ax = plt.subplots()
prediction_intervals=[1.0, 5.0]
for i,(train_dict,adv_train_dict) in enumerate(zip(dataset_rolled,dataset_rolled_adv)):
    start_ = [train_dict['start']]
    adv_start_ = [adv_train_dict['start']]

    interim_target = train_dict['target'].reshape((1,-1))
    interim_target_adv = adv_train_dict['target'].reshape((1,-1))


    fdr =  [plotting_test_feat_dynamic[0][:,:test_length + prediction_length - (i*(step_size))]]
    adv_fdr =  [adversarial_fdr[0][:,:test_length + prediction_length - (i*(step_size))]]

    train_data = ListDataset([{FieldName.TARGET: target,FieldName.START: start,FieldName.FEAT_DYNAMIC_REAL: FDR} for (target, start, FDR)
                              in zip(interim_target, start_, fdr)], freq ="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data, predictor.predict(train_data)):
        to_pandas(test_entry)[validation_length:].plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
        if i!=0:
            forecast.plot(color='g', prediction_intervals=prediction_intervals, show_mean=True)

    #plotting the adversarial one
    train_data_adv = ListDataset([{FieldName.TARGET: target, FieldName.START: start, FieldName.FEAT_DYNAMIC_REAL: FDR} for (target, start, FDR)
                                in zip(interim_target_adv, adv_start_, adv_fdr)], freq="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data_adv, predictor.predict(train_data_adv)):
        if i!=0:
            forecast.plot(color='r', prediction_intervals=prediction_intervals, show_mean=True)


plt.title(f"Forecasts on Training Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth",'_nolegend_', "Regular Median prediction"] + [f"Regular {k}% prediction interval" for k in prediction_intervals][::-1]
legend_adv =["Adversarial Median prediction"] + [f" Adversarial {k}% prediction interval" for k in prediction_intervals][::-1]
legend += legend_adv
plt.xlabel("Time (business days)")
plt.legend(legend, loc="upper left")

filename = dir_path + file_head + "4_adv_testing_no_ci" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =400)
if args.show_figs:
    plt.show()
else:
    plt.clf()

########################################################################
#####################  metrics of TEST DATA  ###########################
########################################################################

mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
    dataset_rolled, predictor=predictor, num_samples=100)
forecasts = list(forecast_it)  # this is the pd series
tss = list(ts_it)  # this is the dataframe
mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
    dataset_rolled, predictor=predictor, num_samples=100)
testing_agg_metrics, _ = Evaluator(num_workers=0)(ts_it, forecast_it)


test_list_samples = np.array([[]])
testing_target_vals = np.array([])
testing_pred_vals = np.array([])
testing_dates = np.array([], dtype = np.datetime64)
for i,(target, forecast) in enumerate(zip(tss, forecasts)):
    #first get the dates. they are added in opposite order so thats why accuracy was wrong before.
    testing_dates = np.concatenate((target.index[-step_size:],testing_dates))
    #Now get the target values
    testing_target_vals = np.concatenate((target.values[-step_size:].reshape((-1,)),testing_target_vals))
    #Lastly, get the predictions
    testing_pred_vals = np.concatenate((np.mean(forecast.samples, axis =0).reshape((-1,)), testing_pred_vals))
    if i != 0:
        test_list_samples = np.concatenate([forecast.samples.reshape((100,prediction_length)),test_list_samples],axis =1)
    else:
        test_list_samples = forecast.samples.reshape((100,prediction_length))

testing_target_vals = testing_target_vals[prediction_length:]
testing_pred_vals = testing_pred_vals[prediction_length:]
testing_dates = testing_dates[prediction_length:]
test_forecasted_samples = test_list_samples[:,prediction_length:]

########################################################################
#####################  metrics of ADV DATA  ###########################
########################################################################
mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
    dataset_rolled_adv, predictor=predictor, num_samples=100)
forecasts = list(forecast_it)  # this is the pd series
tss = list(ts_it)  # this is the dataframe
mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
    dataset_rolled_adv, predictor=predictor, num_samples=100)
adv_testing_agg_metrics, _ = Evaluator(num_workers=0)(ts_it, forecast_it)



adv_test_list_samples = np.array([[]])

adv_testing_target_vals = np.array([])
adv_testing_pred_vals = np.array([])
adv_testing_dates = np.array([], dtype = np.datetime64)
for i,(target, forecast) in enumerate(zip(tss, forecasts)):
    #first get the dates. they are added in opposite order so thats why accuracy was wrong before.
    adv_testing_dates = np.concatenate((target.index[-step_size:],adv_testing_dates))
    #Now get the target values
    adv_testing_target_vals = np.concatenate((target.values[-step_size:].reshape((-1,)),adv_testing_target_vals))
    #Lastly, get the predictions
    adv_testing_pred_vals = np.concatenate((np.mean(forecast.samples, axis =0).reshape((-1,)), adv_testing_pred_vals))
    if i != 0:
        adv_test_list_samples = np.concatenate([forecast.samples.reshape((100,prediction_length)),adv_test_list_samples],axis =1)
    else:
        adv_test_list_samples = forecast.samples.reshape((100,prediction_length))


adv_testing_target_vals = adv_testing_target_vals[prediction_length:]
adv_testing_pred_vals = adv_testing_pred_vals[prediction_length:]
adv_testing_dates = adv_testing_dates[prediction_length:]
adv_test_forecasted_samples = adv_test_list_samples[:,prediction_length:]

########################################################################
#############  Plotting the perturbed dataset  #########################
########################################################################
#columns = ["Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score","positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count", "neutral_count","General_score"]
columns = ["Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score","positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count", "neutral_count","General_score"]
regular_fdr = pd.DataFrame(plotting_test_feat_dynamic[0][:,validation_length:].reshape((-1,len(columns))), columns = columns)
adversarial_fdr_plot = pd.DataFrame(adversarial_fdr[0][:,validation_length:].reshape((-1,len(columns))), columns = columns)
fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize = (10,10))

for i in range(0,len(columns)):
    regular_fdr[columns[i]].plot(ax=axes[i],sharex =axes[0])
    adversarial_fdr_plot[columns[i]].plot(ax=axes[i],sharex =axes[0],alpha = 0.75)
    axes[i].set_title(f"{columns[i]}", loc='left', fontsize = 9,alpha = 0.75)
    if i != 0:
        yticks = axes[i].yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
fig.subplots_adjust(hspace=0.2)
# finally we invoke the legend (that you probably would like to customize...)
fig.legend([f"Reg {columns[i]}", f"Adv {columns[i]}"])

filename = dir_path + file_head + "5_adv_features" + file_body + ".png"
#if args.save_figs:
plt.savefig(filename, dpi =300)
if args.show_figs:
    plt.show()
else:
    plt.clf()


########################################################################
########################################################################

t =training_agg_metrics
v =testing_agg_metrics
a = adv_testing_agg_metrics

########################################################################
#############  Binary accuracy shennagins  #############################
########################################################################

new_dates = [pd.Timestamp(training_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,training_dates.shape[0])]
open_vals_training = df_open[new_dates].to_numpy().reshape((-1,training_target_vals.shape[0])).reshape((-1,))
training_target_diff = (np.exp(np.array(training_target_vals))* open_vals_training) - open_vals_training
training_pred_diff = (np.exp(np.array(training_pred_vals))* open_vals_training) - open_vals_training


training_pred_diff_samples = np.multiply(np.exp(training_forecasted_samples),np.tile(open_vals_training,(100,1))) - np.tile(open_vals_training,(100,1))
training_pred_mean = np.mean(training_pred_diff_samples,axis = 0)
training_pred_std = np.std(training_pred_diff_samples,axis = 0)

new_dates = [pd.Timestamp(testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,testing_dates.shape[0])]
open_vals_testing = df_open[new_dates].to_numpy().reshape((-1,testing_target_vals.shape[0])).reshape((-1,))
testing_target_diff = (np.exp(np.array(testing_target_vals))* open_vals_testing) - open_vals_testing
testing_pred_diff = (np.exp(np.array(testing_pred_vals))* open_vals_testing) - open_vals_testing

testing_pred_diff_samples = np.multiply(np.exp(test_forecasted_samples),np.tile(open_vals_testing,(100,1))) - np.tile(open_vals_testing,(100,1))
testing_pred_mean = np.mean(testing_pred_diff_samples,axis = 0)
testing_pred_std = np.std(testing_pred_diff_samples,axis = 0)

new_dates = [pd.Timestamp(adv_testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,adv_testing_dates.shape[0])]
open_vals_adv_testing = df_open[new_dates].to_numpy().reshape((-1,adv_testing_target_vals.shape[0])).reshape((-1,))
adv_testing_target_diff = (np.exp(np.array(adv_testing_target_vals))* open_vals_adv_testing) - open_vals_adv_testing
adv_testing_pred_diff = (np.exp(np.array(adv_testing_pred_vals))* open_vals_adv_testing) - open_vals_adv_testing

adv_testing_pred_diff_samples = np.multiply(np.exp(adv_test_forecasted_samples),np.tile(open_vals_adv_testing,(100,1))) - np.tile(open_vals_adv_testing,(100,1))
adv_testing_pred_mean = np.mean(adv_testing_pred_diff_samples,axis = 0)
adv_testing_pred_std = np.std(adv_testing_pred_diff_samples,axis = 0)

train_bias_acc_up = (np.sum(training_target_diff >= 0, axis=0)) / training_target_diff.shape[0]
train_bias_acc_down =  (np.sum(training_target_diff < 0, axis=0)) / training_target_diff.shape[0]
train_acc = np.sum(np.sign(training_target_diff) == np.sign(training_pred_diff)) / training_target_diff.shape[0]

test_bias_acc_up = (np.sum(testing_target_diff >= 0, axis=0)) / testing_target_diff.shape[0]
test_bias_acc_down = (np.sum(testing_target_diff < 0, axis=0)) / testing_target_diff.shape[0]
test_acc = np.sum(np.sign(testing_target_diff) == np.sign(testing_pred_diff)) / testing_target_diff.shape[0]

adv_test_bias_acc_up = (np.sum(adv_testing_target_diff >= 0, axis=0)) / adv_testing_target_diff.shape[0]
adv_test_bias_acc_down = (np.sum(adv_testing_target_diff < 0, axis=0)) / adv_testing_target_diff.shape[0]
adv_test_acc = np.sum(np.sign(adv_testing_target_diff) == np.sign(adv_testing_pred_diff)) / adv_testing_target_diff.shape[0]


####################################################################
##################  Getting financial metrics ######################
####################################################################
train_open_zero = df_open[pd.Timestamp(training_dates[0]).strftime('%Y-%m-%d')]
train_close_end = df_close[pd.Timestamp(training_dates[-1]).strftime('%Y-%m-%d')]
test_open_zero = df_open[pd.Timestamp(testing_dates[0]).strftime('%Y-%m-%d')]
test_close_end = df_close[pd.Timestamp(testing_dates[-1]).strftime('%Y-%m-%d')]

#passive gain
passive_gain_train = ((((10000/train_open_zero)*train_close_end) -10000)/10000)*100
passive_gain_test = ((((10000/test_open_zero)*test_close_end) -10000)/10000)*100

#greedy gain
greedy_gain_train,train_invest, training_greedy_returns,training_greedy_PDT = greedy_gain(training_dates, training_pred_diff,training_target_diff, df_open,df_close,training_pred_mean,training_pred_std)
greedy_gain_test, test_invest, testing_greedy_returns,testing_greedy_PDT = greedy_gain(testing_dates, testing_pred_diff, testing_target_diff,df_open,df_close,testing_pred_mean,testing_pred_std)
adv_greedy_gain_test, adv_test_invest, adv_testing_greedy_returns,adv_testing_greedy_PDT = greedy_gain(adv_testing_dates, adv_testing_pred_diff, adv_testing_target_diff,df_open,df_close,adv_testing_pred_mean,adv_testing_pred_std)

#greedy gain_Kelly
kelly_greedy_gain_train,kelly_train_invest, kelly_training_greedy_returns,kelly_training_greedy_PDT,f_train = kelly_greedy_gain(training_dates, training_pred_diff,training_target_diff, df_open,df_close,training_pred_mean,training_pred_std)
kelly_greedy_gain_test, kelly_test_invest, kelly_testing_greedy_returns,kelly_testing_greedy_PDT,f_test= kelly_greedy_gain(testing_dates, testing_pred_diff, testing_target_diff,df_open,df_close,testing_pred_mean,testing_pred_std)
kelly_adv_greedy_gain_test, kelly_adv_test_invest, kelly_adv_testing_greedy_returns,kelly_adv_testing_greedy_PDT,f_adv = kelly_greedy_gain(adv_testing_dates, adv_testing_pred_diff, adv_testing_target_diff,df_open,df_close,adv_testing_pred_mean,adv_testing_pred_std)


#threshold gain
threshold_gain_train, training_threshold_returns,training_threshold_PDT = threshold_gain(training_dates, training_pred_diff,training_target_diff,df_open,df_close,training_pred_mean,training_pred_std)
threshold_gain_test,testing_threshold_returns,testing_threshold_PDT = threshold_gain(testing_dates, testing_pred_diff,testing_target_diff,df_open,df_close,testing_pred_mean,testing_pred_std)
adv_threshold_gain_test,adv_testing_threshold_returns,adv_testing_threshold_PDT = threshold_gain(adv_testing_dates, adv_testing_pred_diff,adv_testing_target_diff,df_open,df_close,adv_testing_pred_mean,adv_testing_pred_std)

#threshold gain kelly
kelly_threshold_gain_train, kelly_training_threshold_returns,kelly_training_threshold_PDT = kelly_threshold_gain(training_dates, training_pred_diff,training_target_diff,df_open,df_close,training_pred_mean,training_pred_std)
kelly_threshold_gain_test,kelly_testing_threshold_returns,kelly_testing_threshold_PDT = kelly_threshold_gain(testing_dates, testing_pred_diff,testing_target_diff,df_open,df_close,testing_pred_mean,testing_pred_std)
kelly_adv_threshold_gain_test,kelly_adv_testing_threshold_returns,kelly_adv_testing_threshold_PDT = kelly_threshold_gain(adv_testing_dates, adv_testing_pred_diff,adv_testing_target_diff,df_open,df_close,adv_testing_pred_mean,adv_testing_pred_std)


print(f"Training: RMSE: {round(t['RMSE'],3)} MAPE: {round(t['MAPE'],3)} CRPS: {round(t['mean_wQuantileLoss'],3)}"
      f" Only Up: {round(train_bias_acc_up,3)} Only Down: {round(train_bias_acc_down,3)} Accuracy: {round(train_acc,3)} \n"
      f"Initial Investment: {round(train_invest,2)}$ Passive Return: {round(passive_gain_train,1)}% Greedy Return: {round(greedy_gain_train,1)}%"
      f" Kelly Greedy Return: {round(kelly_greedy_gain_train,1)}% Threshold Return: {round(threshold_gain_train,1)}%"
      f" Kelly Threshold Return {round(kelly_threshold_gain_train,1)}% \n")
print(f"Testing: RMSE: {round(v['RMSE'],3)} MAPE: {round(v['MAPE'],3)} CRPS: {round(v['mean_wQuantileLoss'],3)}"
      f" Only Up: {round(test_bias_acc_up,3)} Only Down: {round(test_bias_acc_down,3)} Accuracy: {round(test_acc,3)} \n"
      f"Initial Investment: {round(test_invest,2)}$ Passive Return: {round(passive_gain_test,1)}% Greedy Return: {round(greedy_gain_test,1)}%"
      f" Kelly Greedy Return: {round(kelly_greedy_gain_test,1)}% Threshold Return: {round(threshold_gain_test,1)}%"
      f" Kelly Threshold Return {round(kelly_threshold_gain_test,1)}% \n")
print(f"Adversarial: RMSE: {round(a['RMSE'],3)} MAPE: {round(a['MAPE'],3)} CRPS: {round(a['mean_wQuantileLoss'],3)}"
      f" Only Up: {round(adv_test_bias_acc_up,3)} Only Down: {round(adv_test_bias_acc_down,3)} Accuracy: {round(adv_test_acc,3)} \n"
      f"Initial Investment: {round(adv_test_invest,2)}$ Passive Return: {round(passive_gain_test,1)}% Greedy Return: {round(adv_greedy_gain_test,1)}%"
      f" Kelly Greedy Return: {round(kelly_adv_greedy_gain_test,1)}% Threshold Return: {round(adv_threshold_gain_test,1)}%"
      f" Kelly Threshold Return {round(kelly_adv_threshold_gain_test,1)}% \n")

ret_dict ={}
training_metrics = {}
testing_metrics = {}
adv_testing_metrics = {}

for feat in ['RMSE','MAPE','mean_wQuantileLoss']:
    training_metrics[feat] = t[feat]
    testing_metrics[feat] = v[feat]
    adv_testing_metrics[feat] = a[feat]

training_metrics['Biased Acc'] = (train_bias_acc_up,train_bias_acc_down)
training_metrics['Pred Acc'] = train_acc
training_metrics['Passive Gain'] = passive_gain_train
training_metrics['Greedy Gain'] = greedy_gain_train
training_metrics['Threshold Gain'] = threshold_gain_train
training_metrics['Kelly Greedy Gain'] = kelly_greedy_gain_train
training_metrics['Kelly Threshold Gain'] = kelly_threshold_gain_train
training_metrics['initial investment'] = train_invest


testing_metrics['Biased Acc'] = (test_bias_acc_up, test_bias_acc_down)
testing_metrics['Pred Acc'] = test_acc
testing_metrics['Passive Gain'] = passive_gain_test
testing_metrics['Greedy Gain'] = greedy_gain_test
testing_metrics['Threshold Gain'] = threshold_gain_test
testing_metrics['Kelly Greedy Gain'] = kelly_greedy_gain_test
testing_metrics['Kelly Threshold Gain'] = kelly_threshold_gain_test
testing_metrics['initial investment'] = test_invest

adv_testing_metrics['Biased Acc'] = (adv_test_bias_acc_up, adv_test_bias_acc_down)
adv_testing_metrics['Pred Acc'] = adv_test_acc
adv_testing_metrics['Passive Gain'] = passive_gain_test
adv_testing_metrics['Greedy Gain'] = adv_greedy_gain_test
adv_testing_metrics['Threshold Gain'] = adv_threshold_gain_test
adv_testing_metrics['Kelly Greedy Gain'] = kelly_adv_greedy_gain_test
adv_testing_metrics['Kelly Threshold Gain'] = kelly_adv_threshold_gain_test
adv_testing_metrics['initial investment'] = adv_test_invest

ret_dict['training'] = training_metrics
ret_dict['testing'] = testing_metrics
ret_dict['adversarial'] = adv_testing_metrics
ret_dict['adv_algo_info'] = perturbations

returns_dict = {'Training':{'Greedy':training_greedy_returns,'Threshold':training_threshold_returns,'Kelly Greedy':kelly_training_greedy_returns,'Kelly Threshold':kelly_training_threshold_returns},
                'Testing': {'Greedy':testing_greedy_returns,'Threshold':testing_threshold_returns,'Kelly Greedy':kelly_testing_greedy_returns,'Kelly Threshold':kelly_testing_threshold_returns},
                'Adversarial': {'Greedy': adv_testing_greedy_returns, 'Threshold': adv_testing_threshold_returns,'Kelly Greedy':kelly_adv_testing_greedy_returns,'Kelly Threshold':kelly_adv_testing_threshold_returns}}

PDT_dict  = {'Training':{'Greedy':training_greedy_PDT,'Threshold':training_threshold_PDT,'Kelly Greedy':kelly_training_greedy_PDT,'Kelly Threshold':kelly_training_threshold_PDT},
                'Testing': {'Greedy':testing_greedy_PDT,'Threshold':testing_threshold_PDT,'Kelly Greedy':kelly_testing_greedy_PDT,'Kelly Threshold':kelly_testing_threshold_PDT},
                'Adversarial': {'Greedy':adv_testing_greedy_PDT,'Threshold':adv_testing_threshold_PDT,'Kelly Greedy':kelly_adv_testing_greedy_PDT,'Kelly Threshold':kelly_adv_testing_threshold_PDT}}
f_dict = {'Training':f_train, 'Testing':f_test,'Adversarial':f_adv}


ret_dict['returns dict'] = returns_dict
ret_dict['pdt'] = PDT_dict
ret_dict['f_dict'] =f_dict

filename = dir_path + file_head + "6_return_distributions" + file_body + ".png"
plot_return_distributions(returns_dict,PDT_dict,f_dict,save_figs = args.save_figs,show_figs=args.show_figs,save_path = filename)

filename = metric_dir_path + file_head + "adv_testing" + file_body + ".p"
if args.save_figs:
    pickle.dump(ret_dict, open(filename, 'wb'))

########################################################################
################## Plotting the open data  #############################
########################################################################
vt = testing_target_vals
vp = testing_pred_vals
adv_vp = adv_testing_pred_vals

new_dates = [pd.Timestamp(testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,testing_dates.shape[0])]
new_dates_plot = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in testing_dates]
open_vals = df_open[new_dates].to_numpy().reshape((-1,vt.shape[0])).reshape((-1,))
close_vp = np.exp(vp) * open_vals
close_vt  =np.exp(vt) * open_vals
close_adv_vp  =np.exp(adv_vp) * open_vals
fig, ax = plt.subplots(figsize = (12,8))
plt.plot_date(new_dates_plot, close_vt, fmt ='-b')
plt.plot_date(new_dates_plot, close_vp, fmt = '-g')
plt.plot_date(new_dates_plot, close_adv_vp, fmt = '-r')
plt.legend(['Ground Truth Price', 'Predicted Price', 'Manipulated Predicted Price'])
plt.title('Close Price for Test set')
plt.xlabel('Time (business days)')
plt.ylabel('Close Price')


ax.xaxis.set_major_locator(mdates.DayLocator(interval=21))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate()
filename = filename = dir_path + file_head + "7_close_price" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()


######################## stupid test ######################################

true = training_target_vals
pred = training_pred_vals


new_dates = [pd.Timestamp(training_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,training_dates.shape[0])]
new_dates_plot = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in training_dates]
open_vals = df_open[new_dates].to_numpy().reshape((-1,true.shape[0])).reshape((-1,))
close_true = np.exp(true) * open_vals
close_pred  =np.exp(pred) * open_vals

fig, ax = plt.subplots(figsize = (12,8))
plt.plot_date(new_dates_plot, close_true, fmt ='-b')
plt.plot_date(new_dates_plot, close_pred, fmt = '-g')
plt.legend(['Ground Truth Price', 'Predicted Price'])
plt.title('Close Price for Training set')
plt.xlabel('Time (business days)')
plt.ylabel('Close Price')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=21))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate()
filename = filename = dir_path + file_head + "8_training_close_price" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()


