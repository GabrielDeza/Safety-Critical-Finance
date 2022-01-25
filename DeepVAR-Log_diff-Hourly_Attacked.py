#1. Imports
import pandas as pd
import numpy as np
import mxnet as mx
import matplotlib.dates as mdates
import pickle
import datetime
import os
from random import choice
from string import ascii_uppercase
import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator, MultivariateEvaluator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.gpvar import GPVAREstimator
from gluonts.mx.trainer import Trainer
import argparse
from gluonts.dataset.rolling_dataset import StepStrategy, generate_rolling_dataset
import seaborn as sns
from fixed_helper_financial_metrics import *
sns.set(style="darkgrid")

#2. Arguments + Hyper Parameters
##############   PARAMETERS & HYPERPARAMETERS  #########################
########################################################################
parser = argparse.ArgumentParser(description='adversarial DeepVAR forecasts')
parser.add_argument('--company', default='ADSK', type=str, help='Company')
parser.add_argument('--train_length', default=200, type=int, help='Length of training set')
parser.add_argument('--validation_length', default=430, type=int, help='Length of validation set')
parser.add_argument('--test_length', default=530, type=int, help='Length of testing set')
parser.add_argument('--prediction_length', default=5, type=int, help='Prediction Length')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=30, type=int, help='Batch size')
parser.add_argument('--nbpe', default=150, type=int, help='Number of batches per epoch')
parser.add_argument('--save_figs', default=False, type = bool, help = 'decides if the figures and txt files should be saved or not')
parser.add_argument('--show_figs', default=True, type=bool, help='decides if the figures should be shown to the screen')
parser.add_argument('--seed', default=6, type=int, help='seed')
parser.add_argument('--adv_dir', default = +1, type= int, help ="Direction of parameter to modify (-1 or +1)")
parser.add_argument('--epsilon', default=0.05, type=float, help='Percent change in dataset at each iteration')
parser.add_argument('--max_iter', default=2, type=int, help='number of iterations on the adv dataset algorithm')
parser.add_argument('--parameter', default='mu', type=str, help='parameter we want to change. its mu sigma nu for student-t',)


args = parser.parse_args()
plottype ="log_diff"
company = args.company
train_length = args.train_length
validation_length = args.validation_length
test_length = args.test_length
prediction_length = args.prediction_length
step_size = prediction_length
epochs = args.epochs
batch_size = args.batch_size
num_batches_per_epoch = args.nbpe
#adv example generation:
direction = args.adv_dir
epsilon = args.epsilon#in percentage
parameter = args.parameter
max_iter = args.max_iter

#3. Setting the seed for randomness. Important for when you actually get forecasts or metrics for plotting or the gradient attack.
seed = args.seed
mx.random.seed(seed)
np.random.seed(seed)

#3. b) some file stuff

if not os.path.isdir('plots'):
    os.mkdir('plots')
plot_path = f"./plots/"

if not os.path.isdir('metrics'):
    os.mkdir('metrics')
metric_path = f"./metrics/"

if not os.path.isdir(f'plots/{company}'):
    os.mkdir(f'plots/{company}')
dir_path = f"./plots/{company}/"

if not os.path.isdir(f'metrics/{company}'):
    os.mkdir(f'metrics/{company}')
metric_dir_path = f"./metrics/{company}/"


file_head = f"{company}_DeepVAR_Hourly_"
file_body = f"_prediction_length={prediction_length}_BS={batch_size}_NBpE={num_batches_per_epoch}_epochs={epochs}" \
               f"_parameter={parameter}_dir={direction}_eps={epsilon}_max_iter={max_iter}"

#4. Reading in the dataset
columns = [plottype,"Open","Close","Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score",
           "positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count",
           "neutral_count","General_score"]

filename = f"/Users/gabriel/Desktop/Fixed_Stock_Project/input_data/{company}_1h.csv"

df =  pd.read_csv(filename, index_col=1)
df = df.drop(['Unnamed: 0'],axis = 1)
df=  df.dropna()
df = df.reset_index()
df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
start_date = datetime.datetime(year=2016, month=1, day=1, hour=0, minute=0, second=0)
dates = []
for i in range(df.shape[0]):
    dates.append(start_date + datetime.timedelta(hours=i))
df['Date'] = dates
df = df.set_index(['Date'])
df_close = df['Close']
df_open = df['Open']
df['log_diff'] = np.log((df['Close']/df['Open']))
max_values = df.max()
min_values = df.min()
N = len(columns)
T = df.shape[0]

#5. Convert from a pandas dataframe to a numpy matrix. Had to do it iteratively because the one line solution was scaling my data for some reason.
target = np.zeros((N, T))
for i,col in enumerate(columns):
    target[i,:] = df[col][:].to_numpy()
#5b: make adv copy of target

adv_target = np.zeros((N, T))
for i,col in enumerate(columns):
    adv_target[i,:] = df[col][:].to_numpy()

#6. Creating the Training dataset for training the model. Has several steps

#6 a) making the univariate version of the dataset (training and validation):
time_series_dicts = []
for time_series in target:
    time_series_dicts.append({"target": time_series[:train_length], "start": df.index[0]}) #N items, each have dimensions (train_length,)
train_dataset = ListDataset(time_series_dicts, freq="H")

val_time_series_dicts = []
for time_series in target:
    val_time_series_dicts.append({"target": time_series[:validation_length], "start": df.index[0]}) #N items, each have dimensions (train_length,)
validation_dataset = ListDataset(val_time_series_dicts, freq="H")

#6 b) Intialize some multivariate groupers:
grouper_train = MultivariateGrouper(max_target_dim=N)
grouper_validation = MultivariateGrouper(max_target_dim=N)
grouper_interim_train = MultivariateGrouper(max_target_dim=N, num_test_dates=1) #the "num_test_dates=1" is so important it's ridiclious... 5 hours down the drain :(

#6 c) #Create the multivariate dataset
train_ds = grouper_train(train_dataset) #1 item, dimension is (N,train_length)
validation_ds = grouper_validation(validation_dataset)

#7 Define the model of interest

estimator = DeepVAREstimator(target_dim=N,
                             prediction_length=prediction_length,
                             freq="H",
                             trainer=Trainer(epochs=epochs,num_batches_per_epoch=num_batches_per_epoch,learning_rate=1e-3,),)

#8 Train the model with the multivariate dataset
grad_str = ''.join(choice(ascii_uppercase) for i in range(200))
grad_str = f"./gradient_dir/" +grad_str
predictor = estimator.train(training_data = train_ds,validation_data = validation_ds, num_workers = 0,grad_str = grad_str)

#9 Create a rolling version of the training dataset. Notice we pass in the univariate version because generate_rolling_dataset does not currently support multivariate version
training_dataset_rolled = generate_rolling_dataset(dataset = train_dataset,
                                                   start_time=pd.Timestamp(df.index[0], freq="H"),
                                                   end_time=pd.Timestamp(df.index[train_length],freq ="H"),
                                                   strategy=StepStrategy(prediction_length=prediction_length, step_size = prediction_length))

#10 Since in 9 we pass in the univariate version, we have to fix that
training_sets = [[] for i in range(train_length//prediction_length)] #ie: how many rolling sets we have
for i,train_dict in enumerate(training_dataset_rolled):
    training_sets[i%(train_length//prediction_length)].append({"target": train_dict['target'], "start": pd.Timestamp(df.index[0], freq="H")})

#11 We can now plot the rolling version of the training dataset:
prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()
for j,ts in enumerate(training_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
            to_pandas(ts[0]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
            if j != 0: #skip the last forecast because it's not in the training set
                forecast.copy_dim(0).plot(color='m', prediction_intervals=prediction_intervals)
plt.grid(which="both")
plt.title(f"Forecasts on Training Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth",'_no_legend_', "Median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Time (business hours)")
plt.legend(legend, loc="upper left")
filename = dir_path + file_head + "1_training" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()

#12 Now we shall get the metrics of the rolling dataset. Similiarily, cause 9 is with the univariate dataset,
# we will take the metric for each individual rolling set and then we will average each of those metrics over the number of rolling periods.

#Btw, ind_metrics is a (N,11 + 2* number of quantiles) dataframe.
quantiles = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
rolling_training_stats = {'RMSE':0,'CRPS':0,'MAPE':0}
for j,ts in enumerate(training_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        agg_metrics, interim_ind_metrics = backtest_metrics(test_dataset=interim_ds,predictor=predictor,evaluator=MultivariateEvaluator(quantiles=quantiles))
        rolling_training_stats['RMSE'] += np.sqrt(interim_ind_metrics['MSE'].values[0])
        rolling_training_stats['MAPE'] += interim_ind_metrics['MAPE'].values[0]
        crps = []
        for q in quantiles:
            crps.append(interim_ind_metrics[f'QuantileLoss[{q}]'].values[0]/interim_ind_metrics['abs_target_sum'].values[0])
        rolling_training_stats['CRPS'] += sum(crps)/len(crps)
rolling_training_stats['RMSE'] = rolling_training_stats['RMSE']/len(training_sets)
rolling_training_stats['CRPS'] = rolling_training_stats['CRPS']/len(training_sets)
rolling_training_stats['MAPE'] = rolling_training_stats['MAPE']/len(training_sets)


#13 Now we will get the actual point forecasts from monte carlo sampling as well as the corresponding target log returns.
# This will be used for computation of binary accuracy and future plotting

training_target_vals = np.array([])
training_pred_vals = np.array([])
training_dates = np.array([], dtype = np.datetime64)
for j,ts in enumerate(training_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecast_it, ts_it = make_evaluation_predictions(interim_ds, predictor=predictor, num_samples=100)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecasts = list(forecast_it)
        tss = list(ts_it)
        for i, (ground_truth, forecast) in enumerate(zip(tss, forecasts)):
            training_dates = np.concatenate((ground_truth.index[-prediction_length:], training_dates))
            training_target_vals = np.concatenate((ground_truth.values[-prediction_length:, 0].reshape((-1,)), training_target_vals))
            training_pred_vals = np.concatenate((np.mean(forecast.samples, axis=0)[:, 0].reshape((-1,)), training_pred_vals))

#for the predictions, remove the first 'prediction_length' (more like context length but they are equal)
correct_training_target_vals = training_target_vals[prediction_length:]
correct_training_pred_vals = training_pred_vals[prediction_length:]
correct_training_dates = training_dates[prediction_length:]


#14 we will now make the testing set

#14 a) Make the univariate dataset
time_series_dicts = []
for time_series in target:
    time_series_dicts.append({"target": time_series[validation_length:test_length], "start": df.index[validation_length]}) #N items, each are of dimension (test_length - train_length,)
test_dataset = ListDataset(time_series_dicts, freq="H")

#14 b) initialize multivariate groupers:

grouper_test = MultivariateGrouper(max_target_dim=N)
grouper_interim_test = MultivariateGrouper(max_target_dim=N, num_test_dates=1)

# 14c) create the multivariate version of the dataset
test_ds = grouper_test(test_dataset) #1 item, dimension is (N,test_length -validation_length)

# 14 d) create the rolling dataset but from the univariate version
testing_dataset_rolled = generate_rolling_dataset(dataset = test_dataset,
                                          start_time = pd.Timestamp(df.index[validation_length], freq="H"),
                                          end_time = pd.Timestamp(df.index[test_length], freq="H"),
                                            strategy=StepStrategy(prediction_length=prediction_length, step_size = prediction_length))

#15 Similiar to step 10, we will hand make the multivariate version of the rolling testing set

testing_sets = [[] for i in range((test_length-validation_length)//prediction_length)] #ie: how many rolling sets we have
for i,train_dict in enumerate(testing_dataset_rolled):
    testing_sets[i%((test_length-validation_length)//prediction_length)].append({"target": train_dict['target'], "start": pd.Timestamp(df.index[validation_length], freq="H")})

#16 Let's plot the testing set

prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
            to_pandas(ts[0]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
            if j != 0:
                forecast.copy_dim(0).plot(color='g', prediction_intervals=prediction_intervals)
plt.grid(which="both")
plt.title(f"Forecasts on Test Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth",'_no_legend_', "Median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Time (business hours)")
plt.legend(legend, loc="upper left")
filename = dir_path + file_head + "2_testing" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()

###############################################################################################
###################################### Make Adv dataset #######################################
#For loop
for iter in range(0,max_iter): #number of times to iterate over testing set
    print(f"On iter {iter}")
    adv_time_series_dicts = []
    for time_series in adv_target:
        adv_time_series_dicts.append({"target": time_series[validation_length:test_length], "start": df.index[
            validation_length]})  # N items, each are of dimension (test_length - train_length,)
    interim_adv_test_dataset = ListDataset(adv_time_series_dicts, freq="H")
    adv_testing_dataset_rolled = generate_rolling_dataset(dataset=interim_adv_test_dataset,start_time=pd.Timestamp(df.index[validation_length], freq="H"),
                                                      end_time=pd.Timestamp(df.index[test_length], freq="H"),strategy=StepStrategy(prediction_length=prediction_length,step_size=prediction_length))

    interim_adv_testing_sets = [[] for i in range((test_length - validation_length) // prediction_length)]  # ie: how many rolling sets we have
    for qq, train_dict in enumerate(adv_testing_dataset_rolled):
        interim_adv_testing_sets[qq % ((test_length - validation_length) // prediction_length)].append({"target": train_dict['target'], "start": pd.Timestamp(df.index[validation_length], freq="H")})
    for i,ts in enumerate(interim_adv_testing_sets):
        interim_dict = ListDataset(ts, freq="H")  # N times (1,truncated length)
        interim_adv_ds = grouper_interim_train(interim_dict)  # is (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        for dd, (test_entry, forecast) in enumerate(zip(interim_adv_ds, predictor.predict(interim_adv_ds))):
            for t in range(0,prediction_length):
                grad_filename = grad_str + f"_{t}.p"
                with open(grad_filename, 'rb') as f:
                    gradients = pickle.load(f)
                grad = gradients[parameter][t+2, 3:]
                sorted_grads = np.argsort(np.abs(grad))
                sorted_grads = sorted_grads[::-1]
                time_idx = test_length - (i * (step_size)) + t
                for tries in range(0,len(columns)-3):
                    idx = sorted_grads[tries] + 3
                    before = adv_target[idx,time_idx]
                    after = adv_target[idx,time_idx] * (1+ np.sign(grad[idx-3]) * direction * epsilon)
                    if after >= max_values[columns[idx]] or after <= min_values[columns[idx]]:
                        if tries == len(columns) -1:
                            break
                        else:
                            continue
                    if after < max_values[columns[idx]] and after > min_values[columns[idx]]:
                        adv_target[idx, time_idx] = after
                        break

adv_time_series_dicts = []
for time_series in adv_target:
    adv_time_series_dicts.append({"target": time_series[validation_length:test_length], "start": df.index[validation_length]})  # N items, each are of dimension (test_length - train_length,)

adv_test_dataset = ListDataset(adv_time_series_dicts, freq="H")
grouper_adv_test = MultivariateGrouper(max_target_dim=N)
adv_test_ds = grouper_adv_test(adv_test_dataset)

#Step 4:
adv_testing_dataset_rolled = generate_rolling_dataset(dataset = adv_test_dataset,
                                                      start_time = pd.Timestamp(df.index[validation_length], freq="H"),
                                                      end_time = pd.Timestamp(df.index[test_length], freq="H"),
                                                      strategy=StepStrategy(prediction_length=prediction_length,
                                                      step_size = prediction_length))
#Step 5:
adv_testing_sets = [[] for i in range((test_length-validation_length)//prediction_length)] #ie: how many rolling sets we have
for i,train_dict in enumerate(adv_testing_dataset_rolled):
    adv_testing_sets[i%((test_length-validation_length)//prediction_length)].append({"target": train_dict['target'], "start": pd.Timestamp(df.index[validation_length], freq="H")})

#Super simple, plot both adv and reg sets:

prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()
for j,(ts,adv_ts) in enumerate(zip(testing_sets,adv_testing_sets)): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)

        interim_adv_dict = ListDataset(adv_ts, freq="H")  # N times (1,truncated length)
        interim_adv_ds = grouper_interim_train(interim_adv_dict)  # is (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
            to_pandas(ts[0]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
            if j != 0:
                forecast.copy_dim(0).plot(color='g', prediction_intervals=prediction_intervals)
        mx.random.seed(seed)
        np.random.seed(seed)
        for i, (test_entry, forecast) in enumerate(zip(interim_adv_ds, predictor.predict(interim_adv_ds))):
            if j != 0:
                forecast.copy_dim(0).plot(color='r', prediction_intervals=prediction_intervals)

plt.title(f"Forecasts on Training Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth",'_no_legend_', "Regular Median prediction"] + [f"Regular {k}% prediction interval" for k in prediction_intervals][::-1]
legend_adv =["Adversarial Median prediction"] + [f" Adversarial {k}% prediction interval" for k in prediction_intervals][::-1]
legend += legend_adv
plt.xlabel("Time (business hours)")
plt.legend(legend, loc="upper left")
filename = dir_path + file_head + "3_adv_testing" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()

#plot with no CI:
prediction_intervals = [1.0, 5.0]
fig, ax = plt.subplots()
for j, (ts, adv_ts) in enumerate(zip(testing_sets, adv_testing_sets)):  # iterating over the rolling sets
    interim_dict = ListDataset(ts, freq="H")  # N times (1,truncated length)
    interim_ds = grouper_interim_train(interim_dict)  # is (N,truncated length)

    interim_adv_dict = ListDataset(adv_ts, freq="H")  # N times (1,truncated length)
    interim_adv_ds = grouper_interim_train(interim_adv_dict)  # is (N,truncated length)
    mx.random.seed(seed)
    np.random.seed(seed)
    for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
        to_pandas(ts[0]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
        if j != 0:
            forecast.copy_dim(0).plot(color='g', prediction_intervals=prediction_intervals)
    mx.random.seed(seed)
    np.random.seed(seed)
    for i, (test_entry, forecast) in enumerate(zip(interim_adv_ds, predictor.predict(interim_adv_ds))):
        if j != 0:
            forecast.copy_dim(0).plot(color='r', prediction_intervals=prediction_intervals)

plt.title(f"Forecasts on Training Set")
plt.ylabel("Log difference in Open and Close Price")
legend = ["Ground Truth",'_no_legend_', "Regular Median prediction"] + [f"Regular {k}% prediction interval" for k in
                                                          prediction_intervals][::-1]
legend_adv = ["Adversarial Median prediction"] + [f" Adversarial {k}% prediction interval" for k in
                                                  prediction_intervals][::-1]
legend += legend_adv
plt.xlabel("Time (business hours)")
plt.legend(legend, loc="upper left")
filename = dir_path + file_head + "4_adv_testing_no_ci" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi=100)
if args.show_figs:
    plt.show()
else:
    plt.clf()


#20 Let's now calculate the metrics over the rolling test set. For similiar reasons as the training set, we do the following:
quantiles=(0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9)
rolling_testing_stats = {'RMSE':0,'MAPE':0,'CRPS':0}
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_test(interim_dict) # (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        agg_metrics,interim_ind_metrics = backtest_metrics(test_dataset=interim_ds,predictor=predictor,evaluator=MultivariateEvaluator(quantiles=quantiles))
        rolling_testing_stats['RMSE'] += np.sqrt(interim_ind_metrics['MSE'].values[0])
        rolling_testing_stats['MAPE'] += interim_ind_metrics['MAPE'].values[0]
        crps = []
        for q in quantiles:
            crps.append(interim_ind_metrics[f'QuantileLoss[{q}]'].values[0] / interim_ind_metrics['abs_target_sum'].values[0])
        rolling_testing_stats['CRPS'] += sum(crps) / len(crps)
rolling_testing_stats['RMSE'] = rolling_testing_stats['RMSE']/len(testing_sets)
rolling_testing_stats['CRPS'] = rolling_testing_stats['CRPS']/len(testing_sets)
rolling_testing_stats['MAPE'] = rolling_testing_stats['MAPE']/len(testing_sets)

#21 Calculate actual targets and predictions:
testing_target_vals = np.array([])
testing_pred_vals = np.array([])
testing_dates = np.array([], dtype = np.datetime64)
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_test(interim_dict) # (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecast_it, ts_it = make_evaluation_predictions(interim_ds, predictor=predictor, num_samples=100)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecasts = list(forecast_it)
        tss = list(ts_it)
        for i, (target, forecast) in enumerate(zip(tss, forecasts)):
            testing_dates = np.concatenate((target.index[-5:], testing_dates))
            testing_target_vals = np.concatenate((target.values[-5:, 0].reshape((-1,)), testing_target_vals))
            testing_pred_vals = np.concatenate((np.mean(forecast.samples, axis=0)[:, 0].reshape((-1,)), testing_pred_vals))

correct_testing_target_vals = testing_target_vals[5:]
correct_testing_pred_vals = testing_pred_vals[5:]
correct_testing_dates = testing_dates[5:]

#do the same for the adversarial set:

quantiles=(0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9)
adv_rolling_testing_stats = {'RMSE':0,'MAPE':0,'CRPS':0}
for j,ts in enumerate(adv_testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_test(interim_dict) # (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        agg_metrics,interim_ind_metrics = backtest_metrics(test_dataset=interim_ds,predictor=predictor,evaluator=MultivariateEvaluator(quantiles=quantiles))
        adv_rolling_testing_stats['RMSE'] += np.sqrt(interim_ind_metrics['MSE'].values[0])
        adv_rolling_testing_stats['MAPE'] += interim_ind_metrics['MAPE'].values[0]
        crps = []
        for q in quantiles:
            crps.append(interim_ind_metrics[f'QuantileLoss[{q}]'].values[0] / interim_ind_metrics['abs_target_sum'].values[0])
        adv_rolling_testing_stats['CRPS'] += sum(crps) / len(crps)
adv_rolling_testing_stats['RMSE'] = adv_rolling_testing_stats['RMSE']/len(adv_testing_sets)
adv_rolling_testing_stats['CRPS'] = adv_rolling_testing_stats['CRPS']/len(adv_testing_sets)
adv_rolling_testing_stats['MAPE'] = adv_rolling_testing_stats['MAPE']/len(adv_testing_sets)

#21 Calculate actual targets and predictions:
adv_testing_target_vals = np.array([])
adv_testing_pred_vals = np.array([])
adv_testing_dates = np.array([], dtype = np.datetime64)
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="H") #N times (1,truncated length)
        interim_ds = grouper_interim_test(interim_dict) # (N,truncated length)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecast_it, ts_it = make_evaluation_predictions(interim_ds, predictor=predictor, num_samples=100)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecasts = list(forecast_it)
        tss = list(ts_it)
        for i, (target, forecast) in enumerate(zip(tss, forecasts)):
            adv_testing_dates = np.concatenate((target.index[-5:], adv_testing_dates))
            adv_testing_target_vals = np.concatenate((target.values[-5:, 0].reshape((-1,)), adv_testing_target_vals))
            adv_testing_pred_vals = np.concatenate((np.mean(forecast.samples, axis=0)[:, 0].reshape((-1,)), adv_testing_pred_vals))

adv_correct_testing_target_vals = adv_testing_target_vals[5:]
adv_correct_testing_pred_vals = adv_testing_pred_vals[5:]
adv_correct_testing_dates = adv_testing_dates[5:]

########################################################################
#############  Plotting the perturbed dataset  #########################
########################################################################
columns_plots = ["Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score","positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count", "neutral_count","General_score"]
target = np.zeros((len(columns_plots), test_length-validation_length))
for i,col in enumerate(columns_plots):
    target[i,:] = df[col][validation_length:test_length].to_numpy()
regular_fdr = pd.DataFrame(np.transpose(target), columns = columns_plots)
adversarial_fdr_plot = pd.DataFrame(np.transpose(adv_target[3:,validation_length:test_length]), columns = columns_plots)

fig, axes = plt.subplots(nrows=len(columns_plots), ncols=1, figsize = (10,10))
for i in range(0,len(columns_plots)):
    regular_fdr[columns_plots[i]].plot(ax=axes[i],sharex =axes[0],alpha =0.75)
    adversarial_fdr_plot[columns_plots[i]].plot(ax=axes[i],sharex =axes[0],alpha =0.75)
    axes[i].set_title(f"{columns_plots[i]}", loc='left', fontsize = 9)
    if i != 0:
        yticks = axes[i].yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
fig.subplots_adjust(hspace=0.2)
# finally we invoke the legend (that you probably would like to customize...)
fig.legend([f"Reg {columns_plots[i]}", f"Adv {columns_plots[i]}"])

filename = dir_path + file_head + "5_adv_features" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()


#22 We are now at the stage that we want to calculate additional metrics.

#22) a) Binary Accuracy:
#Depending if its the log difference or the difference, we have to undo the log
if plottype == 'log_diff':
    new_dates = [pd.Timestamp(correct_training_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d %H:%M:%S') for i in range(0, correct_training_dates.shape[0])]
    open_vals_training = df_open[new_dates].to_numpy().reshape((-1, correct_training_target_vals.shape[0])).reshape((-1,))
    training_target_diff = (np.exp(np.array(correct_training_target_vals)) * open_vals_training) - open_vals_training
    training_pred_diff = (np.exp(np.array(correct_training_pred_vals)) * open_vals_training) - open_vals_training

    new_dates = [pd.Timestamp(correct_testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d %H:%M:%S') for i in range(0, correct_testing_dates.shape[0])]
    open_vals_testing = df_open[new_dates].to_numpy().reshape((-1, correct_testing_target_vals.shape[0])).reshape((-1,))
    testing_target_diff = (np.exp(np.array(correct_testing_target_vals)) * open_vals_testing) - open_vals_testing
    testing_pred_diff = (np.exp(np.array(correct_testing_pred_vals)) * open_vals_testing) - open_vals_testing

    new_dates = [pd.Timestamp(adv_correct_testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d %H:%M:%S') for i in range(0, adv_correct_testing_dates.shape[0])]
    adv_open_vals_testing = df_open[new_dates].to_numpy().reshape((-1, adv_correct_testing_target_vals.shape[0])).reshape((-1,))
    adv_testing_target_diff = (np.exp(np.array(adv_correct_testing_target_vals)) * adv_open_vals_testing) - adv_open_vals_testing
    adv_testing_pred_diff = (np.exp(np.array(adv_correct_testing_pred_vals)) * adv_open_vals_testing) - adv_open_vals_testing


train_bias_acc_up = (np.sum(training_target_diff >= 0, axis=0)) / training_target_diff.shape[0]
train_bias_acc_down =  (np.sum(training_target_diff < 0, axis=0)) / training_target_diff.shape[0]
train_acc = np.sum(np.sign(training_target_diff) == np.sign(training_pred_diff)) / training_target_diff.shape[0]

test_bias_acc_up = (np.sum(testing_target_diff >= 0, axis=0)) / testing_target_diff.shape[0]
test_bias_acc_down = (np.sum(testing_target_diff < 0, axis=0)) / testing_target_diff.shape[0]
test_acc = np.sum(np.sign(testing_target_diff) == np.sign(testing_pred_diff)) / testing_target_diff.shape[0]

adv_test_bias_acc_up = (np.sum(adv_testing_target_diff >= 0, axis=0)) / adv_testing_target_diff.shape[0]
adv_test_bias_acc_down = (np.sum(adv_testing_target_diff < 0, axis=0)) / adv_testing_target_diff.shape[0]
adv_test_acc = np.sum(np.sign(adv_testing_target_diff) == np.sign(adv_testing_pred_diff)) / adv_testing_target_diff.shape[0]

# 22 b) Financial Metrics
train_open_zero = df_open[pd.Timestamp(correct_training_dates[0]).strftime('%Y-%m-%d %H:%M:%S')]
train_close_end = df_close[pd.Timestamp(correct_training_dates[-1]).strftime('%Y-%m-%d %H:%M:%S')]
test_open_zero = df_open[pd.Timestamp(correct_testing_dates[0]).strftime('%Y-%m-%d %H:%M:%S')]
test_close_end = df_close[pd.Timestamp(correct_testing_dates[-1]).strftime('%Y-%m-%d %H:%M:%S')]

#passive gain
passive_gain_train = ((((10000/train_open_zero)*train_close_end) -10000)/10000)*100
passive_gain_test = ((((10000/test_open_zero)*test_close_end) -10000)/10000)*100
#greedy gain
greedy_gain_train,train_invest,training_greedy_returns,training_greedy_PDT = greedy_gain(correct_training_dates, training_pred_diff,training_target_diff,df_open,df_close,freq_type='hour')
greedy_gain_test, test_invest,testing_greedy_returns,testing_greedy_PDT= greedy_gain(correct_testing_dates, testing_pred_diff, testing_target_diff,df_open,df_close,freq_type='hour')
adv_greedy_gain_test, adv_test_invest, adv_testing_greedy_returns,adv_testing_greedy_PDT = greedy_gain(adv_correct_testing_dates, adv_testing_pred_diff, adv_testing_target_diff,df_open,df_close,freq_type='hour')

#greedy gain_Kelly
kelly_greedy_gain_train,kelly_train_invest, kelly_training_greedy_returns,kelly_training_greedy_PDT,f_train = kelly_greedy_gain(correct_training_dates, training_pred_diff,training_target_diff, df_open,df_close,freq_type='hour')
kelly_greedy_gain_test, kelly_test_invest, kelly_testing_greedy_returns,kelly_testing_greedy_PDT,f_test= kelly_greedy_gain(correct_testing_dates, testing_pred_diff, testing_target_diff,df_open,df_close,freq_type='hour')
kelly_adv_greedy_gain_test, kelly_adv_test_invest, kelly_adv_testing_greedy_returns,kelly_adv_testing_greedy_PDT,f_adv = kelly_greedy_gain(adv_correct_testing_dates, adv_testing_pred_diff, adv_testing_target_diff,df_open,df_close,freq_type='hour')


#threshold gain
threshold_gain_train,training_threshold_returns,training_threshold_PDT = threshold_gain(correct_training_dates, training_pred_diff,training_target_diff,df_open,df_close,freq_type='hour')
threshold_gain_test,testing_threshold_returns,testing_threshold_PDT = threshold_gain(correct_testing_dates, testing_pred_diff,testing_target_diff,df_open,df_close,freq_type='hour')
adv_threshold_gain_test,adv_testing_threshold_returns,adv_testing_threshold_PDT = threshold_gain(adv_correct_testing_dates, adv_testing_pred_diff, adv_testing_target_diff,df_open,df_close,freq_type='hour')

#threshold gain kelly
kelly_threshold_gain_train, kelly_training_threshold_returns,kelly_training_threshold_PDT = kelly_threshold_gain(correct_training_dates, training_pred_diff,training_target_diff,df_open,df_close,freq_type='hour')
kelly_threshold_gain_test,kelly_testing_threshold_returns,kelly_testing_threshold_PDT = kelly_threshold_gain(correct_testing_dates, testing_pred_diff,testing_target_diff,df_open,df_close,freq_type='hour')
kelly_adv_threshold_gain_test,kelly_adv_testing_threshold_returns,kelly_adv_testing_threshold_PDT = kelly_threshold_gain(adv_correct_testing_dates, adv_testing_pred_diff,adv_testing_target_diff,df_open,df_close,freq_type='hour')


# 23 Printing the various metrics:
t = rolling_training_stats
v = rolling_testing_stats
a = adv_rolling_testing_stats
print(f"Training: RMSE: {round(t['RMSE'],3)} MAPE: {round(t['MAPE'],3)} CRPS: {round(t['CRPS'],3)}"
      f" Only Up: {round(train_bias_acc_up,3)} Only Down: {round(train_bias_acc_down,3)} Accuracy: {round(train_acc,3)} \n"
      f"Initial Investment: {round(train_invest,2)}$ Passive Return: {round(passive_gain_train,1)}% Greedy Return: {round(greedy_gain_train,1)}%"
      f" Kelly Greedy Return: {round(kelly_greedy_gain_train,1)}% Threshold Return: {round(threshold_gain_train,1)}%"
      f" Kelly Threshold Return {round(kelly_threshold_gain_train,1)}% \n")
print(f"Testing: RMSE: {round(v['RMSE'],3)} MAPE: {round(v['MAPE'],3)} CRPS: {round(v['CRPS'],3)}"
      f" Only Up: {round(test_bias_acc_up,3)} Only Down: {round(test_bias_acc_down,3)} Accuracy: {round(test_acc,3)} \n"
      f"Initial Investment: {round(test_invest,2)}$ Passive Return: {round(passive_gain_test,1)}% Greedy Return: {round(greedy_gain_test,1)}%"
      f" Kelly Greedy Return: {round(kelly_greedy_gain_test,1)}% Threshold Return: {round(threshold_gain_test,1)}%"
      f" Kelly Threshold Return {round(kelly_threshold_gain_test,1)}% \n")
print(f"Testing: RMSE: {round(a['RMSE'],3)} MAPE: {round(a['MAPE'],3)} CRPS: {round(a['CRPS'],3)}"
      f" Only Up: {round(adv_test_bias_acc_up,3)} Only Down: {round(adv_test_bias_acc_down,3)} Accuracy: {round(adv_test_acc,3)} \n"
      f"Initial Investment: {round(adv_test_invest,2)}$ Passive Return: {round(passive_gain_test,1)}% Greedy Return: {round(adv_greedy_gain_test,1)}%"
      f" Kelly Greedy Return: {round(kelly_adv_greedy_gain_test,1)}% Threshold Return: {round(adv_threshold_gain_test,1)}%"
      f" Kelly Threshold Return {round(kelly_adv_threshold_gain_test,1)}% \n")


ret_dict ={}
training_metrics = {}
testing_metrics = {}
adv_testing_metrics = {}

for feat in ['RMSE','MAPE','CRPS']:
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

returns_dict = {'Training':{'Greedy':training_greedy_returns,'Threshold':training_threshold_returns,'Kelly Greedy':kelly_training_greedy_returns,'Kelly Threshold':kelly_training_threshold_returns},
                'Testing': {'Greedy':testing_greedy_returns,'Threshold':testing_threshold_returns,'Kelly Greedy':kelly_testing_greedy_returns,'Kelly Threshold':kelly_testing_threshold_returns},
                'Adversarial': {'Greedy': adv_testing_greedy_returns, 'Threshold': adv_testing_threshold_returns,'Kelly Greedy':kelly_adv_testing_greedy_returns,'Kelly Threshold':kelly_adv_testing_threshold_returns}}

PDT_dict  = {'Training':{'Greedy':training_greedy_PDT,'Threshold':training_threshold_PDT,'Kelly Greedy':kelly_training_greedy_PDT,'Kelly Threshold':kelly_training_threshold_PDT},
                'Testing': {'Greedy':testing_greedy_PDT,'Threshold':testing_threshold_PDT,'Kelly Greedy':kelly_testing_greedy_PDT,'Kelly Threshold':kelly_testing_threshold_PDT},
                'Adversarial': {'Greedy':adv_testing_greedy_PDT,'Threshold':adv_testing_threshold_PDT,'Kelly Greedy':kelly_adv_testing_greedy_PDT,'Kelly Threshold':kelly_adv_testing_threshold_PDT}}
f_dict = {'Training':f_train, 'Testing':f_test,'Adversarial':f_adv}
filename = dir_path + file_head + "6_return_distributions" + file_body + ".png"
plot_return_distributions(returns_dict,PDT_dict,f_dict,save_figs = args.save_figs,show_figs=args.show_figs,save_path = filename)

filename = metric_dir_path + file_head + "adv_testing" + file_body + ".p"
if args.save_figs:
    pickle.dump(ret_dict, open(filename, 'wb'))


########################################################################
################## Plotting the open data  #############################
########################################################################
vt = correct_testing_target_vals
vp = correct_testing_pred_vals
adv_vp = adv_correct_testing_pred_vals

new_dates = [pd.Timestamp(correct_testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d %H:%M:%S') for i in range(0,correct_testing_dates.shape[0])]
new_dates_plot = [pd.Timestamp(d).strftime('%Y-%m-%d %H:%M:%S') for d in correct_testing_dates]
open_vals = df_open[new_dates].to_numpy().reshape((-1,vt.shape[0])).reshape((-1,))
close_vp = np.exp(vp) * open_vals
close_vt  =np.exp(vt) * open_vals
adv_close_vt  =np.exp(adv_vp) * open_vals
fig, ax = plt.subplots(figsize = (12,8))
plt.plot_date(new_dates_plot, close_vt, fmt ='-b')
plt.plot_date(new_dates_plot, close_vp, fmt = '-g')
plt.plot_date(new_dates_plot, adv_close_vt, fmt = '-r')


plt.legend(['Ground Truth Price', 'Predicted Price', 'Manipulated Predicted Price'])
plt.title('Close Price for Test set')
plt.xlabel('Time (business hours)')
plt.ylabel('Close Price')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=21))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gcf().autofmt_xdate()
filename = filename = dir_path + file_head + "7_close_price" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()


######################## stupid test ######################################

true = correct_training_target_vals
pred = correct_training_pred_vals


new_dates = [pd.Timestamp(correct_training_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d %H:%M:%S') for i in range(0,correct_training_dates.shape[0])]
new_dates_plot = [pd.Timestamp(d).strftime('%Y-%m-%d %H:%M:%S') for d in correct_training_dates]
open_vals = df_open[new_dates].to_numpy().reshape((-1,true.shape[0])).reshape((-1,))
close_true = np.exp(true) * open_vals
close_pred  =np.exp(pred) * open_vals

fig, ax = plt.subplots(figsize = (12,8))
plt.plot_date(new_dates_plot, close_true, fmt ='-b')
plt.plot_date(new_dates_plot, close_pred, fmt = '-g')
plt.legend(['Ground Truth Price', 'Predicted Price'])
plt.title('Close Price for Training set')
plt.xlabel('Time (business hours)')
plt.ylabel('Close Price')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=21))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gcf().autofmt_xdate()
filename = filename = dir_path + file_head + "8_training_close_price" + file_body + ".png"
if args.save_figs:
    plt.savefig(filename, dpi =100)
if args.show_figs:
    plt.show()
else:
    plt.clf()