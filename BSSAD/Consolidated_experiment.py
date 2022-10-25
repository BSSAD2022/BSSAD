import time
import random
import os
import numpy as np
from numpy import savetxt
from framework.models import BSSAD
from framework.preprocessing.data_loader import *
from framework.HPOptimizer.Hyperparameter import UniformIntegerHyperparameter,ConstHyperparameter,UniformFloatHyperparameter
from framework.HPOptimizer import HPOptimizers
from framework.preprocessing import normalize_and_encode_signals
from framework.utils.metrics import bf_search
from framework.utils import negative_sampler
import logging
import tensorflow as tf
import argparse
logging.getLogger('tensorflow').setLevel(logging.ERROR)

'''
    Dataset : WADI, SWAT, PUMP, HTTP, ASD
'''
dataset = "WADI"


'''
    filterType : PF, EnKF, UKF-PF
'''
filterType = "PF"

'''
    resample_method : systematic
'''
resample_method = "systematic"

'''
    n_particles (PF) : 500, 1000, 2000
    n_particles (EnKF) : 10, 20, 50
'''
n_particles = 1000


'''
    Load datasets
'''
if dataset == "WADI":
    train_df,val_df,test_df,signals = load_wadi_data()
    seqL = 12
    kf = BSSAD(signals, tau=seqL, input_range=seqL*3)

if dataset == "SWAT":
    train_df,val_df,test_df,signals = load_swat_data()
    seqL = 12
    kf = BSSAD(signals, tau=seqL, input_range=seqL*3)

if dataset == "PUMP":
    train_df,val_df,test_df,signals = load_pump_data()
    seqL = 5
    kf = BSSAD(signals, tau=seqL, input_range=seqL*3)
    
if dataset == "HTTP":
    train_df,val_df,test_df,signals = load_http_data()
    seqL = 12
    kf = BSSAD(signals, tau=seqL, input_range=seqL*3)

if dataset == "ASD":
    train_df, val_df, test_df, signals = load_ASD_data()
    seqL = 12
    kf = BSSAD(signals, tau=seqL, input_range=seqL*3)


train_df = normalize_and_encode_signals(train_df,signals,scaler='min_max') 
train_x,train_u,train_y,_ = kf.extract_data(train_df)
x_train = [train_x,train_u]
y_train = [train_x,train_y]

'''
    To retrain the model, set this to True
'''
retrain_model = False
if retrain_model:
    file = open(dataset + "_train_params.txt", "w")
    x,u,y = [],[],[]
    for i in range(20):
        r = 0.05*i
        negative_df = negative_sampler.apply_negative_samples(val_df, signals, sample_ratio=r, sample_delta=0.05)
        negative_df = normalize_and_encode_signals(negative_df,signals,scaler='min_max')
        neg_x,neg_u,_,neg_labels = kf.extract_data(negative_df,purpose='AD',freq=seqL,label='class_label')
        neg_labels = neg_labels.sum(axis=1)
        neg_labels[neg_labels<seqL]=0
        neg_labels[neg_labels==seqL]=1
        x.append(neg_x)
        u.append(neg_u)
        y.append(neg_labels)
    x = np.concatenate(x)
    u = np.concatenate(u)
    y = np.concatenate(y)
    print(list(y).count(1),len(y))
    x_neg = [x,u]
    y_neg = y
    
    hp_list = []
    hp_list.append(UniformIntegerHyperparameter('z_dim',1,200))
    hp_list.append(UniformIntegerHyperparameter('hnet_hidden_layers',1,3))  
    hp_list.append(UniformIntegerHyperparameter('fnet_hidden_layers',1,3))
    hp_list.append(UniformIntegerHyperparameter('fnet_hidden_dim',32,256))
    hp_list.append(UniformIntegerHyperparameter('uencoding_layers',1,3))
    hp_list.append(UniformIntegerHyperparameter('uencoding_dim',32,256))
    hp_list.append(UniformFloatHyperparameter('l2',0,0.05))
    hp_list.append(ConstHyperparameter('epochs',100))
    hp_list.append(ConstHyperparameter('save_best_only',True))
    hp_list.append(ConstHyperparameter('validation_split',0.1))
    hp_list.append(ConstHyperparameter('batch_size',256*16))
    hp_list.append(ConstHyperparameter('verbose',2))
    
    optor = HPOptimizers.RandomizedGS(kf, hp_list,x_train, y_train,x_neg,y_neg)
    kf,optHPCfg,bestScore = optor.run(n_searches=10,verbose=1)
    if dataset == "WADI":
        kf = kf.save_model('./results/WADI')
    
    if dataset == "SWAT":
        kf = kf.save_model('./results/SWAT')

    if dataset == "PUMP":
        kf = kf.save_model('./results/PUMP')
        
    if dataset == "HTTP":
        kf = kf.save_model('./results/HTTP')
        
    if dataset == "ASD":
        kf = kf.save_model('./results/ASD1')
    
    print('optHPCfg',optHPCfg)
    print('bestScore',bestScore)
    file.write("OptHPCfg " + str(optHPCfg) + "\n" + "Best Score: " + str(bestScore) + "\n" + "SeqL: " + str(seqL))
    file.close()

else:
    if dataset == "WADI":
        kf = kf.load_model('./results/WADI')
        print("Loaded model...")
    
    if dataset == "SWAT":
        kf = kf.load_model('./results/SWAT')
        print("Loaded model...")

    if dataset == "PUMP":
        kf = kf.load_model('./results/PUMP')
        print("Loaded model...")
    
    if dataset == "HTTP":
        kf = kf.load_model('./results/HTTP')
        print("Loaded model...")
        
    if dataset == "ASD":
        kf = kf.load_model('./results/ASD1')


val_df = normalize_and_encode_signals(val_df,signals,scaler='min_max') 
val_x,val_u,val_y,_ = kf.extract_data(val_df)
test_df = normalize_and_encode_signals(test_df,signals,scaler='min_max')
test_x,test_u,_,labels = kf.extract_data(test_df,purpose='AD',freq=seqL,label='label')
labels = labels.sum(axis=1)
labels[labels>0]=1

print("Estimating noise...")
kf.estimate_noise(val_x,val_u,val_y)


'''
    Every iteration tests the model on a different random seed
    Iteration is random seed number
'''
f1 = []
auc = []
mcc = []
bestSeed = 2022
bestF1 = 0.0
iterations = 50

for k in range(0, iterations):
    newSeed = bestSeed + k
    random.seed(newSeed)
    np.random.seed(newSeed)
    tf.random.set_seed(newSeed)
    fileName = dataset + "_results_Final" + "n = " + str(n_particles) + "seed=" + str(newSeed) + "Filtertype=" + filterType + ".txt"
    writePath = "./results/experiment_results"
    completeName = os.path.join(writePath, fileName)
    if not os.path.exists(writePath):
        os.mkdir(writePath)
    file = open(completeName, "w")
    
    '''
        Score samples: kf.score_samples(test_x, test_u, n_particles, resample_method, reset_hidden_states=True) 
        
        Parameters
        ----------
        test_x
        test_u
        n_particles
        filterType
        resample_method
        reset_hidden_states
    '''
    
    print("Filter type: " + filterType)
    print("Resample_method: " + resample_method)
    print("Number of Particles: "+ str(n_particles))
    print("Seed: ", newSeed)
    
    z_scores_ekf = kf.score_samples(test_x, test_u, newSeed, n_particles, filterType, resample_method, reset_hidden_states=True)
    
    z_scores_ekf = np.nan_to_num(z_scores_ekf)
    t, th = bf_search(z_scores_ekf, labels[1:],start=0,end=np.percentile(z_scores_ekf,99.9),step_num=10000,display_freq=50,verbose=False)
    print('BSSAD_' + filterType)
    print('best-f1', t[0])
    print('precision', t[1])
    print('recall', t[2])
    print('accuracy',(t[3]+t[4])/(t[3]+t[4]+t[5]+t[6]))
    print('TP', t[3])
    print('TN', t[4])
    print('FP', t[5])
    print('FN', t[6])
    print("AUC", t[7])
    print("MCC", t[8])
    print()
    
    if t[0] > bestF1:
        bestF1 = t[0]
        bestSeed = newSeed
    f1.append(t[0])
    auc.append(t[7])
    mcc.append(t[8])
    
    file.write("BSSAD_" + filterType + "\n" + "Best-F1: " + str(t[0]) + "\n"
               + "Precision: " + str(t[1]) + "\n" + "Recall: " + str(t[2]) + "\n"
               + "Accuracy: " + str((t[3]+t[4])/(t[3]+t[4]+t[5]+t[6])) + "\n"
               + "TP: " + str(t[3]) + "\n" + "TN: " + str(t[4]) + "\n"
               + "FP: " + str(t[5]) + "\n" + "FN: " + str(t[6]) + "\n"
               + "AUC: " + str(t[7]) + "\n" + "MCC: " + str(t[8]) + "\n")
    file.close()

npyf1 = np.array(f1)
npyAuc = np.array(auc)
npyMcc = np.array(mcc)

bestF1 = np.max(npyf1)
#bestSeed = np.argmax(npyf1)
#bestSeed = bestSeed + 2022
bestAuc = np.max(auc)
bestMcc = np.max(mcc)


print("Best F1: ", bestF1)
print("Best AUC: ", bestAuc)
print("Best MCC: ", bestMcc)
print("Best Score Seed: ", bestSeed)

fileName = str(dataset) + "_n=" + str(n_particles) + "_bestSeed.txt"
writePath = "./results/experiment_results"
completeName = os.path.join(writePath, fileName)
file = open(completeName, "w")
file.write("BestF1: " + str(bestF1) + "\n" +
           "Best AUC: " + str(bestAuc) + "\n" +
           "Best MCC: " + str(bestMcc) + "\n" +
           "Best Seed: " + str(bestSeed) + "\n")
file.close()

f1file = dataset + "_" + str(n_particles) + "_bestF1.csv"
aucfile = dataset + "_" + str(n_particles) + "_bestAUC.csv"
mccfile = dataset + "_" + str(n_particles) + "_bestMCC.csv"

savetxt(f1file, npyf1, delimiter = ",")
savetxt(aucfile, npyAuc, delimiter = ", ")
savetxt(mccfile, npyMcc, delimiter = ", ")

    