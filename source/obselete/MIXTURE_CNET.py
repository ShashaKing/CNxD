#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 18:20:46 2019

Class of Mixture Cutset Network

@author: 
"""


from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
#from sklearn.calibration import CalibratedClassifierCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import datasets
#import scipy.sparse as sps
#from sklearn.neural_network import MLPClassifier
#import math
import sys
import time
from Util import *
import pickle

#import utilM
from CNET_class import CNET
#import JT

class MIXTURE_CNET():
    
    def __init__(self):
        self.n_components = 0
        self.mixture_weight = None
        # weigths associated with each record in mixture
        # n_componets * n_var
        #self.clt_weights_list = None
        self.cnet_list =[]   # chow-liu tree list
        
    '''
        Learn the structure of the Chow-Liu Tree using the given dataset
    '''
    def learnStructure(self, dataset, n_components, depth):
        print ("Mixture of CNET ......" )
        # Shuffle the dataset
        
        self.n_components = n_components
        self.mixture_weight = np.full(n_components , 1.0 /n_components )
        #print ("mixture weights: ", self.mixture_weight)
        data_shuffle = np.copy(dataset)
        np.random.shuffle(data_shuffle)
        n_data = data_shuffle.shape[0] / self.n_components
        
        
        for c in xrange(self.n_components):
            if c == self.n_components - 1:   # the last portion
                data_slice = data_shuffle[c*n_data : , : ]
                
            else:
                data_slice = data_shuffle[c*n_data: ((c+1)*n_data), :]
            
            cnet = CNET(depth)
            cnet.learnStructure(data_slice)
            
            self.cnet_list.append(cnet)
    
    # Learning parameters using EM
    def EM(self, dataset, max_iter, epsilon):
        
        start = time.time()
        
        #print ("epsilon: ", epsilon)
        structure_update_flag = False
        
        cnet_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        ll_score = -np.inf
        ll_score_prev = -np.inf
        for itr in xrange(max_iter):
            
            #print ( "iteration: ", itr)
            
            if itr > 0:
                #print (np.einsum('ij->i', clt_weights_list))
                self.mixture_weight = Util.normalize(np.einsum('ij->i', cnet_weights_list) + 1.0)  # smoothing and Normalize
                #mixture_weights = np.sum(clt_weights_list, axis = 1)
                #print (self.mixture_weight)
                
                # update tree structure: the first 50 iterations, afterward, every 50 iterations
                if itr < 50 or itr % 50 == 0:
                    structure_update_flag = True
                    
                for c in xrange(self.n_components):
                    self.cnet_list[c].update_exact(dataset, cnet_weights_list[c], structure_update_flag)
                    #self.clt_list[c].update_exact(dataset, clt_weights_list[c])
                
                structure_update_flag = False
            
            ll_score_prev = ll_score
            
            log_mixture_weights = np.log(self.mixture_weight)
            for c in xrange(self.n_components):
                cnet_weights_list[c] = self.cnet_list[c].getWeights(dataset) + log_mixture_weights[c]
            
            

            
            #print ("shape: ", self.clt_weights_list.shape[1])
            # Normalize weights 
            # input is in log format, output is in normal
            cnet_weights_list, ll_score = Util.m_step_trick(cnet_weights_list)
            #print (self.clt_weights_list)
            
            #print ("LL score diff : ", ll_score - ll_score_prev)
            if abs(ll_score - ll_score_prev) < epsilon:
                print ("converged")
                break
            # more than 48 hours   
            if time.time() -  start >= 172800:
                print ('reach 24 hours')
                break
        
        print ("Total iterations: ", itr)
        print('Train set LL scores: ', ll_score / dataset.shape[0])
        print ("difference in LL score: ", ll_score - ll_score_prev)
    
    
    def computeLL(self, dataset):
        
        cnet_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        log_mixture_weights = np.log(self.mixture_weight)
        
        for c in xrange(self.n_components):
            cnet_weights_list[c] = self.cnet_list[c].getWeights(dataset) + log_mixture_weights[c]
            

        cnet_weights_list, ll_score = Util.m_step_trick(cnet_weights_list)
        
        return ll_score
    
    def computeLL_each_datapoint(self, dataset):
        
        cnet_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        log_mixture_weights = np.log(self.mixture_weight)
        
        for c in xrange(self.n_components):
            cnet_weights_list[c] = self.cnet_list[c].getWeights(dataset) + log_mixture_weights[c]
            

        ll_scores = Util.get_ll_trick(cnet_weights_list)
        
        return ll_scores
    

def reload_mcnet(filename):
    
    with open(filename + '.pkl', 'rb') as input:
        reload_mcnet = pickle.load(input)
    
    return reload_mcnet


            


def main_mixture_cnet():
    #train_filename = sys.argv[1]
    
            
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    #seq = sys.argv[6]
    n_components = int(sys.argv[6])
    max_depth = int(sys.argv[8])
    max_iter = int(sys.argv[10])  
    epsilon = float(sys.argv[12])  
    
    
    #train_filename = '../dataset/nltcs.ts.data'
    #test_filename = train_filename[:-8] + '.test.data'
    #valid_filename = train_filename[:-8] + '.valid.data'

    #train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    #valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    
    
    train_name = dataset_dir + data_name +'.ts.data'
    valid_name = dataset_dir + data_name +'.valid.data'
    test_name = dataset_dir + data_name +'.test.data'
    data_train = np.loadtxt(train_name, delimiter=',', dtype=np.uint32)
    data_valid = np.loadtxt(valid_name, delimiter=',', dtype=np.uint32)
    data_test = np.loadtxt(test_name, delimiter=',', dtype=np.uint32)
    
    #epsilon *= data_train.shape[0]
    
    print("Learning Mixture of Cutset Network on original data ......")
    #n_components = 5
    start = time.time()
    mix_cnet = MIXTURE_CNET()
    mix_cnet.learnStructure(data_train, n_components, max_depth)
    mix_cnet.EM(data_train, max_iter, epsilon)
    running_time = time.time() -  start
    
    #print("done")
    
#    save_list = []    
#    #save_dict['weights'] = mix_clt.mixture_weight
#    #print (save_dict['weights'])
#    for i in xrange(n_components):
#        new_dict = dict()
#        new_dict['xprob'] = mix_clt.clt_list[i].xprob
#        new_dict['xyprob'] = mix_clt.clt_list[i].xyprob
#        new_dict['topo_order'] = mix_clt.clt_list[i].topo_order
#        new_dict['parents'] = mix_clt.clt_list[i].parents
#        new_dict['log_cond_cpt'] = mix_clt.clt_list[i].log_cond_cpt
#        new_dict['tree'] = mix_clt.clt_list[i].Tree
#        save_list.append(new_dict)
    
    output_dir = '../mcnet/'
    # save the module
    with open(output_dir + 'module/' + data_name +'_'+str(n_components) +'_'+str(max_depth) + '.pkl', 'wb') as output:
        pickle.dump(mix_cnet, output, pickle.HIGHEST_PROTOCOL)
    # save the time
    np.savetxt(output_dir + 'time/' + data_name +'_'+str(n_components) +'_'+str(max_depth) + '.txt',np.array([running_time]), delimiter=',')
    # save the ll
    train_ll = mix_cnet.computeLL(data_train) / data_train.shape[0]
    valid_ll = mix_cnet.computeLL(data_valid) / data_valid.shape[0]
    test_ll = mix_cnet.computeLL(data_test) / data_test.shape[0]
    ll_score = np.zeros(3)
    ll_score[0] = train_ll
    ll_score[1] = valid_ll
    ll_score[2] = test_ll
    np.savetxt(output_dir + 'll_score/' + data_name +'_'+str(n_components) +'_'+str(max_depth) + '.txt',ll_score, delimiter=',')
    
    
    #out_file = '../output/' + data_name +'_'+str(seq)+'_'+str(n_components) +'.npz'
    
    #np.savez('save_dict.npz', save_dict)
    #np.savez_compressed(out_file, clt_component=save_list, weights=mix_cnet.mixture_weight, valid_ll = valid_ll, test_ll = test_ll)
    print('Train set LL scores')
    print(train_ll)
    
    print('Valid set LL scores')
    print(valid_ll)
    
    print('Test set LL scores')
    print(test_ll)
    
    
#    # reload
#    with open(output_dir + 'module/' + data_name +'_'+str(n_components) +'_'+str(max_depth) + '.pkl', 'rb') as input:
#        reload_mcnet = pickle.load(input)
#        print ('train: ', reload_mcnet.computeLL(data_train) / data_train.shape[0])
#        print ('valid: ', reload_mcnet.computeLL(data_valid) / data_valid.shape[0])
#        print ('test: ',reload_mcnet.computeLL(data_test) / data_test.shape[0])

    
   


    
    
    #print('Train set LL scores')
    #print(mix_clt.computeLL(train_dataset) / train_dataset.shape[0], "Mixture-Chow-Liu")
    #print(mix_clt.computeLL(train_dataset), "Mixture-Chow-Liu")



    


if __name__=="__main__":
    #main_cutset()
    #main_clt()
    start = time.time()
    main_mixture_cnet()
    print ('Total running time: ', time.time() - start)       