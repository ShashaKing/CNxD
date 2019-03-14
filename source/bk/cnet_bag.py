#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 21:53:10 2019
"""

import sys
import time
from Util import *
#import pickle
import numpy as np
import random
import utilM

# for bootstrap
#from sklearn import cross_validation

#import utilM
from CNET_class import CNET

class BAG_CNET():
    
    def __init__(self):
        self.n_components = 0        # how many bags
        self.mixture_weight = None   # bag weights
        self.cnet_list =[]   # cnet list
        
    '''
        Learn the structure of the bags of CNET using the given dataset
    '''
    # node_sel_option = 0 means optimaly select OR node  using MI
    #                 = 1 means select OR node from 0.5 percent of all varibles
    # depth_sel_option = 0 means all cnets have the same depth (max depth)
    #                  = 1 means the depth of cnets are randomly choosing from 1 to 6
    #                  = 2 means the depht of cnets are choosed seqencially from 1 to 6
    def learnStructure(self, dataset, n_components, max_depth, node_sel_option = 0, depth_sel_option =0):
        print ("BAGS of CNET ......" )
        # Shuffle the dataset
        
        self.n_components = n_components
        self.mixture_weight = np.full(n_components , 1.0 /n_components )

        
        for c in xrange(self.n_components):
            
            # bootstrap the data
            data_ind = np.random.choice(dataset.shape[0], dataset.shape[0], replace=True)
            data_slice = dataset[data_ind]
            
            # find the depth
            if depth_sel_option == 0:
                depth = max_depth
            elif depth_sel_option == 1:
                depth = random.randint(1,6)
            elif depth_sel_option == 2:
                depth = c % 6 +1
            else:
                depth = max_depth
            
            #print 'depth: ', depth
            cnet = CNET(depth)
            if node_sel_option == 0:
                cnet.learnStructure(data_slice)
            else:
                cnet.learnStructure_portion(data_slice, 0.5)
                
            
            self.cnet_list.append(cnet)
            
            self.mixture_weight[c] = np.sum(cnet.getWeights(dataset)) / dataset.shape[0]
        
        # Normalize
        self.mixture_weight -= np.max(self.mixture_weight)
        self.mixture_weight = np.exp(self.mixture_weight)
        self.mixture_weight /= np.sum(self.mixture_weight)
        #print self.mixture_weight
        

    
    
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
    
#    
#    """
#    FOR CNET_deep
#    """
#    
#    def get_node_marginal(self, evid_list, var):
#        #log_mixture_weight = np.log(self.mixture_weight)
#        xprob_all = np.zeros(2)
#        for i, t in enumerate(self.cnet_list):
#            #print (t.topo_order)
#            #print (cond_cpt_list[i])
#            #p_xy =  utilM.get_prob_matrix(t.topo_order, t.parents, cond_cpt_list[i], ids)
#            xprob =  t.get_node_marginal(evid_list, var)
#            xprob_all += xprob * self.mixture_weight[i]
#
#        
#        #normalize
#        xprob_all[0] =  xprob_all[0] / (xprob_all[0] + xprob_all[1])
#        xprob_all[1] = 1.0 - xprob_all[0]
#        
#        return xprob_all
#    
#    
#    def get_edge_marginal(self, evid_list, edges):
#        #log_mixture_weight = np.log(self.mixture_weight)
#        xyprob_all = np.zeros((edges.shape[0],2,2))
#        for i, t in enumerate(self.clt_list):
#                                
#            xyprob =  t.get_edge_marginal(evid_list, edges)
#            xyprob_all += xyprob * self.mixture_weight[i]
#
#        
#        #normalize
#        xyprob_all =  Util.normalize1d_in_2d(xyprob_all)
#        
#        return xyprob_all

#def reload_bcnet(filename):
#    
#    with open(filename + '.pkl', 'rb') as input:
#        reload_bcnet = pickle.load(input)
#    
#    return reload_bcnet


'''    
    load the pre trained bags of cnet from disk
'''
def load_bcn(in_dir, data_name):
    
    #print in_dir + data_name +'_component_weights.txt'
    cm_weights = np.loadtxt(in_dir + data_name +'_component_weights.txt')

    #out_file = '../module/' + data_name + '.npz'
    reload_bcn = {}
    reload_bcn['cm_weights'] = cm_weights
    reload_bcn['n_components'] = cm_weights.shape[0]
    
    cnet_list =[]
    for i in xrange(cm_weights.shape[0]):
        cn_file = in_dir + data_name +'_' +str(i) + '.npz'
        cn = np.load(cn_file)['module'].item()
        cnet_list.append(cn)

    reload_bcn['cnet_list'] = cnet_list
    return reload_bcn


'''    
    Compute the LL score from the reloaded bcn
'''  
def compute_ll_from_disk(reload_bcn, dataset):
    n_components = reload_bcn['n_components']
    
    cnet_weights_list = np.zeros((n_components, dataset.shape[0]))
        
    log_cm_weights = np.log(reload_bcn['cm_weights'])
        
    for c in xrange(n_components):
        cnet_weights_list[c] = utilM.computeLL_reload(reload_bcn['cnet_list'][c], dataset) + log_cm_weights[c]
            

    cnet_weights_list, ll_score = Util.m_step_trick(cnet_weights_list)
        
    return ll_score          


def main_bag_cnet():
    #train_filename = sys.argv[1]
    
            
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    n_components = int(sys.argv[6])
    max_depth = int(sys.argv[8])
    sel_option = int (sys.argv[10])
    depth_option = int (sys.argv[12])
    
    
    
    
    train_name = dataset_dir + data_name +'.ts.data'
    valid_name = dataset_dir + data_name +'.valid.data'
    test_name = dataset_dir + data_name +'.test.data'
    data_train = np.loadtxt(train_name, delimiter=',', dtype=np.uint32)
    data_valid = np.loadtxt(valid_name, delimiter=',', dtype=np.uint32)
    data_test = np.loadtxt(test_name, delimiter=',', dtype=np.uint32)
    
    #new_dataset = np.concatenate((data_train, data_valid), axis=0)


    
    print("Learning Bags of Cutset Network on original data ......")
    #start = time.time()
    bag_cnet = BAG_CNET()
    bag_cnet.learnStructure(data_train, n_components, max_depth, node_sel_option = sel_option, depth_sel_option =depth_option)
    #bag_cnet.learnStructure(new_dataset, n_components, max_depth, node_sel_option = sel_option, depth_sel_option =depth_option)
    #running_time = time.time() -  start
    
    
#    # save the ll
    train_ll = bag_cnet.computeLL(data_train) / data_train.shape[0]
    valid_ll = bag_cnet.computeLL(data_valid) / data_valid.shape[0]
    test_ll = bag_cnet.computeLL(data_test) / data_test.shape[0]
    ll_score = np.zeros(3)
    ll_score[0] = train_ll
    ll_score[1] = valid_ll
    ll_score[2] = test_ll

    print('Train set LL scores')
    print(train_ll)
    
    print('Valid set LL scores')
    print(valid_ll)
    
    print('Test set LL scores')
    print(test_ll)
    
    
    output_dir = '../bcnet_output/'
    
    for i in xrange(n_components):
        main_dict = {}
        utilM.save_cutset(main_dict, bag_cnet.cnet_list[i].tree, np.arange(data_train.shape[1]), ccpt_flag = True)
        np.savez_compressed(output_dir + data_name + '_' + str(i), module = main_dict)
    
    # save the component weights
    np.savetxt(output_dir  + data_name +'_component_weights.txt',bag_cnet.mixture_weight, delimiter=',')
    
    
#    
#    reload_bcn = load_bcn(output_dir, data_name)
#    train_ll_rl = compute_ll_from_disk (reload_bcn, data_train) / data_train.shape[0]
#    valid_ll_rl = compute_ll_from_disk (reload_bcn, data_valid) / data_valid.shape[0]
#    test_ll_rl = compute_ll_from_disk (reload_bcn, data_test) / data_test.shape[0]
#    print (train_ll_rl)
#    print (valid_ll_rl)
#    print (test_ll_rl)

    


if __name__=="__main__":

    start = time.time()
    main_bag_cnet()
    print ('Total running time: ', time.time() - start)       