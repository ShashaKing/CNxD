#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:26:30 2018

Class of Mixture CLT

@author: 
"""

from __future__ import print_function

import numpy as np

import sys
import time
from Util import *

from CLT_class import CLT
import JT




class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0
        self.mixture_weight = None

        self.clt_list =[]   # chow-liu tree list
        self.jt_list = []  # junction tree list
        
    '''
        Initialize the structure and parameters of Mixture of Chow-Liu Tree using the given dataset
    '''
    def learnStructure(self, dataset, n_components):
        #print ("Mixture of Chow-Liu Tree ......" )
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
            
            clt = CLT()
            clt.learnStructure(data_slice)
            
            self.clt_list.append(clt)
            
    '''
        Update both structure and parameters by using EM algorithm
    '''
    def EM(self, dataset, max_iter, epsilon):
        
        structure_update_flag = False
        
        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        ll_score = -np.inf
        ll_score_prev = -np.inf
        for itr in xrange(max_iter):
        
            
            if itr > 0:
                self.mixture_weight = Util.normalize(np.einsum('ij->i', clt_weights_list) + 1.0)  # smoothing and Normalize
                
                # update tree structure: the first 50 iterations, afterward, every 50 iterations
                if itr < 50 or itr % 50 == 0:
                    structure_update_flag = True
                    
                for c in xrange(self.n_components):
                    self.clt_list[c].update_exact(dataset, clt_weights_list[c], structure_update_flag)
                
                structure_update_flag = False
            
            ll_score_prev = ll_score
            
            log_mixture_weights = np.log(self.mixture_weight)
            for c in xrange(self.n_components):
                clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]
            

            # for clt_weights_list, input is in log format, output is not in log
            clt_weights_list, ll_score = Util.m_step_trick(clt_weights_list)
           
            if abs(ll_score - ll_score_prev) < epsilon:
                print ("converged")
                break
                
        
        print ("Total iterations: ", itr)        
        print ("difference in LL score: ", ll_score - ll_score_prev)
        print ('Train set LL score: ', ll_score / dataset.shape[0])
        
    
    """
        Compute the log-likelihood score for the input dataset
    """
    def computeLL(self, dataset):
        
        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        log_mixture_weights = np.log(self.mixture_weight)
        
        for c in xrange(self.n_components):
            clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]
            

        clt_weights_list, ll_score = Util.m_step_trick(clt_weights_list)
        
        return ll_score
    
    
    """
        Compute the log-likelihood score for the each datapoint in the input dataset
    """
    def computeLL_each_datapoint(self, dataset):
        
        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        log_mixture_weights = np.log(self.mixture_weight)
        
        for c in xrange(self.n_components):
            clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]
            

        ll_scores = Util.get_ll_trick(clt_weights_list)
        
        return ll_scores
    
    
    """
        Get all single vairble and  pairwised marginal probabilities
    """
    def inference(self,evid_list, ids):
        dim = ids.shape[0]
        p_xy_all = np.zeros((dim, dim, 2, 2))
        p_x_all = np.zeros((dim, 2))
        for i, t in enumerate(self.clt_list):

            cond_cpt = t.instantiation(evid_list)
                
            p_xy =  t.inference(cond_cpt, ids)
            p_xy_all += p_xy * self.mixture_weight[i]
        


        p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
        p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
        
        p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
        p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
        
        
        # Normalize        
        p_x_all = Util.normalize1d(p_x_all)
        
        
        for i in xrange (ids.shape[0]):
            p_xy_all[i,i,0,0] = p_x_all[i,0] - 1e-8
            p_xy_all[i,i,1,1] = p_x_all[i,1] - 1e-8
            p_xy_all[i,i,0,1] = 1e-8
            p_xy_all[i,i,1,0] = 1e-8
        
        p_xy_all = Util.normalize2d(p_xy_all)

        
        return p_xy_all, p_x_all
    
    
    """
        FOR CNET_deep
    """
    
    def get_node_marginal(self, evid_list, var):

        xprob_all = np.zeros(2)
        for i, t in enumerate(self.clt_list):

            if len(evid_list) == 0:  # no evidence
                temp_cond_cpt = np.copy(t.cond_cpt)
            else:
                temp_cond_cpt = t.instantiation(evid_list)
            xprob =  t.get_node_marginal(temp_cond_cpt, var)
            xprob_all += xprob * self.mixture_weight[i]

        
        #normalize
        xprob_all[0] =  xprob_all[0] / (xprob_all[0] + xprob_all[1])
        xprob_all[1] = 1.0 - xprob_all[0]
        
        return xprob_all
    
    
    def get_edge_marginal(self, evid_list, edges):

        xyprob_all = np.zeros((edges.shape[0],2,2))
        for i, t in enumerate(self.clt_list):

            if len(evid_list) == 0:  # no evidence
                temp_cond_cpt = np.copy(t.cond_cpt)
            else:
                temp_cond_cpt = t.instantiation(evid_list)
                
                
            xyprob =  t.get_edge_marginal(temp_cond_cpt, edges)
            xyprob_all += xyprob * self.mixture_weight[i]

        
        #normalize
        xyprob_all =  Util.normalize1d_in_2d(xyprob_all)
        
        return xyprob_all
    
    
    
    """
        Get the pairwised marginals using junction tree
        Used in cnet_jt.py
    """
    def inference_jt(self,evid_list, ids):
        dim = ids.shape[0]
        p_xy_all = np.zeros((dim, dim, 2, 2))
        p_x_all = np.zeros((dim, 2))
        for i, jt in enumerate(self.jt_list):
            p_xy = JT.get_marginal_JT(jt, evid_list, ids)
            p_xy_all += p_xy * self.mixture_weight[i]


        p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
        p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
        
        p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
        p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
        
        
        # Normalize        
        p_x_all = Util.normalize1d(p_x_all)
        
        
        for i in xrange (ids.shape[0]):
            p_xy_all[i,i,0,0] = p_x_all[i,0] - 1e-8
            p_xy_all[i,i,1,1] = p_x_all[i,1] - 1e-8
            p_xy_all[i,i,0,1] = 1e-8
            p_xy_all[i,i,1,0] = 1e-8
        
        p_xy_all = Util.normalize2d(p_xy_all)

        
        return p_xy_all, p_x_all
    
    
    
'''    
    load the pre trained MT from disk
'''
def load_mt(in_dir,data_name):
    infile = in_dir+ data_name + '.npz'
    reload_dict = np.load(infile)
    reload_mix_clt = MIXTURE_CLT()
    reload_mix_clt.mixture_weight = reload_dict['weights']
    reload_mix_clt.n_components = reload_mix_clt.mixture_weight.shape[0]
    
    reload_clt_component = reload_dict['clt_component']
    
    for i in xrange(reload_mix_clt.n_components):
        clt_c = CLT()
        curr_component = reload_clt_component[i]
        clt_c.xyprob = curr_component['xyprob']
        clt_c.xprob = curr_component['xprob']
        clt_c.topo_order = curr_component['topo_order']
        clt_c.parents = curr_component['parents']
        clt_c.log_cond_cpt = curr_component['log_cond_cpt']
        clt_c.cond_cpt = np.exp(clt_c.log_cond_cpt)   #deep
        
        reload_mix_clt.clt_list.append(clt_c)
    
    return reload_mix_clt
            

'''
   Main function for Learning the mixture of clt 
'''
def main_mixture_clt(parms_dict):
    
    print ("----------------------------------------------------")
    print ("Learning Mixture of Chow-Liu Tree on original data  ")
    print ("----------------------------------------------------")
            
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    n_components = int(parms_dict['ncomp'])
    max_iter = int(parms_dict['max_iter'])  
    epsilon = float(parms_dict['eps'])  
    out_dir = parms_dict['output_dir']
    
    
    
    train_name = dataset_dir + data_name +'.ts.data'
    valid_name = dataset_dir + data_name +'.valid.data'
    test_name = dataset_dir + data_name +'.test.data'
    data_train = np.loadtxt(train_name, delimiter=',', dtype=np.uint32)
    data_valid = np.loadtxt(valid_name, delimiter=',', dtype=np.uint32)
    data_test = np.loadtxt(test_name, delimiter=',', dtype=np.uint32)
    
    
    mix_clt = MIXTURE_CLT()
    mix_clt.learnStructure(data_train, n_components)
    mix_clt.EM(data_train, max_iter, epsilon)
    
    
    save_list = []    
    for i in xrange(n_components):
        new_dict = dict()
        new_dict['xprob'] = mix_clt.clt_list[i].xprob
        new_dict['xyprob'] = mix_clt.clt_list[i].xyprob
        new_dict['topo_order'] = mix_clt.clt_list[i].topo_order
        new_dict['parents'] = mix_clt.clt_list[i].parents
        new_dict['log_cond_cpt'] = mix_clt.clt_list[i].log_cond_cpt
        new_dict['tree'] = mix_clt.clt_list[i].Tree
        save_list.append(new_dict)
   

    valid_ll = mix_clt.computeLL(data_valid) / data_valid.shape[0]
    test_ll = mix_clt.computeLL(data_test) / data_test.shape[0]
    
    out_file = out_dir + data_name +'_'+str(n_components) +'.npz'
    
    np.savez_compressed(out_file, clt_component=save_list, weights=mix_clt.mixture_weight, valid_ll = valid_ll, test_ll = test_ll)
    
    
    print('Valid set LL score: ', valid_ll)
    print('Test set LL score : ', test_ll)

   


    
    


    