#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 19:38:16 2018
The class of CNet

@author:
"""

from __future__ import print_function

import numpy as np
from Util import *
from CLT_class import CLT
import sys
import copy
import utilM

        
'''
The Cutset network learned from dataset
'''
class CNET:
    def __init__(self,depth=100, min_rec=10, min_var=5):
        self.nvariables=0
        self.depth=depth
        self.tree=[]        
        # 2 thresholds to stop going deeper
        self.min_rec = min_rec   
        self.min_var = min_var
        # for get node and edge potential
        self.internal_list = []
        self.internal_var_list = []
        self.leaf_list = []
        self.leaf_ids_list = []
    

    '''
        Recursively learn the structure and parameter
    '''    
    def learnStructureHelper(self,dataset,ids):
        curr_depth=self.nvariables-dataset.shape[1]
        #print ("curr_dept: ", curr_depth)
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(dataset)

                    
            return clt
        xycounts = Util.compute_xycounts(dataset) + 1  # laplace correction
        xcounts = Util.compute_xcounts(dataset) + 2  # laplace correction
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_edge_weights(xycounts, xcounts)
        np.fill_diagonal(edgemat, 0) #  #
        
    
        scores = np.sum(edgemat, axis=0)
        variable = np.argmax(scores)
        
        new_dataset1=np.delete(dataset[dataset[:,variable]==1],variable,1)
        p1=float(new_dataset1.shape[0])+1.0
        new_ids=np.delete(ids,variable,0)
        
        new_dataset0 = np.delete(dataset[dataset[:, variable] == 0], variable, 1)
        p0 = float(new_dataset0.shape[0]) +1.0
        
        return [variable,ids[variable],p0,p1,self.learnStructureHelper(new_dataset0,new_ids),
                self.learnStructureHelper(new_dataset1,new_ids)]
        
        
    def learnStructure(self, dataset):
        self.nvariables = dataset.shape[1]
        ids=np.arange(self.nvariables)
        self.tree=self.learnStructureHelper(dataset,ids)
        
    
    """
        Compute the log-likelihood score for the input dataset
    """
    def computeLL(self,dataset):
        prob = 0.0
        for i in range(dataset.shape[0]):
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    prob+=np.log(p1/(p0+p1))
                    node=node1
                else:
                    prob+=np.log(p0/(p0+p1))
                    node = node0
            prob+=node.computeLL(dataset[i:i+1,ids])
        return prob
    
    
    """
        Compute the log-likelihood score for the each datapoint in the input dataset
    """
    def computeLL_each_datapoint(self,dataset):
        probs = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0]):
            prob = 0.0
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    prob+=np.log(p1/(p0+p1))
                    node=node1
                else:
                    prob+=np.log(p0/(p0+p1))
                    node = node0
            prob+=node.computeLL(dataset[i:i+1,ids])
            probs[i] = prob
        return probs
    
    
    '''
        Update the parameters 
        1) if all datapoints has the same weight, update directly
        2) if different weights associated with each datapoint, update approximately
           using sampling
    '''
    def update(self,dataset_, weights=np.array([])):
        if weights.shape[0]==dataset_.shape[0]:
            norm_weights = Util.normalize(weights)
            indices = np.argwhere(np.random.multinomial(dataset_.shape[0], norm_weights)).ravel()
            dataset = dataset_[indices, :]
        else:
            dataset=dataset_
        for i in range(dataset.shape[0]):
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                p0_index=2
                p1_index=3
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    node[p1_index]=p1+1.0
                    node=node1
                else:
                    node[p0_index]=p0+1.0
                    node = node0
            node.update(dataset[i:i+1,ids])



    '''
        Recursively learn the structure and parameter using weighted data
    '''
    def learn_structure_weight(self, dataset, weights, ids, smooth):
        curr_depth=self.nvariables-dataset.shape[1]
        
        
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(dataset) 
            clt.xyprob = np.zeros((1, 1, 2, 2))
            clt.xprob = np.zeros((1, 2))             
            return clt
        
        
        self.xycounts = Util.compute_weighted_xycounts(dataset, weights) + smooth
        self.xcounts = Util.compute_weighted_xcounts(dataset, weights) + 2.0 *smooth
        edgemat = Util.compute_edge_weights(self.xycounts, self.xcounts)
        
        np.fill_diagonal(edgemat, 0)
        
        scores = np.sum(edgemat, axis=0)

        variable = np.argmax(scores)
        
        
        index1 = np.where(dataset[:,variable]==1)[0]
        index0 = np.where(dataset[:,variable]==0)[0]
        
        new_dataset =  np.delete(dataset, variable, axis = 1)
        
        new_dataset1 = new_dataset[index1]
        new_weights1 = weights[index1]
        p1= np.sum(new_weights1)+smooth
                
        new_dataset0 = new_dataset[index0]
        new_weights0 = weights[index0]
        p0 = np.sum(new_weights0)+smooth
        
        # Normalize
        p0 = p0/(p0+p1)
        p1 = 1.0 - p0
        
        
        new_ids=np.delete(ids,variable,0)
        
        return [variable,ids[variable],p0,p1,self.learn_structure_weight(new_dataset0,new_weights0,new_ids, smooth),
                self.learn_structure_weight(new_dataset1,new_weights1, new_ids, smooth)]
    
    
    
    '''
        Update the parameters using weighted data
    '''
    def update_parameter(self, node, dataset, weights, ids, smooth):
        
        if dataset.shape[0] == 0:
            return
        
        # internal nodes, not reach the leaf
        if isinstance(node,list):
            id,x,p0,p1,node0,node1 = node
            index1 = np.where(dataset[:,x]==1)[0]
            index0 = np.where(dataset[:,x]==0)[0]
            
            
            new_weights1 = weights[index1]
            new_weights0 = weights[index0]
            new_dataset1 = dataset[index1]
            new_dataset0 = dataset[index0]
            
            p1 = np.sum(new_weights1) + smooth
            p0 = np.sum(new_weights0) + smooth
            
            # Normalize
            p0 = p0/(p0+p1)
            p1 = 1.0 - p0
            
            
            node[2] = p0
            node[3] = p1
            
            new_ids = np.delete(ids, id)
            
            self.update_parameter(node0, new_dataset0, new_weights0, new_ids, smooth)
            self.update_parameter(node1, new_dataset1, new_weights1, new_ids, smooth)
        
        else:
            clt_dataset = dataset[:, ids]
            node.update_exact(clt_dataset, weights, structure_update_flag = False)
            return

            
            

        

    '''
        Update the CNet using weighted samples, exact update
    '''
    def update_exact(self, dataset, weights, structure_update_flag = False):
        
        if dataset.shape[0] != weights.shape[0]:
            print ('ERROR: weight size not equal to dataset size!!!')
            exit()
        # Perform based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        # try to avoid sum(weights = 0
        smooth = max(np.sum(weights), 1.0) / dataset.shape[0]
        ids = np.arange(dataset.shape[1])
        self.nvariables = dataset.shape[1]
       
        
        if structure_update_flag == True:
            # update the structure as well as parameters
            self.tree = self.learn_structure_weight(dataset, weights, ids, smooth)
        else:
            # only update parameters
            node=self.tree
            self.update_parameter(node, dataset, weights, ids,smooth)
            
    

    '''
        The recursivley part for function getWeights()
    '''   
    def get_prob_each(self, node, samples, row_index, ids, probs):
        
        
        if isinstance(node,list):
            id,x,p0,p1,node0,node1=node
            p0 = p0 / float(p0+p1)
            p1 = 1.0 - p0

            
            index1 = np.where(samples[:,id]==1)[0]
            index0 = np.where(samples[:,id]==0)[0]

            
            row_index1 = row_index[index1]
            row_index0 = row_index[index0]
            
            probs[row_index1] += np.log(p1)
            probs[row_index0] += np.log(p0)

            
            new_samples =  np.delete(samples, id, axis = 1)
            new_samples1 = new_samples[index1]
            new_samples0 = new_samples[index0]
            
            new_ids = np.delete(ids, id)
            
            if new_samples0.shape[0] > 0:
                self.get_prob_each(node0, new_samples0, row_index0, new_ids, probs)
            if new_samples1.shape[0] > 0:
                self.get_prob_each(node1, new_samples1, row_index1, new_ids, probs)
        
        # leaf node
        else:
            clt_prob = node.getWeights (samples)
            probs[row_index] += clt_prob
            
            
    '''
        Recursivley get the LL score for each datapoint
        Much faster than computeLL_each_datapoint
    ''' 
    def getWeights(self, dataset):
        
        probs = np.zeros(dataset.shape[0])
        row_index = np.arange(dataset.shape[0])
        ids=np.arange(self.nvariables)
        node=self.tree
        
        self.get_prob_each(node, dataset, row_index, ids, probs)
        return probs
        
    
    
    '''
        For bags of CNet
    '''
    def learnStructureP_Helper(self,dataset,ids, portion):
        curr_depth=self.nvariables-dataset.shape[1]
        #print ("curr_dept: ", curr_depth)
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(dataset)
            
            return clt
        xycounts = Util.compute_xycounts(dataset) + 1  # laplace correction
        xcounts = Util.compute_xcounts(dataset) + 2  # laplace correction
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_edge_weights(xycounts, xcounts)
        np.fill_diagonal(edgemat, 0) #  #
        
        scores = np.sum(edgemat, axis=0)
        ind_portion = np.random.choice(ids.shape[0], int(ids.shape[0] * portion), replace=False )
        scores_portion = scores[ind_portion]
        
        variable = ind_portion[np.argmax(scores_portion)]

        
        new_dataset1=np.delete(dataset[dataset[:,variable]==1],variable,1)
        p1=float(new_dataset1.shape[0])+1.0
        new_ids=np.delete(ids,variable,0)
        
        new_dataset0 = np.delete(dataset[dataset[:, variable] == 0], variable, 1)
        p0 = float(new_dataset0.shape[0]) +1.0
        
        return [variable,ids[variable],p0,p1,self.learnStructureP_Helper(new_dataset0,new_ids, portion),
                self.learnStructureP_Helper(new_dataset1,new_ids, portion)]
        
    '''
        For bags of CNet, argument 'portion' means the percentage of total variables
        that will be random selected from all varibles to generate the 'OR' node
    '''
    def learnStructure_portion(self, dataset,portion_percent):
        self.nvariables = dataset.shape[1]
        ids=np.arange(self.nvariables)
        self.tree=self.learnStructureP_Helper(dataset,ids, portion_percent)
        
    
    

                
                
                    
            

    
'''
   Main function for Learning Cutset Network from Data by given depth
'''    
def main_cutset():
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    depth = int(sys.argv[6])

    
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    

    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    

    print("Learning Cutset Networks.....")


    cnet = CNET(depth=depth)
    cnet.learnStructure(train_dataset)
    

    train_ll =  np.sum(cnet.getWeights(train_dataset)) / train_dataset.shape[0]
    valid_ll =  np.sum(cnet.getWeights(valid_dataset)) / valid_dataset.shape[0]
    test_ll  =  np.sum(cnet.getWeights(test_dataset))  / test_dataset.shape[0]

    print (train_ll)
    print (valid_ll)
    print (test_ll)
    


'''
   Main function for Learning an optimal Cutset Network from Data bounded by max depth
   Store the learned Cutset Network
'''  
def main_cutset_opt(parms_dict):
    
    print ("----------------------------------------------------")
    print ("Learning Cutset Networks on original data           ")
    print ("----------------------------------------------------")
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    max_depth = int(parms_dict['max_depth']) 
    out_dir = parms_dict['output_dir']

    

    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    

    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    

    train_ll = np.zeros(max_depth)
    valid_ll = np.zeros(max_depth)
    test_ll = np.zeros(max_depth)
    
    best_valid = -np.inf
    best_module = None
    for i in range(1, max_depth+1):
        cnet = CNET(depth=i)
        cnet.learnStructure(train_dataset)
        train_ll[i-1] = np.sum(cnet.getWeights(train_dataset)) / train_dataset.shape[0]
        valid_ll[i-1] = np.sum(cnet.getWeights(valid_dataset)) / valid_dataset.shape[0]
        test_ll[i-1] = np.sum(cnet.getWeights(test_dataset))  / test_dataset.shape[0]
        
        if best_valid < valid_ll[i-1]:
            best_valid = valid_ll[i-1]
            best_module = copy.deepcopy(cnet)
            
    
    print('Train set cnet LL scores')
    for l in xrange(max_depth):
        print (train_ll[l], l+1)
    print()
    
    print('Valid set cnet LL scores')
    for l in xrange(max_depth):
        print (valid_ll[l], l+1)
    print()   
    
    print('test set cnet LL scores')
    for l in xrange(max_depth):
        print (test_ll[l], l+1)
        
    best_ind = np.argmax(valid_ll)
    
    print ()
    print ('Best Validation ll score achived in layer: ', best_ind )    
    print( 'Train set LL score: ', np.sum(best_module.getWeights(train_dataset)) / train_dataset.shape[0])
    print( 'valid set LL score: ', np.sum(best_module.getWeights(valid_dataset)) / valid_dataset.shape[0])
    print( 'test set LL score : ',np.sum(best_module.getWeights(test_dataset)) / test_dataset.shape[0])
    
    main_dict = {}
    utilM.save_cutset(main_dict, best_module.tree, np.arange(train_dataset.shape[1]), ccpt_flag = True)
    np.savez_compressed(out_dir + data_name, module = main_dict)
    

            

            
            
            
            
            
            
            
