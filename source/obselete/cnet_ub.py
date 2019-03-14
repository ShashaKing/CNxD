#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:17:29 2018

@author:

unbalanced Cutset network 
Using KL divergence to check, if a chow-liu tree attached to cnent has a very good
approximation of the TUM, then we will stop going deeper
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
import copy
from Util import *

import utilM
import util_ub

from collections import deque
import time
from CLT_class import CLT
from MIXTURE_CLT import MIXTURE_CLT
import JT

class CNET_unbalace:
    def __init__(self,tree, depth=100):
        self.nvariables=0
        self.depth=depth
        self.tree=tree
       
    #def learnStructureHelper(self,tum, ids, evid_list, next_id = -1, next_weights = np.zeros(2)):
    def learnStructureHelper(self,tum, dataset, ids, lamda,  beta_function, evid_list, data_ind, next_id = -1, next_weights = np.zeros(2)):
        curr_depth=self.nvariables - ids.shape[0]
        print ("curr_depth: ", curr_depth)
        
        if len(evid_list) == 0:    # the first run
            #alpha = 1.0 * lamda
            sub_dataset = dataset
        else:
            #evid_arr = np.asarray(evid_list)
            #print ('evid_arr:', evid_arr)
            #evid_ind = evid_arr[:,0]
            #print ('evid_ind:', evid_ind)
            #evid_value = evid_arr[:,1]
            #print ('evid_value:', evid_value)
            #data_ind = np.where((dataset[:,evid_ind] == evid_value).all(axis=1))[0]
            #print ('data_ind', data_ind)
            if data_ind.shape[0] == 0:
                sub_dataset = np.array([])
                #alpha = 0.0
            else:
                sub_dataset = dataset[data_ind,:][:, ids]
                #print (sub_dataset.shape)            
        alpha = utilM.updata_coef(sub_dataset.shape[0], dataset.shape[0], lamda, beta_function)
        #print ('alpha: ', alpha)
        #if True: 
        if next_id == -1:
            #print ('hhhhhhhhhhh')
            # tum part
            p_xy, p_x = tum.inference_jt(evid_list,ids)
            
            if alpha > 0:
                # dataset part
                xycounts = Util.compute_xycounts(sub_dataset) + 1  # laplace correction
                xcounts = Util.compute_xcounts(sub_dataset) + 2  # laplace correction
                p_xy_d = Util.normalize2d(xycounts)
                #print p_xy
                p_x_d = Util.normalize1d(xcounts)
                #print (p_xy)
                # leaf node
                
                p_xy = alpha * p_xy_d + (1-alpha) * p_xy
                p_x = alpha * p_x_d + (1-alpha) * p_x
            
            
            # compute mutual information score for all pairs of variables
            # weights are multiplied by -1.0 because we compute the minimum spanning tree
            edgemat = Util.compute_MI_prob(p_xy, p_x)
            
            # reset self mutual information to be 0
            np.fill_diagonal(edgemat, 0)
            #for i in xrange(self.nvariables):
                #print (edgemat[i,i])
            
            #print ("edgemat: ", edgemat)
            scores = np.sum(edgemat, axis=0)
            #print (scores)
            variable = np.argmax(scores)   
            
            #variable = 7 ####test
            variable_id = ids[variable] # the index in the original file
            print ('variable_id: ', variable_id)
            
            p1 =p_x[variable,1]
            p0 =p_x[variable,0]
            
            evid_list.append(np.array([variable_id, -1]))   # -1 means not determined yet
            
            
            #-----------------------------------------------------------------
            # Start to generate CLT
            #-----------------------------------------------------------------
            extend_flag =  True
            
            clt_leaf=CLT()
            clt_leaf.learnStructure_MI(edgemat)
            #edgemat = None # Save memory
            clt_leaf.xyprob = p_xy
            clt_leaf.xprob = p_x
            #print ('a')
            clt_leaf.get_log_cond_cpt() 
            #print ('b')
            
            #threshhold = 0.7
            if curr_depth <= self.depth:
                # get edges, the node id attached to each egde should be their original id
                edges_leaf = np.vstack((clt_leaf.topo_order[1:], clt_leaf.parents[clt_leaf.topo_order[1:]])).T
                #print ('edges_leaf: ', edges_leaf)
                # edges_proj convert the edges to the real ids, for tum inference purpose
                edges_proj = np.zeros((edges_leaf.shape[0], edges_leaf.shape[1]), dtype = int)
                edges_proj[:,0] = ids[edges_leaf[:,0]] 
                edges_proj[:,1] = ids[edges_leaf[:,1]]
                # multipy the root node potential to the first edge
                #print ('edges_proj: ')
                #print (edges_proj)
                edge_potential = np.copy(clt_leaf.cond_cpt[1:])
                #print ('edge potential before: ')
                #print (clt_leaf.cond_cpt)
                edge_potential[0] *= clt_leaf.cond_cpt[0].T                
                log_edge_potential = np.log(edge_potential)
                #print ('log edge potential: ')
                #print (log_edge_potential)
                # compute the KL divergence
                kl_value = -tum.get_kl_divergence(edges_proj, log_edge_potential)
                print ('KL value: ', kl_value / ids.shape[0])
                # stop extending
                if kl_value / ids.shape[0] < util_ub.KL_ThreshHold:
                    extend_flag = False
                    
 
                
            
            # attach the generated CLT as leaf node and stop going deeper
            if curr_depth >= self.depth or extend_flag == False:
                # Try to save the memory
                clt_leaf.xyprob = np.zeros((1, 1, 2, 2))       #   0809
                #p_xy = None # save memory

                
                save_info = {}
                #save_info['cond_cpt_list'] =  cond_cpt_list
                #save_info['dataset'] = dataset       #1
                save_info['ids'] = ids           #2
                #save_info['p_xy'] = p_xy          #3
                #save_info['p_x'] = p_x           #4
                save_info['next_id'] = variable_id
                save_info['next_weights'] = np.array([p0,p1])
                #print ('f' , evid_list)
                save_info['evid_list'] = evid_list 
                save_info['data_ind'] = data_ind
                save_info['ex_flg'] = extend_flag
                
                
                clt_leaf.save_info = save_info
                return clt_leaf
        
        else:
            variable_id = next_id
            p0 = next_weights[0]
            p1 = next_weights[1]
            variable = np.where(ids==variable_id)[0][0]
        
        
        
        

        
        evid_list_0 = copy.deepcopy(evid_list) 
        evid_list_1 = copy.deepcopy(evid_list)
        evid_list_0[-1][1] = 0
        evid_list_1[-1][1] = 1
        new_ids=np.delete(ids,variable)
        

        
        if alpha> 0:
            #print ('0+1', data_ind.shape[0])
            new_data_ind0 = data_ind[np.where(sub_dataset[:,variable] ==0)[0]]
            #print ('0:',new_data_ind0.shape[0])
            new_data_ind1 = data_ind[np.where(sub_dataset[:,variable] ==1)[0]]
            #print ('1:',new_data_ind1.shape[0])
        else:
            new_data_ind0 = np.array([])
            new_data_ind1 = np.array([])
        
        new_ids=np.delete(ids,variable)
        
        
        #print ("p0, p1: ", p0, p1)

        
        #return [variable,variable_id,p0,p1,self.learnStructureHelper(tum, new_ids, evid_list_0),
        #        self.learnStructureHelper(tum,  new_ids, evid_list_1)]
        return [variable,variable_id,p0,p1,self.learnStructureHelper(tum, dataset, new_ids, lamda,  beta_function, evid_list_0, new_data_ind0),
                self.learnStructureHelper(tum,dataset, new_ids, lamda, beta_function, evid_list_1, new_data_ind1)]
        
        
        
    def learnStructure(self, tum, dataset, lamda, beta_function):
        self.nvariables = dataset.shape[1]
        ids=np.arange(self.nvariables)
        #total_rec = dataset.shape[0]
    
        
        #cond_cpt_list =[]
        for t in tum.clt_list:
            #cond_cpt = np.exp(t.log_cond_cpt)
            #cond_cpt_list.append(cond_cpt)
            t.nvariables = dataset.shape[1]
            t.get_tree_path()
        
        
        #self.tree=self.learnStructureHelper(tum, cond_cpt_list,  dataset, ids, lamda)
        #cond_cpt_arr = np.asarray(cond_cpt_list)
        #print (cond_cpt)
        # First time learn
        
        if len(self.tree) == 0:
            #self.tree=self.learnStructureHelper(tum,  ids, [])
            self.tree=self.learnStructureHelper(tum, dataset, ids, lamda, beta_function, [],np.arange(dataset.shape[0]))
        else:
            #print ('****in SAVE MODE****')
            node=self.tree
            nodes_to_process = deque()
            nodes_to_process.append(node)
            while nodes_to_process:
                curr_node = nodes_to_process.popleft()
                id,x,p0,p1,node0,node1=curr_node
                if isinstance(node0,list):
                    nodes_to_process.append(node0)
                # Chow-liu tree leaf node   
                else:
                    #print ("leaf")
                    #node0 = self.learnStructureHelper(tum, node0.save_info['cond_cpt_list'],  node0.save_info['ids'],  lamda, node0.save_info['p_xy'], node0.save_info['p_x'])
                    #node0 = self.learnStructureHelper(tum, node0.save_info['cond_cpt_list'], node0.save_info['dataset'], node0.save_info['ids'], lamda, total_rec, beta_function, node0.xyprob, node0.xprob)
                    #new_node0 = self.learnStructureHelper(tum,  node0.save_info['ids'],  node0.save_info['evid_list'], node0.save_info['next_id'], node0.save_info['next_weights'])
                    if node0.save_info['ex_flg'] == True: # need to extend
                        new_node0 = self.learnStructureHelper(tum,dataset, 
                                    node0.save_info['ids'], lamda, beta_function, 
                                    node0.save_info['evid_list'], node0.save_info['data_ind'],
                                    node0.save_info['next_id'], 
                                    node0.save_info['next_weights'])
                        node0 = None # save the memory
                        curr_node[4] = new_node0

                if isinstance(node1,list):
                    nodes_to_process.append(node1)
                else:
                    #print ("leaf")
                    #node1 = self.learnStructureHelper(tum, node1.save_info['cond_cpt_list'], node1.save_info['ids'],  lamda, node1.save_info['p_xy'], node1.save_info['p_x'])
                    #node1 = self.learnStructureHelper(tum, node1.save_info['cond_cpt_list'], node1.save_info['dataset'], node1.save_info['ids'], lamda, total_rec, beta_function, node1.xyprob, node1.xprob)
                    if node1.save_info['ex_flg'] == True: # need to extend
                        new_node1 = self.learnStructureHelper(tum, dataset, 
                                    node1.save_info['ids'], lamda,  beta_function, 
                                    node1.save_info['evid_list'], node1.save_info['data_ind'],
                                    node1.save_info['next_id'], 
                                    node1.save_info['next_weights'])
                        node1 = None
                        curr_node[5] = new_node1
            self.tree = node
        
        
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
                    prob+=np.log(p1)
                    node=node1
                else:
                    prob+=np.log(p0)
                    node = node0
            prob+=node.computeLL(dataset[i:i+1,ids])
        return prob
    
    # computer the log likelihood score for each datapoint in the dataset
    # returns a numpy array
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
                    prob+=np.log(p1)
                    node=node1
                else:
                    prob+=np.log(p0)
                    node = node0
            prob+=node.computeLL(dataset[i:i+1,ids])
            probs[i] = prob
        return probs
    
    def update(self,node, ids, dataset, lamda):
        
        #node=self.tree
        #ids=np.arange(self.nvariables)
        
        if isinstance(node,list):
            id,x,p0,p1,node0,node1=node
            p0_index=2
            p1_index=3
            
            
            new_dataset1=np.delete(dataset[dataset[:,id]==1],id,1)        
            new_dataset0 = np.delete(dataset[dataset[:, id] == 0], id, 1)
            
            new_p1 = (float(new_dataset1.shape[0]) + 1.0) / (dataset.shape[0] + 2.0) # smoothing 
            new_p0 = 1 - new_p1
            
            node[p0_index] = (1-lamda) * p0 + lamda * new_p0
            node[p1_index] = (1-lamda) * p1 + lamda * new_p1
            
            
            
            ids=np.delete(ids,id,0)
     
            self.update (node0, ids, new_dataset0, lamda)
            self.update (node1, ids, new_dataset1, lamda)                
        else:
            return 
    
    
    # using weighted sample generated by Baysien Net work to update
    # The last column of dataset is the weight
    def update_by_BN(self,node, ids, dataset, lamda):
        
        #node=self.tree
        #ids=np.arange(self.nvariables)
        
        if isinstance(node,list):
            id,x,p0,p1,node0,node1=node
            p0_index=2
            p1_index=3
            
            
            new_dataset1=np.delete(dataset[dataset[:,id]==1],id,1)        
            new_dataset0 = np.delete(dataset[dataset[:, id] == 0], id, 1)
            
            weight0 = np.sum(new_dataset0[:,-1])
            weight1 = np.sum(new_dataset1[:,-1])
            
            #print ("weight0, weight1 : ", weight0, weight1)
            
            new_p1 = float(weight0) / (weight0 + weight1)
            new_p0 = 1 - new_p1
            
            node[p0_index] = (1-lamda) * p0 + lamda * new_p0
            node[p1_index] = (1-lamda) * p1 + lamda * new_p1
            
            
            
            ids=np.delete(ids,id,0)
     
            self.update_by_BN (node0, ids, new_dataset0, lamda)
            self.update_by_BN (node1, ids, new_dataset1, lamda)                
        else:
            return 
        
        """
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
        """
        
        


    
def main_cutset_unbalace():
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    lamda = float(sys.argv[6])  # using validation dataset 
    beta_function = sys.argv[8]  # 'linear', square, root (square root)
    min_depth = int(sys.argv[10])
    max_depth = int(sys.argv[12])
    util_ub.KL_ThreshHold = float(sys.argv[14])
    
    
    
    print('------------------------------------------------------------------')
    print('Results using Data and TUM')
    print('------------------------------------------------------------------')
    
    
    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    out_file = '../module/' + data_name + '.npz'
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    #print ("********* Using Validation Dataset in Training ************")
    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    #print("Learning Chow-Liu Trees on original data ......")
    #clt = CLT()
    #clt.learnStructure(train_dataset)
    #print("done")
    #print ('total recrod: ', train_dataset.shape[0])
    
    n_variables = train_dataset.shape[1]

    ### Load the trained mixture of clt
    print ('Start reloading ...')
    

    reload_dict = np.load(out_file)
    
    #print ("mixture weights: ", reload_dict['weights'])
    
    reload_mix_clt = MIXTURE_CLT()
    reload_mix_clt.mixture_weight = reload_dict['weights']
    reload_mix_clt.n_components = reload_mix_clt.mixture_weight.shape[0]
    
    reload_clt_component = reload_dict['clt_component']
    
    #print (reload_clt_component)
    for i in xrange(reload_mix_clt.n_components):
        clt_c = CLT()
        #str_id = str(i)
        curr_component = reload_clt_component[i]
        clt_c.xyprob = curr_component['xyprob']
        clt_c.xprob = curr_component['xprob']
        clt_c.topo_order = curr_component['topo_order']
        clt_c.parents = curr_component['parents']
        clt_c.log_cond_cpt = curr_component['log_cond_cpt']
        
        reload_mix_clt.clt_list.append(clt_c)
    
    # Set information for MT
    for t in reload_mix_clt.clt_list:

        t.nvariables = n_variables
        #t.get_tree_path()
        # learn the junction tree for each clt
        jt = JT.JunctionTree()
        jt.learn_structure(t.topo_order, t.parents, t.cond_cpt)
        reload_mix_clt.jt_list.append(jt)
    
    #lamda_array = np.arange(11) / 10.0
    #lamda_array = [0.9]
    #best_cnet = None
    #best_ll = -np.inf
    print("Learning Cutset Networks by inferece.....")
    #for lamda in lamda_array:
    print ("Current Lamda: ", lamda)
    print ("Current Function: ", beta_function)
    #n_variable = valid_dataset.shape[1]
    #cnets = []
    tree = []
    #max_depth = 11
    output_cnet = '../cnet_ub_module/'
    output_dir = '../cnet_ub_results/' + data_name + '/'
    valid_out_file = output_dir +  'valid_'  + str(lamda) + '_' + beta_function + '.txt'
    test_out_file = output_dir + 'test_' +  str(lamda) + '_' + beta_function + '.txt'
    
    train_ll_score = np.zeros(max_depth)
    valid_ll_score = np.zeros(max_depth)
    test_ll_score = np.zeros(max_depth)
    for i in range(min_depth, max_depth):
    #for i in range(10, 20):
        cnet  = CNET_unbalace(tree, depth=i)       
        cnet.learnStructure(reload_mix_clt, train_dataset, lamda, beta_function)         
        #cnets.append(cnet)        
        tree = copy.deepcopy(cnet.tree)
        
        # compute ll score
        train_ll_score[i-1] = cnet.computeLL(train_dataset) / train_dataset.shape[0]
        valid_ll_score[i-1] = cnet.computeLL(valid_dataset) / valid_dataset.shape[0]
        test_ll_score[i-1] = cnet.computeLL(test_dataset) / test_dataset.shape[0]
        
        with open(valid_out_file, 'w') as f_handle:
            np.savetxt(f_handle, valid_ll_score[:i], delimiter=',')
            
        with open(test_out_file, 'w') as f_handle:
            np.savetxt(f_handle, test_ll_score[:i], delimiter=',')
        #print ('Layer: ', i)
        #print (train_ll_score[i-1])
        #print (valid_ll_score[i-1])
        #print (test_ll_score[i-1])
        #print ()
        
        # save cnet module to file        
        #print ("save module: ", i)
        main_dict = {}
        utilM.save_cutset(main_dict, cnet.tree, np.arange(n_variables), ccpt_flag = True)
        #np.save(output_cnet + data_name + '_' + str(i), main_dict)
        np.savez_compressed(output_cnet + data_name + '_' + str(lamda) + '_'  + beta_function + '_'  + str(i), module = main_dict)
    
        """
        save_dict = np.load('../plots/' + data_name + '.npy').item()
        save_dict['tum'] = test_ll_score
        np.save('../plots/' + data_name, save_dict)
        """
        
        #alpha_dict = np.load('../alpha/' + data_name + '.npy').item()
        #alpha_dict[str(lamda)] = valid_ll_score
        #np.save('../alpha/' + data_name, alpha_dict)
    print("done")
    
    #print (train_ll_score)
    print('Train set cnet LL scores')
    for l in xrange(max_depth-1):
        print (train_ll_score[l], l+1)
        
    print('valid set cnet LL scores')
    for l in xrange(max_depth-1):
        print (valid_ll_score[l], l+1)
        
    
    print('Test set cnet LL scores')
    for l in xrange(max_depth-1):
        print (test_ll_score[l], l+1)
    
    #alpha_dict = np.load('../alpha/' + data_name + '.npy').item()
    #alpha_dict[str(lamda)] = valid_ll_score
    #np.save('../alpha/' + data_name, alpha_dict)
    
    


if __name__=="__main__":
    #main_cutset()
    #main_clt()
    start = time.time()
    #main_cutset_clt()
    main_cutset_unbalace()
    print ('Total running time: ', time.time() - start)