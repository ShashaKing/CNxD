"""
# Using alpha and function to combine data and tum (MAP intractable model)
# Instead of VE, using junction tree to calculate pairwise marginal
"""

from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
from collections import deque
import time

from Util import *
import utilM

from CLT_class import CLT
from MIXTURE_CLT import MIXTURE_CLT, load_mt
import JT



# The mutual information is got from inference of TUM (mixture of CLT)
class CNXD:
    def __init__(self,tree, depth=100):
        self.nvariables=0
        self.depth=depth
        self.tree=tree
       

    def learnStructureHelper(self,tum, dataset, ids, lamda,  beta_function, evid_list, data_ind, next_id = -1, next_weights = np.zeros(2)):
        
        curr_depth=self.nvariables - ids.shape[0]
        
        if len(evid_list) == 0:    # the first run
            sub_dataset = dataset
        else:

            if data_ind.shape[0] == 0:
                sub_dataset = np.array([])
            else:
                sub_dataset = dataset[data_ind,:][:, ids]          
        alpha = utilM.updata_coef(sub_dataset.shape[0], dataset.shape[0], lamda, beta_function)
        #if True: 
        if next_id == -1:
            # tum part
            p_xy, p_x = tum.inference_jt(evid_list,ids)
            
            if alpha > 0:
                # dataset part
                xycounts = Util.compute_xycounts(sub_dataset) + 1  # laplace correction
                xcounts = Util.compute_xcounts(sub_dataset) + 2  # laplace correction
                p_xy_d = Util.normalize2d(xycounts)
                p_x_d = Util.normalize1d(xcounts)

                p_xy = alpha * p_xy_d + (1-alpha) * p_xy
                p_x = alpha * p_x_d + (1-alpha) * p_x
            
            
            # compute mutual information score for all pairs of variables
            # weights are multiplied by -1.0 because we compute the minimum spanning tree
            edgemat = Util.compute_MI_prob(p_xy, p_x)
            
            # reset self mutual information to be 0
            np.fill_diagonal(edgemat, 0)

            scores = np.sum(edgemat, axis=0)
            variable = np.argmax(scores)   
            
            variable_id = ids[variable] # the index in the original file
            
            p1 =p_x[variable,1]
            p0 =p_x[variable,0]
            
            evid_list.append(np.array([variable_id, -1]))   # -1 means not determined yet
        
            if curr_depth >= self.depth:
                clt_leaf=CLT()
                clt_leaf.learnStructure_MI(edgemat)
                clt_leaf.xyprob = p_xy
                clt_leaf.xprob = p_x
                clt_leaf.get_log_cond_cpt()
                # Try to save the memory
                clt_leaf.xyprob = np.zeros((1, 1, 2, 2))

                
                save_info = {}
                save_info['ids'] = ids
                save_info['next_id'] = variable_id
                save_info['next_weights'] = np.array([p0,p1])
                save_info['evid_list'] = evid_list 
                save_info['data_ind'] = data_ind 
                
                
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
            
            new_data_ind0 = data_ind[np.where(sub_dataset[:,variable] ==0)[0]]
            new_data_ind1 = data_ind[np.where(sub_dataset[:,variable] ==1)[0]]
        else:
            new_data_ind0 = np.array([])
            new_data_ind1 = np.array([])
        
        new_ids=np.delete(ids,variable)
        
        
    
        return [variable,variable_id,p0,p1,self.learnStructureHelper(tum, dataset, new_ids, lamda,  beta_function, evid_list_0, new_data_ind0),
                self.learnStructureHelper(tum,dataset, new_ids, lamda, beta_function, evid_list_1, new_data_ind1)]
        
        
        
    def learnStructure(self, tum, dataset, lamda, beta_function):
        self.nvariables = dataset.shape[1]
        ids=np.arange(self.nvariables)

        # First time learn       
        if len(self.tree) == 0:
            self.tree=self.learnStructureHelper(tum, dataset, ids, lamda, beta_function, [],np.arange(dataset.shape[0]))
        else:
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
    
    """
        computer the log likelihood score for each datapoint in the dataset
        returns a numpy array
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
                    prob+=np.log(p1)
                    node=node1
                else:
                    prob+=np.log(p0)
                    node = node0
            prob+=node.computeLL(dataset[i:i+1,ids])
            probs[i] = prob
        return probs
    

    
    '''
        using weighted sample generated by Baysien Net work to update
        The last column of dataset is the weight
    '''
    def update_by_BN(self,node, ids, dataset, lamda):
        
        
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
        
        
        


    
def main_cnxd(parms_dict):
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    lamda = 1.0-float(parms_dict['a'])
    beta_function = parms_dict['f']
    min_depth = int(parms_dict['min_depth'])
    max_depth = int(parms_dict['max_depth']) 
    mt_dir =  parms_dict['input_dir']
    tum_module = parms_dict['input_module']
    module_dir = parms_dict['output_dir']
    
    
    print('------------------------------------------------------------------')
    print('Learning CNxD using Data and MAP Intractable Model')
    print('------------------------------------------------------------------')
    
    
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')

    
    n_variables = train_dataset.shape[1]

    ### Load the trained mixture of clt
    print ('Start reloading MT...')
    reload_mix_clt = load_mt(mt_dir, tum_module)


     # Set information for MT
    for t in reload_mix_clt.clt_list:

        t.nvariables = n_variables
        # learn the junction tree for each clt
        jt = JT.JunctionTree()
        jt.learn_structure(t.topo_order, t.parents, t.cond_cpt)
        reload_mix_clt.jt_list.append(jt)
        

    print ("Current Alpha: ", lamda)
    print ("Current Function: ", beta_function)
    
    tree = []

    #module_dir = '../cnxd_output/' + data_name +'/'
    
    train_ll_score = np.zeros(max_depth)
    valid_ll_score = np.zeros(max_depth)
    test_ll_score = np.zeros(max_depth)
    learning_time = np.zeros(max_depth)
    for i in range(min_depth, max_depth+1):
        start = time.time()
        cnet  = CNXD(tree, depth=i)       
        cnet.learnStructure(reload_mix_clt, train_dataset, lamda, beta_function)  
        learning_time[i-1] = time.time() - start      
        tree = copy.deepcopy(cnet.tree)
        
        # compute ll score
        train_ll_score[i-1] = cnet.computeLL(train_dataset) / train_dataset.shape[0]
        valid_ll_score[i-1] = cnet.computeLL(valid_dataset) / valid_dataset.shape[0]
        test_ll_score[i-1] = cnet.computeLL(test_dataset) / test_dataset.shape[0]
        
        
        main_dict = {}
        utilM.save_cutset(main_dict, cnet.tree, np.arange(n_variables), ccpt_flag = True)
        np.savez_compressed(module_dir + data_name + '_' + str(lamda) + '_'  + beta_function + '_'  + str(i), module = main_dict)
        
    
    print('CNxD train set LL scores')
    for l in xrange(max_depth):
        print (train_ll_score[l], l+1)
    print()
        
    print('CNxD valid set LL scores')
    for l in xrange(max_depth):
        print (valid_ll_score[l], l+1)
    print()   
    
    print('CNxD test set LL scores')
    for l in xrange(max_depth):
        print (test_ll_score[l], l+1)
    print()
    
    print ('CNxD learning times: ')
    for l in xrange(max_depth):
        print (np.sum(learning_time[0:l+1]), l+1)
    print()
        
        
        
    


