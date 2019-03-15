"""
Learn a very deep cutset network
The structure is random, only parameters are learnt
"""

from __future__ import print_function
import numpy as np
import sys
import copy
from Util import *

import utilM

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

from collections import deque
import time
from random import randint
from CLT_class import CLT
from MIXTURE_CLT import MIXTURE_CLT, load_mt


'''
    The structure of the tree is randomly built
    The parameter of the tree is learnt
'''
class rand_tree():
    
    def __init__(self):
        self.nvariables = 0
        self.topo_order = []
        self.parents = []
        self.log_cond_cpt = []  
        self.cond_cpt = []
        self.xprob = []
        self.save_info = None
        self.ids = None  # the real node id in original dataset
        self.edge_prob_t=[] # the edge probablities from TUM
        self.edge_prob_d=[] # the edge probablities from data
    
    
    '''
        Learn a random structure
    '''
    def learnStructure(self, n_variable, ids):

        self.nvariables = n_variable
        self.ids = ids
        edgemat = np.random.rand(self.nvariables, self.nvariables)
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)

    '''
        learning parameter only from TUM
    '''
    def learnParm(self, tum, evid_list):
        # pairwised egde CPT in log format based on tree structure

        edges = np.vstack((self.topo_order[1:], self.parents[self.topo_order[1:]])).T
        # edges_proj convert the edges to the real ids, for tum inference purpose
        edges_proj = np.zeros((edges.shape[0], edges.shape[1]))
        edges_proj[:,0] = self.ids[edges[:,0]] 
        edges_proj[:,1] = self.ids[edges[:,1]]
        edge_prob = tum.get_edge_marginal(evid_list, edges_proj)

        
        # get node marginals, indexing from 0 to nvaribles-1
        node_prob = np.zeros((self.nvariables, 2))
        node_prob[self.topo_order[1:],0] = edge_prob[:,0,0] + edge_prob[:,0,1]
        node_prob[self.topo_order[1:],1] = edge_prob[:,1,0] + edge_prob[:,1,1]
        # for root node
        node_prob[0,0] = edge_prob[0,0,0] + edge_prob[0,1,0]
        node_prob[0,1] = edge_prob[0,0,1] + edge_prob[0,1,1]

        
        # compute conditional cpt
        self.cond_cpt = np.zeros((self.nvariables,2,2))
        self.cond_cpt[0, 0, :] = node_prob[0, 0]
        self.cond_cpt[0, 1, :] = node_prob[0, 1]
        
        
        self.cond_cpt[1:,0, :] = edge_prob[:,0,:] / node_prob[edges[:,1], :]
        self.cond_cpt[1:,1, :] = edge_prob[:,1,:] / node_prob[edges[:,1], :]
        
        # convert nan to 0 if exist
        self.cond_cpt = np.nan_to_num(self.cond_cpt)
        
        self.log_cond_cpt  = np.log(self.cond_cpt)
        

    '''
        learning parameter from TUM and dataset
    '''
    def learnParm_DT(self, tum, dataset, evid_list, ids):
        self.ids = ids
        self.nvariables = self.ids.shape[0]
    
        edges = np.vstack((self.topo_order[1:], self.parents[self.topo_order[1:]])).T
        # edges_proj convert the edges to the real ids, for tum inference purpose
        edges_proj = np.zeros((edges.shape[0], edges.shape[1]))
        edges_proj[:,0] = self.ids[edges[:,0]] 
        edges_proj[:,1] = self.ids[edges[:,1]]
        self.edge_prob_t = tum.get_edge_marginal(evid_list, edges_proj)
        
        if dataset.shape[0] > 0:
            edge_xy_counts = Util.compute_xycounts_edges(dataset, edges) + 1  # laplace correction
            self.edge_prob_d = Util.normalize1d_in_2d(edge_xy_counts)
        else:
            self.edge_prob_d = np.zeros((edges.shape[0],2,2))
        

        

    '''
        Compute the Log-likelihood score of the dataset
    '''  
    def computeLL(self,dataset):
        return utilM.get_tree_dataset_ll(dataset, self.topo_order, self.parents, self.log_cond_cpt)
        


'''
    Learn a very deep cutset network
'''
class CNET_deep:
    def __init__(self,tree, depth=100):
        self.nvariables=0
        self.depth=depth 
        self.tree=tree
       

    def learnStructureHelper(self,ids):
        
        
        curr_nvariables = ids.shape[0]
        curr_depth=self.nvariables - curr_nvariables

        
        # creat a leaf node (random tree)
        if curr_depth >= self.depth:
            rt_leaf = rand_tree()
            rt_leaf.learnStructure(curr_nvariables, ids) #structure is randomly assigned
            

            
            save_info = {}
            save_info['ids'] = ids           
            rt_leaf.save_info = save_info
            return rt_leaf
        
        # randomly pick a 'OR' node
        variable = randint(0, curr_nvariables-1)                    
        variable_id = ids[variable] # the index in the original file
        
        
        p0 = p1 = 0.5  # initial to 0.5
        
        new_ids=np.delete(ids,variable)
        
        
        return [variable,variable_id,p0,p1,self.learnStructureHelper(new_ids),
                self.learnStructureHelper(new_ids)]

        
        
        
    def learnStructure(self, nvariables):
        self.nvariables = nvariables
        ids=np.arange(self.nvariables)
    
        # First time learn        
        if len(self.tree) == 0:
            self.tree=self.learnStructureHelper(ids)
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
                    new_node0 = self.learnStructureHelper(node0.save_info['ids'])
                    curr_node[4] = new_node0

                if isinstance(node1,list):
                    nodes_to_process.append(node1)
                else:
                    new_node1 = self.learnStructureHelper(node1.save_info['ids'])
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
    
    '''
        computer the log likelihood score for each datapoint in the dataset
        returns a numpy array
    '''
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
    This function is used when we fixed the structure of the cnet, only try to
    update the paramters
'''    

def learnParm(cnet,tum, dataset, evid_list, ids):
    # n_record is a constant, the number of records of the original dataset  
    if cnet['type'] == 'internal':
        vid = cnet['id']
        x  = cnet['x']
        c0 = cnet['c0']
        c1 = cnet['c1']
        
        
        cnode_marginal = tum.get_node_marginal(evid_list, x)
        # No trainning data is available      
        
        data_marginal = np.zeros(2) # the marginals from data
        if dataset.shape[0] == 0:
            #print ('cnode_marginal: ', cnode_marginal)                        
            new_dataset0 = np.asarray([])
            new_dataset1 = np.asarray([])
        else:    
            new_dataset1 = np.delete(dataset[dataset[:,vid] == 1], vid, 1)
            new_dataset0 = np.delete(dataset[dataset[:,vid] == 0], vid, 1)
            data_marginal[0] = new_dataset0.shape[0] / float(dataset.shape[0])
            data_marginal[1] = 1-data_marginal[0]
            
        cnet['p_t'] = cnode_marginal  # the marginals from TUM
        cnet['p_d'] = data_marginal  # the marginals from TUM
        cnet['n_records'] = dataset.shape[0]
        
        evid_list_0 = copy.copy(evid_list) 
        evid_list_1 = copy.copy(evid_list)
        evid_list_0.append(np.array([x, 0]))
        evid_list_1.append(np.array([x, 1]))
        new_ids=np.delete(ids,vid)
        
        
        learnParm(c0,tum, new_dataset0, evid_list_0, new_ids)
        learnParm(c1,tum, new_dataset1, evid_list_1, new_ids)
    
    # reach the leaf clt
    elif cnet['type'] == 'leaf':
        leaf_tree = rand_tree()
        leaf_tree.topo_order = cnet['topo_order'] 
        leaf_tree.parents = cnet['parents']
        leaf_tree.learnParm_DT(tum, dataset, evid_list, ids)
        
        cnet['ids'] = leaf_tree.ids
        cnet['edge_prob_t'] = leaf_tree.edge_prob_t
        cnet['edge_prob_d'] = leaf_tree.edge_prob_d
        cnet['n_records'] = dataset.shape[0]


    else:
        print ("*****ERROR: invalid node type******")
        exit()



"""
    The function for buliding a skeleton of CNR
    1) randomly build the structure
    2) parameters are just uniform distribution
"""
def main_cnr_structure(parms_dict):
    
    print('------------------------------------------------------------------')
    print('Learning the structure of Deep Random Cutset Network')
    print('------------------------------------------------------------------')
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    min_depth = int(parms_dict['min_depth'])
    max_depth = int(parms_dict['max_depth'])
    output_dir = parms_dict['output_dir']
    
    train_filename = dataset_dir + data_name + '.ts.data'
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    
    n_variables = train_dataset.shape[1]

    max_depth = min (n_variables-2, max_depth)  # at least 2 nodes in the leaf to bulid the chow-liu tree
    
    tree = []    
    
    for i in range(min_depth, max_depth+1):
        cnet  = CNET_deep(tree, depth=i)       
        cnet.learnStructure(n_variables)
        tree = copy.deepcopy(cnet.tree)

        main_dict = {}
        utilM.save_cutset(main_dict, cnet.tree, np.arange(n_variables), ccpt_flag = True)
        np.savez_compressed(output_dir + data_name + '_structure_'  + str(i), module = main_dict)
    


"""
    The function for learning paramters of CNR
    1) store the marginals from both DATA and MAP intractable module
"""
    
def cnr_learn_parm(train_dataset, data_name, tum_dir, tum_name, depth, module_dir):
    
    
    print ('-----Learning parameters from GIVEN structure----')

    # load the structure of the cnet
    input_module = module_dir + data_name + '_structure_' + str(depth) + '.npz'
    # reload the structure
    cnet_module = np.load(input_module)['module'].item()

    
    # load mt
    reload_mix_clt = load_mt(tum_dir, tum_name)
    
    tum = reload_mix_clt
    learnParm(cnet_module,tum, train_dataset, [], np.arange(train_dataset.shape[1]))
    
    # save the module
    np.savez_compressed(module_dir + data_name + '_parm_' + str(depth), module = cnet_module)

    

"""
    This function is use to combine vairious option of lamda and beta_function 
    to find the paramter that achieves the best validation set LL score
"""
def getExactParm(cnet,total_records,lamda, beta_function):
    # n_record is a constant, the number of records of the original dataset 
    queue = deque()
    queue.append(cnet)
    
    while (len(queue) > 0):
        curr_cnet = queue.popleft()
        alpha = utilM.updata_coef(curr_cnet['n_records'], total_records, lamda, beta_function)
        if curr_cnet['type'] == 'internal':
            c0 = curr_cnet['c0']
            c1 = curr_cnet['c1']
            p_t = curr_cnet['p_t']   # the marginals from TUM
            p_d = curr_cnet['p_d']   # the marginals from TUM
                    
            curr_cnet['p0'] = alpha * p_d[0] + (1-alpha) * p_t[0]
            curr_cnet['p1'] = alpha * p_d[1] + (1-alpha) * p_t[1]
            
            queue.append(c0)
            queue.append(c1)
            
        # reach the leaf clt
        elif curr_cnet['type'] == 'leaf':
            # get node marginals, indexing from 0 to nvaribles-1
            ids = curr_cnet['ids']
            edge_prob_t = curr_cnet['edge_prob_t']
            edge_prob_d = curr_cnet['edge_prob_d']
            topo_order = curr_cnet['topo_order']
            parents = curr_cnet['parents']
            
            edges = np.vstack((topo_order[1:], parents[topo_order[1:]])).T            
            edge_prob = alpha * edge_prob_d + (1-alpha) * edge_prob_t
            
            node_prob = np.zeros((ids.shape[0], 2))
            node_prob[topo_order[1:],0] = edge_prob[:,0,0] + edge_prob[:,0,1]
            node_prob[topo_order[1:],1] = edge_prob[:,1,0] + edge_prob[:,1,1]
            # for root node
            node_prob[0,0] = edge_prob[0,0,0] + edge_prob[0,1,0]
            node_prob[0,1] = edge_prob[0,0,1] + edge_prob[0,1,1]

            
            # compute conditional cpt
            cond_cpt = np.zeros((ids.shape[0],2,2))
            cond_cpt[0, 0, :] = node_prob[0, 0]
            cond_cpt[0, 1, :] = node_prob[0, 1]
            

            cond_cpt[1:,0, :] = edge_prob[:,0,:] / node_prob[edges[:,1], :]
            cond_cpt[1:,1, :] = edge_prob[:,1,:] / node_prob[edges[:,1], :]
            
            # convert nan to 0 if exist
            cond_cpt = np.nan_to_num(cond_cpt)
            
            log_cond_cpt  = np.log(cond_cpt)
            curr_cnet['log_cond_cpt'] = log_cond_cpt
            
    
        else:
            print ("*****ERROR: invalid node type******")
            exit()


"""
    This function is use to tune the lamda and beta_function to find the 
    best Cutset Network under current structure
"""
def cnr_tune_paramter(train_dataset, valid_dataset, test_dataset, data_name, depth, module_dir):
    
    
    print ('-----Tuning Paramters----')
        
    # load the structure of the cnet
    input_module = module_dir + data_name + '_parm_' + str(depth) + '.npz'
    
    # reload the structure
    cnet_load = np.load(input_module)['module'].item()
    
    functions = ['linear', 'square', 'root'] # currently support
    lamda = np.arange(11) / 10.0
    
    # the module has highest validation ll score
    best_func_val = ''
    best_lam_val = 0.0
    best_module_val = None
    best_ll_val = -np.inf
    ll_results = np.zeros(3)
    
    for func in functions:
        for lam in lamda:
            cnet_module = copy.deepcopy(cnet_load)
            getExactParm(cnet_module,train_dataset.shape[0],lam, func)

            # Get LL score
            train_ll = np.sum(utilM.computeLL_reload(cnet_module, train_dataset)) /  train_dataset.shape[0]
            valid_ll = np.sum(utilM.computeLL_reload(cnet_module, valid_dataset)) /valid_dataset.shape[0]
            test_ll = np.sum(utilM.computeLL_reload(cnet_module, test_dataset)) / test_dataset.shape[0]
            

            if valid_ll > best_ll_val:
                best_ll_val = valid_ll
                best_func_val =  func
                best_lam_val = lam
                best_module_val = copy.deepcopy(cnet_module)
                ll_results[0] = train_ll
                ll_results[1] = valid_ll
                ll_results[2] = test_ll
                
    #print ('Best function: ', best_func_val)    
    #print ('Best Lamda: ', best_lam_val)     
    print ('Train LL score for CNR: ', ll_results[0])          
    print ('Valid LL score for CNR: ', ll_results[1])      
    print ('Test LL score for CNR : ', ll_results[2])              
    
    # save the module
    np.savez_compressed(module_dir + data_name +'_'+str(depth), module = best_module_val)



"""
    The Main function for Random Deep CNet
"""
def main_cnr_parm(parms_dict):
    
    print('------------------------------------------------------------------')
    print('Learning Parameters of a Deep Random Cutset Network')
    print('------------------------------------------------------------------')
    
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    depth = int(parms_dict['depth'])
    tum_dir =  parms_dict['input_dir']
    tum_name = parms_dict['input_module']
    output_dir = parms_dict['output_dir']
   
    
    # load the dataset
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    

    
    cnr_learn_parm(train_dataset, data_name, tum_dir, tum_name, depth, output_dir)
    cnr_tune_paramter(train_dataset, valid_dataset, test_dataset, data_name, depth, output_dir)
    





