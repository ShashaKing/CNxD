# All functions are the same as inference.py, the only difference is this version use previously
# generated evidence file instead of generate evidence file itself

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

#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import minimum_spanning_tree
#from scipy.sparse.csgraph import depth_first_order

from collections import deque
import time
import pickle



class CLT:
    def __init__(self):
        self.nvariables = 0
        self.xprob = np.zeros((1, 2))
        self.topo_order = np.array([])
        self.parents = np.array([])
        self.log_cond_cpt = np.array([])  
        self.log_value = 0.0
        self.ids = np.array([])
        self.as_child =np.array([])   # indices as child based on topo_order for evidences
        self.as_parent =[]   # indices as as_parent based on topo_order  for evidences
        self.map_tuple = []
        
    
        
    
    
    def getWeights(self, samples):
        
        
        probs = utilM.get_sample_ll(samples,self.topo_order, self.parents, self.log_cond_cpt)
        return probs
     
    
    
    # set the evidence
    def instantiation(self, evid_list):
        #print ('in instantiation')
        #print (evid_list)
        

        cond_cpt = np.exp(self.log_cond_cpt)
        #print ("before:")
        #print (cond_cpt)
        for i in xrange (len(evid_list)):
            variable_id = evid_list[i][0]
            value = evid_list[i][1]
            
            index_c = np.where(self.topo_order==variable_id)[0][0]
            # variable as parent
            varible_child = np.where(self.parents ==variable_id)[0]
            ix = np.isin(self.topo_order, varible_child)
            index_p = np.where(ix)[0]
            #print (index_p)
            
            # set varible value = 0
            if value == 0:                    
                cond_cpt[index_c, 1,:] = 0
                cond_cpt[index_p, :,1] = 0
            
            # set varible value = 1                   
            elif value == 1:                    
                cond_cpt[index_c, 0,:] = 0
                cond_cpt[index_p, :,0] = 0

            else:
                print ('error in value: ', value)
                exit()
            #print ("after: ")
            #print (cond_cpt)
            
        return cond_cpt
    
    def instantiation_log(self, e_value):
        
        log_cond_cpt = np.copy(self.log_cond_cpt)
        for i, value in enumerate(e_value):
            
            index_c = self.as_child[i]
            index_p = self.as_parent[i]
        
            # set varible value = 0
            if value == 0:                    
                log_cond_cpt[index_c, 1,:] = utilM.LOG_ZERO
                log_cond_cpt[index_p, :,1] = utilM.LOG_ZERO
            
            # set varible value = 1                   
            elif value == 1:                    
                log_cond_cpt[index_c, 0,:] = utilM.LOG_ZERO
                log_cond_cpt[index_p, :,0] = utilM.LOG_ZERO
    
            else:
                print ('error in value: ', value)
                exit()
            #print ("after: ")
            #print (cond_cpt)
        
        return log_cond_cpt
        
    def inference(self, cond_cpt, ids):
        
        #return utilM.get_prob_matrix(self.topo_order, self.parents, cond_cpt, ids)
        return utilM.get_prob_matrix(self.topo_order, self.parents, cond_cpt, ids, self.tree_path)
    
    def ind_as_parent(self, e_var):
        
        for i, e in enumerate(e_var):
        
            # evidence variable as parent
            e_child = np.where(self.parents ==e)[0]
            ix = np.isin(self.topo_order, e_child)
            index_p = np.where(ix)[0]

            self.as_parent.append(index_p)
            
            
            
            
    def ind_as_child(self, e_var):
        
        self.as_child = np.zeros(e_var.shape[0], dtype = int)
        
        
        #print ('e_var: ', e_var)
        for i, e in enumerate(e_var):
            
            # evidence variable as child
            index_c = np.where(self.topo_order==e)[0][0]
            #print (index_c)
            self.as_child[i] = index_c


   


class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0
        self.mixture_weight = None
        # weigths associated with each record in mixture
        # n_componets * n_var
        #self.clt_weights_list = None
        self.clt_list =[]   # chow-liu tree list

    
    def computeLL(self, dataset):
        
        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        log_mixture_weights = np.log(self.mixture_weight)
        
        for c in xrange(self.n_components):
            clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]
            

        clt_weights_list, ll_score = Util.m_step_trick(clt_weights_list)
        
        return ll_score
    
    def computeLL_each_datapoint(self, dataset):
        
        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))
        
        log_mixture_weights = np.log(self.mixture_weight)
        
        for c in xrange(self.n_components):
            clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]
            

        ll_scores = Util.get_ll_trick(clt_weights_list)
        
        return ll_scores
    
    


class CNode:



    def __init__(self, var, weights, ids, id):
        
        self.var = var  # the variable id
        self.var_assign = 0  # the assignment of var in map tuple
        self.children = []    # only has 2 child
        self.weights = weights 
        #print ('self.weights :', self.weights)
        self.log_weights = np.log(weights)
        #self.log_inst_weight = np.array([])  # when doing instantiation
        self.log_value = utilM.LOG_ZERO
        self.ids = ids
        self.id = id
        
        

    def add_child(self, child):

        self.children.append(child)


    def set_weights(self, weights):
                
        self.weights = weights
        self.log_weights = np.log(self.weights)
    
    
    def sumout(self, log_weights):
        
        self.log_val = np.logaddexp(self.children[0].log_value + log_weights[0],
                                    self.children[1].log_value + log_weights[1])
        
        
    def maxout(self, log_weights):
        
        left = self.children[0].log_value + log_weights[0]
        right = self.children[1].log_value + log_weights[1]
        
        if left >= right:
            self.log_value = left
            self.var_assign = 0
        else:
            self.log_value = right
            self.var_assign = 1
            
            
        
        #self.log_value = max(self.children[0].log_value + log_weights[0],
        #                            self.children[1].log_value + log_weights[1])
        #print ('cnet nodes: ', self.var)
        #print ('child_log value:' , self.children[0].log_value,  self.children[1].log_value)
        #print ('log weights: ', log_weights)
        #print ( 'log_val:', self.log_value)
        #print (self.children[0])
        #print (self.children[1])
    
    """    
    def instantiation(self, flag = False, log_weight = None):
        
        if flag == False:
            self.log_inst_weight = np.copy(self.log_weights)
        else:
            self.log_inst_weight = log_weight
    """   
        
        
         

    


    
    
class CNet:
    def __init__(self,load_info, depth, n_variables):
        self.nvariables=n_variables
        self.depth=depth
        self.root=self.recover_structure(load_info)
        
    
    
    def recover_structure(self, load_info):
        
        ids=np.arange(self.nvariables)
        root_var = load_info['x']
        root_id = load_info['id']
        root_p0 = load_info['p0']
        root_p1 = load_info['p1']
        root_weights = np.array([root_p0, root_p1]) / (root_p0+ root_p1)
        # the root node
        root = CNode(root_var, root_weights, ids, root_id)        
        root.add_child(load_info['c0'])
        root.add_child(load_info['c1'])
        
        cent_nodes =deque()
        #cent_nodes.append(root.children[0])
        #cent_nodes.append(root.children[1])
        cent_nodes.append(root)
        #print ('a', root)
        
        #print (cnet['type'])
        while cent_nodes:
            cnet = cent_nodes.popleft()
            ids = np.delete(cnet.ids, cnet.id)
            left_child = cnet.children[0]
            right_child = cnet.children[1]
            
            #new_node_left = None
            #new_node_right = None
            # for the left child
            if left_child['type'] == 'internal':
                
                id = left_child['id']
                x  = left_child['x']
                p0 = left_child['p0']
                p1 = left_child['p1']
                c0 = left_child['c0']
                c1 = left_child['c1']
                new_node_left = CNode(x, np.array([p0,p1])/(p0+p1), ids, id)   
                new_node_left.add_child(c0)
                new_node_left.add_child(c1)
                
                cent_nodes.append(new_node_left) # append to the queue
                cnet.children[0] = new_node_left # reassign 
            
            if left_child['type'] == 'leaf':
                new_node_left = CLT()
                new_node_left.log_cond_cpt = left_child['log_cond_cpt']
                new_node_left.topo_order = left_child['topo_order']
                new_node_left.parents = left_child['parents']
                new_node_left.xprob = left_child['p_x']
                new_node_left.ids = ids
                cnet.children[0] = new_node_left
                
            if right_child['type'] == 'internal':
                
                id = right_child['id']
                x  = right_child['x']
                p0 = right_child['p0']
                p1 = right_child['p1']
                c0 = right_child['c0']
                c1 = right_child['c1']
                new_node_right = CNode(x, np.array([p0,p1])/(p0+p1), ids, id)   
                new_node_right.add_child(c0)
                new_node_right.add_child(c1)
                
                cent_nodes.append(new_node_right)
                cnet.children[1] = new_node_right
            
            if right_child['type'] == 'leaf':
                new_node_right = CLT()
                new_node_right.log_cond_cpt = right_child['log_cond_cpt']
                new_node_right.topo_order = right_child['topo_order']
                new_node_right.parents = right_child['parents']
                new_node_right.xprob = right_child['p_x']
                new_node_right.ids = ids 
                cnet.children[1] = new_node_right
                
                
        return root
    
    
    def get_node_list(self):
        
        internal_list = []
        leaf_list = []
        
        nodes_to_process = deque()
        nodes_to_process.append(self.root)
        while nodes_to_process:
            curr_node = nodes_to_process.popleft()
            #print (curr_node)
            if isinstance(curr_node, CNode):
                internal_list.append(curr_node)
                nodes_to_process.append(curr_node.children[0])
                nodes_to_process.append(curr_node.children[1])
            elif isinstance(curr_node, CLT):
                leaf_list.append(curr_node)
            else:
                print ('Error, invalid node')
                exit()
                
        return internal_list, leaf_list

        
    def instantiation(self, datapoint, e_var, evid_flag, internal_list, leaf_list):
        
        #log_sum_value = 0.0
        log_max_value = 0.0
        
        # instantiate the leaf node
        log_cond_cpt_list = []
        
        for t in leaf_list:
            #t.ind_as_parent(e_var)
            #t.ind_as_child(e_var)
            
            t_evalue = datapoint[t.ids][t.evar]
            log_cond_cpt = t.instantiation_log(t_evalue)
            log_cond_cpt_list.append(log_cond_cpt)
        
            #print ('log_cond_cpt_list:')
            #print (log_cond_cpt)
        
        log_inst_weight_list = []
        for c in internal_list:
            log_weights =  np.copy(c.log_weights)
            #set_weight_flag = False
            
            if evid_flag[c.var] == 1: # evidence
                #set_weight_flag =  True
                
                if datapoint[c.var] == 0:  # evidence = 0
                    log_weights[1] = utilM.LOG_ZERO
                else:       # evidence = 0
                    log_weights[0] = utilM.LOG_ZERO
            
            log_inst_weight_list.append(log_weights)
            
        """
        # Get sumout value
        for i, t in enumerate(leaf_list):
            #sample = datapoint[t.ids]
            t.log_val = utilM.ve_tree_bin_log(t.topo_order, t.parents, log_cond_cpt_list[i])
        
        # in reverse order
        for j in xrange(len(internal_list)-1,-1,-1):
            c = internal_list[j]
            c.sumout(log_inst_weight_list[j])
            
        log_sum_value = internal_list[0].log_val
        """
        
        # Get maxout value
        for i, t in enumerate(leaf_list):
            #sample = datapoint[t.ids]
            #t.log_value = utilM.max_tree_bin_log(t.topo_order, t.parents, log_cond_cpt_list[i])
            t.log_value, t.map_tuple = utilM.max_tree_bin_map(t.topo_order, t.parents, log_cond_cpt_list[i])
            
            #print ('leaf node value: ', t.log_value)
            #print ('leaf node ids: ', t.ids)
            #print (leaf_list[i].log_val)
            #print (t)
        
        #print ('log_inst_weight_list', log_inst_weight_list)
        # in reverse order
        for j in xrange(len(internal_list)-1,-1,-1):
            c = internal_list[j]
            c.maxout(log_inst_weight_list[j])
            
        log_max_value = internal_list[0].log_value
    
        
        # back propergate, from root to leaf to find the map tuple
        max_tuple = np.zeros(datapoint.shape[0], dtype = int)
        back_node = internal_list[0]
        #back_queue.append(internal_list[0])
        
        while isinstance(back_node, CNode):
           
            max_tuple[back_node.var] = back_node.var_assign
            
            if back_node.var_assign == 0:
                back_node =   back_node.children[0]
            else:
                back_node =   back_node.children[1]
        
        # reach the leaf CLT node        
        max_tuple[back_node.ids] = back_node.map_tuple   
        
        #print ('log_sum_value: ', log_sum_value)
        #print ('log_max_value', log_max_value)
        #return log_sum_value, log_max_value
        return log_max_value, max_tuple
    


class MCnet():
    
    def __init__(self, n_variable):
        self.cnet_dict_list=[]
        self.weights=[]
        self.internal_list = []
        self.leaf_list = []
        self.n_variable = n_variable
        self.n_components = 0
        
    def structure_redefine(self, load_info):
        
        self.weights = load_info.mixture_weight
        self.n_components = load_info.n_components
        for cn in load_info.cnet_list:
            main_dict = {}
            utilM.save_cutset(main_dict, cn.tree, np.arange(self.n_variable), ccpt_flag = True)
            cnet_component = CNet(main_dict, 0, self.n_variable)
            self.cnet_dict_list.append(cnet_component)
            
            internal_list, leaf_list = cnet_component.get_node_list()
            self.internal_list.append(internal_list)
            self.leaf_list.append(leaf_list)
            

def compute_xmax_mcnet(load_info, dataset, x_var, e_var):
    
    #print ('In compute_xmax_mcnet')
    x_map_list = []    # store the map tuple
    evid_flag = np.zeros(dataset.shape[1], dtype = int)  # value = 1 means evidence
    evid_flag[e_var] = 1
    
    mcnet = MCnet(dataset.shape[1])
    mcnet.structure_redefine(load_info)
    
    log_weights = np.log(mcnet.weights)   
    
    #print ('log_weights: ', log_weights)    
    
    for i in xrange(len(mcnet.leaf_list)):
        for t in mcnet.leaf_list[i]:
            t.evar = np.where(evid_flag[t.ids]==1)[0]
            
            #print ('--------------------------------------')
            #print ('topo_order: ', t.topo_order)
            #print ('parents: ', t.parents)
            #print ('t_evar: ', t.evar)
            t.ind_as_parent(t.evar)
            t.ind_as_child(t.evar)
            
    
    xmax_prob_arr = np.zeros(dataset.shape[0])
            
    for i in xrange(dataset.shape[0]):
        datapoint = dataset[i]
        xmap_temp =[]
        xmax_prob_temp = np.zeros(mcnet.n_components)
        
        for j in xrange(mcnet.n_components):
            
            cnet = mcnet.cnet_dict_list[j]
            xmax_prob_temp[j], x_map = cnet.instantiation(datapoint, e_var, evid_flag, mcnet.internal_list[j], mcnet.leaf_list[j])
            xmap_temp.append(x_map)
            
            
        #print ('xmax_prob_temp:', xmax_prob_temp)
        xmax_prob_temp += log_weights
        #print ('xmax_prob_temp:', xmax_prob_temp)
        best_ind = np.argmax(xmax_prob_temp)
        #print ('best_ind: ', best_ind)
        # find the best one
        x_map_list.append(xmap_temp[best_ind])
        xmax_prob_arr[i] = xmax_prob_temp[best_ind]
        
    
    return  xmax_prob_arr, np.asarray(x_map_list)

        

    
      
def compute_xmax_cnet(cnet_module, depth, dataset, x_var, e_var):   
    
    # test purpose
    #x_var = np.arange(8, dtype=int)*2
    #e_var = np.arange(8, dtype=int)*2 + 1
    #print ('x_var', x_var)
    #print ('e_var', e_var)
    
    x_map_list = []    # store the map tuple
    evid_flag = np.zeros(dataset.shape[1], dtype = int)  # value = 1 means evidence
    evid_flag[e_var] = 1
    
    #print ('evid_flag: ', evid_flag)
    
    cnet = CNet(cnet_module, depth, dataset.shape[1])
    
    internal_list, leaf_list = cnet.get_node_list()
    
    """
    print ('internal nodes:')
    for c in internal_list:
        print ('var: ', c.var)
        print ('weights:', c.weights)
        
    
    print ('Leaf nodes:')
    for t in leaf_list:
        print ('ids: ', t.ids)
        print ('topo_order: ', t.topo_order.shape[0])
    """
    
    for t in leaf_list:
        t.evar = np.where(evid_flag[t.ids]==1)[0]
        
        #print ('--------------------------------------')
        #print ('topo_order: ', t.topo_order)
        #print ('parents: ', t.parents)
        #print ('t_evar: ', t.evar)
        t.ind_as_parent(t.evar)
        t.ind_as_child(t.evar)
        
        #print ('as child: ', t.as_child)
        #print ('as parent: ', t.as_parent)
    
    
    #evid_prob_arr = np.zeros(dataset.shape[0])
    xmax_prob_arr = np.zeros(dataset.shape[0])
            
    for i in xrange(dataset.shape[0]):
        datapoint = dataset[i]
        xmax_prob_arr[i], x_map = cnet.instantiation(datapoint, e_var, evid_flag, internal_list, leaf_list)
        x_map_list.append(x_map)
    
    return  xmax_prob_arr, np.asarray(x_map_list)


def compute_xmax_mt(mt_module, dataset, x_var, e_var):   
    
    # get the child index list and parent index list according to topo order 
    #as_child_list = []
    #as_parent_list =[]
    
    # test purpose
    #e_var = np.arange(8, dtype=int)*2
    #x_var = np.arange(8, dtype=int)*2 + 1    
    #print ('x_var', x_var)
    #print ('e_var', e_var)
    
    for t in mt_module.clt_list:
        t.ind_as_parent(e_var)
        t.ind_as_child(e_var)
        
    
    x_map_list = []
    log_mixture_weight = np.log(mt_module.mixture_weight)
    #print ('log mixture weights: ', log_mixture_weight)       
    #xmax_prob_arr = np.zeros(dataset.shape[0]) 
    #xmax_dataset = []   # the new created dataset contains x_max and e_value 
    for i in xrange(dataset.shape[0]):
        datapoint = dataset[i]
        e_value = datapoint[e_var]  
        # the instantiated log conditional cpt based on the evidence value
        log_cond_cpt_list = []
        # the x_max for each mt component
        x_max_list = []
        # the max probablity for each mt component
        maxout_value_list = np.zeros(mt_module.n_components)
        
        for j, t in enumerate(mt_module.clt_list):
            log_cond_cpt = t.instantiation_log(e_value)
            log_cond_cpt_list.append(log_cond_cpt)
            
            maxout_value, x_max = utilM.max_tree_bin_map(t.topo_order, t.parents, log_cond_cpt)
            #if np.round(maxout_value- utilM.get_single_ll(x_max,t.topo_order, t.parents, log_cond_cpt),decimals = 10) != 0:
            #    print ('error')
            #print (j, maxout_value, x_max)
            maxout_value_list[j] = maxout_value + log_mixture_weight[j]
            x_max_list.append(x_max)
            
            # p(e)
            #evid_prob_arr[i] += utilM.ve_tree_bin_log(t.topo_order, t.parents, log_cond_cpt) * mt_module.mixture_weight[j]
        
        
        
        ind = np.argmax(maxout_value_list)
        x_map = x_max_list[ind]
        x_map_list.append(x_map)
        #print (ind, maxout_value_list[ind], x_map)
        

    return np.asarray(x_map_list)



def main():
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    max_depth = int(sys.argv[6])
    e_percent = float(sys.argv[8])  #  #evidence/ # total variables
    seq = int(sys.argv[10])  # the sequence of evidence record in pre-generated evidence file
    module_type = sys.argv[12]
    
    #train_filename = dataset_dir + data_name + '.ts.data'
    #train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    
    
    print('------------------------------------------------------------------')
    print('Calculate log(p(x_data|e)) - log(p(x_max|e))                      ')
    print('------------------------------------------------------------------')
        
    

    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    output_dir = '../infer_output/'
    e_file = output_dir + data_name + '_evid_'+ str(int(e_percent*100)) + '.txt'
    #mt_file = output_dir + data_name + '_mt_'+ str(int(e_percent*100)) + '.txt'
    #tim_file = output_dir + data_name + '_tim_'+ str(int(e_percent*100)) + '.txt'
       
    #tim_new_data_file = output_dir + data_name + '_tim_data_'+ str(int(e_percent*100)) + '.txt'
    mt_new_data_file = output_dir + data_name + '_mt_data_'+ str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    
    # Test purpose
    #test_dataset = np.array(test_dataset[13:14])
    #print (test_dataset)
    
    # randomly select x and e
    #variables = np.arange(n_variables, dtype = int)
    #np.random.shuffle(variables)
    #num_evar = int(np.round(n_variables*e_percent, decimals = 0))
    #e_var = variables[0:num_evar] # the evidence variable
    #x_var = variables[num_evar:] 
    
    #e_var = np.array([13,2,0])
    #x_var = np.setdiff1d(np.arange(n_variables), e_var)
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    #for i in xrange(10):
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)


    
    ### Load the trained cutset network
    print ('Start reloading TIM / cutset network ...')
    #diff_arr = np.zeros(max_depth -1 )
    #for depth in xrange(1, max_depth):
    #for depth in xrange(max_depth-1, max_depth):
        #print ('depth: ', depth)
    if module_type == 'cnxd':
        cnet_file = '../best_module_upd/' + data_name + '_' + str(max_depth) + '.npz'
    if module_type == 'deep':
        cnet_file = '../best_module_deep/' + data_name + '_' + str(max_depth) + '.npz'
    if module_type == 'tim':
        cnet_file = '../best_module/' + data_name + '_' + str(max_depth) + '.npz'
    
    cnet_module = np.load(cnet_file)['module'].item()
    # the joint probablity of x_data and e  P(x_data, e)
    #data_prob_cnet = utilM.computeLL_reload(cnet_module, test_dataset)
    # the joint probablity of x_max and e P(x_data, e), and P(e)
    start = time.time()
    xmax_prob_cnet, map_dataset_cnet = compute_xmax_cnet(cnet_module, max_depth, test_dataset, x_var, e_var)
    print ('running time for TIM Cnet: ', time.time()-start)
    
    
    #map_prob = utilM.computeLL_reload(cnet_module, map_dataset_cnet)
    #for i in xrange(map_dataset_cnet.shape[0]):
    #    print (xmax_prob_cnet[i], map_prob[i], xmax_prob_cnet[i]- map_prob[i])
    
    # save the max tuple
    tim_new_data_file = output_dir + module_type + '/' + data_name + str(int(e_percent*100)) +'_'+str(max_depth) +'_'+ str(seq)+'.txt'
    #with open(tim_new_data_file) as f_handle:
    np.savetxt(tim_new_data_file, np.asarray(map_dataset_cnet).astype(int), fmt='%i', delimiter=',')
    
    
    ### Load the trained mixture of clt
    print ('Start reloading TUM / mixture of trees ...')
    #tum_file = '../module/' + data_name + '.npz'
    tum_file = '../module_mt/' + data_name + '.npz'
    reload_dict = np.load(tum_file)
    
    #print ("mixture weights: ", reload_dict['weights'])
    
    mt_module = MIXTURE_CLT()
    mt_module.mixture_weight = reload_dict['weights']
    mt_module.n_components = mt_module.mixture_weight.shape[0]
    
    reload_clt_component = reload_dict['clt_component']
    
    #print (reload_clt_component)
    for i in xrange(mt_module.n_components):
        clt_c = CLT()
        #str_id = str(i)
        curr_component = reload_clt_component[i]
        #clt_c.xyprob = curr_component['xyprob']
        clt_c.xprob = curr_component['xprob']
        clt_c.topo_order = curr_component['topo_order']
        clt_c.parents = curr_component['parents']
        clt_c.log_cond_cpt = curr_component['log_cond_cpt']
        clt_c.nvariables = n_variables
        
        mt_module.clt_list.append(clt_c)
    
    
    
    #print('Mixture of trees Test set LL scores')
    #data_prob_mt = mt_module.computeLL_each_datapoint(test_dataset)
    start2 = time.time()
    map_dataset = compute_xmax_mt(mt_module, test_dataset, x_var, e_var)
    print ('running time for TUM MT: ', time.time()-start2)
    #xmax_prob_mt = mt_module.computeLL_each_datapoint(map_dataset)
    
    
    #diff = xmax_prob_mt - data_prob_mt
    #for i in xrange(test_dataset.shape[0]):
    #    print (diff[i])
    #mt_diff = np.sum(np.abs(xmax_prob_mt - data_prob_mt)) / test_dataset.shape[0]
    #print ('MT difference: ', mt_diff)
    #with open(mt_file, 'a') as f_handle:
    #    np.savetxt(f_handle, np.array([mt_diff]))
        
    
    # save the map tuple to file
    #with open(mt_new_data_file) as f_handle:
    mt_new_file =  output_dir + module_type + '/' + data_name + str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    np.savetxt(mt_new_file, np.asarray(map_dataset).astype(int), fmt='%i', delimiter=',')
    
    #total_diff = np.log(np.sum((np.exp(xmax_prob_mt) - np.exp(data_prob_mt)) / evid_prob_mt))
    #data_prob_mt -= evid_prob_mt
    #xmax_prob_mt -= evid_prob_mt
    #max_value = np.max(xmax_prob_mt)
    #data_prob_mt -= max_value
    #xmax_prob_mt -= max_value
    #total_diff2 = np.log(np.sum(np.exp(xmax_prob_mt)) - np.sum(np.exp(data_prob_mt))) + max_value


def main_mcnet():
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    max_depth = int(sys.argv[6])
    e_percent = float(sys.argv[8])  #  #evidence/ # total variables
    seq = int(sys.argv[10])  # the sequence of evidence record in pre-generated evidence file
    module_type = sys.argv[12]
    
    #train_filename = dataset_dir + data_name + '.ts.data'
    #train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    
    
    print('------------------------------------------------------------------')
    print('MPE inference for MCNets                                          ')
    print('------------------------------------------------------------------')
        
    

    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    output_dir = '../infer_output/'
    e_file = output_dir + data_name + '_evid_'+ str(int(e_percent*100)) + '.txt'
    #mt_file = output_dir + data_name + '_mt_'+ str(int(e_percent*100)) + '.txt'
    #tim_file = output_dir + data_name + '_tim_'+ str(int(e_percent*100)) + '.txt'
       

    
    # Test purpose
    #test_dataset = np.array(test_dataset[13:14])
    #print (test_dataset)
    
    # randomly select x and e
    #variables = np.arange(n_variables, dtype = int)
    #np.random.shuffle(variables)
    #num_evar = int(np.round(n_variables*e_percent, decimals = 0))
    #e_var = variables[0:num_evar] # the evidence variable
    #x_var = variables[num_evar:] 
    
    #e_var = np.array([13,2,0])
    #x_var = np.setdiff1d(np.arange(n_variables), e_var)
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    #for i in xrange(10):
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)

    #print ('x_var', x_var)
    #print ('e_var', e_var)
    
    # write the evidence file
    #e_list = []
    #e_list.append(e_var)
    #with open(e_file, 'a') as f_handle:
    #    np.savetxt(f_handle, np.asarray(e_list).astype(int), fmt='%i', delimiter=',')lla
    
    ### Load the trained cutset network
    print ('Start reloading TIM / cutset network ...')

#    module_dir = '../mcnet/module/'
#    print (module_dir+data_name + '.pkl')
#    
#    with open(module_dir+data_name + '.pkl', 'rb') as input:
#        reload_mcnet = pickle.load(input)
    # the joint probablity of x_data and e  P(x_data, e)
    #data_prob_cnet = utilM.computeLL_reload(cnet_module, test_dataset)
    # the joint probablity of x_max and e P(x_data, e), and P(e)
    
    module_dir = '../mcnet/module/'
    with open(module_dir+data_name + '.pkl', 'rb') as input:
        reload_mcnet = pickle.load(input)

    start = time.time()
    xmax_prob_cnet, map_dataset_cnet = compute_xmax_mcnet(reload_mcnet, test_dataset, x_var, e_var)
    print ('running time for TIM Cnet: ', time.time()-start)
    
    
    #map_prob = utilM.computeLL_reload(cnet_module, map_dataset_cnet)
    #for i in xrange(map_dataset_cnet.shape[0]):
    #    print (xmax_prob_cnet[i], map_prob[i], xmax_prob_cnet[i]- map_prob[i])
    
    # save the max tuple
    tim_new_data_file = output_dir + module_type + '/' + data_name + str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    #with open(tim_new_data_file) as f_handle:
    np.savetxt(tim_new_data_file, np.asarray(map_dataset_cnet).astype(int), fmt='%i', delimiter=',')
            

if __name__=="__main__":
    #start = time.time()
    #for i in xrange(10):
    if sys.argv[12] == 'mcnet':
        main_mcnet()
    else:
        main()
    #print ('Total running time: ', time.time() - start)
