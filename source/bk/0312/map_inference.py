"""
    MAP Inference
    Stroe the MAP tuple, and Get the LL score for the MAP tuple
"""


from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import sys
from Util import *

import utilM
import time

from cnet_dfs import  CNET_dfs, Leaf_tree
from MIXTURE_CLT import MIXTURE_CLT
from cnet_bag import load_bcn, compute_ll_from_disk




class bcnet_dfs():
    
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
            cnet_component = CNET_dfs(main_dict,self.n_variable)
            self.cnet_dict_list.append(cnet_component)
            
            internal_list, leaf_list = cnet_component.get_node_list()
            self.internal_list.append(internal_list)
            self.leaf_list.append(leaf_list)
            
"""
    Get the max tuple and its probablity using bags of cnets
"""
def compute_xmax_bcnet(bcnet, dataset, x_var, e_var):
    
    #print ('In compute_xmax_bcnet')
    x_map_list = []    # store the map tuple
    evid_flag = np.zeros(dataset.shape[1], dtype = int)  # value = 1 means evidence
    evid_flag[e_var] = 1
    
    n_variables = dataset.shape[1]
#    cm_weights = np.loadtxt(bcnet_dir +'_component_weights.txt')
#    log_weights = np.log(cm_weights)  
#    
#    n_components = cm_weights.shape[0]
#    
#    bcnet_dict_list = []
#    bcnet_internal_list =[]
#    bcnet_leaf_list =[]
#    for i in xrange(n_components):
#        cn_file = bcnet_dir +'_' +str(i) + '.npz'
#        cn = np.load(cn_file)['module'].item()
#        cnet_component = CNET_dfs(cn,n_variables)
#        bcnet_dict_list.append(cnet_component)
#        
#        internal_list, leaf_list = cnet_component.get_node_list()
#        bcnet_internal_list.append(internal_list)
#        bcnet_leaf_list.append(leaf_list)
    
    
    cm_weights = bcnet['cm_weights']
    log_weights = np.log(cm_weights)  
    
    n_components = cm_weights.shape[0]
    
    bcnet_dict_list = []
    bcnet_internal_list =[]
    bcnet_leaf_list =[]
    for i in xrange(n_components):
        cn = bcnet['cnet_list'][i]
        cnet_component = CNET_dfs(cn,n_variables)
        bcnet_dict_list.append(cnet_component)
        
        internal_list, leaf_list = cnet_component.get_node_list()
        bcnet_internal_list.append(internal_list)
        bcnet_leaf_list.append(leaf_list)
     

    #print ('log_weights: ', log_weights)    
    
    for i in xrange(len(bcnet_leaf_list)):
        for t in bcnet_leaf_list[i]:
            t.evar = np.where(evid_flag[t.ids]==1)[0]
            t.ind_as_parent(t.evar)
            t.ind_as_child(t.evar)
            
    
    xmax_prob_arr = np.zeros(dataset.shape[0])
            
    for i in xrange(dataset.shape[0]):
        datapoint = dataset[i]
        xmap_temp =[]
        xmax_prob_temp = np.zeros(n_components)
        
        for j in xrange(n_components):
            
            cnet = bcnet_dict_list[j]
            xmax_prob_temp[j], x_map = cnet.instantiation(datapoint, e_var, evid_flag, bcnet_internal_list[j],bcnet_leaf_list[j])
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

        

    
"""
    Get the max tuple  and its probablity using cnet
"""      
def compute_xmax_cnet(cnet_module, depth, dataset, x_var, e_var):   
    
    
    x_map_list = []    # store the map tuple
    evid_flag = np.zeros(dataset.shape[1], dtype = int)  # value = 1 means evidence
    evid_flag[e_var] = 1
    
    
    cnet = CNET_dfs(cnet_module,dataset.shape[1])
    
    internal_list, leaf_list = cnet.get_node_list()

    
    for t in leaf_list:
        t.evar = np.where(evid_flag[t.ids]==1)[0]
        
        t.ind_as_parent(t.evar)
        t.ind_as_child(t.evar)
        
    xmax_prob_arr = np.zeros(dataset.shape[0])
            
    for i in xrange(dataset.shape[0]):
        datapoint = dataset[i]
        xmax_prob_arr[i], x_map = cnet.instantiation(datapoint, e_var, evid_flag, internal_list, leaf_list)
        x_map_list.append(x_map)
    
    return  xmax_prob_arr, np.asarray(x_map_list)


"""
    Get the max tuple and its probablity using mt
""" 
def compute_xmax_mt(mt_module, dataset, x_var, e_var):   
    
    
    for t in mt_module.clt_list:
        t.ind_as_parent(e_var)
        t.ind_as_child(e_var)
        
    
    x_map_list = []
    log_mixture_weight = np.log(mt_module.mixture_weight)

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
            
            maxout_value_list[j] = maxout_value + log_mixture_weight[j]
            x_max_list.append(x_max)
        
        
        ind = np.argmax(maxout_value_list)
        x_map = x_max_list[ind]
        x_map_list.append(x_map)
        
        

    return np.asarray(x_map_list)



'''
    For All types of cutset networks, including CNxD RCN and CN
    Note: for CN, we don't need the depth, so depth can be any number
'''
def main_cnet():
    
    module_type = sys.argv[2]
    dataset_dir = sys.argv[4]
    data_name = sys.argv[6]
    max_depth = int(sys.argv[8])
    e_percent = float(sys.argv[10])  #  #evidence/ # total variables
    seq = int(sys.argv[12])  # the sequence of evidence record in pre-generated evidence file
    

    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    output_dir = '../infer_output/'
    e_file = output_dir + 'evid_file/'+ data_name + '_evid_'+ str(int(e_percent*100)) + '.txt'
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)


    
    ### Load the trained cutset network
    print ('Start reloading cutset network ...')
    if module_type == 'cnxd':
        cnet_file = '../cnxd_output/' + data_name + '_' + str(max_depth) + '.npz'
    if module_type == 'rcn':
        cnet_file = '../rcn_output/' + data_name + '_' + str(max_depth) + '.npz'
    if module_type == 'cn':
        cnet_file = '../cn_output/' + data_name  + '.npz'
    
    
    print ('Getting the MAP tuple...')
    cnet_module = np.load(cnet_file)['module'].item()
    xmax_prob_cnet, map_dataset_cnet = compute_xmax_cnet(cnet_module, max_depth, test_dataset, x_var, e_var)

    
    
    # save the max tuple
    if module_type == 'cn':
        new_data_file = output_dir + module_type + '/' + data_name + '_' + str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    else:   
        new_data_file = output_dir + module_type + '/' + data_name + '_' + str(int(e_percent*100)) +'_'+str(max_depth) +'_'+ str(seq)+'.txt'

    map_dataset_cnet = np.asarray(map_dataset_cnet).astype(int)
    np.savetxt(new_data_file, map_dataset_cnet, fmt='%i', delimiter=',')
    
    ll_score = np.sum(utilM.computeLL_reload(cnet_module, map_dataset_cnet)) / map_dataset_cnet.shape[0]
    
    print ('MAP dataset Set LL Score: ', ll_score)
    


    
'''
    For mixture of trees
    Get the MAP tuple of the test dataset
'''
def main_mt():
    
    module_type = sys.argv[2]
    dataset_dir = sys.argv[4]
    data_name = sys.argv[6]
    e_percent = float(sys.argv[8])  #  #evidence/ # total variables
    seq = int(sys.argv[10])  # the sequence of evidence record in pre-generated evidence file
    tum_module = sys.argv[12]
    
        
    
    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    output_dir = '../infer_output/'
    e_file = output_dir + 'evid_file/'+ data_name + '_evid_'+ str(int(e_percent*100)) + '.txt'
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)



    ### Load the trained mixture of clt
    print ('Start reloading mixture of trees ...')
    tum_file = '../mt_output/' + tum_module + '.npz'
    reload_dict = np.load(tum_file)
    
    #print ("mixture weights: ", reload_dict['weights'])
    
    mt_module = MIXTURE_CLT()
    mt_module.mixture_weight = reload_dict['weights']
    mt_module.n_components = mt_module.mixture_weight.shape[0]
    
    reload_clt_component = reload_dict['clt_component']
    
    #print (reload_clt_component)
    for i in xrange(mt_module.n_components):
        clt_c = Leaf_tree()
        #str_id = str(i)
        curr_component = reload_clt_component[i]
        #clt_c.xyprob = curr_component['xyprob']
        clt_c.xprob = curr_component['xprob']
        clt_c.topo_order = curr_component['topo_order']
        clt_c.parents = curr_component['parents']
        clt_c.log_cond_cpt = curr_component['log_cond_cpt']
        clt_c.nvariables = n_variables
        
        mt_module.clt_list.append(clt_c)
    
        
    print ('Getting the MAP tuple...')
    start2 = time.time()
    map_dataset = compute_xmax_mt(mt_module, test_dataset, x_var, e_var)
    #print ('running time for TUM MT: ', time.time()-start2)
    
    map_dataset = np.asarray(map_dataset).astype(int)
    mt_new_file =  output_dir + module_type + '/' + data_name + str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    np.savetxt(mt_new_file, map_dataset, fmt='%i', delimiter=',')
    
    ll_score = mt_module.computeLL (map_dataset) / map_dataset.shape[0]    
    print ('MAP dataset Set LL Score: ', ll_score)
    

'''
    For bags of cnets
    Get the MAP tuple of the test dataset
'''
def main_bcnet():
    
    module_type = sys.argv[2]
    dataset_dir = sys.argv[4]
    data_name = sys.argv[6]
    e_percent = float(sys.argv[8])  #  #evidence/ # total variables
    seq = int(sys.argv[10])  # the sequence of evidence record in pre-generated evidence file
    

    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    output_dir = '../infer_output/'
    e_file = output_dir + 'evid_file/'+ data_name + '_evid_'+ str(int(e_percent*100)) + '.txt'
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)
    
    ### Load the trained cutset network
    print ('Start reloading Bags of CNets...')

    bcnet_dir = '../bcnet_output/'
    reload_bcn = load_bcn(bcnet_dir, data_name)
    
    print ('Getting the MAP tuple...')
    #start = time.time()
    xmax_prob_cnet, map_dataset_bcnet = compute_xmax_bcnet(reload_bcn, test_dataset, x_var, e_var)
    #print ('running time for bags of Cnet: ', time.time()-start)
    map_dataset_bcnet = np.asarray(map_dataset_bcnet).astype(int)
    
    
    # save the max tuple
    tim_new_data_file = output_dir + module_type + '/' + data_name + str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    #with open(tim_new_data_file) as f_handle:
    np.savetxt(tim_new_data_file, map_dataset_bcnet, fmt='%i', delimiter=',')
            
    
    ll_score = compute_ll_from_disk (reload_bcn, map_dataset_bcnet) / map_dataset_bcnet.shape[0]    
    print ('MAP dataset Set LL Score: ', ll_score)

if __name__=="__main__":
    #start = time.time()
    #for i in xrange(10):
    if sys.argv[2] == 'bcnet':
        main_bcnet()
    elif sys.argv[2] == 'mt':
        main_mt()
    elif sys.argv[2] in ['cn','cnxd','rcn']:
        main_cnet()    
    else:
        print ('**** ERROR: invalid module type ****')
        exit()
    
    #print ('Total running time: ', time.time() - start)
