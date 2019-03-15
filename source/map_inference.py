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


            
"""
    Get the max tuple and its probablity using bags of cnets
"""
def compute_xmax_bcnet(bcnet, dataset, x_var, e_var):
    
    
    x_map_list = []    # store the map tuple
    evid_flag = np.zeros(dataset.shape[1], dtype = int)  # value = 1 means evidence
    evid_flag[e_var] = 1
    
    n_variables = dataset.shape[1]
    
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
            
            
        xmax_prob_temp += log_weights
        best_ind = np.argmax(xmax_prob_temp)
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
    For All types of cutset networks, including CNxD CNR and CNd
    Note: for CN, we don't need the depth, so depth can be any number
'''
def main_map_cnet(parms_dict):
    
    print('------------------------------------------------------------------')
    print('MAP inference of Cutset Network'                                   )
    print('------------------------------------------------------------------')
    
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    depth = int(parms_dict['depth'])
    e_file = parms_dict['efile']
    e_percent = float(parms_dict['e'])
    output_dir = parms_dict['output_dir']
    module_type = parms_dict['t']
    seq = int(parms_dict['seq'])
    
    input_dir =  parms_dict['input_dir']
    input_module = parms_dict['input_module']
    

    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)


    
    ### Load the trained cutset network
    print ('Start reloading cutset network ...')
    
    cnet_file = input_dir + input_module + '.npz'
    print ('Getting the MAP tuple...')
    cnet_module = np.load(cnet_file)['module'].item()
    xmax_prob_cnet, map_dataset_cnet = compute_xmax_cnet(cnet_module, depth, test_dataset, x_var, e_var)

    
    
    # save the max tuple
    if module_type == 'cn':
        new_data_file = output_dir + data_name + '_' + module_type + '_' + str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    else:   
        new_data_file = output_dir + data_name + '_' + module_type + '_' + str(int(e_percent*100)) +'_'+str(depth) +'_'+ str(seq)+'.txt'

    map_dataset_cnet = np.asarray(map_dataset_cnet).astype(int)
    np.savetxt(new_data_file, map_dataset_cnet, fmt='%i', delimiter=',')
    
    ll_score = np.sum(utilM.computeLL_reload(cnet_module, map_dataset_cnet)) / map_dataset_cnet.shape[0]
    
    print ('MAP dataset Set LL Score: ', ll_score)
    


    
'''
    For mixture of trees
    Get the MAP tuple of the test dataset
'''
def main_map_mt(parms_dict):
    
    print('------------------------------------------------------------------')
    print('MAP inference of Mixture Chow_Liu Tree'                            )
    print('------------------------------------------------------------------')
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    e_file = parms_dict['efile']
    e_percent = float(parms_dict['e'])
    output_dir = parms_dict['output_dir']
    seq = int(parms_dict['seq'])
    
    input_dir =  parms_dict['input_dir']
    input_module = parms_dict['input_module']
        
    
    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)



    ### Load the trained mixture of clt
    print ('Start reloading mixture of trees ...')
    #tum_file = '../mt_output/' + tum_module + '.npz'
    reload_dict = np.load(input_dir + input_module + '.npz')
    
    
    mt_module = MIXTURE_CLT()
    mt_module.mixture_weight = reload_dict['weights']
    mt_module.n_components = mt_module.mixture_weight.shape[0]
    
    reload_clt_component = reload_dict['clt_component']
    
    for i in xrange(mt_module.n_components):
        clt_c = Leaf_tree()
        curr_component = reload_clt_component[i]
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
    mt_new_file =  output_dir  + data_name + '_mt_'+ str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    np.savetxt(mt_new_file, map_dataset, fmt='%i', delimiter=',')
    
    ll_score = mt_module.computeLL (map_dataset) / map_dataset.shape[0]    
    print ('MAP dataset Set LL Score: ', ll_score)
    

'''
    For bags of cnets
    Get the MAP tuple of the test dataset
'''
def main_map_bcnet(parms_dict):
    
    print('------------------------------------------------------------------')
    print('MAP inference of Mixture Chow_Liu Tree'                            )
    print('------------------------------------------------------------------')
    
    
    dataset_dir = parms_dict['dir']
    data_name = parms_dict['dn']
    e_file = parms_dict['efile']
    e_percent = float(parms_dict['e'])
    output_dir = parms_dict['output_dir']
    seq = int(parms_dict['seq'])
    
    input_dir =  parms_dict['input_dir']
    input_module = parms_dict['input_module']
    
    

    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    
    test_filename = dataset_dir + data_name +'.test.data'
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = test_dataset.shape[1]
    e_var_arr = np.loadtxt(e_file, dtype=int, delimiter=',')
    
    e_var = e_var_arr[seq]
    x_var = np.setdiff1d(np.arange(n_variables), e_var)
    
    ### Load the trained cutset network
    print ('Start reloading Bags of CNets...')
    reload_bcn = load_bcn(input_dir, input_module)
    
    print ('Getting the MAP tuple...')
    #start = time.time()
    xmax_prob_cnet, map_dataset_bcnet = compute_xmax_bcnet(reload_bcn, test_dataset, x_var, e_var)
    map_dataset_bcnet = np.asarray(map_dataset_bcnet).astype(int)
    
    
    # save the max tuple
    tim_new_data_file = output_dir + data_name +'_bcnet_'+ str(int(e_percent*100)) +'_'+ str(seq)+'.txt'
    #with open(tim_new_data_file) as f_handle:
    np.savetxt(tim_new_data_file, map_dataset_bcnet, fmt='%i', delimiter=',')
            
    
    ll_score = compute_ll_from_disk (reload_bcn, map_dataset_bcnet) / map_dataset_bcnet.shape[0]    
    print ('MAP dataset Set LL Score: ', ll_score)

if __name__=="__main__":
    
    if sys.argv[2] == 'bcnet':
        main_map_bcnet()
    elif sys.argv[2] == 'mt':
        main_map_mt()
    elif sys.argv[2] in ['cn','cnxd','cnr']:
        main_map_cnet()    
    else:
        print ('**** ERROR: invalid module type ****')
        exit()
    

