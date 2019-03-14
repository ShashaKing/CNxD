#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 11:39:28 2018
# This is the workable verion that only works for tree converted junction tree
# No log during msg passing
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
from CLT_class import CLT
import copy
import time
import sys
#import numba
#mport utilM
from Util import *

#@numba.jit
'''
    add the query varible to the clique
'''
def add_query_var_to_matrix(var_arr, potential_orig, n_varibles, varible_id):
    var_arr[:,-1] = varible_id
    potential_extend = np.zeros((potential_orig.shape[0],2,2,2))
    
    # find indices (cliqure cid) which has varible_id == parent
    vp = np.where(var_arr[:,1]==varible_id)[0]
    potential_extend[vp,:,0,0] = potential_orig[vp,:,0]
    potential_extend[vp,:,1,1] = potential_orig[vp,:,1]
    
    # find indices (cliqure cid) which has varible_id == child
    vc = np.where(var_arr[:,0]==varible_id)[0]
    potential_extend[vc,0,:,0] = potential_orig[vc,0,:]
    potential_extend[vc,1,:,1] = potential_orig[vc,1,:]
    
    # find indices (cliqure cid) which doesn't contain varible id
    vo = np.delete(np.arange(n_varibles), np.append(vp, vc))
    potential_extend[vo,:,:,0] = potential_orig[vo]
    potential_extend[vo,:,:,1] = potential_orig[vo]
    

    return potential_extend

#@numba.jit
def msg_leaf_to_root(topo_order, parents, potential_orig):
    # exclude the root node
    msg = np.ones((parents.shape[0],2,2))
    for i in xrange(topo_order.shape[0]-1, 0, -1):
        cid = topo_order[i]
        cid_pa  = parents[cid] # parent cid
        # get the msg
        msg[cid] = np.einsum('ijk->jk',potential_orig[cid])
#        print msg[cid]
        potential_orig[cid_pa] = np.einsum('ijk,ik->ijk',potential_orig[cid_pa], msg[cid])
    
    return   msg

#@numba.jit
def msg_root_to_leaf(topo_order, children, no_of_chlidren, potential_orig, msg_sent_prev):
    

    for cid in topo_order:
        n_child = no_of_chlidren[cid]
        if n_child == 0:  # no child, pass
            continue
        curr_children  = children[cid, 0: n_child]
        # get the msg
        msg = np.einsum('ijk->ik',potential_orig[cid])
        msg_sent_prev[msg_sent_prev == 0] = 1
        msg = msg / msg_sent_prev   # exclude the msg that sent by self when doing leaf -> root

        potential_orig[curr_children] = np.einsum('cijk,cjk->cijk',potential_orig[curr_children], msg[curr_children])

    return potential_orig

   


#@numba.jit     
def get_marginal(potential_orig, clique_id_var_asChild, pair_var):
    clique_ids = clique_id_var_asChild[pair_var] # the clique ids that will be refered
    marginals =  np.einsum('cijk->cik',potential_orig[clique_ids]) # 'c' stand for clique
    return marginals




# Assume in cliques, all sequence is child|parent
# all pothentail is in log space
class Clique:
     
    def __init__(self,cid, varibles, potential):
        
        # '-1' is reserved to add new node        
        self.cid=cid #the unique id for each clique
        self.var = np.full(3, -1)
        self.var[:2] = varibles  # the varible array that clique contains
        self.potential = potential  # the potential functions
        self.parent = None
        self.children = []
        
       
    # when initial the children list
    def set_child_list(self, child_list):
        self.children = child_list

        
    def set_parent(self, parent):
        self.parent = parent
    
        
    


 


class JunctionTree:
    
    def _init_(self):
        self.clique_list = []
        self.n_cliques = 0
        self.n_varibles = 0
        self.jt_order = None
        self.jt_parents = None
        self.var_in_clique = {} # a dictionary contains the information indicate the  variable in which clique
                                # make sure the first element under each key is the smallest one
                                # which is the one that cantain the actual information about the key var
        
                                 
    
    def learn_structure(self, topo_order, parents, cond_prob):
        self.clique_list = []
        self.n_cliques = topo_order.shape[0]
        self.jt_parents = np.zeros(self.n_cliques)
        self.var_in_clique = {}
        self.n_varibles = topo_order.shape[0]
        
        
        # create a very special clique as root
        root_cpt = np.copy(cond_prob[0])
        root_cpt[0,1] = root_cpt[1,0] = 0
        root_clique = Clique(0, np.array([0, 0]), root_cpt)
        self.clique_list.append(root_clique)
        self.var_in_clique[topo_order[0]] = [topo_order[0]]
        
        # exclude the root node
        for i in xrange(1, topo_order.shape[0]):
            child = topo_order[i]
            parent = parents[child]
            clique_id = i
            new_clique = Clique(clique_id, np.array([child, parent]), cond_prob[i])
            self.clique_list.append(new_clique)
            
            if child in self.var_in_clique:
                self.var_in_clique[child].append(clique_id)
            else:
                self.var_in_clique[child] = [clique_id]                
            if parent in self.var_in_clique:
                self.var_in_clique[parent].append(clique_id)
            else:
                self.var_in_clique[parent] = [clique_id]
                
        
        
        self.clique_to_tree()
        self.clique_to_matrix()
    
    
    
    def clique_to_tree(self):
                
        neighbors = np.zeros((self.n_cliques, self.n_cliques))
        for k in self.var_in_clique.keys():
            
            nb_val = self.var_in_clique[k]
            nb_num = len(nb_val) # how many cliques that conatain this variable
            
            # for cliques connected to root clique
            if k==0:
                for i in xrange(nb_num):
                    neighbors[0, nb_val[i]] =1
                    neighbors[nb_val[i], 0] =1                    
                continue
                
            
            if nb_num > 1:
                for i in xrange(nb_num):
                    for j in xrange(i+1, nb_num):                        
                        # connect only parent and child, for tree only
                        if self.clique_list[nb_val[i]].var[0] == self.clique_list[nb_val[j]].var[1] \
                        or self.clique_list[nb_val[i]].var[1] == self.clique_list[nb_val[j]].var[0] :
                            neighbors[nb_val[i], nb_val[j]] =1
                            neighbors[nb_val[j], nb_val[i]] =1
                    
                    
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(neighbors * (-1)))
        # Convert the spanning tree to a Bayesian network
        self.jt_order, self.jt_parents = depth_first_order(Tree, 0, directed=False)

        for i in xrange(self.n_cliques):
            child_index = np.where(self.jt_parents==i)[0]
            
            if child_index.shape[0] > 0:
                child_list = []
                for c in child_index:
                    child_list.append(self.clique_list[c])
                self.clique_list[i].set_child_list(child_list)
            
            if self.jt_parents[i] != -9999:
                self.clique_list[i].set_parent(self.clique_list[self.jt_parents[i]])
            
        
        
    def set_evidence(self, evid_list):
        # no evidence
        if len(evid_list) == 0:
            return
        for k in xrange(len(evid_list)):
            evid_id = evid_list[k][0]
            evid_val = evid_list[k][1]

            ind = self.var_in_clique[evid_id]
          
            # leaf node in original clt
            ops_val = 1-evid_val  # the oppsite value
            if ind.shape[0] == 0:
                self.clique_potential[ind[0],ops_val,:] = 0
            else:
                # as child
                self.clique_potential[ind[0],ops_val,:] = 0
                # as parent
                self.clique_potential[ind[1:],:,ops_val] = 0
            

        
                

    '''
    Start from here, we convert cliques to matrix, no clique will be available
    '''

    # remove object clique, convert everthing to matrix
    def clique_to_matrix(self):
        self.clique_var_arr = np.zeros((self.n_cliques, 3), dtype = int)    # The variable each clique contains
        self.clique_potential = np.zeros((self.n_cliques, 2,2))  # the potential functions
        # parent is the same as jt.parent
        self.clique_children = None
    
        # Under our assumption, this is the 'clique id' where the 'var' acctually has information
        self.clique_id_var_asChild = np.zeros(self.n_varibles, dtype = int) 
       
        
        
        max_width = np.max(np.bincount(self.jt_parents[1:])) # the max number of child in jt
        self.clique_children = np.full((self.n_cliques, max_width),-1)
        self.no_of_chlidren = np.zeros(self.n_cliques, dtype = int)  # how many child for each clique
        
        for clq in self.clique_list:
            self.clique_var_arr[clq.cid] = clq.var
            self.clique_potential[clq.cid] = clq.potential
            self.clique_id_var_asChild[clq.var[0]] = clq.cid
            
            for j, ch in enumerate(clq.children):
                self.clique_children[clq.cid,j] = ch.cid
            self.no_of_chlidren[clq.cid] = len(clq.children)
        
        # Delete the clique list
        self.clique_list = None
        
        
        # convert from list to numpy array
        for j in xrange(0, self.n_varibles):
            # convert list to numpy array
            self.var_in_clique[j] = np.asarray(self.var_in_clique[j])
            

        
        
    def add_query_var(self, var_id):
       
        clique_potential_extend = add_query_var_to_matrix(self.clique_var_arr, self.clique_potential, self.n_varibles, var_id)

        return clique_potential_extend

    
    def propagation(self, potential):
        # From leaf to root
        clique_msg_out = msg_leaf_to_root(self.jt_order, self.jt_parents, potential)

        msg_root_to_leaf(self.jt_order, self.clique_children, self.no_of_chlidren, potential, clique_msg_out)


        

                    
            
    def get_pairwise_marginal(self, potential, query_id):
        
        pairwise_marginal = get_marginal(potential , self.clique_id_var_asChild, query_id)
        return pairwise_marginal
                
        
def get_marginal_JT(jt, evid_list, query_var):
     

     jt_var = copy.deepcopy(jt)
     jt_var.set_evidence(evid_list)
     n_varible = query_var.shape[0]
     marginal_all = np.zeros((n_varible, n_varible, 2,2))

     
     for i  in xrange(n_varible-1):
         var = query_var[i] 
         new_potential = jt_var.add_query_var(var)
         
 
         jt_var.propagation(new_potential)
         
         pair_var = query_var[i+1:]
         pairwise_marginal = jt_var.get_pairwise_marginal(new_potential, pair_var)
         marginal_all[i+1:, i] = pairwise_marginal
         pairwise_marginal_dia = np.copy(pairwise_marginal)
         #swap
         pairwise_marginal_dia[:,1,0], pairwise_marginal_dia[:,0,1] = pairwise_marginal[:,0,1], pairwise_marginal[:,1,0]
 
         marginal_all[i, i+1:] = pairwise_marginal_dia
                  
 
     return marginal_all


def main_jt():
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
       
    train_name = dataset_dir + data_name +'.ts.data'
    valid_name = dataset_dir + data_name +'.valid.data'
    test_name = dataset_dir + data_name +'.test.data'
    data_train = np.loadtxt(train_name, delimiter=',', dtype=np.uint32)
    data_valid = np.loadtxt(valid_name, delimiter=',', dtype=np.uint32)
    data_test = np.loadtxt(test_name, delimiter=',', dtype=np.uint32)
    
    clt = CLT()
    clt.learnStructure(data_train)
    print 'clt testset loglikihood score: ', clt.computeLL(data_test) / data_test.shape[0]
    
    n_variable = data_train.shape[1]
    clt.get_log_cond_cpt()

    
    jt = JunctionTree()
    jt.learn_structure(clt.topo_order, clt.parents, clt.cond_cpt)
    
    
    evid_list = []
    query_var = np.arange(n_variable)
    
    
    start = time.time()
    marginal = get_marginal_JT(jt, evid_list, query_var)

    print '------Marginals------'
    for i in xrange (query_var.shape[0]):

        print marginal[i]
    print 'running time for new: ', time.time()-start
    

if __name__=="__main__":

    #start = time.time()
    main_jt()
    #print ('Total running time: ', time.time() - start)