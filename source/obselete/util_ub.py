#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:04:51 2018

@author:
"""
# The util file for unbalaced Cnet
import numpy as np
import utilM
import numba

KL_ThreshHold = 1.0

#@numba.jit
# The varible eliminate for tree structure with only binary variables
# The sub routing for KL divergence
# Note here edge_potential is in log format
def ve_kl_bin(topo_order, parents, cond_cpt, edge_var, log_edge_potential):
    
#    print ('-------------------IN Util_ub---------------')
#    print ('topo_order: ', topo_order)
#    print ('edges: ')
#    print np.vstack((topo_order[1:], parents[topo_order[1:]])).T
#    print ('edge var:', edge_var)
#    print ('log_edge_potential: ', log_edge_potential)
#    print ('cond cpt:')
#    print (cond_cpt)
    
    # same edge found
    if parents[edge_var[0]] == edge_var[1]:
        cond_cpt_cp = np.copy(cond_cpt)
        ind = np.where(topo_order == edge_var[0])[0]
        cond_cpt_cp[ind,:,:] *= log_edge_potential
        #print cond_cpt_cp
        return utilM.ve_tree_bin(topo_order, parents, cond_cpt_cp)
    
    # same edge found, but in reversed order
    if parents[edge_var[1]] == edge_var[0]:
        cond_cpt_cp = np.copy(cond_cpt)
        ind = np.where(topo_order == edge_var[1])[0]
        adjust_potential = np.copy(log_edge_potential)
        # swap the rows
        adjust_potential[0,1],adjust_potential[1,0] = adjust_potential[1,0],adjust_potential[0,1]
        cond_cpt_cp[ind,:,:] *= adjust_potential
        #print cond_cpt_cp
        return utilM.ve_tree_bin(topo_order, parents, cond_cpt_cp)
    
    
    
    #print (topo_order)
    #print (parents)
    #print (np.exp(log_cond_cpt))
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))
    cond_cpt_cp = np.copy(cond_cpt)
    edge_cp = np.copy(edge_var)
    edge_potential_cp = np.copy (log_edge_potential)
    
    # loop in reverse order, this loop exclude the root 
    for i in xrange(nvariables-1, 0, -1):
        
        
        x = topo_order[i]
        
        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        #single_cpt = np.copy(cond_cpt[i])
        single_cpt =  cond_cpt_cp[i]
        #print "single cpt: ", single_cpt
        cpt = np.ones(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        if x == edge_cp[0]:
            #print 'x is the first node of the edge: ', x
            if y == edge_cp[1]:
                #print 'x, y now connected'
                single_cpt *= edge_potential_cp
                edge_cp[:] = -1
                edge_potential_cp =  None
                #print "cpt before: ", np.exp( cpt)
                cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
                cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])        
                #print ("cpt: ", np.exp(cpt))
                cpt_income[y] *= cpt
                #print 'cpt income: ', cpt_income
        
        #print np.exp(cpt_income)
            else:
                edge_cp[0] = y
                #print 'update edge: ', edge_cp
                temp1 = single_cpt * edge_potential_cp
                # swap the 2 columns
                edge_potential_cp[:,[0, 1]] = edge_potential_cp[:,[1, 0]]
                temp2 = single_cpt * edge_potential_cp
                temp3 = temp1[0,:] + temp1[1,:]
                temp4 = temp2[0,:] + temp2[1,:]
                edge_potential_cp [0,0] = temp3[0]
                edge_potential_cp [0,1] = temp4[0]
                edge_potential_cp [1,0] = temp4[1]
                edge_potential_cp [1,1] = temp3[1]
                
                #print 'edge potential cp: ', edge_potential_cp
            continue
    
        if x == edge_cp[1]:
            #print 'x is the second node of the edge: ', x
            if y == edge_cp[0]:
                #print 'x, y now connected'
                edge_potential_cp[0,1],edge_potential_cp[1,0] = edge_potential_cp[1,0],edge_potential_cp[0,1]
                single_cpt *= edge_potential_cp                
                edge_potential_cp =  None
                edge_cp[:] = -1
                #print "cpt before: ", np.exp( cpt)
                cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
                cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])        
                #print ("cpt: ", np.exp(cpt))
                cpt_income[y] *= cpt
                #print 'cpt income: ', cpt_income
            else:
                edge_cp[1] = y
                #print 'update edge: ', edge_cp
                edge_potential_cp = edge_potential_cp.T
                temp1 = single_cpt * edge_potential_cp
                edge_potential_cp[:,[0, 1]] = edge_potential_cp[:,[1, 0]]
                temp2 = single_cpt * edge_potential_cp
                temp3 = temp1[0,:] + temp1[1,:]
                temp4 = temp2[0,:] + temp2[1,:]
                edge_potential_cp [0,0] = temp3[0]
                edge_potential_cp [0,1] = temp4[1]
                edge_potential_cp [1,0] = temp4[0]
                edge_potential_cp [1,1] = temp3[1]
                #print 'edge potential cp: ', edge_potential_cp
            continue
    
        
        #print "cpt before: ", np.exp( cpt)
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
        
        #print ("cpt: ", np.exp(cpt))
        
        cpt_income[y] *= cpt
        
        #print np.exp(cpt_income)
        
        
    
    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] * cond_cpt[0,:,0]
    
    
    #print "cpt income" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    return root_cpt[0]+ root_cpt[1]



## compute P(X)log(Q(X))
## cltP, the P distribution, 1 tree from Mixture of clt
## cltQ, the Q distribution, a leaf node of cutset network
#def kl_div(cltP, edges, log_edge_potential):
#    
#    partial_value = 0.0
#    for i,e in enumerate(edges):
#        partial_value += ve_kl_bin(cltP.topo_order, cltP.parents, cltP.inst_cond_cpt, e, log_edge_potential[i])
#    
#    return partial_value
#        

"""
# using very simple example to test
"""
#n_variable = 4
#topo_order = np.array([0,2,1,3])
#parents = np.array([-9999,0,0,1])    
#topo_order = np.array([0,1,2,3])
#parents = np.array([-9999,0,1,1]) 
#cond_cpt = np.zeros((4,2,2))
#cond_cpt[0,0,0] = 0.3
#cond_cpt[0,0,1] = 0.3
#cond_cpt[0,1,0] = 0.7
#cond_cpt[0,1,1] = 0.7
#cond_cpt[1,0,0] = 0.2
#cond_cpt[1,0,1] = 0.4
#cond_cpt[1,1,0] = 0.8
#cond_cpt[1,1,1] = 0.6
#cond_cpt[2,0,0] = 0.3
#cond_cpt[2,0,1] = 0.1
#cond_cpt[2,1,0] = 0.7
#cond_cpt[2,1,1] = 0.9
#cond_cpt[3,0,0] = 0.8
#cond_cpt[3,0,1] = 0.7
#cond_cpt[3,1,0] = 0.2
#cond_cpt[3,1,1] = 0.3
##log_cond_cpt = np.log(cond_cpt)
#
#edge_var = np.zeros(2,dtype = int)
#edge_var[0] = 2
#edge_var[1] = 3
#log_edge_potential = np.zeros((2,2))
#log_edge_potential[0,0] = -0.1
#log_edge_potential[0,1] = -0.2
#log_edge_potential[1,0] = -0.3
#log_edge_potential[1,1] = -0.4
#
#
#kl_value = ve_kl_bin(topo_order, parents, cond_cpt, edge_var, log_edge_potential)
#print ('The test KL value: ', kl_value)
##print cond_cpt
##print log_edge_potential