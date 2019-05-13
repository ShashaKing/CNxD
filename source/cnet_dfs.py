
"""
    When training CNet, we using recursive methods. But we store the trained 
    Module in a DFS manner and we need the evalution to be done in dfs way


"""

from __future__ import print_function
import numpy as np

import utilM
from collections import deque



class Leaf_tree:
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

        cond_cpt = np.exp(self.log_cond_cpt)

        for i in xrange (len(evid_list)):
            variable_id = evid_list[i][0]
            value = evid_list[i][1]
            
            index_c = np.where(self.topo_order==variable_id)[0][0]
            # variable as parent
            varible_child = np.where(self.parents ==variable_id)[0]
            ix = np.isin(self.topo_order, varible_child)
            index_p = np.where(ix)[0]
            
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
            
        return cond_cpt
    
    # set the evidence, using log space
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
        
        return log_cond_cpt
        

    
    def ind_as_parent(self, e_var):
        
        for i, e in enumerate(e_var):
        
            # evidence variable as parent
            e_child = np.where(self.parents ==e)[0]
            ix = np.isin(self.topo_order, e_child)
            index_p = np.where(ix)[0]

            self.as_parent.append(index_p)
            
            
            
            
    def ind_as_child(self, e_var):
        
        self.as_child = np.zeros(e_var.shape[0], dtype = int)
        
        for i, e in enumerate(e_var):
            
            # evidence variable as child
            index_c = np.where(self.topo_order==e)[0][0]
            self.as_child[i] = index_c


   



    
    


class CNode:



    def __init__(self, var, weights, ids, id):
        
        self.var = var  # the variable id
        self.var_assign = 0  # the assignment of var in map tuple
        self.children = []    # only has 2 child
        self.weights = weights 
        self.log_weights = np.log(weights)
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
            
            
        
        
         

    


    
    
class CNET_dfs:
    def __init__(self,load_info, n_variables):
        self.nvariables=n_variables
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
        cent_nodes.append(root)

        while cent_nodes:
            cnet = cent_nodes.popleft()
            ids = np.delete(cnet.ids, cnet.id)
            left_child = cnet.children[0]
            right_child = cnet.children[1]
            
            
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
                new_node_left = Leaf_tree()
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
                new_node_right = Leaf_tree()
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
            if isinstance(curr_node, CNode):
                internal_list.append(curr_node)
                nodes_to_process.append(curr_node.children[0])
                nodes_to_process.append(curr_node.children[1])
            elif isinstance(curr_node, Leaf_tree):
                leaf_list.append(curr_node)
            else:
                print ('Error, invalid node')
                exit()
                
        return internal_list, leaf_list

        
    def instantiation(self, datapoint, e_var, evid_flag, internal_list, leaf_list):
        
        log_max_value = 0.0
        
        # instantiate the leaf node
        log_cond_cpt_list = []
        
        for t in leaf_list:
            
            t_evalue = datapoint[t.ids][t.evar]
            log_cond_cpt = t.instantiation_log(t_evalue)
            log_cond_cpt_list.append(log_cond_cpt)
        
        
        log_inst_weight_list = []
        for c in internal_list:
            log_weights =  np.copy(c.log_weights)
            
            if evid_flag[c.var] == 1: # evidence
                
                if datapoint[c.var] == 0:  # evidence = 0
                    log_weights[1] = utilM.LOG_ZERO
                else:       # evidence = 0
                    log_weights[0] = utilM.LOG_ZERO
            
            log_inst_weight_list.append(log_weights)
            
        
        # Get maxout value
        for i, t in enumerate(leaf_list):
            
            t.log_value, t.map_tuple = utilM.max_tree_bin_map(t.topo_order, t.parents, log_cond_cpt_list[i])
            

        # in reverse order
        for j in xrange(len(internal_list)-1,-1,-1):
            c = internal_list[j]
            c.maxout(log_inst_weight_list[j])
            
        log_max_value = internal_list[0].log_value
    
        
        # back propergate, from root to leaf to find the map tuple
        max_tuple = np.zeros(datapoint.shape[0], dtype = int)
        back_node = internal_list[0]
        
        while isinstance(back_node, CNode):
           
            max_tuple[back_node.var] = back_node.var_assign
            
            if back_node.var_assign == 0:
                back_node =   back_node.children[0]
            else:
                back_node =   back_node.children[1]
        
        # reach the leaf tree node        
        max_tuple[back_node.ids] = back_node.map_tuple   
        

        return log_max_value, max_tuple
    
