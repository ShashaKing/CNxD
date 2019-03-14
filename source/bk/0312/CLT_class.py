
"""
Define the Chow_liu Tree class
"""

#

from __future__ import print_function
import numpy as np

from Util import *

import utilM

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

import sys
import time



'''
Class Chow-Liu Tree.
Members:
    nvariables: Number of variables
    xycounts: 
        Sufficient statistics: counts of value assignments to all pairs of variables
        Four dimensional array: first two dimensions are variable indexes
        last two dimensions are value indexes 00,01,10,11
    xcounts:
        Sufficient statistics: counts of value assignments to each variable
        First dimension is variable, second dimension is value index [0][1]
    xyprob:
        xycounts converted to probabilities by normalizing them
    xprob:
        xcounts converted to probabilities by normalizing them
    topo_order:
        Topological ordering over the variables
    parents:
        Parent of each node. Parent[i] gives index of parent of variable indexed by i
        If Parent[i]=-9999 then i is the root node
'''
class CLT:
    def __init__(self):
        self.nvariables = 0
        self.xycounts = np.ones((1, 1, 2, 2), dtype=int)
        self.xcounts = np.ones((1, 2), dtype=int)
        self.xyprob = np.zeros((1, 1, 2, 2))
        self.xprob = np.zeros((1, 2))
        self.topo_order = []
        self.parents = []
        self.Tree = None 
        self.log_cond_cpt = []    # the log based conditional CPT
        self.save_info = None     # Used to save the clt to output file
        self.tree_path = []
        self.inst_cond_cpt = []   # save the instantiated cond cpt, used for unbalanced cnet
    '''
        Learn the structure of the Chow-Liu Tree using the given dataset
    '''
    def learnStructure(self, dataset):
        self.nvariables = dataset.shape[1]
        self.xycounts = Util.compute_xycounts(dataset) + 1 # laplace correction
        self.xcounts = Util.compute_xcounts(dataset) + 2 # laplace correction
        self.xyprob = Util.normalize2d(self.xycounts)
        self.xprob = Util.normalize1d(self.xcounts)
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
        edgemat[edgemat == 0.0] = 1e-20  # sha1225  # to avoid tree not connected
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)  
        self.get_log_cond_cpt()


    '''
        Learn the structure of the Chow-Liu Tree using the given p_xy and p_x
        
    '''
    def learnStructure_prob(self, p_xy, p_x):
        self.nvariables = p_x.shape[0]
        self.xyprob = p_xy
        self.xprob = p_x
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_MI_prob(self.xyprob, self.xprob) * (-1.0)
        edgemat[edgemat == 0.0] = 1e-20  # sha1225  # to avoid tree not connected
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
        self.get_log_cond_cpt()
    
    
    '''
        Learn the structure of the Chow-Liu Tree using the given mutual information
          
        Used only in specail cases
    '''
    def learnStructure_MI(self, mi):
        self.nvariables = mi.shape[0]
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = mi * (-1.0)
        # compute the minimum spanning tree
        edgemat[edgemat == 0.0] = 1e-20  # sha1225  # to avoid tree not connected
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
  
        
    
    
    '''
        Update the Chow-Liu Tree using weighted samples
    '''
    def update(self, dataset_, weights=np.array([])):
        # Perform Sampling importance resampling based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        if weights.shape[0]==dataset_.shape[0]:
            norm_weights = Util.normalize(weights)
            indices = np.argwhere(np.random.multinomial(dataset_.shape[0], norm_weights)).ravel()
            dataset = dataset_[indices, :]
        else:
            dataset=dataset_
            print ("Not using weight to update")
        self.xycounts += Util.compute_xycounts(dataset)
        self.xcounts += Util.compute_xcounts(dataset)
        self.xyprob = Util.normalize2d(self.xycounts)
        self.xprob = Util.normalize1d(self.xcounts)
        edgemat = Util.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
        
    '''
        Update the Chow-Liu Tree using weighted samples, exact update
    '''
    def update_exact(self, dataset_, weights=np.array([]), structure_update_flag = False):
        # Perform based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        # try to avoid sum(weights = 0
        
        if weights.shape[0]==dataset_.shape[0] and np.sum(weights > 0):
    
            smooth = max (np.sum(weights), 1.0) / dataset_.shape[0]
            self.xycounts = Util.compute_weighted_xycounts(dataset_, weights) + smooth
            self.xcounts = Util.compute_weighted_xcounts(dataset_, weights) + 2.0 *smooth
        else:
            dataset=dataset_
            print ("Not using weight to update")
            self.xycounts += Util.compute_xycounts(dataset)
            self.xcounts += Util.compute_xcounts(dataset)
        
        self.xyprob = Util.normalize2d(self.xycounts)
        self.xprob = Util.normalize1d(self.xcounts)
        
        if structure_update_flag == True:

            edgemat = Util.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
            Tree = minimum_spanning_tree(csr_matrix(edgemat))
            self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
    
    '''
        Compute the Log-likelihood score of the dataset
    '''    
    def computeLL(self,dataset):
        prob=0.0
        

        if self.xyprob.shape[0] != dataset.shape[1]:
            return utilM.get_tree_dataset_ll(dataset,self.topo_order, self.parents, self.log_cond_cpt)
        
        for i in range(dataset.shape[0]):
            for x in self.topo_order:
                assignx=dataset[i,x]
                # if root sample from marginal
                if self.parents[x] == -9999:
                    prob+=np.log(self.xprob[x][assignx])
                else:
                    # sample from p(x|y)
                    y = self.parents[x]
                    assigny = dataset[i,y]
                    prob+=np.log(self.xyprob[x, y, assignx, assigny] / self.xprob[y, assigny])
        return prob
    def generate_samples(self, numsamples):
        samples = np.zeros((numsamples, self.nvariables), dtype=int)
        for i in range(numsamples):
            for x in self.topo_order:
                # if root sample from marginal
                if self.parents[x] == -9999:
                    samples[i, x] = int(np.random.random() > self.xprob[x, 0])
                else:
                    # sample from p(x|y)
                    y = self.parents[x]
                    assigny = samples[i, y]
                    prob=self.xyprob[x, y, 0, assigny] / self.xprob[y, assigny]
                    samples[i, x] = int(np.random.random() > prob)
        return samples
    
    '''
        Get the log based conditional CPT
    '''
    def get_log_cond_cpt(self):
        # pairwised egde CPT in log format based on tree structure
        self.cond_cpt = Util.compute_conditional_CPT(self.xyprob,self.xprob,self.topo_order, self.parents)
        self.log_cond_cpt = np.log(self.cond_cpt)
    
    
    '''
        Get the weights of each sample in samples
    '''
    def getWeights(self, samples):
        
        self.get_log_cond_cpt()
        
        probs = utilM.get_sample_ll(samples,self.topo_order, self.parents, self.log_cond_cpt)
        return probs
    
    '''
        Find the path from each node to root
    '''
    def get_tree_path(self):
        
        self.tree_path.append([0])
        for i in xrange(1,self.nvariables):
            single_path = []
            single_path.append(i)
            curr = i
            while curr!=0:
                curr = self.parents[curr]
                single_path.append(curr)
        
            self.tree_path.append(single_path)
    

    '''
        set the evidence
    '''
    def instantiation(self, evid_list):

        self.inst_cond_cpt = []
        if len(evid_list) == 0:  # no evidence
            self.cond_cpt = np.exp(self.log_cond_cpt)
            self.inst_cond_cpt = np.copy(self.cond_cpt)
            return self.cond_cpt
        

        cond_cpt = np.copy(self.cond_cpt)
        
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
        
       
        self.inst_cond_cpt =  cond_cpt    # for unbalaced cnet
        return cond_cpt
    

    '''
        Get the pairwised probablity matrix
    '''    
    def inference(self, cond_cpt, ids):
        return utilM.get_prob_matrix(self.topo_order, self.parents, cond_cpt, ids)
        
    
    
    """
        FOR CUTSET_deep
    """
    
    def get_node_marginal(self, cond_cpt, var):
        
        return utilM.get_var_prob(self.topo_order, self.parents, cond_cpt, var)
    
    def get_edge_marginal(self, cond_cpt, edges):
        
        return utilM.get_edge_prob(self.topo_order, self.parents, cond_cpt, edges)
    

  


    """
        For knowing the structure, update paramter only from data and TUM
    """
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
   Main function for Learning clt 
'''
def main_clt():
            
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    
    
    train_name = dataset_dir + data_name +'.ts.data'
    valid_name = dataset_dir + data_name +'.valid.data'
    test_name = dataset_dir + data_name +'.test.data'
    data_train = np.loadtxt(train_name, delimiter=',', dtype=np.uint32)
    data_valid = np.loadtxt(valid_name, delimiter=',', dtype=np.uint32)
    data_test = np.loadtxt(test_name, delimiter=',', dtype=np.uint32)
    
    
    print("Learning Chow-Liu Trees on original data ......")
    clt = CLT()
    clt.learnStructure(data_train)
    
   
    valid_ll = clt.computeLL(data_valid) / data_valid.shape[0]
    test_ll = clt.computeLL(data_test) / data_test.shape[0]
    
      
    print('Test set LL scores')
    print(test_ll, "Mixture-Chow-Liu")
   


    print('Valid set LL scores')
    print(valid_ll, "Mixture-Chow-Liu")
    


    


if __name__=="__main__":

    start = time.time()
    main_clt()
    print ('Total running time: ', time.time() - start)       