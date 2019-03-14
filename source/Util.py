import numpy as np
import time

class Util(object):
    def __init__(self):
        self.times=0

    @staticmethod
    def compute_xycounts_slow(dataset,timing=False):
        start = time.time()
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        for i in range(nvariables):
            for j in range(nvariables):
                prob_xy[i][j][0][0] = np.count_nonzero((dataset[:, i] == 0) & (dataset[:, j] == 0))
                prob_xy[i][j][0][1] = np.count_nonzero((dataset[:, i] == 0) & (dataset[:, j] == 1))
                prob_xy[i][j][1][0] = np.count_nonzero((dataset[:, i] == 1) & (dataset[:, j] == 0))
                prob_xy[i][j][1][1] = np.count_nonzero((dataset[:, i] == 1) & (dataset[:, j] == 1))
        if timing:
            print (time.time()-start)
        return prob_xy

    @staticmethod
    def compute_xcounts_slow(dataset,timing=False):
        start = time.time()
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        for i in range(nvariables):
            prob_x[i][0] = np.count_nonzero(dataset[:, i] == 0)
            prob_x[i][1] = dataset.shape[0]-prob_x[i][0]
        if timing:
            print(time.time() - start)
        return prob_x

    @staticmethod
    def compute_xycounts(dataset,timing=False):
        start = time.time()
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        
        prob_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 0).astype(int))
        prob_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 1).astype(int))
        prob_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 0).astype(int))
        prob_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 1).astype(int))
        if timing:
            print(time.time() - start)
        return prob_xy

    @staticmethod
    def compute_xcounts(dataset,timing=False):
        start = time.time()
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        prob_x[:,0]=np.einsum('ij->j',(dataset == 0).astype(int))
        prob_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int))
        if timing:
            print(time.time() - start)
        return prob_x
    
    @staticmethod
    def compute_weighted_xycounts(dataset,weights, timing=False):
        start = time.time()
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        
        prob_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int)* weights[:, np.newaxis], (dataset == 0).astype(int))
        prob_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int)* weights[:, np.newaxis], (dataset == 1).astype(int))
        prob_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int)* weights[:, np.newaxis], (dataset == 0).astype(int))
        prob_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int)* weights[:, np.newaxis], (dataset == 1).astype(int))
        if timing:
            print(time.time() - start)
        
        return prob_xy
    

    @staticmethod
    def compute_weighted_xcounts(dataset,weights, timing=False):
        start = time.time()
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        prob_x[:,0]=np.einsum('ij->j',(dataset == 0).astype(int) * weights[:, np.newaxis])
        prob_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int) * weights[:, np.newaxis])
        if timing:
            print(time.time() - start)
        return prob_x
    

    # compute the probability from dataset based on edges
    @staticmethod
    def compute_xycounts_edges(dataset,edges,timing=False):
        start = time.time()
        nedges=edges.shape[0]
        prob_xy = np.zeros((nedges, 2, 2))
        edge1_data = dataset[:,edges[:,0]]
        edge2_data = dataset[:,edges[:,1]]
        
        
        prob_xy[:, 0, 0] = np.einsum('ij,ij->j', (edge1_data == 0).astype(int), (edge2_data == 0).astype(int))
        prob_xy[:, 0, 1] = np.einsum('ij,ij->j', (edge1_data == 0).astype(int), (edge2_data == 1).astype(int))
        prob_xy[:, 1, 0] = np.einsum('ij,ij->j', (edge1_data == 1).astype(int), (edge2_data == 0).astype(int))
        prob_xy[:, 1, 1] = np.einsum('ij,ij->j', (edge1_data == 1).astype(int), (edge2_data == 1).astype(int))
        if timing:
            print(time.time() - start)
        return prob_xy

    @staticmethod
    def normalize2d(xycounts):
        xycountsf=xycounts.astype(np.float64)
        norm_const=np.einsum('ijkl->ij',xycountsf)
        return xycountsf/norm_const[:,:,np.newaxis,np.newaxis]

    @staticmethod
    def normalize1d(xcounts):
        xcountsf = xcounts.astype(np.float64)
        norm_const = np.einsum('ij->i', xcountsf)
        return xcountsf/norm_const[:,np.newaxis]

    @staticmethod
    def normalize(weights):
        norm_const=np.sum(weights)
        return weights/norm_const
    

    @staticmethod
    def normalize1d_in_2d(xycounts):
        xycountsf=xycounts.astype(np.float64) 
        norm_const=np.einsum('ijk->i',xycountsf)
        return xycountsf/norm_const[:,np.newaxis,np.newaxis]
    
    @staticmethod
    # normalize the matirx for each columns, and compute ll score
    # input weights are in log form
    # return normalized weights, Not in log form and ll score
    def m_step_trick(log_weights):

        max_arr = np.max(log_weights, axis = 0)

        
        weights = np.exp(log_weights - max_arr[np.newaxis,:])
        norm_const = np.einsum('ij->j', weights)
        weights = weights / norm_const[np.newaxis,:]
        
        ll_score = np.sum(np.log(norm_const)) + np.sum(max_arr)
        
        
        return weights, ll_score
    

    @staticmethod
    # normalize the matirx for each columns, and compute ll score
    # input weights are in log form
    # return normalized weights, Not in log form and ll score
    def get_ll_trick(log_weights):
        max_arr = np.max(log_weights, axis = 0)
        
        
        weights = np.exp(log_weights - max_arr[np.newaxis,:])
        norm_const = np.einsum('ij->j', weights)
        
        ll_scores = np.log(norm_const) + max_arr
        

        return ll_scores

    @staticmethod
    def compute_edge_weights_slow(xycounts, xcounts,timing=False):
        start = time.time()
        p_xy=Util.normalize2d(xycounts)
        p_x=Util.normalize1d(xcounts)
        log_px = np.log(p_x)
        log_pxy = np.log(p_xy)
        nvariables=p_x.shape[0]
        sum_xy = np.zeros((nvariables, nvariables))
        for i in range(nvariables):
            for j in range(nvariables):
                sum_xy[i][j] += p_xy[i][j][0][0] * (log_pxy[i][j][0][0] - log_px[i][0] - log_px[j][0])
                sum_xy[i][j] += p_xy[i][j][0][1] * (log_pxy[i][j][0][1] - log_px[i][0] - log_px[j][1])
                sum_xy[i][j] += p_xy[i][j][1][0] * (log_pxy[i][j][1][0] - log_px[i][1] - log_px[j][0])
                sum_xy[i][j] += p_xy[i][j][1][1] * (log_pxy[i][j][1][1] - log_px[i][1] - log_px[j][1])
        if timing:
            print(time.time() - start)
        return sum_xy

    @staticmethod
    def compute_edge_weights(xycounts,xcounts,timing=False):
        start = time.time()
        p_xy = Util.normalize2d(xycounts)
        p_x_r = np.reciprocal(Util.normalize1d(xcounts))
        
        
        sum_xy=np.zeros((p_x_r.shape[0], p_x_r.shape[0]))
        sum_xy += p_xy[:,:,0,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,0],p_x_r[:,0],p_x_r[:,0]))
        sum_xy += p_xy[:,:,0,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,1],p_x_r[:,0],p_x_r[:,1]))
        sum_xy += p_xy[:,:,1,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,0],p_x_r[:,1],p_x_r[:,0]))
        sum_xy += p_xy[:,:,1,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,1],p_x_r[:,1],p_x_r[:,1]))
        if timing:
            print(time.time() - start)
            print sum_xy

        return sum_xy
    
    
    @staticmethod
    #  basically the same as compute_edge_weights
    # The only difference is the input in probablity, not count
    def compute_MI_prob(p_xy,p_x,timing=False):
        start = time.time()
        p_x_r = np.reciprocal(p_x)
        
        sum_xy=np.zeros((p_x_r.shape[0], p_x_r.shape[0]))
        sum_xy += p_xy[:,:,0,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,0],p_x_r[:,0],p_x_r[:,0]))
        sum_xy += p_xy[:,:,0,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,1],p_x_r[:,0],p_x_r[:,1]))
        sum_xy += p_xy[:,:,1,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,0],p_x_r[:,1],p_x_r[:,0]))
        sum_xy += p_xy[:,:,1,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,1],p_x_r[:,1],p_x_r[:,1]))
        if timing:
            print(time.time() - start)
            print sum_xy

        return sum_xy
    

    @staticmethod
    def compute_conditional_CPT(xyprob,xprob,topo_order, parents, timing=False):
        
        start = time.time()        
        nvariables = xprob.shape[0]
        cond_cpt = np.zeros((nvariables,2,2))
        
        # for the root we have a redundant representation
        root = topo_order[0]
        cond_cpt[0, 0, :] = xprob[root, 0]
        cond_cpt[0, 1, :] = xprob[root, 1]
        
        

        
        for i in xrange(1, nvariables):
            x = topo_order[i]
            y = parents[x]
            
            # id, child, parent
            
            if (xprob[y, 0] == 0):
                cond_cpt[i, 0, 0] = 0
                cond_cpt[i, 1, 0] = 0
            else:
                cond_cpt[i, 0, 0] = xyprob[x, y, 0, 0] / xprob[y, 0]
                cond_cpt[i, 1, 0] = xyprob[x, y, 1, 0] / xprob[y, 0]
            
            if (xprob[y, 1] == 0):
                cond_cpt[i, 0, 1] = 0
                cond_cpt[i, 1, 1] = 0
            else:
                cond_cpt[i, 0, 1] = xyprob[x, y, 0, 1] / xprob[y, 1]
                cond_cpt[i, 1, 1] = xyprob[x, y, 1, 1] / xprob[y, 1]
                
                
                
        

    
        
  
        
        if timing:
            print(time.time() - start)
        return cond_cpt
    
    @staticmethod
    def compute_edge_potential(xyprob, parents, timing=False):
        
                
        nvariables = parents.shape[0]
        edge_potential = np.zeros((nvariables,2,2))
        
        # for convinient, the first item is redundent
        edge_potential[0, :, :] = 0
        
        

        
        for x in xrange(1, nvariables):
            y = parents[x]
            
     
            edge_potential[x, :, :] = xyprob[x, y, :, :] 
        
        return edge_potential
