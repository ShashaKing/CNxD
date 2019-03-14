from __future__ import print_function
import sys
import time

from CLT_class import main_clt
from MIXTURE_CLT import main_mixture_clt
from CNET_class import main_cutset_opt
from CNXD import main_cnxd
from cnet_bag import main_bag_cnet
from rcn import main_rcn_structure, main_rcn_parm
from map_inference import main_map_cnet, main_map_mt, main_map_bcnet


def main():
    
    # set default value
    parms_dict =  {}
    parms_dict ['p'] = 'CNXD'
    parms_dict['dir'] = '../dataset/'
    parms_dict['dn'] = ''  # data_name
    parms_dict['ncomp'] = 2    # numbers of components, for mt or bcnet
    parms_dict['max_iter'] = 100    # max itertaion, for mt
    parms_dict['eps'] = 1e-7    # epilon, for mt or mt
    parms_dict['min_depth'] = 1    
    parms_dict['max_depth'] = 10
    parms_dict['depth'] = 5
    parms_dict['a'] = 0.1
    parms_dict['f'] = 'linear'
    parms_dict['sp'] = 0   #sel_opt
    parms_dict['dp'] = 0   #depth_option
    
    parms_dict['t'] = 'structure'

    
    parms_dict['seq'] = 0
    parms_dict['e'] = 0.2    
    parms_dict['efile'] = '../evidence_file/nltcs_0.2.txt'
    parms_dict['input_dir'] = '../input/'
    parms_dict['input_module'] = 'nltcs'
    parms_dict['output_dir'] = '../output/'
    

    for i in xrange(len(sys.argv)/ 2):
        ind = sys.argv[2*i+1][1:]
        val = sys.argv[2*i+2]
        if ind not in parms_dict.keys():        
            print ('****** ERROR, invalid paramter: ', ind)
            exit()
        else:
            parms_dict[ind] = val

    if parms_dict['p'] == 'clt':
        main_clt(parms_dict)
    elif parms_dict['p'] == 'mt':
        main_mixture_clt(parms_dict)
    elif parms_dict['p'] == 'cnd':
        main_cutset_opt(parms_dict)
    elif parms_dict['p'] == 'cnxd':
        main_cnxd(parms_dict)
    elif parms_dict['p'] == 'bcnet':
        main_bag_cnet(parms_dict)
    elif parms_dict['p'] == 'rcn':
        if parms_dict['t'] == 'parm':
            main_rcn_parm(parms_dict)
        else:
            main_rcn_structure(parms_dict)
    elif parms_dict['p'] == 'map':
        if parms_dict['t'] in ['cnxd','rcn','cnd']:
            main_map_cnet(parms_dict)
        elif parms_dict['t'] == 'mt':
            main_map_mt(parms_dict)
        elif parms_dict['t'] == 'bcnet':
            main_map_bcnet(parms_dict)
        else:
            print ('****** ERROR, invalid module name in MAP inference: ', parms_dict['t'])
        
    else:
        print ('****** ERROR, invalid program: ', parms_dict['p'])
        exit()
        
        
        
    

if __name__=="__main__":

    start = time.time()
    main()
    print ('Total running time: ', time.time() - start)      