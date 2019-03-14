# CNxD
Learning an Accurate Cutset Networks via Compilation
Running environment: Python 2.7
----------------------------------------HELP----------------------------------------------
-p                The program name
-dir              The directory of input dataset
-dn               The name of the dataset
-ncomp            The number of components in mixture or bag
-max_iter         The maximun iterations to stop training (only used in mt)
-eps              The training stop criteria (only used in mt)
-depth            The depth of cutset network
-min_depth        The minimun depth when training a set of cutset networks
-max_depth        The maximun depth when training a set of cutset networks
-a                A hyper parameter, used to tune the percentage of data statistic used 
                    when train CNxd and RCN. 0<=a<=1.0
-f                A hyper parameter, adjust a by number_of_records_left / total_records.
                    Now only support 'root', 'linear' and square'
-sp               The 'OR' nodes selection option.
                    Only used in 'Bag of CNets (bcnet)', could be 0 or 1.
                    0 means optimaly select OR node using MI; 
                    1 means select OR node from 0.5 percent of all variables
-dp               The depth_option. 
                    Only used in 'Bag of CNets (bcnet)', could be 0,1 or 2 
                    0 means all cnets have the same depth (max depth)
                    1 means the depth of cnets are randomly choosing from 1 to 6
                    2 means the depht of cnets are choosed seqencially from 1 to 6
-t                Type
                    'structure' or 'parm' when learning RCN
                    'cnxd', 'cnd', 'rcn', 'mt', 'bcnet' during MAP inference
-e                The percentage of evidence variables
-seq              The index of which set of evidence is used
-input_dir        The directory of MAP intractable module used in training CNxD or RCN
-input_module     The MAP intractable module used in training CNxD or RCN
-output_dir       The output dir to store the trained modules
-efile            The full path of evidence files
    


Module training examples:
1) Learning Chow_Liu tree:
    python main.py -p 'clt' -dir  '../dataset/'  -dn  'nltcs'
2) Learning Mixture of Chow_Liu tree:
    python main.py -p 'mt' -dir  '../dataset/'  -dn  'nltcs' -ncomp   5  -max_iter   100   -eps   1e-7 -output_dir '../output/mt/'
3) Learning Bags of Cutset networks
    python main.py -p 'bcnet'  -dir   '../dataset/'   -dn   'nltcs'  -ncomp   5 -max_depth 5  -sp   0 -dp 0 -input_dir '../output/mt/' -input_module 'nltcs_5' -output_dir '../output/bcnet/'
4) Learning Cutset Network from Data
    python main.py -p 'cnd' -dir  '../dataset/'  -dn  'nltcs'  -max_depth   10   -output_dir '../output/cnd/'
5) Learning CNxD
    python main.py -p 'cnxd'  -dir   '../dataset/'   -dn   'nltcs'  -a  0.5  -f  'root' -min_depth 1 -max_depth 5  -input_dir '../output/mt/' -input_module 'nltcs_5' -output_dir '../output/cnxd/nltcs/'
6) Learning Random Cutset Network (RCN): structure is random while parameters are learnt  
    i) Get the structure
        python main.py -p 'rcn' -dir   '../dataset/'   -dn   'nltcs'   -t 'structure' -min_depth 1 -max_depth 10 -output_dir '../output/rcn/'
    ii) learn parameters
        python main.py -p 'rcn' -dir   '../dataset/'   -dn   'nltcs'   -t 'parm' -depth 4 -input_dir '../output/mt/' -input_module 'nltcs_5' -output_dir '../output/rcn/'


MAP inference examples:
    python main.py -p 'map' -dir   '../dataset/'   -dn   'nltcs'  -t 'cnxd' -depth 5 -e 0.2 -seq 0 -efile '../efile/nltcs_evid_20.txt' -input_dir '../output/cnxd/' -input_module 'nltcs_5' -output_dir '../map/cnxd/'
    python main.py -p 'map' -dir   '../dataset/'   -dn   'nltcs'  -t 'cnd' -depth 5 -e 0.2 -seq 0 -efile '../efile/nltcs_evid_20.txt' -input_dir '../output/cnd/' -input_module 'nltcs' -output_dir '../map/cnd/'
    python main.py -p 'map' -dir   '../dataset/'   -dn   'nltcs'  -t 'rcn' -depth 4 -e 0.2 -seq 0 -efile '../efile/nltcs_evid_20.txt' -input_dir '../output/rcn/' -input_module 'nltcs_4' -output_dir '../map/rcn/'
    python main.py -p 'map' -dir   '../dataset/'   -dn   'nltcs'  -t 'mt' -e 0.2 -seq 0 -efile '../efile/nltcs_evid_20.txt' -input_dir '../output/mt/' -input_module 'nltcs_5' -output_dir '../map/mt/'
    python main.py -p 'map' -dir   '../dataset/'   -dn   'nltcs'  -t 'bcnet' -e 0.2 -seq 0 -efile '../efile/nltcs_evid_20.txt' -input_dir '../output/bcnet/' -input_module 'nltcs' -output_dir '../map/bcnet/'
