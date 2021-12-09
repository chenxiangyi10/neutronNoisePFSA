# neutronNoisePFSA

PC used for this work:

    The PC used to produce this work is with the following hardware:
        * i9-11850 cpu
        * 128 Gb memory
        * 1 Tb ssd
    The system is ubuntu 20.10:
        * 128 Gb swap (swap is virtual memory, uses hard disk to provide low speed extra memory)
        
    The MS windows system is also tested:
        * the PFSA written in python works with MS windows system
        * however, the parallel features does not work (you do not need to worry about this; it still runs, just slower)
        * the memory needed for PFSA to run on windows system is significantly lower than ubuntu
        * the reason could be the parallel features cannot be applied?



In this folder, the following files are included:

    - stochastic reactor julia-for_paper ipynb file (Written in JULIA script): 
        * it contains the model developed in this paper
        * it generates the figures of the model given in the paper
        
    - PFSA py file (A PFSA python class file which is imported in the PFSA neutron noise analysis ipynb files):
        * the PFSA module has much more features that were not used in this paper
        * the specific features that are used can be found in the following files
        
    - PFSA neutron noise analysis1 ipynb file (Written in python script):
        * analyze the window size effect on the anomaly detection accuracy for in core detector
        
    - PFSA neutron noise analysis2 ipynb file (Written in python script):
        * analyze the window size effect on the anomaly detection accuracy for ex core detector
        
    - PFSA neutron noise analysis3 ipynb file (Written in python script): 
    	* analyze the alphabet size effect on the anomaly detection accuracy for in core detector
    	
    - PFSA neutron noise analysis4 ipynb file (Written in python script): 
    	* analyze the alphabet size effect on the anomaly detection accuracy for ex core detector
    	
    - PFSA neutron noise analysis5 ipynb file (Written in python script): 
    	* analyze the frequency of sampling (fs=100) effect on the anomaly detection accuracy for in core detector
    	
    - PFSA neutron noise analysis6 ipynb file (Written in python script): 
    	* analyze the frequency of sampling (fs=10) effect on the anomaly detection accuracy for in core detector
    	
    - PFSA neutron noise analysis7 ipynb file (Written in python script): 
    	* analyze the frequency of sampling (fs=10) effect on the anomaly detection accuracy for ex core detector

    - PFSA neutron noise analysis8 ipynb file (Written in python script): 
    	* analyze the frequency of sampling (fs=100) effect on the anomaly detection accuracy for ex core detector
    	
    - PFSA neutron noise analysis9 ipynb file (Written in python script): 
    	* online anomaly detection
        
    - plot figure 8 (a) ipynb file (Written in python script):
        * as the name indicated
        
    - plot figure 8 (b) ipynb file  (Written in python script):
        * as the name indicated    
        
    - plto figure 9 ipynb file (Written in python script):
    	* as the name indicated
      
How to reproduce the work of the paper (step by step):

    (1) Run the stochastic reactor julia-for_paper ipynb file which generates the time series data.

    (2) Run the PFSA neutron noise analysis files.
