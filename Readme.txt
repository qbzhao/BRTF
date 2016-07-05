Matlab Code of Bayesian Robust Tensor Factorization (Written by Qibin Zhao 2014)


To run the code:
1. Change Matlab work directory to '\BayesRobustTensorFactorizationP\'.
2. Run  " loadpah "  command to add the current folder and subfolders into Matlab path searching list.
3. Open and run the demo files. 


We provide three demo codes:

I. DemoBayesRCP.m     Demonstration on synthesic data. 

The experimental settings for evaluations include that 
1) Tensor Size
2) True CP Rank
3) Data type (deterministic signals, random data)
4) Outliers ratio 
5) Noise SNR

The settings of algorithm does not need any changes, but if you like you can test
1) Initialization method ('ml', 'rand')
2) Initial Rank
3) If the components will be pruned out or not.
4) Convergence condition ('tol', 'maxiters')
5) Visualization during model learning ('verbose')


After the model learning, the results will be shown including
1. Performance RSE, RMSE, Estimated Rank, TimeCost, Noise estimation
2. Visualization of true latent factors and the estimated factors
3. visualization of observed tensor Y,  the estimated low-CP-rank tensor X and the sparse tensor S by cubic style visualization


The classical CP factorization algorithm is also performed for comparisons. 



II. DemoBayesRCP_TC.m     Demonstration on synthetic data for the case of incomplete tensor 

The experimental settings are similar to the previous one with an additional setting for Observation ratio.

The settings of algorithm is similar to the case of complete tensor. 


III. DemoVBM.m    Demonstration for video background modeling.  
 
We provide four videos as examples for test.


After model learning, the visualization of results are also provided. 