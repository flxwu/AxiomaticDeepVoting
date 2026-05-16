--------------------------------------------------------------------------------------------
---                                                                                         
SetTransV2 n (NW,C,P)         100.0  100.0     100.00   100.0   50.00   90.00               
SetTransV3 n (NW,C,P)         100.0  100.0      93.00   100.0   66.00   91.80               
============================================================================================
===                                                                                         
                                                                                            
  vs n-WEC:  ▲ 3.26% improvement in average axiom satisfaction                              
  vs v2:     ▲ 1.80% improvement in average axiom satisfaction                              
                                                                                            
  Per-axiom vs v2:                                                                          
    Anonymity      :  100.0% (v2: 100.0%) = 0.0%                                            
    Neutrality     :  100.0% (v2: 100.0%) = 0.0%                                            
    Condorcet      :   93.0% (v2: 100.0%) ▼ 7.0%                                            
    Pareto         :  100.0% (v2: 100.0%) = 0.0%                                            
    Independence   :   66.0% (v2: 50.0%) ▲ 16.0%                                            
                                                                                            
Runtime: 631.1 minutes                                                                      
Results saved to: ./results/exp3/SetTransformerV3/exp3_2026-05-15_10-24-01_IC               
                                                                                            
======================================================================                      
  EXPERIMENT COMPLETE                                                                       
  Results at: ./results/exp3/SetTransformerV3/exp3_2026-05-15_10-24-01_IC                   
======================================================================                      
  Output files:                                                                             
    training_progress.png  — loss + axiom satisfaction + LR                                 
    final_comparison.png   — bar chart vs v2 vs n-WEC                                       
    results.json           — full numerical results                                         
    model.pth              — saved model weights                                            
====================================================================== 