# A movement-aware trajectory similarity measure and its application for ride-sharing path extraction in a road network

This repository contains the codes and testing datasets of the DSPD proposed in the study for measuring the trajectory simialrity which has been published by International Journal of Geographical Information Science. 

The proposed DSPD is a powerful parameter-free similarity measure that can effectively distinguish trajectories with different movement characteristics. Compared to most existing similarity measures, the DSPD is robust in measuring the similarity of crowdsourced trajectories with different lengths, noise, or small perturbances.

If you would like to use the code and data from this study, please cite the following paper.  

[Peng J, Deng Min, Tang Jianbo*, et al. A movement-aware trajectory similarity measure and its application for ride-sharing path extraction in road network[J].International Journal of Geographical Information Science, 2024.  ](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2353695)

Also, it is noted that the implementation code for several baseline methods (sowd, discrete frechet, sspd, lcss, dtw, hausdorff) in the pydist folder  is reused from the public trajectory_distance package by Besse et al. (2016). https://github.com/bguillouet/traj-dist




## Requirement

- Python3 （tested on 3.8)

- numpy

- sklearn

- geopandas

- pandas

- networkx

- shapely

- hdbscan

- scipy

- pickle

- pandas

- matplotlib

  ​

## Description of the datasets

1. We provide six simulated datasets used in this paper.
   - It includes six trajectory pairs and five datasets of Cross, I5, I5-direct, I5SIM, I5SIM-direct.
   - The dataset of six trajectory paris provides the different possible situations of the trajectory pair.
   - The Cross, I5 ,I5SIM are provided by Morris and Trivedi (2009). 
   - The i5-direct and i5sim-direct are modified from the i5 and i5sim datasets by modifying the trajectory directions.
   - Field "IDX" in the I5, I5-direct, I5SIM, I5SIM-direct dataset describes the ground truth the trajectory cluster. 

2. The dataset used in the case study can be downloaded from this link: https://www.microsoft.com/en-us/download/details.aspx?id=52367

   ​


## Simulated datasets

  <img src="figs/Figure 3.jpg" alt="Six pairs of simulated trajectories. X and Y indicate the coordinates of the trajectory points.">
 

**Figure 1.** Six pairs of simulated trajectories. X and Y indicate the coordinates of the trajectory points.


  <img src="figs/Figure 4.jpg" alt="Simulated trajectory datasets (different colors represent different trajectory clusters)">
  **Figure 2.** Simulated trajectory datasets (different colors represent different trajectory clusters).



## Description of the code files

1. We provide the python and matlab implementations of the proposed trajectory similarity measure "DSPD".

2. In the provided python code, the DSPD and its optimization implementation are both provided. You can find them in the DSPD.py 
   - function "directed_spd(*args)" 
   - function "fast_directed_spd(*args)" 

3. In the file "calculate_distance_matrix.py", you can calculate the distance matrices of the proposed DSPD and baseline measures.

4. In the file "simulated_datasets_clustering_comparison_Ex.py", we provide the clustering comparison experiments using the DSPD and baselines on the five simulated datasets.

5. In the file "six_trajectory_pairs_Ex.py", the comparison of the DSPD and baselines was conducted on the six trajectory pairs. 

6. Files "case_study1.p" and "case_study2.py" provide the code of the two case studies in the paper. 

7. For different parts of experiments in this paper, you may need to manually change the parameter if the parameter is required.

   ​


## Illustration of the proposed DSPD  



### <img src="figs/Figure 1.jpg" alt="Illustrations of the projected distance and segment-path distance of (a) projection distance from $p_i$ to $T^2$ and (b) segment-path distance of $ T^1$ to $T^2$">

**Figure 3.** Illustrations of the projected distance and segment-path distance of (a) projection distance from $$p_i$$ to $$T_2$$ and (b) segment-path distance of $$ T_1$$ to $$T_2$$



  <img src="figs/Figure 2.jpg" alt="Illustration of the calculation of direction similarity">

  **Figure 4.** Illustration of the calculation of direction similarity



## Running time  comparison of the  DSPD and baseline measures

  <img src="figs/Figure 5.jpg" alt="Running times of different trajectory similarity measures on the Cross dataset (with 1900 trajectories), the I5 dataset (with 806 trajectories), and the I5SIM dataset (with 800 trajectories)..">
**Figure 5.** Running times of different trajectory similarity measures on the Cross dataset (with 1900 trajectories), the I5 dataset (with 806 trajectories), and the I5SIM dataset (with 800 trajectories).