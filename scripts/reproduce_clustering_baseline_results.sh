for i in 0.01 0.02154435  0.04641589  0.1  0.21544347 0.46415888 1. 2.15443469 4.64158883 10. 
do
    python run_experiment.py --config "configurations/fashionMNIST.yaml" --run_name "r_10_fashionMNIST_dirichlet_exp" --use_kmeans_for_clustering True --split_with_dirichlet True --dirichlet_param $i
    python run_experiment.py --config "configurations/fashionMNIST.yaml" --run_name "r_10_fashionMNIST_dirichlet_exp" --use_kmeans_for_clustering False --split_with_dirichlet True --dirichlet_param $i
done

