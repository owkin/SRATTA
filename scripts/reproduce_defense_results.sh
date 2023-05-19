python run_experiment.py --config "configurations/defenses/dna_search_lr.yaml" --prun_risky_update_threshold 0 --run_name "dna"

python run_experiment.py --config "configurations/defenses/dna_search_lr.yaml" --prun_risky_update_threshold 1 --run_name "dna_q1"
python run_experiment.py --config "configurations/defenses/dna_search_lr.yaml" --prun_risky_update_threshold 4 --run_name "dna_q4"
python run_experiment.py --config "configurations/defenses/dna_search_lr.yaml"  --prun_risky_rel_lambda_threshold 0.9 --run_name "dna_beta9" 
python run_experiment.py --config "configurations/defenses/dna_search_lr.yaml"  --prun_risky_rel_lambda_threshold 0.99 --run_name "dna_beta99" 

python run_experiment.py --config "configurations/defenses/fashion_mnist_search_lr.yaml"  --prun_risky_update_threshold 0 --run_name "fashion_mnist"

python run_experiment.py --config "configurations/defenses/fashion_mnist_search_lr.yaml"  --prun_risky_update_threshold 1 --run_name "fashion_mnist_q1"
python run_experiment.py --config "configurations/defenses/fashion_mnist_search_lr.yaml"  --prun_risky_update_threshold 4 --run_name "fashion_mnist_q4"
python run_experiment.py --config "configurations/defenses/fashion_mnist_search_lr.yaml"  --prun_risky_rel_lambda_threshold 0.9 --run_name "fashion_mnist_beta9" 
python run_experiment.py --config "configurations/defenses/fashion_mnist_search_lr.yaml"  --prun_risky_rel_lambda_threshold 0.99 --run_name "fashion_mnist_beta99" 

python run_experiment.py --config "configurations/defenses/cifar10_search_lr.yaml"  --prun_risky_update_threshold 0 --run_name "cifar10"

python run_experiment.py --config "configurations/defenses/cifar10_search_lr.yaml"  --prun_risky_update_threshold 1 --run_name "cifar10_q1"
python run_experiment.py --config "configurations/defenses/cifar10_search_lr.yaml"  --prun_risky_update_threshold 4 --run_name "cifar10_q4"
python run_experiment.py --config "configurations/defenses/cifar10_search_lr.yaml"  --prun_risky_rel_lambda_threshold 0.9 --run_name "cifar10_beta9" 
python run_experiment.py --config "configurations/defenses/cifar10_search_lr.yaml"  --prun_risky_rel_lambda_threshold 0.99 --run_name "cifar10_beta99" 
