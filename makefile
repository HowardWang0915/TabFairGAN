# Makefile for running TabFairGAN.py commands

# Default target
all: no_fairness with_fairness

# Target for running TabFairGAN.py with no fairness
no_fairness_adult:
	python TabFairGAN.py no_fairness experiments/adult/adult.csv 300 256 experiments/adult/fake_adult.csv 32561

# Target for running TabFairGAN.py with fairness
with_fairness_adult:
	python TabFairGAN.py with_fairness experiments/adult/adult.csv "sex" "income" " Female" " >50K" 10 256 30 0.5 experiments/adult/fake_adult_with_fairness.csv 32561

dp_no_fairness_adult:
	python TabFairGAN.py no_fairness experiments/adult/adult.csv 300 256 experiments/adult/fake_adult_with_dp.csv 32561 --with_dp

no_fairness_bank:
	python TabFairGAN.py no_fairness experiments/bank/bank_full_modified.csv 300 256 experiments/bank/fake_bank.csv 45212

with_fairness_bank:
	python TabFairGAN.py with_fairness experiments/bank/bank_full_modified.csv "age_binary" "y" " older" " yes" 10 256 30 0.5 experiments/bank/fake_bank_with_fairness.csv 45212
	
# Clean target (optional)
clean:
	# Add commands to clean up if necessary (e.g., remove generated files)

# Phony target to prevent conflicts with files named as target names
.PHONY: all no_fairness with_fairness clean
