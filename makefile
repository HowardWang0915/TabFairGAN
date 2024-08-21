# Makefile for running TabFairGAN.py commands

# Default target
all: no_fairness with_fairness

# Target for running TabFairGAN.py with no fairness
no_fairness_adult:
	python TabFairGAN.py no_fairness experiments/adult/adult.csv 300 256 experiments/adult/fake_adult.csv 32561

# Target for running TabFairGAN.py with fairness
with_fairness_adult:
	python TabFairGAN.py with_fairness experiments/adult/adult.csv "sex" "income" " Female" " >50K" 100 256 30 0.05 experiments/adult/fake_adult_with_fairness.csv 32561
with_cond_ind_adult:
	python TabFairGAN.py with_cond_ind experiments/adult/adult_race_processed.csv "sex" "income" "race" " Female" " >50K" " White" 200 256 30 0.05 experiments/adult/fake_adult_with_cond_ind.csv 32561
with_cond_ind_adult_all:
	python CI-Constraint.py experiments/adult/adult.csv "sex" "income" "race" 200 256 30 0.005 experiments/adult/fake_adult_with_cond_ind_all.csv 32561
no_fairness_bank:
	python TabFairGAN.py no_fairness experiments/bank/bank_full_modified.csv 300 256 experiments/bank/fake_bank.csv 45212

with_fairness_bank:
	python TabFairGAN.py with_fairness experiments/bank/bank_full_modified.csv "age_binary" "y" "older" "yes" 200 256 20 0.05 experiments/bank/fake_bank_with_fairness.csv 45212
no_fairness_compas:
	python TabFairGAN.py no_fairness experiments/compas/Compas_Raw_Scores_Modified.csv 200 256 experiments/compas/fake_compas.csv 16268
	
with_fairness_compas:
	python TabFairGAN.py with_fairness experiments/compas/Compas_Raw_Scores_Modified.csv "Ethnic_Code_Text" "binary_text" "African-American" "Low_Chance" 200 256 45 0.05 experiments/compas/fake_compas_with_fairness.csv 16268
	
# Clean target (optional)
clean:
	# Add commands to clean up if necessary (e.g., remove generated files)

# Phony target to prevent conflicts with files named as target names
.PHONY: all no_fairness with_fairness clean
