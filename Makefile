# ======================================================================
# Makefile for MLOps Training Project 
# ======================================================================

# ---------------------------- VARIABLES --------------------------------
INPUT_DATA_PATH   ?= datastores/raw_data/housing.csv
OUTPUT_FILENAME   ?= clean_housing.csv
INPUT_TRAIN_DATA  ?= datastores/splits_data/train_data.csv
INPUT_TEST_DATA   ?= datastores/splits_data/test_data.csv
MODEL_FILENAME    ?= LinearRegression.joblib
CONDA_ENV         ?= ml_env

# --------------------------- DEFAULT TARGETS ------------------------------
.PHONY: env_update install_dependencies update_dependencies clean split train pipeline

# ======================================================================
# ENVIRONMENT MANAGEMENT
# ======================================================================

env_update:
	@echo "=> Updating conda environment from conda.yaml (env: $(CONDA_ENV))"
	conda env update -f conda.yaml 
	@echo "=> Conda environment '$(CONDA_ENV)' updated successfully."

# ======================================================================
# DATA PREPROCESSING, SPLITING & TRAINING
# ======================================================================

clean:
	@echo "=> Running data preprocessing..."
	python ml_houseprice_prediction/src/ml_houseprice_prediction/data_preprocessing/preprocessing.py \
		--input_data_path $(INPUT_DATA_PATH) \
		--output_data_filename $(OUTPUT_FILENAME)
	@echo "=> Data preprocessing completed. Clean data saved to $(OUTPUT_FILENAME)."

split:
	@echo "=> Running splits data ..."
	python ml_houseprice_prediction/src/ml_houseprice_prediction/data_splits/splits.py \
		--input_data_path datastores/clean_data/$(OUTPUT_FILENAME)
	@echo "=> Splits completed successfully."

train:
	@echo "=> Running train model..."
	python ml_houseprice_prediction/src/ml_houseprice_prediction/train_model/train.py \
		--input_train_data $(INPUT_TRAIN_DATA) \
		--input_test_data $(INPUT_TEST_DATA) \
		--model_filename $(MODEL_FILENAME)
	@echo "=> Train model completed successfully."

# ======================================================================
# ALL-IN-ONE WORKFLOW
# ======================================================================

pipeline: clean split train
	@echo "All tasks completed successfully."
