#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

import sys
import json
import os

import src
import warnings


def main(targets):
    warnings.filterwarnings("ignore")

    config_dir = '/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/config'
    run_all = False

    if 'all' in targets:
        run_all = True

    if 'pred' in targets:
        # Load model and predict
        print('Prediction target recognized.')

        with open(os.path.join(config_dir, 'prediction-params.json'), 'r') as f:
            pred_params = json.load(f)

        print('Prediction configuration loaded.')

        print('Predicting labels.')
        src.models.pred(**pred_params)
        print('Prediction complete.')
      
    if 'dtpred' in targets:
        # Load model and predict
        print('dt Prediction target recognized.')
        src.models.dtpred()
        print('dt Prediction complete.')

    if 'test' in targets:
        # Test all targets with the configuration and example data found in
        # the ./test/ directory.
        config_dir = 'test/testconfig'
        run_all = True
        
    if 'data' in targets or run_all:
        # Load, clean, and preprocess data. Then store preprocessed data to
        # intermediate directory.

        print('Data target recognized.')

        with open(os.path.join(config_dir, 'data-params.json'), 'r') as f:
            data_params = json.load(f)
        print('Data configuration loaded.')
        
        print('Running ETL pipeline.')
        src.data.pipeline(**data_params)
        print('ETL pipeline complete.')

    if 'features' in targets or run_all:
        # Load preprocessed data, chunk each file into smaller windows of time,
        # then engineer features and write sets of features and labels to table.

        print('Features target recognized.')

        with open(os.path.join(config_dir, 'features-params.json'), 'r') as f:
            features_params = json.load(f)
        print('Features configuration loaded.')

        print('Engineering features.')
        src.features.pipeline(**features_params)
        print('Feature engineering complete.')

    if 'train' in targets or run_all:
        # Train a model on feature-engineered data and report its ROC AUC.

        print('Train target recognized.')

        with open(os.path.join(config_dir, 'train-params.json'), 'r') as f:
            train_params = json.load(f)
        print('Train configuration loaded.')

        print('Training model.')
        src.models.train(**train_params)
        print('Model training complete.')
        
        
    return
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    targets = sys.argv[1:]
    main(targets)