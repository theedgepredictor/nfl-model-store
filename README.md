# nfl-model-store

****

## ETL Process for training ML models and generating downstream predictions

****

### [MLOps](https://ml-ops.org/)

- Dataset registry
- Model registry
- Model versioning
- Data drift tracking
- Offline prediction use case

### Local MLFLow server

```commandline
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlflow_data

```

The whole idea with this is to run experiments, track runs, log metrics and determine the best model to register to the model-store. This is an additional step / piece that we add to the end of the MLFlow cycle by storing a metadata file of the experiment_id, run_id of the best model. 