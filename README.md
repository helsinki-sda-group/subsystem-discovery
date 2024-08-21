# Get the data (optional)
- Download preprocessed weather dataset and MTGNN trained weights from zenodo: https://zenodo.org/records/13357396

# Training
- Note that you probably want to run the model training with JAX on a GPU (or a TPU?)
- PyTorch is required only for the dataloaders, so its CPU version is enough
- `pip install -r requirements.txt`
- Either setup a script like `experiments/test.sh` or change the parameters in `run.py` and run `python run.py`

# Preprocessing
- Run `preprocessing/us_weather_process_data.py` to recreate the preprocessed weather dataset (or download from zenodo link above)

# Evaluation
- Run `evaluation/clustering_evaluate.py`

# Map visualization
- Run `evaluation/plot_weather_maps_avg.py` (you need to download 
![alt text](https://github.com/helsinki-sda-group/subsystem-discovery-high-dimensional-time-series-masked-autoencoders/blob/main/output/maps/Proposed_model_avg_map.png)
