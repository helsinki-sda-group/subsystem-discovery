# Code for ECAI2024 paper: Subsystem Discovery in High-Dimensional Time-Series Using Masked Autoencoders

Link to full paper (open access, green button for PDF) https://ebooks.iospress.nl/volumearticle/69939

## Get the data (optional)
- 3x double pendulums code adapted from https://matplotlib.org/stable/gallery/animation/double_pendulum.html
  - Generate more via https://github.com/helsinki-sda-group/subsystem-discovery/blob/main/dataloaders/pendulum_dataloader.py#L240-L269
  - Pregenerated 40k steps of pendulum data already included in the repository, in the `data` directory

https://github.com/user-attachments/assets/2959e95f-4713-45a8-b202-e73229169f26


- Download preprocessed weather dataset and MTGNN trained weights from zenodo: https://zenodo.org/records/13357396

## Training
- Note that you probably want to run the model training with Jax on a GPU (or a TPU?)
- PyTorch is required only for the dataloaders, so its CPU version is enough
- `pip install -r requirements.txt` this will only install the CPU versions of Jax and PyTorch
- Either setup a script like `experiments/test.sh` or change the parameters in `run.py` and run it

## Preprocessing
- Run `preprocessing/us_weather_process_data.py` to recreate the preprocessed weather dataset (or download from zenodo link above)

## Evaluation
- Run `evaluation/clustering_evaluate.py`

## Map visualization
- Run `evaluation/plot_weather_maps_avg.py` (you need to download the full weather dataset for this, see notes in the file)
- Proposed model resulting map (check `output/maps` for others)
![alt text](https://github.com/helsinki-sda-group/subsystem-discovery-high-dimensional-time-series-masked-autoencoders/blob/main/output/maps/Proposed_model_avg_map.png)

## Citation
Coming soon!
