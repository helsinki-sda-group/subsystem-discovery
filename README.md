# Code for ECAI2024 paper: Subsystem Discovery in High-Dimensional Time-Series Using Masked Autoencoders

Link to full paper (open access, green button for PDF) https://ebooks.iospress.nl/volumearticle/69939

### TL;DR
Structure learning directly from time-series observations of up to ~2000 variables, millions of timesteps

### Abstract
Deep neural networks are increasingly used for time series tasks, yet they often struggle to interpretably model high-dimensional data. In this context, we consider the task of learning easy to understand connections between time-series variables, and organizing them into subsystems, directly from observed data. Our approach reconstructs multivariate time-series with a masked autoencoder, where all information between individual variables is mediated by a learned adjacency matrix. This intuitive pairwise relationship enables grouping of variables without prior knowledge of cluster quantity or size, and is particularly useful for analyzing complex sensor systems with unknown structural interdependencies. Our method simultaneously learns a useful signal representation and aids in understanding the underlying processes. We show that we can learn the correct subsystems from simulated data, and demonstrate identification of plausible subsystem structure from high-dimensional real-world data. In addition, we show that the model retains high predictive performance.

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
```
@article{subsystem2024sarapisto,
  title = "Subsystem Discovery in High-Dimensional Time-Series Using Masked Autoencoders",
  keywords = "113 Computer and information sciences",
  author = "Teemu Sarapisto and Haoyu Wei and Keijo Heljanko and Arto Klami and Laura Ruotsalainen",
  year = "2024",
  doi = "10.3233/FAIA240844",
  language = "English",
  volume = "392",
  pages = "3031 -- 3038",
  journal = "Frontiers in artificial intelligence and applications",
  issn = "0922-6389",
  publisher = "IOS PRESS",
  note = "European Conference on Artificial Intelligence, ECAI 2024 ; Conference date: 19-10-2024 Through 24-10-2024",
  url = "https://www.ecai2024.eu/",
}
```
