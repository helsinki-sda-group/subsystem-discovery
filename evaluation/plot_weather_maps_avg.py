#%%
from scipy.spatial import distance_matrix
import numpy as np
import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from glob import glob
import os
import csv
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

from evaluation.evaluation_adjacencies import get_ours, get_mtgnns, get_corrs, get_mutuals
from config import US_WEATHER_DATA_PATH
#%%


def sort_columns_by_fiedler(column_names, fiedler_vector, station_names):
    # Map station names to Fiedler values
    fiedler_dict = {name: value for name, value in zip(station_names, fiedler_vector)}
    
    # Create a list to store tuples of column names and their corresponding Fiedler values
    columns_fiedler = []
    
    # Assign Fiedler values to columns
    for col in column_names:
        for station in station_names:
            if station in col:
                columns_fiedler.append((col, fiedler_dict[station]))
                break
    
    # Sort columns based on the Fiedler values
    sorted_columns = sorted(columns_fiedler, key=lambda x: x[1])
    
    # Extract the sorted column names
    sorted_column_names = [col[0] for col in sorted_columns]
    return sorted_columns

def get_indices_of(array1, array2, allow_missing=False):
    matches = (array1[:, None] == array2).argmax(axis=1)
    matches[~(array1[:, None] == array2).any(axis=1)] = -1
    if not allow_missing:
        assert -1 not in matches, 'Some columns not found'
    return matches

def dist_to_similarity(dist_matrix):
    similarity_matrix = np.exp(-dist_matrix ** 2 / (2. * np.std(dist_matrix)**2))
    # Ensure that the diagonal is zero
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix

def get_fiedler_indices(stations_data, df):
    station_set = get_unique_stations(df)
    station_coords = get_station_coords(station_set)
    dist_matrix = distance_matrix(list(station_coords.values()), list(station_coords.values()))
    fiedler = get_fiedler_vector(dist_to_similarity(dist_matrix))
    column_names = df.columns

    station_names = list(station_coords.keys())
    cols_by_fiedler_values = sort_columns_by_fiedler(column_names, fiedler, station_names)

    cols_by_fiedler = [col[0] for col in cols_by_fiedler_values]
    indices = get_indices_of(np.array(cols_by_fiedler), np.array(df.columns))
    assert np.all(np.array(df.columns)[indices] == np.array(cols_by_fiedler))
    return indices, station_coords, fiedler, station_names

def get_fiedler_vector(similarity_matrix):
    graph_lap_new = laplacian(similarity_matrix, normed=True)
    _, eigenvectors_new = eigh(graph_lap_new)
    fiedler_vector_new = eigenvectors_new[:, 1]
    fiedler_normalized_new = (fiedler_vector_new - min(fiedler_vector_new)) / (max(fiedler_vector_new) - min(fiedler_vector_new))
    return fiedler_normalized_new


stations_file = os.path.join('data', 'stations.tsv')

# Location of the weather dataset, download from here https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/
# e.g https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/snapshots/CRNS0101-05-202408190550.zip
# Due to a naming mismatch with the column names and stations.tsv, need to map from the filenames of the folders...
base_path = '/Users/tesatesa/devaus/datasets/CRNS0101-05-202403110550/CRNS0101-05-202403110550'
def fetch_lat_lon(station_name):
    file_name = f"CRNS0101-05-2024-{station_name}.txt"
    file_path = os.path.join(base_path, "2024", file_name)
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                latitude = float(parts[7])  # Assuming latitude is at index 7
                longitude = float(parts[6])  # Assuming longitude is at index 6
                return (latitude, longitude)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return None

def get_station_coords(station_set):
    station_coords = {}
    for station in station_set:
        coordinates = fetch_lat_lon(station)
        if coordinates:
            station_coords[station] = coordinates
        else:
            raise ValueError(f"Coordinates not found for station: {station}")
    
    return station_coords


def get_unique_stations(df):
    folders = glob(f'{base_path}/2024/**')
    station_files = [f.split('/')[-1] for f in folders]
    # names have format CRNS0101-MM-YYYY-StationName.txt
    station_names = [f.split('-')[-1].split('.')[0] for f in station_files]

    station_set = set()
    for col in df.columns:
        for station in station_names:
            if station in col:
                station_set.add(station)

    len(station_set)
    return station_set

def get_cols_to_stations(df, station_names, indices):
    cols_to_stations = {}
    for col in df.columns[indices]:
        for station in station_names:
            if station in col:
                cols_to_stations[col] = station
                break

    return cols_to_stations



def get_variable_types(df, station_names):
    variable_types = []
    for col in df.columns:
        for station in station_names:
            if station in col:
                variable = col.split(station)[0][:-1]
                if variable not in variable_types:
                    variable_types.append(variable)
                break
    return variable_types

def get_col_to_var(df, variable_types):
    col_to_variable_type = {}
    for col in df.columns:
        for variable in variable_types:
            if variable in col:
                col_to_variable_type[col] = variable
                break
    return col_to_variable_type

def plot_separate_maps_for_variable_types(station_coords, fiedler, column_names, weather_adjacency, cols_to_stations, variable_types, line_strength):
    maps = {}
    
    for variable in variable_types:
        coordinates = np.array(list(station_coords.values()))
        map_center = [coordinates[:, 0].mean(), coordinates[:, 1].mean()]
        station_map = folium.Map(location=map_center, zoom_start=3, tiles='CartoDB positron')
        
        polyline_ids = {}
        
        for (station, (lat, lon)), color_value in zip(station_coords.items(), fiedler):
            color = plt.cm.viridis(color_value)
            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color=mcolors.rgb2hex(color),
                fill=True,
                fill_color=mcolors.rgb2hex(color),
                fill_opacity=0.7,
                popup=f"{station}, Fiedler: {color_value:.2f}",
                id=station
            ).add_to(station_map)
        
        # Direct connections for the specific variable type
        connection_id = 0
        for i, var_i in enumerate(column_names):
            if var_i.startswith(variable):
                for j, var_j in enumerate(column_names):
                    if i != j and var_j.startswith(variable):
                        stat1 = cols_to_stations[var_i]
                        stat2 = cols_to_stations[var_j]
                        if stat1 == stat2:
                            continue
                        connection_strength = weather_adjacency[i, j]
                        line = folium.PolyLine(
                            [station_coords[stat1], station_coords[stat2]],
                            color='blue',  # General color for simplicity, modify as needed
                            weight=connection_strength.tolist() * line_strength,
                            opacity=0.5,
                            id=f'line_{connection_id}'
                        ).add_to(station_map)
                        polyline_ids[f'line_{connection_id}'] = (stat1, stat2)
                        connection_id += 1
        
        maps[variable] = station_map

    return maps

def plot_aggregated_station_connections_basemap_normalized(station_coords, fiedler, column_names, weather_adjacency, cols_to_stations, col_to_var, line_strength, plot_name, top_n=5, normalization='mean'):
    fig, ax = plt.subplots(figsize=(20, 10))

    # Determine map boundaries
    lats, lons = zip(*station_coords.values())
    margin = 3  # Degree margin for map boundaries
    lat_min, lat_max = min(lats) - margin, max(lats) + margin
    lon_min, lon_max = min(lons) - margin, max(lons) + margin

    # Set up Basemap
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='i', ax=ax)
    m.etopo()
    m.drawcoastlines(False)
    m.drawcountries(False)

    # Plot stations
    for (station, (lat, lon)), color_value in zip(station_coords.items(), fiedler):
        x, y = m(lon, lat)
        m.scatter(x, y, color='teal', s=50, edgecolor='k', label=f"{station}, Fiedler: {color_value:.2f}")

    # Aggregated lines with variable normalization
    aggregated_strength = {}
    variable_values = []

    # Collect all strengths and compute normalization factor
    for i, var_i in enumerate(column_names):
        for j, var_j in enumerate(column_names):
            if j > i:
                stat1, stat2 = cols_to_stations[var_i], cols_to_stations[var_j]
                if stat1 == stat2:
                    continue
                strength = weather_adjacency[i, j]
                variable_values.append(strength)
                pair_key = tuple(sorted([stat1, stat2]))
                aggregated_strength.setdefault(pair_key, []).append(strength)

    # Calculate normalization factors based on the selected method
    if normalization == 'max':
        norm_factor = max(abs(val) for val in variable_values)
    elif normalization == 'mean':
        norm_factor = sum(variable_values) / len(variable_values)
    else:
        raise ValueError("Normalization method not supported. Use 'mean' or 'max'.")

    # Normalize strengths and calculate averages
    avg_strengths = {}
    for pair, strengths in aggregated_strength.items():
        avg_strengths[pair] = sum(strengths) / len(strengths) / norm_factor

    # Determine the top N connections for each station and normalize
    station_connections = {station: [] for station in station_coords.keys()}
    for (stat1, stat2), strength in avg_strengths.items():
        station_connections[stat1].append((stat2, strength))
        station_connections[stat2].append((stat1, strength))

    normalized_strengths = {}
    for station, connections in station_connections.items():
        connections.sort(key=lambda x: x[1], reverse=True)
        top_connections = connections[:top_n]
        max_strength = top_connections[0][1] if top_connections else 1.0
        for neighbor, strength in top_connections:
            normalized_strengths[(station, neighbor)] = strength / max_strength

    # Plot the top N strongest connections for each station
    for (stat1, stat2), norm_strength in normalized_strengths.items():
        lat1, lon1 = station_coords[stat1]
        lat2, lon2 = station_coords[stat2]
        x1, y1 = m(lon1, lat1)
        x2, y2 = m(lon2, lat2)
        m.plot([x1, x2], [y1, y2], color='red', linewidth=norm_strength * line_strength, alpha=0.5)

    plt.savefig(f"output/maps/{plot_name}_normalized_map.png", dpi=300, bbox_inches='tight')
    plt.show()
    return None, normalized_strengths, station_connections


def plot_aggregated_station_connections_basemap(station_coords, fiedler, column_names, weather_adjacency, cols_to_stations, col_to_var, line_strength, plot_name, top_n=5):
    fig, ax = plt.subplots(figsize=(20, 10))

    # Determine map boundaries
    lats, lons = zip(*station_coords.values())
    margin = 3  # Degree margin for map boundaries
    lat_min, lat_max = min(lats) - margin, max(lats) + margin
    lon_min, lon_max = min(lons) - margin, max(lons) + margin

    # Set up Basemap
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                 llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='i', ax=ax)

    m.etopo()
    m.drawcoastlines(False)
    m.drawcountries(False)

    # Plot stations
    for (station, (lat, lon)), color_value in zip(station_coords.items(), fiedler):
        x, y = m(lon, lat)
        #color = plt.cm.viridis(color_value)
        m.scatter(x, y, color='teal', s=50, edgecolor='k', label=f"{station}, Fiedler: {color_value:.2f}")

    # Aggregated lines with averaged strength
    aggregated_strength = {}
    for i, var_i in enumerate(column_names):
        for j, var_j in enumerate(column_names):
            if j > i:
                stat1 = cols_to_stations[var_i]
                stat2 = cols_to_stations[var_j]
                if stat1 == stat2:
                    continue
                pair_key = tuple(sorted([stat1, stat2]))
                aggregated_strength.setdefault(pair_key, []).append(weather_adjacency[i, j])

    # Calculate average of connection strengths over all weather variable types between each pair of stations
    avg_strengths = {pair: sum(strengths) / len(strengths) for pair, strengths in aggregated_strength.items()}

    # Determine the top N connections for each station and normalize
    station_connections = {station: [] for station in station_coords.keys()}
    for (stat1, stat2), strength in avg_strengths.items():
        station_connections[stat1].append((stat2, strength))
        station_connections[stat2].append((stat1, strength))

    # Normalize strengths for each station's top N connections
    normalized_strengths = {}
    for station, connections in station_connections.items():
        connections.sort(key=lambda x: x[1], reverse=True)
        top_connections = connections[:top_n]
        max_strength = top_connections[0][1] if top_connections else 1.0
        for neighbor, strength in top_connections:
            normalized_strengths[(station, neighbor)] = strength
            # normalized_strengths[(station, neighbor)] = strength / max_strength
            # normalized_strengths[(neighbor, station)] = strength / max_strength  # Ensure symmetry

    # Normalize by divididing all by the mean strength
    mean_strength = sum(normalized_strengths.values()) / len(normalized_strengths)
    normalized_strengths = {k: v / mean_strength for k, v in normalized_strengths.items()}

    # Plot the top N strongest connections for each station
    for (stat1, stat2), norm_strength in normalized_strengths.items():
        lat1, lon1 = station_coords[stat1]
        lat2, lon2 = station_coords[stat2]
        x1, y1 = m(lon1, lat1)
        x2, y2 = m(lon2, lat2)
        m.plot([x1, x2], [y1, y2], color='red', linewidth=norm_strength * line_strength, alpha=0.5)

    #plt.title("Top N Strongest Connections Between Stations")
    #plt.legend(loc='upper right')
    plt.savefig(f"output/maps/{plot_name}_avg_map.png", dpi=300, bbox_inches='tight')
    plt.show()
    return None, normalized_strengths, station_connections

def plot_aggregated_station_connections(station_coords, fiedler, column_names, weather_adjacency, cols_to_stations, line_strength, top_n=5):
    coordinates = np.array(list(station_coords.values()))
    map_center = [coordinates[:, 0].mean(), coordinates[:, 1].mean()]
    station_map = folium.Map(location=map_center, zoom_start=3, tiles='CartoDB positron')

    for (station, (lat, lon)), color_value in zip(station_coords.items(), fiedler):
        color = plt.cm.viridis(color_value)
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color=mcolors.rgb2hex(color),
            fill=True,
            fill_color=mcolors.rgb2hex(color),
            fill_opacity=0.7,
            popup=f"{station}, Fiedler: {color_value:.2f}"
        ).add_to(station_map)

    # Aggregated lines with averaged strength
    aggregated_strength = {}
    for i, var_i in enumerate(column_names):
        for j, var_j in enumerate(column_names):
            if j > i:
                stat1 = cols_to_stations[var_i]
                stat2 = cols_to_stations[var_j]
                if stat1 == stat2:
                    continue
                pair_key = tuple(sorted([stat1, stat2]))
                aggregated_strength.setdefault(pair_key, []).append(weather_adjacency[i, j])

    # Calculate average of connection strengths over all weather variable types between each pair of stations
    avg_strengths = {pair: sum(strengths) / len(strengths) for pair, strengths in aggregated_strength.items()}

    # Determine the top N connections for each station and normalize
    station_connections = {station: [] for station in station_coords.keys()}
    for (stat1, stat2), strength in avg_strengths.items():
        station_connections[stat1].append((stat2, strength))
        station_connections[stat2].append((stat1, strength))

    # Normalize strengths for each station's top N connections
    normalized_strengths = {}
    for station, connections in station_connections.items():
        connections.sort(key=lambda x: x[1], reverse=True)
        top_connections = connections[:top_n]
        max_strength = top_connections[0][1] if top_connections else 1.0
        for neighbor, strength in top_connections:
            normalized_strengths[(station, neighbor)] = strength
            # normalized_strengths[(station, neighbor)] = strength / max_strength
            # normalized_strengths[(neighbor, station)] = strength / max_strength  # Ensure symmetry

    # Plot the top N strongest connections for each station
    for (stat1, stat2), norm_strength in normalized_strengths.items():
        folium.PolyLine(
            [station_coords[stat1], station_coords[stat2]],
            color='blue',  # General color for simplicity, modify as needed
            weight=norm_strength.tolist() * line_strength,  # Adjust weight scaling as needed
            opacity=0.5
        ).add_to(station_map)

    return station_map, normalized_strengths


def plot_aggregated_adjacency():
    aggregated_strength = {}

    target_adjacency = weather_adjacency[indices][:, indices]
    for i, var_i in enumerate(column_names[indices]):
        for j, var_j in enumerate(column_names[indices]):
            if i != j:
                stat1 = cols_to_stations[var_i]
                stat2 = cols_to_stations[var_j]
                if stat1 == stat2:
                    continue
                pair_key = tuple(sorted([stat1, stat2]))
                aggregated_strength.setdefault(pair_key, []).append(target_adjacency[i, j])

    aggregated_mean_strength = {k: np.mean(v) for k, v in aggregated_strength.items()}
    # Extract unique station names
    stations = sorted(set(cols_to_stations.values()))

    # Create an empty adjacency matrix
    adjacency_matrix = pd.DataFrame(0, index=stations, columns=stations, dtype=float)

    # Fill the adjacency matrix with the mean values
    for (stat1, stat2), mean_val in aggregated_mean_strength.items():
        adjacency_matrix.at[stat1, stat2] = mean_val
        adjacency_matrix.at[stat2, stat1] = mean_val  # Symmetric matrix

    # Plot the adjacency matrix using seaborn
    plt.figure(figsize=(20, 20))
    #sns.heatmap(adjacency_matrix, annot=False, cmap='coolwarm', linewidths=.5)
    plt.imshow(adjacency_matrix)
    plt.title('Aggregated Strength Adjacency Matrix')
    plt.show()

#%%
if __name__ == '__main__':
    #%%
    df = pd.read_parquet(US_WEATHER_DATA_PATH)
    #%%
    column_names = df.columns
    #%%
    stations_path = './data/stations.tsv'
    stations_data = pd.read_csv(stations_path, delimiter='\t')
    indices, station_coords, fiedler, station_names = get_fiedler_indices(stations_data, df)
    cols_to_stations = get_cols_to_stations(df, station_names, indices)
    #%%
    variable_types = get_variable_types(df, station_names)
    cols_to_vars = get_col_to_var(df, variable_types)
    #%%
    weather_idx = 1

    corr_adj = get_corrs()[weather_idx]
    our_adj = get_ours()[weather_idx]
    mtgnn_adj = get_mtgnns()[weather_idx]
    mutual_adj = get_mutuals()[weather_idx]
    #%%
    names = [
        # ['Correlation', 10],
        # ['Proposed_model', 15],
        # ['MTGNN', 1],
        # ['Mutual_info', 3]
        ['Correlation', 1],
        ['Proposed_model', 1],
        ['MTGNN', 1],
        ['Mutual_info', 1]
    ]
    adjacencies = [corr_adj, our_adj, mtgnn_adj, mutual_adj]
    for [name, line_strength], adjacency in zip(names, adjacencies):
        aggregated_map, norm_strengths, station_connections = plot_aggregated_station_connections_basemap(station_coords, fiedler, df.columns, adjacency, cols_to_stations, cols_to_vars, plot_name=name, line_strength=line_strength, top_n=10)

        # maps = plot_separate_maps_for_variable_types(station_coords, fiedler, column_names, adjacency, cols_to_stations, variable_types, line_strength)

        # for variable_type in variable_types:
        #     maps[variable_type].save(f"output/maps/{name}_{variable_type}_map.html")
# %%
