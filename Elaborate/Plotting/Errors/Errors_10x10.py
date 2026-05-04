import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import re
import os

def plot_energy_vs_params_manual(manual_data, save_path=None):
    """
    Plots Energy vs Number of Parameters from manually inserted values.
    """
    plt.figure(figsize=(9, 6))
    
    for item in manual_data:
        label = item.get('label', 'Unknown')
        params = item.get('params')
        energy = item.get('energy')
        color = item.get('color', 'black')
        marker = item.get('marker', 'o')
        
        if isinstance(params, (int, float)) and energy is not None:
            plt.scatter(params, energy, label=label, color=color, marker=marker, 
                        s=item.get('size', 120), 
                        alpha=0.95 if item.get('zorder', 3) > 3 else 0.8, 
                        edgecolors=item.get('edgecolors', 'face'),
                        linewidths=item.get('linewidths', 1.0),
                        zorder=item.get('zorder', 3))
        elif energy is not None:
            # Plot as a horizontal line if parameters are missing or non-numeric (e.g., DMRG)
            plt.axhline(y=energy, color=color, linestyle='--', label=f"{label} ({params if params else 'N/A'})", alpha=0.8, zorder=1)

    plt.xlabel("Number of Parameters", fontsize=14)
    plt.ylabel("Energy", fontsize=14)
    plt.title("Energy vs Number of Parameters (10x10)", fontsize=15)
    
    # Setting x-axis to log scale is generally good for parameter counts
    plt.xscale("log")  
    plt.grid(True, linestyle="--", alpha=0.6, zorder=0)
    
    # Avoid duplicate labels in legend if you use the same label multiple times
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Move the legend outside the plot so it doesn't overlap the data points
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()

if __name__ == "__main__":
    # =====================================================================
    # MANUALLY INSERT YOUR VALUES HERE
    # =====================================================================
    table_data = [
        
        {
            "Energy per site": -0.494757,
            "Uncertainty": 12,
            "Wave function": "CNN",
            "Parameters": None,
            "Marshall prior": "No",
            "Reference": "[23]",
            "Year": 2020
        },
        {
            "Energy per site": -0.4947359,
            "Uncertainty": 1,
            "Wave function": "Shallow CNN",
            "Parameters": 11009,
            "Marshall prior": "Not available",
            "Reference": "[22]",
            "Year": 2018
        },
        {
            "Energy per site": -0.49516,
            "Uncertainty": 1,
            "Wave function": "Deep CNN",
            "Parameters": 7676,
            "Marshall prior": "Yes",
            "Reference": "[21]",
            "Year": 2019
        },
        {
            "Energy per site": -0.495502,
            "Uncertainty": 1,
            "Wave function": "PEPS + Deep CNN",
            "Parameters": 3531,
            "Marshall prior": "No",
            "Reference": "[34]",
            "Year": 2021
        },
        {
            "Energy per site": -0.495530,
            "Uncertainty": None,
            "Wave function": "DMRG",
            "Parameters": "8192 SU(2) states",
            "Marshall prior": "No",
            "Reference": "[32]",
            "Year": 2014
        },
        {
            "Energy per site": -0.495627,
            "Uncertainty": 6,
            "Wave function": "aCNN",
            "Parameters": 6538,
            "Marshall prior": "Yes",
            "Reference": "[35]",
            "Year": 2023
        },
        {
            "Energy per site": -0.49575,
            "Uncertainty": 3,
            "Wave function": "RBM-fermionic",
            "Parameters": 2000,
            "Marshall prior": "Yes",
            "Reference": "[16]",
            "Year": 2019
        },
        {
            "Energy per site": -0.49586,
            "Uncertainty": 4,
            "Wave function": "CNN",
            "Parameters": 10952,
            "Marshall prior": "Yes",
            "Reference": "[36]",
            "Year": 2023
        },
        {
            "Energy per site": -0.4968,
            "Uncertainty": 4,
            "Wave function": "RBM (p = 1)",
            "Parameters": None,
            "Marshall prior": "Yes",
            "Reference": "[37]",
            "Year": 2022
        },
        {
            "Energy per site": -0.49717,
            "Uncertainty": 1,
            "Wave function": "Deep CNN",
            "Parameters": 106529,
            "Marshall prior": "Yes",
            "Reference": "[29]",
            "Year": 2022
        },
        {
            "Energy per site": -0.497437,
            "Uncertainty": 7,
            "Wave function": "GCNN",
            "Parameters": 67548,
            "Marshall prior": "No",
            "Reference": "[28]",
            "Year": 2023
        },
        {
            "Energy per site": -0.497468,
            "Uncertainty": 1,
            "Wave function": "Deep CNN",
            "Parameters": 421953,
            "Marshall prior": "Yes",
            "Reference": "[31]",
            "Year": 2022
        },
        {
            "Energy per site": -0.497627,
            "Uncertainty": 1,
            "Wave function": "Deep CNN",
            "Parameters": 146320,
            "Marshall prior": "Yes",
            "Reference": "[30]",
            "Year": 2023
        },
        {
            "Energy per site": -0.497629,
            "Uncertainty": 1,
            "Wave function": "RBM+PP",
            "Parameters": 13132,
            "Marshall prior": "Yes",
            "Reference": "[38]",
            "Year": 2021
        },
        {
            "Energy per site": -0.497634,
            "Uncertainty": 1,
            "Wave function": "Deep ViT",
            "Parameters": 267720,
            "Marshall prior": "No",
            "Reference": "Present work",
            "Year": 2023
        },
        {
            "Energy per site": -0.4962,
            "Uncertainty": 1,
            "Wave function": "HFDS",
            "Parameters": 37728,
            "Marshall prior": "No",
            "Reference": "Present work",
            "Year": 2026
        },
        {
            "Energy per site": -0.4966751884584341,
            "Uncertainty": 1,
            "Wave function": "ViT",
            "Parameters": 13812,
            "Marshall prior": "No",
            "Reference": "Present work",
            "Year": 2026
        }
    ]
    
    # Dynamically generate markers and colors
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'h', 'H', '<', '>', '8', 'd']
    cmap = plt.get_cmap('tab20')

    manual_values = []
    for i, item in enumerate(table_data):
        wf_name = item.get('Wave function', 'Unknown')
        
        # Default styling
        color = cmap(i % 20)
        size = 120
        zorder = 3
        edgecolors = 'face'
        linewidths = 1.0
        
        # Highlight specific models
        if wf_name == 'HFDS':
            color = 'tab:blue'
            size = 100
            zorder = 5
            edgecolors = 'black'
            linewidths = 1.5
        elif wf_name == 'ViT':
            color = 'tab:orange'
            size = 100
            zorder = 5
            edgecolors = 'black'
            linewidths = 1.5
            
        manual_values.append({
            'label': wf_name,
            'params': item.get('Parameters'),
            'energy': item.get('Energy per site'),
            'color': color,
            'marker': markers[i % len(markers)],
            'size': size,
            'zorder': zorder,
            'edgecolors': edgecolors,
            'linewidths': linewidths
        })

    save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/Energy_vs_Params_10x10.png"
    
    plot_energy_vs_params_manual(manual_values, save_path=save_dir)