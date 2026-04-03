import numpy as np
import pandas as pd
import networkx as nx
import yfinance as yf


def clipping(returns, window):
    corr_matrix = returns.corr()

    # CLIPPING VALUES TO AVOID NUMERICAL INSTABILITIES (MARKET )
    lambda_max = (1+np.sqrt(10/window))**2
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    is_noise = eigenvalues < lambda_max
    noise_mean = np.mean(eigenvalues[is_noise])
    eigenvalues_denoised = eigenvalues.copy()
    eigenvalues_denoised[is_noise] = noise_mean

    C_clean = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T

    d = np.diag(C_clean)
    inv_std = np.diag(1 / np.sqrt(d))
    C_final = inv_std @ C_clean @ inv_std
    # Convert back to DataFrame with proper index and columns
    C_final_df = pd.DataFrame(C_final, index=corr_matrix.index, columns=corr_matrix.columns)
    return C_final_df

def mst (clipped_corr_matrix):
   
    clipped_corr_matrix = clipped_corr_matrix.fillna(0)
    distance_matrix = np.sqrt(2 * (1 - clipped_corr_matrix))
    distance_matrix = distance_matrix.fillna(0) #ultrametric distance
    #BUILDING MST
    G = nx.from_pandas_adjacency(distance_matrix) 
    mst = nx.minimum_spanning_tree(G)

    return mst
