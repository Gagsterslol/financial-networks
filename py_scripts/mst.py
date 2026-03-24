import numpy as np
import pandas as pd
import networkx as nx
import yfinance as yf



def mst (returns):

    #OBTAINING DISTANCE MATRIX
    #correlation matrix (p_ij) (.corr() -> Pearson correlation coefficient)
    #Pearson used in the Mantegna paper -> Pearson deals with continuous data (while Spearman and Kendall is for ordinal)
    corr_matrix = returns.corr()
    
    #corr_matrix -> dist_matrix ( d(i, j ) = 1 − ρ_ij^2 )
    #not sure which one to use, I've read online that sqrt(2(1-p)) is favorable, one because it preserves direction, 2. because its euclidian as well
    dist_matrix = (1-(corr_matrix ** 2))
    dist_matrix_euclidian = np.sqrt(2 * (1 - corr_matrix))

    #BUILDING MST
    #graph G, nodes := stocks, edges := dist_matrix entry
    G = nx.from_pandas_adjacency(dist_matrix_euclidian) 
    mst = nx.minimum_spanning_tree(G)

    return mst

def rolling_window_mst ():
    #OBTAINING DATA
    trees = []
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'PEP']
    data = yf.download(tickers, start="2023-01-01", progress=False)['Close']
    #OBTAINING RETURNS
    #Yi = lnPi(t) − lnPi(t − 1)
    returns = np.log(data /data.shift(1)).dropna()
    window = 60
   

    for i in range(window, len(returns)):
        window_slice = returns.iloc[i-window : i]
        current_date = returns.index[i]
        curr_tree = mst.mst(window_slice)
        trees.append((current_date, curr_tree))

    df_trees = pd.DataFrame(trees, columns=['Date', 'MST'])
    return df_trees
