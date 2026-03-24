import sys
import os
import yfinance as yf
import numpy as np
import pandas as pd
import networkx as nx

#EXTRACT PROPERTIES OF THE MSTS
def trees_to_properties (trees):

    tree_properties = {
        'dates': [],
        'medoids': [],
        'avg_path_lengths': [],
        'total_weights': [],
        'survival_ratio': [],
        'jaccard_similarity': [],
    }
    
    prev = None
    for date, tree in trees:
        medoid = nx.center(tree)[0]
        avg_path_length = nx.average_shortest_path_length(tree)
        total_weight = tree.size(weight="weight")

        #static properties
        tree_properties['dates'].append([date])
        tree_properties['medoids'].append([medoid])
        tree_properties['avg_path_lengths'].append([avg_path_length])
        tree_properties['total_weights'].append([total_weight])

        #time-dependent properties
        #survival ratio & jaccard
        if prev is not None:
            prev_edges = set(tuple(sorted(edge)) for edge in prev.edges())
            curr_edges = set(tuple(sorted(edge)) for edge in tree.edges())
            survival_ratio = len(prev_edges & curr_edges)/(len(curr_edges))
            jaccard_similarity = len(prev_edges & curr_edges)/(len(prev_edges | curr_edges))
            tree_properties['survival_ratio'].append([survival_ratio])
            tree_properties['jaccard_similarity'].append([jaccard_similarity])
        else:
            tree_properties['survival_ratio'].append([1.0])
            tree_properties['jaccard_similarity'].append([1.0])


        prev = tree
        
    return (pd.DataFrame(tree_properties).set_index('dates'))

def medoid_analysis(medoids):
    contiguous_duration = {}
    max_contiguous_duration = {}
    total_duration = {}
    prev = None
    for medoid in medoids:
        if medoid not in total_duration or contiguous_duration:
            contiguous_duration[medoid] = 0
            contiguous_duration[medoid] = 0
            total_duration[medoid] = 0
            
        else:
            total_duration[medoid] += 1
            if medoid == prev:
                contiguous_duration[medoid] += 1
                total_duration[medoid] += 1
            else:
                if max_contiguous_duration[prev] < contiguous_duration[prev]:
                    max_contiguous_duration[prev] = contiguous_duration[prev]
                contiguous_duration[prev] = 0
        prev = medoid
    sorted_medoids = dict(sorted(max_contiguous_duration.items(), key=lambda item: item[1], reverse=True))
    return sorted_medoids

def edge_analysis(df, window=5):
    #1.calc moving avg. (reduce noise)
    #assign back to the dataframe so .diff() can access them
    df['ma_avg_path_lengths'] = df['avg_path_lengths'].rolling(window).mean()
    df['ma_total_weights'] = df['total_weights'].rolling(window).mean()
    #2.calc velocity
    df['velocity_avg_path_l'] = df['ma_avg_path_lengths'].diff()
    df['velocity_total_weights'] = df['ma_total_weights'].diff()
    #3.calc acceleration
    df['acc_avg_path_l'] = df['velocity_avg_path_l'].diff()
    df['acc_total_weights'] = df['velocity_total_weights'].diff()

    return df
