# Michael Pien, 4/25/22

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


classes = ['ER', 'ERDD', 'GEO', 'GEOGD', 'HGG', 'SF', 'SFDD', 'Sticky','Original']
data_locations = [r"D:\file\Research\P-value\data\AT"+c for c in classes] # replace with paths to data on your computer

# Kenny's code (modified)
def combine_data(data_location,classes):
    df_comb = pd.DataFrame()
    i = 0
    for protein in data_location:
        df = pd.read_csv(protein, header = None, sep = ' ')
        df['class'] = classes[i]
        df_comb = pd.concat([df, df_comb])
        i += 1
    return df_comb

def data(dataframe, class_name):
    return dataframe[dataframe['class']==class_name].drop("class",axis=1).to_numpy()

def plot_empirical_CDF(data, Density,title, equal):
    '''
    return x,y data of CDF
    if density==True, y is percentage where y[-1]=1 
    '''
    f, ax = plt.subplots(1,1,figsize=(6,4))
    unequal_bins = set()
    greatest = max(data)
    while len(unequal_bins) < len(data):
        unequal_bins.add(random.uniform(0, max(data)))
    unequal_bins.add(max(data))
    unequal_bins = sorted(unequal_bins)
##    print(unequal_bins)
##    print(len(unequal_bins))
    if equal:
        h = ax.hist(data,bins=len(data),density=Density,cumulative=True,histtype='stepfilled')
    else:
        h = ax.hist(data,bins=unequal_bins,density=Density,cumulative=True,histtype='stepfilled')
    x = h[1][:-1]
    y = h[0]
    ax.plot(x,y,color='k', label='Empirical')
    ax.set_title(title)
    ax.legend()
    plt.show()
    return x,y

def data_distance(data):
    shortest_distance = [0]*len(data)
    for i in range(len(data)):
        x = np.delete(data,i,0)
        temp = (x-data[i])**2
        d = np.sqrt(np.sum(temp,axis=1))
        shortest_distance[i] = d.min()
    
    return np.array(shortest_distance)


df_comb = combine_data(data_locations,classes)

# CDFs with equal bin sizes
for c in classes[:-1]:
    dd = data_distance(data(df_comb, c))
    plot_empirical_CDF(dd, True, 'AT'+c, True)

# CDFs with unequal bin sizes
for c in classes[:-1]:
    dd = data_distance(data(df_comb, c))
    plot_empirical_CDF(dd, True, 'AT'+c, False)
    

