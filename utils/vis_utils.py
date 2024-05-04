import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def kdeplot_plot(data, data_type):
    
    if data_type == 'syn_route':
        plt.xlim(0, 100)
        plt.xticks(np.arange(0,100,1))
        sns.histplot(data,bins=100)
    elif data_type in 'qed':
        plt.xlim(0, 1)
        plt.xticks(np.arange(0, 1, 0.1))
        sns.histplot(data,bins=50)
    elif data_type in 'vina':
        plt.xlim(0, 1)
        plt.xticks(np.arange(-10, 0, 0.1))
        sns.histplot(data,bins=100)
    else:
        raise NotImplementedError
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('/home/haowei/Desktop/repos/DRR/benchmarks/crossdocked/metrics.csv')
    kdeplot_plot(df['vina'], 'vina')