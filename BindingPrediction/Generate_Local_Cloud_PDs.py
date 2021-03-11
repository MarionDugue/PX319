from biopandas.mol2 import PandasMol2
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')
from concurrent import futures
from gtda.homology import VietorisRipsPersistence


npoints = 15

persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=[0,1,2],
            collapse_edges=True,
            n_jobs = None
    )


def get_local_cloud(prot_res):
        prot, res = prot_res
        tempdf = df.loc[prot]
        center = tempdf.loc[res, ['x', 'y', 'z']].to_numpy()

        tempdf['dist'] = np.sqrt((tempdf['x'] - center[0])**2 + (tempdf['y'] - center[1])**2 + (tempdf['z'] - center[2])**2)

        localcloud = tempdf.nsmallest(npoints, 'dist')[['x','y','z']].to_numpy()
        return localcloud

get_local_cloud = np.vectorize(get_local_cloud, otypes=[np.ndarray],)

    
def get_diagrams(localclouds):
        x = persistence.fit_transform(localclouds)
        return x    
    

if __name__ == '__main__':

    
    df = pd.read_pickle('protdf.pkl')
    


    localclouds = np.stack(get_local_cloud(prot_res))
    persistence = get_diagrams(localclouds)
    
    df['persistence'] = None
    df['persistence'] = df.persistence.astype(object)
    df['persistence'] = list(persistence)
    
    
    df.to_pickle('protdiag_'+ str(npoints)+'.pkl')
