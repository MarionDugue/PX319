from biopandas.mol2 import PandasMol2
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')
from concurrent import futures
from gtda.homology import VietorisRipsPersistence

aa3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()

df = pd.DataFrame()


def func(protname):
    
    try:

        protdf = pd.DataFrame()
        prot = PandasMol2().read_mol2(os.path.join('scPDB',protname,'protein.mol2'))
        binding_site = PandasMol2().read_mol2(os.path.join('scPDB',protname,'site.mol2'))

        protdf = prot.df[prot.df['subst_name'].apply(lambda x: x[:3] in aa3)]

        protdf['prot'] = protname
        protdf['amino_acid'] = protdf['subst_name'].apply(lambda x: x[:3]).astype('category')
        protdf['res_num'] = protdf['subst_name'].apply(lambda x: x[3:]).astype('int')
        if (protdf[protdf['atom_name'] == 'CA'].groupby('res_num').size() > 1).any():
            raise Exception 


        #protdf['element'] = protdf['atom_type'].apply(lambda x : x.split('.')[0]).astype('category')
        sitedf = binding_site.df[binding_site.df['subst_name'].apply(lambda x: x[:3] in aa3)]

        protdf['site'] = 0
        for index, row in sitedf.iterrows():
            protdf.loc[(protdf['x'] == row.x) &(protdf['y'] == row.y) &(protdf['z'] == row.z), 'site'] = 1

        protdf['site'] = protdf['site'].astype('category')
        protdf.set_index('res_num', inplace=True,drop=False)

        protdf = protdf[protdf['atom_name'] == 'CA']

        #protdf['res_x'] = protdf.loc[protdf['atom_name'] == 'CA', 'x']
        #protdf['res_y'] = protdf.loc[protdf['atom_name'] == 'CA', 'y']
        #protdf['res_z'] = protdf.loc[protdf['atom_name'] == 'CA', 'z']

        protdf.drop(columns = ['atom_id', 'atom_name', 'atom_type', 'subst_id', 'subst_name', 'charge'], inplace=True)
        protdf.dropna(inplace=True)
        return protdf
    except:
        return pd.DataFrame()
        




if __name__ == '__main__':
    


    with futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(func, os.listdir('scPDB')))
    
    df = pd.concat(results)
    df.set_index(['prot','res_num'], inplace=True, drop=False)
    
    df.to_pickle('protdb.pkl')
    
    


    

    
