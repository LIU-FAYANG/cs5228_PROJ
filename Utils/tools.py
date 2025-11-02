import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

def haversine_matrix(hdb_lat:np.ndarray,hdb_lon:np.ndarray,mall_lat:np.ndarray,mall_lon:np.ndarray):
    R=6371
    
    dlat=hdb_lat[:,None]-mall_lat[None,:] #为了便于计算，将hdb_lat和hdb_lon扩展为矩阵
    dlon=hdb_lon[:,None]-mall_lon[None,:]

    a=np.sin(dlat/2)**2+np.cos(hdb_lat)[:, None]*np.cos(mall_lat)[None, :]*np.sin(dlon/2)**2
    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c

def create_auxiliary_location_features(hdb_coords:pd.DataFrame,
                        auxilliary_df:pd.DataFrame,
                        radii=(1.0,2.0,5.0),
                        batch_size:int |None=None,
                        feature_prefix:str='MALL'
                        )->pd.DataFrame: #hdb_df is the dataframe got from get_hdb_coordinates_df
    mlat=np.radians(auxilliary_df['LATITUDE'].to_numpy())
    mlon=np.radians(auxilliary_df['LONGITUDE'].to_numpy())
    hlat=np.radians(hdb_coords['LATITUDE'].to_numpy())
    hlon=np.radians(hdb_coords['LONGITUDE'].to_numpy())

    #create a empty container to store the mall features
    n=len(hlat)
    nearest=np.full((n,),np.inf,dtype=np.float32)
    counts=[np.zeros(n,dtype=np.int32) for _ in radii]

    if batch_size is None:
        dist=haversine_matrix(hlat,hlon,mlat,mlon)
        nearest=dist.min(axis=1).astype(np.float32)
        for i,r in enumerate(radii):
            counts[i]=np.array((dist<r).sum(axis=1),dtype=np.int32)
    else:
       for i in tqdm(range(0, n, batch_size),total=(n+batch_size-1)//batch_size, desc="Creating mall features"):
            j = min(i+batch_size, n)
            dist = haversine_matrix(hlat[i:j], hlon[i:j], mlat, mlon)
            nearest[i:j] = dist.min(axis=1).astype(np.float32)
            for k, r in enumerate(radii):
                counts[k][i:j] = (dist <= r).sum(axis=1).astype(np.int32)
    output=pd.DataFrame({
        f'NEAREST_{feature_prefix}_KM': nearest,
        **{f'{feature_prefix}_COUNT_{r}KM': c for r, c in zip(radii, counts)}
    }, index=hdb_coords.index)
    
    return output
            