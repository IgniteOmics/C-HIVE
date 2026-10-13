import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.linalg import svd

def varimax(x: pd.DataFrame, normalize: bool = True, eps: float = 1e-5):
    '''
    Adjusted from https://github.com/BaderLab/sciRED
    Perform varimax rotation on a factor loading matrix.

    Parameters
    ----------
    x : pd.DataFrame
        The factor loading matrix, shape (features, components)
    normalize : bool
        Whether to normalize the loading vectors before rotation
    eps : float
        Convergence threshold

    Returns
    -------
    dict with:
        'rotloading' : pd.DataFrame
            Rotated loading matrix, same index and columns as input
        'rotmat' : np.ndarray
            Rotation matrix used
    '''
    if not isinstance(x, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    index = x.index
    columns = x.columns
    x_values = x.values

    nc = x_values.shape[1]
    if nc < 2:
        return {'rotloading': x.copy(), 'rotmat': np.eye(nc)}
    
    if normalize:
        sc = np.sqrt(np.sum(x_values**2, axis=1))
        x_values = x_values / sc[:, np.newaxis]
    
    p = x_values.shape[0]
    TT = np.eye(nc)
    d = 0
    for _ in range(1000):
        z = np.dot(x_values, TT)
        B = np.dot(x_values.T, z**3 - np.dot(z, np.diag(np.sum(z**2, axis=0)) / p))
        u, sB, vh = svd(B, full_matrices=False)
        TT = np.dot(u, vh)
        dpast = d
        d = np.sum(sB)
        if d < dpast * (1 + eps):
            break
    
    z = np.dot(x_values, TT)
    if normalize:
        z = z * sc[:, np.newaxis]

    rotloading_df = pd.DataFrame(z, index=index, columns=columns)
    return {'rotloading': rotloading_df, 'rotmat': TT}


def get_rotated_scores(factor_scores, rotmat):
    '''
    calculate the rotated factor scores
    factor_scores: the factor scores matrix
    rotmat: the rotation matrix
    '''
    return np.dot(factor_scores, rotmat)


def t_stat(X, labels):
    t_stat_matrix = pd.DataFrame(index=np.unique(labels), columns=X.columns)

    for label in np.unique(labels):
        in_class = labels == label
        out_class = labels != label
        
        for dim in X.columns:
            x1 = X.loc[in_class, dim]
            x2 = X.loc[out_class, dim]
            
            if len(np.unique(x1)) <= 1 or len(np.unique(x2)) <= 1:
                t_stat = np.nan
            else:
                t_stat, _ = ttest_ind(x1, x2, equal_var=False)
            t_stat_matrix.loc[label, dim] = t_stat

    max_abs_col_index = t_stat_matrix.abs().idxmax(axis=1)  
    max_values = t_stat_matrix.apply(lambda row: row[max_abs_col_index[row.name]], axis=1)
    result = pd.DataFrame({
        'Label': max_abs_col_index.index,
        'Topic': max_abs_col_index,
        'Tvalue': max_values
    })

    return t_stat_matrix, result

def correct_loadings(loading, t_stat, priority_cluster=None):
    corrected_loading = loading.copy()
    for topic in loading.columns:
        abs_vals = t_stat[topic].abs()
        max_val = abs_vals.max()
        
        candidates = abs_vals[abs_vals == max_val].index.tolist()

        if priority_cluster in candidates:
            max_cluster = priority_cluster
        else:
            max_cluster = candidates[0]  

        sign = np.sign(t_stat.loc[max_cluster, topic])

        corrected_loading[topic] *= sign
    ranked_indices_dict = {
        topic: corrected_loading[topic].sort_values(ascending=False).index.tolist()
        for topic in corrected_loading.columns
    }
    return corrected_loading.T, ranked_indices_dict