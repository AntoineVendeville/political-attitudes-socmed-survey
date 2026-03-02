# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from numba import njit
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.stats import wasserstein_distance, pearsonr, linregress
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity

# Ignore specific warnings from diptest and scipygit
warnings.filterwarnings("ignore", category=UserWarning, module="diptest")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def apply_pca(df, dimensions, n_components):

    # Standardize the data (important for PCA)
    df_scaled = StandardScaler().fit_transform(df[dimensions]) 

    # Apply PCA
    pca = PCA(n_components = n_components)  # Keep all components
    principal_components = pca.fit_transform(df_scaled)

    # Create a DataFrame with principal components
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

    # Explained variance
    explained_variance = pca.explained_variance_ratio_

    return pca, pca_df, explained_variance

def get_pca_loadings(pca, dimensions, n_components):
    return pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=dimensions)

@njit
def weighted_mean(x, w):
    return np.average(x, weights=w)

@njit
def weighted_cov(x, y, w):
    mean_x = weighted_mean(x, w)
    mean_y = weighted_mean(y, w)
    cov = np.sum(w * (x - mean_x) * (y - mean_y)) / np.sum(w)
    return cov

@njit
def weighted_corr(x, y, w):
    cov_xy = weighted_cov(x, y, w)
    std_x = np.sqrt(weighted_cov(x, x, w))
    std_y = np.sqrt(weighted_cov(y, y, w))
    return cov_xy / (std_x * std_y)

@njit
def weighted_std(x, w, ddof=0):
    mean_x = np.sum(w * x) / np.sum(w)
    variance = np.sum(w * (x - mean_x)**2) / (np.sum(w) - ddof)
    return np.sqrt(variance)


def weighted_corr_matrix(df, cols, weights):
    corr_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    for i in cols:
        for j in cols:
            corr_matrix.loc[i, j] = weighted_corr(df[i].values, df[j].values, df[weights].values)
    return corr_matrix

@njit
def weighted_corr_matrix_np(X, weights):
    """
    Compute a weighted correlation matrix for a 2D NumPy array X.
    
    Parameters:
    - X: 2D NumPy array of shape (n_samples, n_features)
    - weights: 1D NumPy array of shape (n_samples,)
    
    Returns:
    - corr_matrix: 2D NumPy array of shape (n_features, n_features)
    """
    n_features = X.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            corr_matrix[i, j] = weighted_corr(X[:, i], X[:, j], weights)
    
    return corr_matrix

def effective_dimensionality(df, dimensions, weight_col=None, eps=1e-12):
    n_dim = len(dimensions)
    if weight_col==None:
        corr_mat = df[dimensions].corr().values
    else:
        corr_mat = weighted_corr_matrix_np(df[dimensions].values, df[weight_col].values)
    eigenvalues = np.linalg.eigvals(corr_mat)
    eigenvalues = eigenvalues / n_dim
    eigenvalues = np.clip(eigenvalues, eps, None) # clip small eigenvalues to avoid floating-point issues
    eigenvalues = np.power(eigenvalues, -eigenvalues)
    eigenvalues = np.real_if_close(eigenvalues, tol=1000) # remove complex components due to floating-point arithmetic
    return np.prod(eigenvalues)

def effective_dimensionality_from_corrmat(corr_mat, eps=1e-12):
    n_dim = corr_mat.shape[0]
    eigenvalues = np.linalg.eigvals(corr_mat)
    eigenvalues = eigenvalues / n_dim
    eigenvalues = np.clip(eigenvalues, eps, None) # clip small eigenvalues to avoid floating-point issues
    eigenvalues = np.power(eigenvalues, -eigenvalues)
    eigenvalues = np.real_if_close(eigenvalues, tol=1000) # remove complex components due to floating-point arithmetic
    return np.prod(eigenvalues)

def weightedPCA(df, columns, weight_col, n_components=None):
    """
    Perform weighted PCA on selected columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric data and a weight column.
    columns : list of str
        Names of columns to include in the PCA.
    weight_col : str, default="weight"
        Name of the column containing sample weights.
    n_components : int, optional
        Number of principal components to keep. If None, keep all.

    Returns
    -------
    loadings_df : pandas.DataFrame
        Loadings (eigenvectors), features x components.
    scores_df : pandas.DataFrame
        Row coordinates (PCA scores), samples x components (aligned with df.index).
    explained_var_df : pandas.DataFrame
        Explained variance ratio for each component.
    """
    # Extract PCA matrix and weights
    X = df[columns].to_numpy(dtype=float)
    weights = df[weight_col].to_numpy(dtype=float)
    weights = weights / np.sum(weights)

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    # Weighted mean
    mean = np.average(X, axis=0, weights=weights)
    X_centered = X - mean

    # Weighted covariance
    cov = (X_centered.T * weights) @ X_centered

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Keep desired components
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]
    explained_var_ratio = eigvals / np.sum(eigvals)

    # Compute PCA scores (row coordinates)
    scores = X_centered @ eigvecs

    # Create DataFrames
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(eigvecs, index=columns, columns=pc_names)
    scores_df = pd.DataFrame(scores, index=df.index, columns=pc_names)

    return loadings_df, scores_df, explained_var_ratio

def compute_polarization(df, col, weighted=False, weight_col=None):

    # prepare
    value_proba = dict()

    if weighted:
        total_weights = df[weight_col].sum()
        for val in df[col].unique():
            weight_tmp = df[df[col]==val][weight_col].sum() / total_weights
            value_proba[val] = weight_tmp
    else:
        value_proba = dict(df[col].value_counts(normalize=True))
    value_proba = {key: val for key,val in value_proba.items() if val>0} # remove zero entries
    value_proba = sorted(value_proba.items(), key=lambda v:v[0])
    
    # iterate
    Ec = 0
    proba_cumul = 0
    for value,proba in value_proba[:-1]:
        proba_cumul += proba
        p,q = proba_cumul, 1-proba_cumul
        Ec += 2 ** (-p*np.log2(p) -q*np.log2(q)) -1
    Ec /= (len(value_proba)-1)

    # end
    return Ec

def polarization_from_counts_df(df, value_col='bin', counts_col='count'):

    # prepare
    value_proba = df.set_index(value_col)[counts_col].to_dict()
    value_proba = sorted(value_proba.items(), key=lambda v:v[0])
    
    # iterate
    Ec = 0
    proba_cumul = 0
    for value,proba in value_proba[:-1]:
        proba_cumul += proba
        p,q = proba_cumul, 1-proba_cumul
        Ec_tmp = 0
        if p>0:
            Ec_tmp += -p*np.log2(p)
        if q>0:
            Ec_tmp += -q*np.log2(q)
        Ec += 2**Ec_tmp -1
    Ec /= (len(value_proba)-1)

    # end
    return Ec

def load_ess_counts(o=False):
    folder = 'ESS-results/' if o else 'ESS_results/'
    ess_counts_full = pd.read_csv(folder + 'weighted_counts_by_variable.csv')
    ess_counts_full['variable'] = [d.replace('.', ' ') for d in ess_counts_full['variable'].values]
    ess_counts_full['variable'] = ess_counts_full['variable'].replace({'GAL TAN':'GAL-TAN',
                                                                    'Left Right':'Left-Right',
                                                                    'Anti elitism':'Anti-elitism',
                                                                    'LGBT rights':'Social liberalism'})
    return ess_counts_full


def perform_pca(DF, Names, weighted, Dimensions, weight_col, df_ches, df_mp, df_follower):

    pca, pca_df, explained_variance, loadings = dict(), dict(), dict(), dict()

    # n comp
    n_components = len(Dimensions)
    for df in DF:
        n_components = min(n_components, df.shape[0])

    for i,df in enumerate(DF):
        name = Names[i]

        if 'ESS' not in name:
            if weighted or name=='CHES':
                pca[name], loadings[name], pca_df[name], explained_variance[name] = weightedPCA(df, Dimensions, weight_col[name], n_components)
            else:
                pca[name], pca_df[name], explained_variance[name] = apply_pca(df, Dimensions, n_components)
                loadings[name] = get_pca_loadings(pca[name], Dimensions,  n_components)

    # append to existing df
    columns = [f'PC{i+1}' for i in range(n_components)]
    df_ches.loc[:, columns] = pca_df['CHES'][columns]
    df_mp.loc[:, columns] = pca_df['MPs'][columns]
    df_follower.loc[:, columns] = pca_df['Followers'][columns]

    return pca, pca_df, explained_variance, loadings, df_ches, df_mp, df_follower, n_components


def perform_pca_ess(df_ess, opa, n_components, explained_variance, loadings):

    columns = [f'PC{i+1}' for i in range(n_components)]

    name = 'ESS-opa' if opa else 'ESS'
    folder = name+'_pca/'

    ess_coordinates = pd.read_csv(folder + 'PCA_coordinates.csv')
    ess_explained_variance = pd.read_csv(folder + 'PCA_explained_variance.csv')

    explained_variance[name] = ess_explained_variance['Proportion'].values
    explained_variance[name] = explained_variance[name]/explained_variance[name].sum()
    df_ess.loc[:, columns] = ess_coordinates[[f'PC{i}' for i in range(1, n_components+1)]].values

    loadings[name] = pd.read_csv(folder + 'PCA_loadings.csv')
    #loadings[name] = loadings[name].rename(columns = {f'Dim.{i}': f'PC{i}' for i in range(1, n_components+1)})
    loadings[name]['Variable'] = [d.replace('.', ' ') for d in loadings[name]['Variable'].values]
    loadings[name]['Variable'] = loadings[name]['Variable'].replace({'GAL TAN':'GAL-TAN',
                                                                    'Left Right':'Left-Right',
                                                                    'Anti elitism':'Anti-elitism',
                                                                    'LGBT rights':'Social liberalism'})
    loadings[name] = loadings[name].set_index(loadings[name]['Variable'].values)
    loadings[name] = loadings[name].drop(columns = 'Variable')

    corrmat_ESS = pd.read_csv(folder + 'Corr_matrix.csv')
    corrmat_ESS['Unnamed: 0'] = [d.replace('.', ' ') for d in corrmat_ESS['Unnamed: 0'].values]
    corrmat_ESS['Unnamed: 0'] = corrmat_ESS['Unnamed: 0'].replace({'GAL TAN':'GAL-TAN',
                                                                    'Left Right':'Left-Right',
                                                                    'Anti elitism':'Anti-elitism',
                                                                    'LGBT rights':'Social liberalism'})
    corrmat_ESS = corrmat_ESS.set_index(corrmat_ESS['Unnamed: 0'].values).drop(columns='Unnamed: 0')
    corrmat_ESS.columns = corrmat_ESS.index

    pval_ESS = pd.read_csv(folder + 'Corr_pval.csv')
    pval_ESS['Unnamed: 0'] = [d.replace('.', ' ') for d in pval_ESS['Unnamed: 0'].values]
    pval_ESS['Unnamed: 0'] = pval_ESS['Unnamed: 0'].replace({'GAL TAN':'GAL-TAN',
                                                                    'Left Right':'Left-Right',
                                                                    'Anti elitism':'Anti-elitism',
                                                                    'LGBT rights':'Social liberalism'})
    pval_ESS = pval_ESS.set_index(pval_ESS['Unnamed: 0'].values).drop(columns='Unnamed: 0')
    pval_ESS.columns = pval_ESS.index

    return df_ess, explained_variance, loadings, corrmat_ESS, pval_ESS


def discretized(df, dim, ess_label_range):
    df_tmp = df.copy()
    bin_labels = ess_label_range[dim]
    bins = np.linspace(0, 10, len(bin_labels)+1)
    df_tmp[dim] = df_tmp[dim].apply(lambda a: min(max(a,0), 10))
    binned = pd.cut(df_tmp[dim], bins=bins, labels=bin_labels, include_lowest=True).astype(int)
    return binned

def get_x_counts(df, dim, weighted=False, weight_col='impression_count', name='Followers'):
    if weighted:
        counts = df.groupby(dim + "_bin")[weight_col].sum().reset_index(name="weighted_proportion").assign(weighted_proportion=lambda x: x["weighted_proportion"] / x["weighted_proportion"].sum())
    else:
        counts = df[dim + "_bin"].value_counts(sort=False, normalize=True).reset_index()
    counts.columns = ['bin', 'count']
    counts['group'] = name
    counts = counts.sort_values(by='bin')
    return counts


def get_ess_counts(ess_counts_full, dim, groupname='ESS'):
    ess_counts = pd.DataFrame()
    ess_counts[dim+'_bin'] = ess_counts_full[ess_counts_full.variable==dim]['value']
    ess_counts['proportion'] = ess_counts_full[ess_counts_full.variable==dim]['weighted_count']
    ess_counts['proportion'] = ess_counts['proportion'] / ess_counts['proportion'].sum()
    ess_counts.columns = ['bin', 'count']
    ess_counts['group'] = groupname
    ess_counts = ess_counts.sort_values(by='bin')
    return ess_counts

def compute_skew_old(ess_label_range, dim, counts, return_lower_higher=False):
    middle = np.mean(ess_label_range[dim])
    higher = counts[counts.bin>middle]['count'].sum()
    lower = counts[counts.bin<middle]['count'].sum()
    
    if higher > lower:
        skew = higher - lower
    elif lower > higher:
        skew = -lower + higher
    else:
        skew = 0

    if return_lower_higher:
        return skew, lower, higher
    else:
        return skew
    
def compute_skew(ess_label_range, dim, counts):
    middle = np.mean(ess_label_range[dim])
    skew = np.average(counts['bin'].values, weights=counts['count'].values)
    skew = (skew-middle)/middle
    return skew

def draw_arrow(x, y, ax, cmap='bwr', zorder=10, num_points=1000):

    # Create points from (-x, -y) to (x, y)
    xs = np.linspace(-x, x, num_points)
    ys = np.linspace(-y, y, num_points)

    # Stack into line segments
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize values for colormap (centered at 0)
    norm_vals = np.linspace(-1, 1, num_points-1)  # from -1 to 1
    norm = plt.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap(cmap)

    # Create gradient line
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=10, zorder=zorder)
    lc.set_array(norm_vals)

    # Plot
    ax.add_collection(lc)

    # draw arrow head
    scale1 = 1.2
    scale2 = 1.3
    color = cmap(-np.inf)
    arrow = mpatches.FancyArrowPatch((0,0), (-x*scale2, -y*scale2), mutation_scale=50, linewidth=0, color=color, zorder=-10)
    ax.add_patch(arrow)
    color = cmap(np.inf)
    arrow = mpatches.FancyArrowPatch((0,0), (x*scale2, y*scale2), mutation_scale=50, linewidth=0, color=color, zorder=-10)
    ax.add_patch(arrow)

    return ax

def draw_arrow_bidim(x, y, ax, center=5, scale=1.1, cmap='bwr', zorder=10, num_points=1000):

    # Create points
    xs = np.linspace(center-x, center+x, num_points)
    ys = np.linspace(center-y, center+y, num_points)

    # Stack into line segments
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize values for colormap (centered at 0)
    norm_vals = np.linspace(-1, 1, num_points-1)  # from -1 to 1
    norm = plt.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap(cmap)

    # Create gradient line
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=7, zorder=zorder)
    lc.set_array(norm_vals)

    # Plot
    ax.add_collection(lc)

    # draw arrow head
    eps = 1e-6
    color = cmap(-np.inf)
    arrow = mpatches.FancyArrowPatch(
        (center - x * eps, center - y * eps),  # almost the same as end point
        (center - x * scale, center - y * scale),
        mutation_scale=40,
        arrowstyle='-|>',     # simple arrow head
        linewidth=0,
        color=color,
        zorder=zorder - 1
    )
    ax.add_patch(arrow)
    color = cmap(np.inf)
    arrow = mpatches.FancyArrowPatch(
        (center + x * eps, center + y * eps),  # almost the same as end point
        (center + x * scale, center + y * scale),
        mutation_scale=40,
        arrowstyle='-|>',     # simple arrow head
        linewidth=0,
        color=color,
        zorder=zorder - 1
    )
    ax.add_patch(arrow)

    return ax


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """
    Varimax rotation of a loadings matrix.
    
    Parameters:
        Phi : (n_features, n_factors) loadings matrix
        gamma : usually 1.0 for Varimax
        q : max iterations
        tol : convergence tolerance
    
    Returns:
        rotated loadings
    """
    p,k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(np.dot(Phi.T, Lambda**3 - (gamma/p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0)))))
        R = np.dot(u, vh)
        d_old = d
        d = np.sum(s)
        if d_old != 0 and d - d_old < tol:
            break
    return np.dot(Phi, R), R