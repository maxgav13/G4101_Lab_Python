from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import gstools as gs
from scipy import stats
from typing import Optional


def gs_loocv(
    model: gs.CovModel = gs.Spherical(dim=2), 
    coords: np.ndarray = None, 
    values: np.ndarray = None, 
    normalizer: Optional[gs.normalizer] = None, 
    kriging_type: str = "ordinary", 
    transformar: bool = False, 
    transformacion: Optional[str] = None
    ):

    """
    Perform leave-one-out cross-validation with GSTools.
    
    Parameters:
    -----------
    model : gs.CovModel, optional
        GSTools covariance model. If None, gs.Spherical(dim=2) is used by default.
    coords : array-like, shape (n, 2)
        Coordinates of the observations.
    values : array-like, shape (n,)
        Observed values.
    normalizer : gs.normalizer, optional
        GSTools normalizer. If None, no normalization is applied.
    kriging_type : str, optional
        Type of kriging to use. Default is "ordinary".
    transformar : bool, optional
        Whether to apply a transformation to the data. Default is False.
    transformacion : str, optional
        Type of transformation to apply. Default is None, options could be "Quantile" or "Box-Cox".

    Returns:
    --------
    dict
        Dictionary with pd.DataFrames kcv_df (observed, pred, pred_var, residual, zscore) and metrics (RMSE, MAE, r, R2, MSDR, bias)
    """

    n = len(values)
    preds = np.empty(n, dtype=float)
    pred_vars = np.empty(n, dtype=float)

    xs = coords[:, 0]
    ys = coords[:, 1]

    for i in range(n):
        mask = np.arange(n) != i
        cond_pos = (xs[mask], ys[mask])
        cond_val = values[mask]

        kr = gs.krige.Krige(
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            normalizer=normalizer,
            fit_normalizer=False,
            unbiased=False if transformar and transformacion == "Quantile" else True,
            mean = 0 if transformar and transformacion == "Quantile" else None
        )

        # Try calling kr() to predict a point; if your gstools version uses another API,
        # replace next line with the appropriate call (e.g. kr.execute('points', [xs[i]],[ys[i]]))
        pred, var = kr((xs[i], ys[i]))  # pred, var may be scalars or 1‑element arrays

        preds[i] = float(np.asarray(pred).ravel()[0])
        pred_vars[i] = float(np.asarray(var).ravel()[0])

    residuals = values - preds
    zscores = residuals / np.sqrt(pred_vars)

    rmse = root_mean_squared_error(values, preds)
    mae = mean_absolute_error(values, preds)
    r = stats.pearsonr(values,preds)[0]
    r2 = r2_score(values, preds)
    msdr = np.mean(residuals**2 / pred_vars)
    bias = residuals.mean()
    
    kcv_df = pd.DataFrame({
        "observed": values,
        "pred": preds,
        "pred_var": pred_vars,
        "residual": residuals,
        "zscore": zscores
    })

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r": r,
        "r2": r2,
        "msdr": msdr,
        "bias": bias
    }

    return {
        "kcv_df": kcv_df,
        "metrics": metrics
    }

def gs_kfold(
    model: gs.CovModel = gs.Spherical(dim=2), 
    coords: np.ndarray = None, 
    values: np.ndarray = None, 
    k: int = 5,
    normalizer: Optional[gs.normalizer] = None, 
    kriging_type: str = "ordinary", 
    transformar: bool = False, 
    transformacion: Optional[str] = None,
    shuffle: bool = True, 
    random_state: Optional[int] = None
    ):

    """
    Perform k-fold cross-validation with GSTools.
    
    Parameters:
    -----------
    model : gs.CovModel, optional
        GSTools covariance model. If None, gs.Spherical(dim=2) is used by default.
    coords : array-like, shape (n, 2)
        Coordinates of the observations.
    values : array-like, shape (n,)
        Observed values.
    k : int, optional
        Number of folds for cross-validation. Default is 5.
    normalizer : gs.normalizer, optional
        GSTools normalizer. If None, no normalization is applied.
    kriging_type : str, optional
        Type of kriging to use. Default is "ordinary".
    transformar : bool, optional
        Whether to apply a transformation to the data. Default is False.
    transformacion : str, optional
        Type of transformation to apply. Default is None, options could be "Quantile" or "Box-Cox".
    shuffle : bool, optional
        Whether to shuffle the data before splitting into folds. Default is True.
    random_state : int, optional
        Random seed for reproducibility when shuffling. Default is None.

    Returns:
    --------
    dict
        Dictionary with pd.DataFrames kcv_df (observed, pred, pred_var, residual, zscore, fold) and metrics (RMSE, MAE, r, R2, MSDR, bias)
    """

    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)
    n = len(values)

    preds = np.full(n, np.nan)
    pred_vars = np.full(n, np.nan)
    groups = np.full(n, np.nan)

    xs = coords[:, 0]
    ys = coords[:, 1]

    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    splits = list(kf.split(coords))
    for fold, (train_idx, test_idx) in enumerate(splits):

        print(f"fold {fold}: train {train_idx.size}, test {test_idx.size}")

        cond_pos = (xs[train_idx], ys[train_idx])
        cond_val = values[train_idx]

        kr = gs.krige.Krige(
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            normalizer=normalizer,
            fit_normalizer=False,
            fit_variogram=False,
            unbiased=False if transformar and transformacion == "Quantile" else True,
            mean = 0 if transformar and transformacion == "Quantile" else None
        )

        pred, var = kr((xs[test_idx], ys[test_idx]))  # batch predict
        pred = np.asarray(pred).ravel()
        var = np.asarray(var).ravel()

        groups[test_idx] = fold
        preds[test_idx] = pred
        pred_vars[test_idx] = var

    residuals = values - preds
    zscores = residuals / np.sqrt(pred_vars)

    rmse = root_mean_squared_error(values, preds)
    mae = mean_absolute_error(values, preds)
    r = stats.pearsonr(values,preds)[0]
    r2 = r2_score(values, preds)

    valid = pred_vars > 0
    msdr = np.nan if valid.sum() == 0 else np.mean(residuals[valid]**2 / pred_vars[valid])

    bias = np.nanmean(residuals)

    kcv_df = pd.DataFrame({
        "observed": values,
        "pred": preds,
        "pred_var": pred_vars,
        "residual": residuals,
        "zscore": zscores,
        "fold": groups
    })

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r": r,
        "r2": r2,
        "msdr": msdr,
        "bias": bias
    }

    return {
        "kcv_df": kcv_df,
        "metrics": metrics
    }


def make_spatial_groups(
    coords, 
    k=5, 
    method="kmeans", 
    block_size=None, 
    random_state=None
    ):

    """
    Perform spatial k-fold cross-validation with GSTools.
    
    Parameters:
    -----------
    coords : array-like, shape (n, 2)
        Coordinates of the observations.
    k : int, optional
        Number of folds for cross-validation. Default is 5.
    method : str, optional
        Method to create spatial groups. Options are "kmeans" or "block". Default is "kmeans".
    block_size : float, optional
        Block size for "block" method. Required if method is "block". Default is None.
    random_state : int, optional
        Random seed for reproducibility when shuffling. Default is None.

    Returns:
    --------
    np.ndarray
        Array of group labels for each point, with values from 0 to k-1.
    """

    coords = np.asarray(coords)
    if method == "kmeans":
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(coords)
        return labels
    elif method == "block":
        if block_size is None:
            raise ValueError("block_size must be provided for method='block'")
        mins = coords.min(axis=0)
        bins = np.floor((coords - mins) / block_size).astype(int)
        # encode 2D bins into single integer label
        labels, _ = np.unique(bins[:,0] * (bins[:,1].max()+1) + bins[:,1], return_inverse=True)
        # return inverse mapping (one label per point)
        return np.floor((coords - mins) / block_size).astype(int)[:,0] * (bins[:,1].max()+1) + np.floor((coords - mins) / block_size).astype(int)[:,1]
    else:
        raise ValueError("Unknown method. Use 'kmeans' or 'block'.")


def gs_spatial_kfold(
    model: gs.CovModel = gs.Spherical(dim=2),
    coords: np.ndarray = None,
    values: np.ndarray = None,
    k: int = 5,
    method: str = "kmeans",
    block_size: float = None,
    normalizer: Optional[gs.normalizer] = None, 
    kriging_type: str = "ordinary", 
    transformar: bool = False, 
    transformacion: Optional[str] = None,
    random_state: Optional[int] = None
):

    """
    Perform spatial k-fold cross-validation with GSTools.
    
    Parameters:
    -----------
    model : gs.CovModel, optional
        GSTools covariance model. If None, gs.Spherical(dim=2) is used by default.
    coords : array-like, shape (n, 2)
        Coordinates of the observations.
    values : array-like, shape (n,)
        Observed values.
    k : int, optional
        Number of folds for cross-validation. Default is 5.
    method : str, optional
        Method to create spatial groups. Options are "kmeans" or "block". Default is "kmeans".
    block_size : float, optional
        Block size for "block" method. Required if method is "block". Default is None.
    normalizer : gs.normalizer, optional
        GSTools normalizer. If None, no normalization is applied.
    kriging_type : str, optional
        Type of kriging to use. Default is "ordinary".
    transformar : bool, optional
        Whether to apply a transformation to the data. Default is False.
    transformacion : str, optional
        Type of transformation to apply. Default is None, options could be "Quantile" or "Box-Cox".
    random_state : int, optional
        Random seed for reproducibility when shuffling. Default is None.

    Returns:
    --------
    dict
        Dictionary with pd.DataFrames kcv_df (observed, pred, pred_var, residual, zscore) and metrics (RMSE, MAE, r, R2, MSDR, bias)
    """

    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)
    n = len(values)
    xs, ys = coords[:,0], coords[:,1]

    groups = make_spatial_groups(coords, k=k, method=method, block_size=block_size, random_state=random_state)

    gkf = GroupKFold(n_splits=k)

    preds = np.full(n, np.nan)
    pred_vars = np.full(n, np.nan)

    # for i in range(k):
    #     train_idx = np.where(groups != i)[0]
    #     test_idx = np.where(groups == i)[0]
    #     cond_pos = (xs[train_idx], ys[train_idx])
    #     cond_val = values[train_idx]


    splits = list(gkf.split(X=coords, y=values, groups=groups))
    for fold_idx, (train_idx, test_idx) in enumerate(splits):

        print(f"fold {fold_idx}: train {train_idx.size}, test {test_idx.size}")
        
        cond_pos = (xs[train_idx], ys[train_idx])
        cond_val = values[train_idx]

        kr = gs.krige.Krige(
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            normalizer=normalizer,
            fit_normalizer=False,
            unbiased=False if transformar and transformacion == "Quantile" else True,
            mean=0 if transformar and transformacion == "Quantile" else None
        )

        # batch predict test points
        pred, var = kr((xs[test_idx], ys[test_idx]))
        pred = np.asarray(pred).ravel()
        var = np.asarray(var).ravel()

        preds[test_idx] = pred
        pred_vars[test_idx] = var

    residuals = values - preds
    zscores = residuals / np.sqrt(pred_vars)

    kcv_df = pd.DataFrame({
        "observed": values,
        "pred": preds,
        "pred_var": pred_vars,
        "residual": residuals,
        "zscore": zscores,
        "fold": groups
    })

    valid_mask = ~np.isnan(preds)
    metrics = {
        "rmse": root_mean_squared_error(values[valid_mask], preds[valid_mask]),
        "mae": mean_absolute_error(values[valid_mask], preds[valid_mask]),
        "r": stats.pearsonr(values[valid_mask], preds[valid_mask])[0],
        "r2": r2_score(values[valid_mask], preds[valid_mask]),
        # MSDR: mean(square(residual)/pred_var) computed only where pred_var > 0
        "msdr": None,
        "bias": np.nanmean(residuals[valid_mask])
    }

    pv_mask = (pred_vars > 0) & valid_mask
    if pv_mask.sum() > 0:
        metrics["msdr"] = np.mean((residuals[pv_mask]**2) / pred_vars[pv_mask])
    else:
        metrics["msdr"] = np.nan

    return {
        "kcv_df": kcv_df,
        "metrics": metrics
    }

def cross_validate_kriging(
    model: gs.CovModel = gs.Spherical(dim=2), 
    coords: np.ndarray = None, 
    values: np.ndarray = None, 
    groups: np.ndarray = None, 
    n_splits: int = 5, 
    max_lag: float = None, 
    kriging_type: str = "ordinary"
    ):
    
    """
    Perform GroupKFold cross-validation for kriging
    
    Parameters:
    -----------
    model : gstools.CovModel
        GSTools covariance model to fit.
    coords : array-like, shape (2, n)
        Coordinates of the observations (2 x n).
    values : array-like, shape (n,)
        Observed values.
    groups : array-like, shape (n,)
        Group labels for each observation, used for GroupKFold splitting.
    n_splits : int, optional
        Number of folds for cross-validation. Default is 5.
    max_lag : float, optional
        Maximum lag distance for variogram estimation. If None, it will be set to half the maximum distance between points.
    kriging_type : str, optional
        Type of kriging to use. Options are "ordinary" or "simple". Default is "ordinary".
    
    Returns:
    --------
    dict
        Dictionary containing cross-validation scores and fold results.
    """
    
    xs, ys = coords[:,0], coords[:,1]

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    
    scores = {'rmse': [], 'r2': [], 'mae': []}
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(coords, values, groups)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"Train groups: {np.unique(groups[train_idx])}")
        print(f"Test groups: {np.unique(groups[test_idx])}")
        
        # Split data
        pos_train = (xs[train_idx], ys[train_idx])
        pos_test = (xs[test_idx], ys[test_idx])
        # coords_train = coords[train_idx]
        coords_test = coords[test_idx]
        values_train = values[train_idx]
        values_test = values[test_idx]
        
        # Fit variogram on training data
        bin_center, gamma = gs.vario_estimate(
            pos_train, 
            values_train,
            max_dist=max_lag
        )
        
        # Fit theoretical variogram model
        fit_model = model
        fit_model.fit_variogram(bin_center, gamma, nugget=True)
        
        print(f"  Fitted parameters: var={fit_model.var:.2f}, len_scale={fit_model.len_scale:.2f}, nugget={fit_model.nugget:.2f}")
        
        # Perform kriging on test locations
        if kriging_type == "ordinary":
            krige = gs.krige.Ordinary(fit_model, pos_train, values_train)
        elif kriging_type == "simple":
            krige = gs.krige.Simple(fit_model, pos_train, values_train, mean=np.mean(values_train))

        preds, var = krige(pos_test, return_var=True)
        preds = np.asarray(preds).ravel()

        # Calculate metrics
        rmse = root_mean_squared_error(values_test, preds)
        r = stats.pearsonr(values_test, preds)[0]
        r2 = r**2 # r2_score(values_test, preds)
        mae = mean_absolute_error(values_test, preds)

        scores['rmse'].append(rmse)
        scores['r2'].append(r2)
        scores['mae'].append(mae)
        
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAE: {mae:.3f}")
        
        fold_results.append({
            'fold': fold + 1,
            'model': fit_model,
            'pred': preds,
            'observed': values_test,
            'var': var,
            'coords': coords_test
        })
    
    # Summary statistics
    print("\n" + "="*50)
    print("Cross-Validation Summary:")
    print(f"Mean RMSE: {np.mean(scores['rmse']):.3f} (+/- {np.std(scores['rmse']):.3f})")
    print(f"Mean R²: {np.mean(scores['r2']):.3f} (+/- {np.std(scores['r2']):.3f})")
    print(f"Mean MAE: {np.mean(scores['mae']):.3f} (+/- {np.std(scores['mae']):.3f})")
    
    return scores, fold_results

# Run cross-validation
# scores, fold_results = cross_validate_kriging(gs.Spherical(dim=2), coords, values, groups, n_splits=5)


def compare_variogram_models(
    coords: np.ndarray = None, 
    values: np.ndarray = None, 
    groups: np.ndarray = None, 
    models_to_test: dict = None,
    max_lag: float = None,
    kriging_type: str = "ordinary"
    ):

    """
    Compare different variogram models using GroupKFold

    Parameters:
    -----------
    coords : array-like, shape (2, n)
        Coordinates of the observations (2 x n).
    values : array-like, shape (n,)
        Observed values.
    groups : array-like, shape (n,)
        Group labels for each observation, used for GroupKFold splitting.
    models_to_test : dict
        Dictionary of model names and corresponding gstools.CovModel classes to test.
    max_lag : float, optional
        Maximum lag distance for variogram estimation. If None, it will be set to half the maximum distance between points.
    kriging_type : str, optional
        Type of kriging to use. Options are "ordinary" or "simple". Default is "ordinary".
    
    Returns:
    --------
    dict
        Dictionary containing RMSE scores for each model across folds.
    """
    
    xs, ys = coords[:,0], coords[:,1]

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    
    model_scores = {name: [] for name in models_to_test.keys()}
    
    for train_idx, test_idx in gkf.split(coords, values, groups):
        pos_train = (xs[train_idx], ys[train_idx])
        pos_test = (xs[test_idx], ys[test_idx])
        # coords_train = coords[train_idx]
        # coords_test = coords[test_idx]
        values_train = values[train_idx]
        values_test = values[test_idx]
        
        # Estimate experimental variogram
        bin_center, gamma = gs.vario_estimate(
            pos_train, values_train, max_dist=max_lag
        )
        
        # Test each model
        for model_name, model_class in models_to_test.items():
            fit_model = model_class(dim=2)
            fit_model.fit_variogram(bin_center, gamma, nugget=True)
            
            if kriging_type == "ordinary":
                krige = gs.krige.Ordinary(fit_model, pos_train, values_train)
            elif kriging_type == "simple":
                krige = gs.krige.Simple(fit_model, pos_train, values_train, mean=np.mean(values_train))

            preds, var = krige(pos_test, return_var=True)
            preds = np.asarray(preds).ravel()

            rmse = root_mean_squared_error(values_test, preds)
            model_scores[model_name].append(rmse)
    
    # Print comparison
    print("\nModel Comparison (RMSE):")
    for model_name, rmses in model_scores.items():
        print(f"{model_name:15s}: {np.mean(rmses):.3f} (+/- {np.std(rmses):.3f})")
    
    return model_scores

# Compare models
models = {
    'Gaussian': gs.Gaussian,
    'Exponential': gs.Exponential,
    'Spherical': gs.Spherical,
    'Matern': gs.Matern,
    'Stable': gs.Stable
}

# model_comparison = compare_variogram_models(coords, values, groups, models)