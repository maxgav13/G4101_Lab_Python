import numpy as np
import pandas as pd
import gstools as gs
from typing import Optional, Union
import warnings

def krige_cv(
    formula: Optional[str] = None,
    locations: Optional[np.ndarray] = None,
    data: pd.DataFrame = None,
    model: Optional[gs.CovModel] = None,
    nfold: Optional[Union[int, np.ndarray]] = None,
    variogram_model: str = 'Spherical',
    variogram_parameters: Optional[dict] = None,
    coordinates: Optional[tuple] = None,
    verbose: bool = False,
    kriging_type: str = 'ordinary'
):
    """
    Perform leave-one-out or n-fold cross-validation for kriging using GSTools.
    
    Parameters:
    -----------
    formula : str, optional
        Formula specifying the dependent variable and covariates (e.g., 'z ~ x + y')
        If None or 'z ~ 1', ordinary kriging is performed
    locations : np.ndarray, optional
        Nx2 array of coordinates [x, y]
    data : pd.DataFrame
        DataFrame containing the data
    model : gs.CovModel, optional
        GSTools covariance model. If None, one will be created from variogram_model
    nmax : int, optional
        Maximum number of neighboring observations for kriging
    nmin : int
        Minimum number of neighboring observations
    maxdist : float, optional
        Maximum search radius
    nfold : int, np.ndarray, or None, optional
        - If None: leave-one-out cross-validation
        - If int: number of folds (random assignment)
        - If np.ndarray: custom fold assignments (length must match data)
          Array values indicate fold membership (e.g., [1,1,2,2,3,3])
    variogram_model : str
        Type of variogram model: 'Gaussian', 'Exponential', 'Spherical', 
        'Matern', 'Stable', 'Linear'
    variogram_parameters : dict, optional
        Parameters for the variogram model:
        - 'var' or 'sill': variance/sill
        - 'len_scale' or 'range': correlation length/range
        - 'nugget': nugget effect
    coordinates : tuple, optional
        Column names for coordinates in data (e.g., ('x', 'y'))
    verbose : bool
        Print progress information
    kriging_type : str
        Type of kriging: 'ordinary', 'simple', 'universal'
        
    Returns:
    --------
    pd.DataFrame with columns:
        - var1.pred: predicted values
        - var1.var: kriging variance
        - observed: observed values
        - residual: residuals (observed - predicted)
        - zscore: standardized residuals
        - fold: fold number
    """
    
    # Parse formula and extract variable names
    if formula is not None:
        parts = formula.split('~')
        dependent = parts[0].strip()
        if len(parts) > 1 and parts[1].strip() and parts[1].strip() != '1':
            # Universal kriging with trend
            covariates = [c.strip() for c in parts[1].split('+')]
            kriging_type = 'universal'
        else:
            kriging_type = kriging_type
            covariates = None
    else:
        # Need to infer dependent variable
        if 'z' in data.columns:
            dependent = 'z'
        else:
            dependent = data.columns[0]
        kriging_type = kriging_type
        covariates = None
    
    # Extract coordinates
    if locations is not None:
        x = locations[:, 0]
        y = locations[:, 1]
    elif coordinates is not None:
        x = data[coordinates[0]].values
        y = data[coordinates[1]].values
    else:
        # Try common coordinate names
        if 'x' in data.columns and 'y' in data.columns:
            x = data['x'].values
            y = data['y'].values
        else:
            raise ValueError("Coordinates must be specified")
    
    # Extract dependent variable
    z = data[dependent].values
    n = len(z)
    
    # Create covariance model if not provided
    if model is None:
        # Set default parameters if not provided
        if variogram_parameters is None:
            # Estimate from data
            var = np.var(z)
            # Estimate range as 1/3 of maximum distance
            max_dist = np.sqrt(np.max((x - x.mean())**2 + (y - y.mean())**2))
            len_scale = max_dist / 3
            nugget = 0.0
        else:
            var = variogram_parameters.get('var', 
                  variogram_parameters.get('sill', np.var(z)))
            len_scale = variogram_parameters.get('len_scale',
                        variogram_parameters.get('range', 1.0))
            nugget = variogram_parameters.get('nugget', 0.0)
        
        # Create model based on type
        model_class = getattr(gs, variogram_model, gs.Spherical)
        model = model_class(dim=2, var=var, len_scale=len_scale, nugget=nugget)
    
    # Setup cross-validation folds
    if nfold is None:
        # Leave-one-out
        folds = [[i] for i in range(n)]
        fold_assignment = np.arange(1, n + 1)
    elif isinstance(nfold, (np.ndarray, list, pd.Series)):
        # Custom fold assignment provided
        fold_assignment = np.array(nfold)
        
        # Validate fold assignment
        if len(fold_assignment) != n:
            raise ValueError(f"Fold assignment length ({len(fold_assignment)}) "
                           f"must match data length ({n})")
        
        # Group indices by fold
        unique_folds = np.unique(fold_assignment)
        folds = [np.where(fold_assignment == fold)[0] for fold in unique_folds]
        
        if verbose:
            print(f"Using custom fold assignment with {len(unique_folds)} folds")
            for i, fold in enumerate(unique_folds):
                n_in_fold = np.sum(fold_assignment == fold)
                print(f"  Fold {fold}: {n_in_fold} observations")
    elif isinstance(nfold, int):
        # n-fold cross-validation with random assignment
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, nfold)
        
        # Create fold assignment array
        fold_assignment = np.zeros(n, dtype=int)
        for fold_idx, fold_indices in enumerate(folds):
            fold_assignment[fold_indices] = fold_idx + 1
    else:
        raise ValueError("nfold must be None, int, or array-like")
    
    # Initialize results
    predictions = np.zeros(n)
    variances = np.zeros(n)
    
    if verbose:
        print(f"\nUsing {kriging_type} kriging\n")

    # Perform cross-validation
    for fold_idx, test_indices in enumerate(folds):
        if verbose:
            print(f"Processing fold {fold_idx + 1}/{len(folds)} "
                  f"({len(test_indices)} test points)")
        
        # Create train/test split
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_indices] = False
        
        x_train = x[train_mask]
        y_train = y[train_mask]
        z_train = z[train_mask]
        x_test = x[test_indices]
        y_test = y[test_indices]
        
        # Prepare conditioning points
        cond_pos = np.array([x_train, y_train])


        # Fit kriging model
        try:
            if kriging_type == 'ordinary':
                krig = gs.krige.Ordinary(
                    model=model,
                    cond_pos=cond_pos,
                    cond_val=z_train
                )
            elif kriging_type == 'simple':
                krig = gs.krige.Simple(
                    model=model,
                    cond_pos=cond_pos,
                    cond_val=z_train,
                    mean=np.mean(z_train)
                )
            elif kriging_type == 'universal':
                # For universal kriging, we need drift functions
                if covariates:
                    # Build drift terms
                    drift_functions = []
                    for cov in covariates:
                        if cov in ['x', 'X']:
                            drift_functions.append(lambda x, y: x)
                        elif cov in ['y', 'Y']:
                            drift_functions.append(lambda x, y: y)
                        elif cov in data.columns:
                            # External drift
                            cov_values = data[cov].values[train_mask]
                            drift_functions.append(lambda x, y, vals=cov_values: vals)
                    
                    krig = gs.krige.Universal(
                        model=model,
                        cond_pos=cond_pos,
                        cond_val=z_train,
                        drift_functions=drift_functions
                    )
                else:
                    # Fall back to ordinary kriging
                    krig = gs.krige.Ordinary(
                        model=model,
                        cond_pos=cond_pos,
                        cond_val=z_train
                    )
            else:
                raise ValueError(f"Unknown kriging type: {kriging_type}")
            
            # Predict at test locations
            field, krig_var = krig(
                [x_test, y_test],
                return_var=True
            )
            
            predictions[test_indices] = field
            variances[test_indices] = krig_var

        
        except Exception as e:
            warnings.warn(f"Kriging failed for fold {fold_idx}: {str(e)}")
            predictions[test_indices] = np.nan
            variances[test_indices] = np.nan
    
    # Calculate residuals and standardized residuals
    residuals = z - predictions
    zscores = residuals / np.sqrt(variances)
    
    # Create results dataframe
    results = pd.DataFrame({
        'var1.pred': predictions,
        'var1.var': variances,
        'observed': z,
        'residual': residuals,
        'zscore': zscores,
        'fold': fold_assignment,
        'x': x,
        'y': y
    })
    
    # Add summary statistics as attributes
    results.attrs['mean_error'] = np.nanmean(residuals)
    results.attrs['rmse'] = np.sqrt(np.nanmean(residuals**2))
    results.attrs['mae'] = np.nanmean(np.abs(residuals))
    results.attrs['mse'] = np.nanmean(residuals**2)
    results.attrs['cor'] = np.corrcoef(z, predictions)[0, 1] if not np.any(np.isnan(predictions)) else np.nan
    results.attrs['msdr'] = np.nanmean(zscores**2)
    
    if verbose:
        print("\nCross-validation results:")
        print(f"Mean Error: {results.attrs['mean_error']:.4f}")
        print(f"RMSE: {results.attrs['rmse']:.4f}")
        print(f"MAE: {results.attrs['mae']:.4f}")
        print(f"MSDR: {results.attrs['msdr']:.4f}")
        print(f"Correlation: {results.attrs['cor']:.4f}")
    
    return results


# Example usage with custom fold assignment:
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 50
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    
    # Create spatially correlated field
    model_true = gs.Gaussian(dim=2, var=2, len_scale=2)
    srf = gs.SRF(model_true, seed=20170519)
    z = srf([x, y])
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z
    })
    
    # Example 1: Leave-one-out
    print("=" * 60)
    print("Example 1: Leave-one-out CV")
    print("=" * 60)
    cv_loo = krige_cv(
        formula='z ~ 1',
        data=df,
        coordinates=('x', 'y'),
        variogram_model='Gaussian',
        variogram_parameters={'var': 2, 'len_scale': 2, 'nugget': 0.1},
        nfold=None,
        verbose=True
    )
    
    # Example 2: Random 5-fold
    print("\n" + "=" * 60)
    print("Example 2: Random 5-fold CV")
    print("=" * 60)
    cv_5fold = krige_cv(
        formula='z ~ 1',
        data=df,
        coordinates=('x', 'y'),
        variogram_model='Gaussian',
        variogram_parameters={'var': 2, 'len_scale': 2, 'nugget': 0.1},
        nfold=5,
        verbose=True
    )
    
    # Example 3: Custom fold assignment (spatial blocks)
    print("\n" + "=" * 60)
    print("Example 3: Custom spatial block CV")
    print("=" * 60)
    
    # Create spatial blocks based on x and y coordinates
    custom_folds = np.ones(n_points, dtype=int)
    custom_folds[(x < 5) & (y < 5)] = 1  # Bottom-left
    custom_folds[(x >= 5) & (y < 5)] = 2  # Bottom-right
    custom_folds[(x < 5) & (y >= 5)] = 3  # Top-left
    custom_folds[(x >= 5) & (y >= 5)] = 4  # Top-right
    
    cv_custom = krige_cv(
        formula='z ~ 1',
        data=df,
        coordinates=('x', 'y'),
        variogram_model='Gaussian',
        variogram_parameters={'var': 2, 'len_scale': 2, 'nugget': 0.1},
        nfold=custom_folds,
        verbose=True
    )
    
    print("\nCustom fold distribution:")
    print(cv_custom['fold'].value_counts().sort_index())
    
    # Example 4: Stratified folds based on value quantiles
    print("\n" + "=" * 60)
    print("Example 4: Stratified CV based on value quantiles")
    print("=" * 60)
    
    quantiles = pd.qcut(z, q=5, labels=False, duplicates='drop') + 1
    
    cv_stratified = krige_cv(
        formula='z ~ 1',
        data=df,
        coordinates=('x', 'y'),
        variogram_model='Gaussian',
        variogram_parameters={'var': 2, 'len_scale': 2, 'nugget': 0.1},
        nfold=quantiles,
        verbose=True
    )
