import pandas as pd
from backend.core.signal_rule_config import SignalRuleConfig

def check_depth_signal(df, depth_col='depth', d_noise=None, min_points=None):
    """
    Rule engine for depth signal health check.
    - Clips small negative values (>-d_noise) to zero and tags as minor_noise.
    - Flags large negative values (<-d_noise) as neg_depth and assigns 'repair' action.
    - Flags out-of-bounds values as physical anomaly.
    Args:
        df: pandas DataFrame containing the depth column.
        depth_col: column name for depth.
        d_noise: threshold for small negative noise (unit: mm). If None, use config default.
        min_points: minimum consecutive points to consider as a segment (not used in this basic version). If None, use config default.
    Returns:
        DataFrame with added columns: 'neg_depth', 'minor_noise', 'depth_action', 'depth_physical_anomaly'.
    """
    if d_noise is None:
        d_noise = SignalRuleConfig.DEPTH_NOISE_THRESHOLD
    if min_points is None:
        min_points = SignalRuleConfig.DEPTH_BAD_SEGMENT_MIN_POINTS
    df = df.copy()
    df['neg_depth'] = 0
    df['minor_noise'] = 0
    df['depth_action'] = None
    df['depth_physical_anomaly'] = 0

    # 1. Physical bounds check
    mask_physical = (df[depth_col] < SignalRuleConfig.DEPTH_MIN) | (df[depth_col] > SignalRuleConfig.DEPTH_MAX)
    df.loc[mask_physical, 'depth_physical_anomaly'] = 1
    df.loc[mask_physical, 'depth_action'] = 'depth_out_of_bounds'

    # 2. Clip small negative values to zero, tag as minor_noise
    mask_minor = (df[depth_col] < 0) & (df[depth_col] >= -d_noise)
    df.loc[mask_minor, depth_col] = 0
    df.loc[mask_minor, 'minor_noise'] = 1
    df.loc[mask_minor, 'depth_action'] = 'minor_noise'

    # 3. Flag large negative values as neg_depth, assign 'repair' action
    mask_bad = (df[depth_col] < -d_noise)
    df.loc[mask_bad, 'neg_depth'] = 1
    df.loc[mask_bad, 'depth_action'] = 'repair'

    return df


def check_velocity_signal(df, velocity_col='velocity'):
    """
    Rule engine for velocity signal health check.
    - Flags out-of-bounds values as physical anomaly.
    - Flags flat or zero velocity as anomaly.
    Returns DataFrame with added columns: 'velocity_physical_anomaly', 'velocity_flat', 'velocity_action'.
    """
    df = df.copy()
    df['velocity_physical_anomaly'] = 0
    df['velocity_flat'] = 0
    df['velocity_action'] = None

    # 1. Physical bounds check
    mask_physical = (df[velocity_col] < SignalRuleConfig.VELOCITY_MIN) | (df[velocity_col] > SignalRuleConfig.VELOCITY_MAX)
    df.loc[mask_physical, 'velocity_physical_anomaly'] = 1
    df.loc[mask_physical, 'velocity_action'] = 'velocity_out_of_bounds'

    # 2. Flat or zero velocity check
    if velocity_col in df.columns:
        if df[velocity_col].max() == 0 or df[velocity_col].std() < SignalRuleConfig.VELOCITY_FLAT_STD:
            df['velocity_flat'] = 1
            df['velocity_action'] = 'velocity_flat_or_dropout'

    return df


def check_flow_signal(df, flow_col='flow'):
    """
    Rule engine for flow signal health check.
    - Flags out-of-bounds values as physical anomaly.
    - Flags no dry weather flow as anomaly.
    Returns DataFrame with added columns: 'flow_physical_anomaly', 'flow_no_dwf', 'flow_action'.
    """
    df = df.copy()
    df['flow_physical_anomaly'] = 0
    df['flow_no_dwf'] = 0
    df['flow_action'] = None

    # 1. Physical bounds check
    mask_physical = (df[flow_col] < SignalRuleConfig.FLOW_MIN) | (df[flow_col] > SignalRuleConfig.FLOW_MAX)
    df.loc[mask_physical, 'flow_physical_anomaly'] = 1
    df.loc[mask_physical, 'flow_action'] = 'flow_out_of_bounds'

    # 2. No dry weather flow check
    if flow_col in df.columns:
        if df[flow_col].max() < SignalRuleConfig.FLOW_NO_DWF:
            df['flow_no_dwf'] = 1
            df['flow_action'] = 'no_dry_weather_flow'

    return df


def detect_no_storm_response(df, depth_col='depth', rainfall_col='rainfall'):
    """
    Detects no storm response: if rainfall > config threshold but depth std < config threshold.
    Adds 'no_storm_response' column and updates action.
    """
    df = df.copy()
    df['no_storm_response'] = 0
    if rainfall_col in df.columns and depth_col in df.columns:
        if df[rainfall_col].max() > SignalRuleConfig.NO_STORM_RAIN_THRESH and df[depth_col].std() < SignalRuleConfig.NO_STORM_STD_THRESH:
            df['no_storm_response'] = 1
            if 'depth_action' in df.columns:
                df['depth_action'] = df['depth_action'].fillna('no_storm_response')
            else:
                df['depth_action'] = 'no_storm_response'
    return df


def detect_scatter_pattern(df, depth_col='depth', velocity_col='velocity'):
    """
    Detects scatter pattern: if IQR/mean is very high (non-physical scatter).
    Adds 'scatter_pattern' column and updates action.
    """
    df = df.copy()
    df['scatter_pattern'] = 0
    for col in [depth_col, velocity_col]:
        if col in df.columns and df[col].mean() != 0:
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            if iqr / abs(df[col].mean()) > SignalRuleConfig.SCATTER_IQR_THRESH:
                df['scatter_pattern'] = 1
                if f'{col}_action' in df.columns:
                    df[f'{col}_action'] = df[f'{col}_action'].fillna('scatter_pattern')
                else:
                    df[f'{col}_action'] = 'scatter_pattern'
    return df


def detect_sensor_drift(df, velocity_col='velocity'):
    """
    Detects sensor drift: rolling mean difference > config threshold.
    Adds 'sensor_drift' column and updates action.
    """
    df = df.copy()
    df['sensor_drift'] = 0
    if velocity_col in df.columns:
        roll_mean = df[velocity_col].rolling(window=SignalRuleConfig.DRIFT_WINDOW, min_periods=1).mean()
        drift = roll_mean.diff().abs().max()
        if drift > SignalRuleConfig.DRIFT_THRESH:
            df['sensor_drift'] = 1
            if 'velocity_action' in df.columns:
                df['velocity_action'] = df['velocity_action'].fillna('sensor_drift')
            else:
                df['velocity_action'] = 'sensor_drift'
    return df


def detect_step_change(df, depth_col='depth'):
    """
    Detects step change in depth: large diff between consecutive points.
    Adds 'step_change' column and updates action.
    """
    df = df.copy()
    df['step_change'] = 0
    if depth_col in df.columns:
        if df[depth_col].diff().abs().max() > SignalRuleConfig.STEP_CHANGE_THRESH:
            df['step_change'] = 1
            if 'depth_action' in df.columns:
                df['depth_action'] = df['depth_action'].fillna('step_change')
            else:
                df['depth_action'] = 'step_change'
    return df


def detect_no_dwf_scatter(df, flow_col='flow'):
    """
    Detects no DWF + wide scatter: flow always near zero, but std high.
    Adds 'no_dwf_scatter' column and updates action.
    """
    df = df.copy()
    df['no_dwf_scatter'] = 0
    if flow_col in df.columns:
        if df[flow_col].max() < SignalRuleConfig.FLOW_NO_DWF and df[flow_col].std() > SignalRuleConfig.NO_DWF_SCATTER_STD_THRESH:
            df['no_dwf_scatter'] = 1
            if 'flow_action' in df.columns:
                df['flow_action'] = df['flow_action'].fillna('no_dwf_scatter')
            else:
                df['flow_action'] = 'no_dwf_scatter'
    return df


def detect_steep_pipe(df, velocity_col='velocity', flow_col='flow'):
    """
    Detects steep pipe: high velocity, low flow, no response.
    Adds 'steep_pipe' column and updates action.
    """
    df = df.copy()
    df['steep_pipe'] = 0
    if velocity_col in df.columns and flow_col in df.columns:
        if df[velocity_col].mean() > SignalRuleConfig.STEEP_PIPE_VEL_THRESH and df[flow_col].mean() < SignalRuleConfig.STEEP_PIPE_FLOW_THRESH:
            df['steep_pipe'] = 1
            if 'velocity_action' in df.columns:
                df['velocity_action'] = df['velocity_action'].fillna('steep_pipe')
            else:
                df['velocity_action'] = 'steep_pipe'
    return df


def detect_volume_imbalance(df):
    """
    Placeholder for volume imbalance detection (requires multi-device data).
    Adds 'volume_imbalance' column and updates action.
    """
    df = df.copy()
    df['volume_imbalance'] = 0  # Not implemented
    return df


def detect_no_clear_profile(df, depth_col='depth', velocity_col='velocity', flow_col='flow'):
    """
    Detects no clear diurnal/profile: std of flow, depth, or velocity is very low.
    Adds 'no_profile' column and updates action.
    """
    df = df.copy()
    df['no_profile'] = 0
    for col in [depth_col, velocity_col, flow_col]:
        if col in df.columns:
            if df[col].std() < SignalRuleConfig.NO_PROFILE_STD_THRESH:
                df['no_profile'] = 1
                if f'{col}_action' in df.columns:
                    df[f'{col}_action'] = df[f'{col}_action'].fillna('no_profile')
                else:
                    df[f'{col}_action'] = 'no_profile'
    return df


def process_all_signals(df, depth_col='depth', velocity_col='velocity', flow_col='flow', rainfall_col='rainfall', d_noise=None, min_points=None):
    """
    Process all signal rules: depth, velocity, flow.
    Returns DataFrame with all rule tags and actions.
    """
    df_checked = check_depth_signal(df, depth_col=depth_col, d_noise=d_noise, min_points=min_points)
    df_checked = check_velocity_signal(df_checked, velocity_col=velocity_col)
    df_checked = check_flow_signal(df_checked, flow_col=flow_col)
    df_checked = detect_no_storm_response(df_checked, depth_col=depth_col, rainfall_col=rainfall_col)
    df_checked = detect_scatter_pattern(df_checked, depth_col=depth_col, velocity_col=velocity_col)
    df_checked = detect_sensor_drift(df_checked, velocity_col=velocity_col)
    df_checked = detect_step_change(df_checked, depth_col=depth_col)
    df_checked = detect_no_dwf_scatter(df_checked, flow_col=flow_col)
    df_checked = detect_steep_pipe(df_checked, velocity_col=velocity_col, flow_col=flow_col)
    df_checked = detect_volume_imbalance(df_checked)
    df_checked = detect_no_clear_profile(df_checked, depth_col=depth_col, velocity_col=velocity_col, flow_col=flow_col)
    return df_checked


def export_for_system(df, depth_col='depth', velocity_col='velocity', flow_col='flow', d_noise=None, min_points=None):
    """
    Export rule engine results for system use (database, frontend, etc).
    Returns DataFrame with original data, tags, and actions.
    """
    df_checked = process_all_signals(df, depth_col=depth_col, velocity_col=velocity_col, flow_col=flow_col, d_noise=d_noise, min_points=min_points)
    # Select columns for system (customize as needed)
    system_cols = [
        'time', depth_col, velocity_col, flow_col,
        'neg_depth', 'minor_noise', 'depth_physical_anomaly', 'depth_action',
        'velocity_physical_anomaly', 'velocity_flat', 'velocity_action',
        'flow_physical_anomaly', 'flow_no_dwf', 'flow_action'
    ]
    # Only keep columns that exist in df
    system_cols = [col for col in system_cols if col in df_checked.columns]
    return df_checked[system_cols]


def export_clean_view(df, depth_col='depth', velocity_col='velocity', flow_col='flow', d_noise=None, min_points=None):
    """
    Export cleaned data view for modeling/analysis.
    Removes bad data (neg_depth, physical anomalies, flat/zero velocity, no DWF), keeps only cleaned values.
    Returns DataFrame with cleaned data.
    """
    df_checked = process_all_signals(df, depth_col=depth_col, velocity_col=velocity_col, flow_col=flow_col, d_noise=d_noise, min_points=min_points)
    # Keep only rows without any physical anomaly or flagged bad data
    clean_mask = (
        (df_checked['neg_depth'] == 0) &
        (df_checked['depth_physical_anomaly'] == 0) &
        (df_checked['velocity_physical_anomaly'] == 0) &
        (df_checked['velocity_flat'] == 0) &
        (df_checked['flow_physical_anomaly'] == 0) &
        (df_checked['flow_no_dwf'] == 0)
    )
    df_clean = df_checked[clean_mask].copy()
    return df_clean 