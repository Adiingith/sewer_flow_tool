import pandas as pd
import numpy as np
from backend.core.signal_rule_config import SignalRuleConfig

def split_data_segments(data, front_ratio=0.2):
    """
    Split data into front and back segments with specified ratio.
    
    Args:
        data: pandas Series or DataFrame to split
        front_ratio: ratio for front segment (default 0.2 for 2:8 split)
    
    Returns:
        tuple: (front_segment, back_segment)
    """
    if len(data) < 10:  # Need minimum data points for segmentation
        return None, None
    
    split_point = int(len(data) * front_ratio)
    # Ensure at least some data in both segments
    split_point = max(1, min(split_point, len(data) - 1))
    
    front_segment = data.iloc[:split_point]
    back_segment = data.iloc[split_point:]
    
    return front_segment, back_segment

def check_negative_velocity(df, velocity_col='velocity'):
    if velocity_col not in df:
        return None, None, None
    # Handle None values
    velocity_data = df[velocity_col].dropna()
    
    # Split data into front and back segments (20%:80%)
    front_segment, back_segment = split_data_segments(velocity_data)
    if front_segment is None or back_segment is None:
        return None, None, None
    
    # Check front segment
    front_negative_count = (front_segment < 0).sum()
    # Check back segment  
    back_negative_count = (back_segment < 0).sum()
    
    # Determine status based on segment comparison
    if front_negative_count >= SignalRuleConfig.NEG_VELOCITY_MAX_COUNT and back_negative_count < SignalRuleConfig.NEG_VELOCITY_MAX_COUNT:
        print(f"[RULE_ENGINE] Negative velocity segment change: front={front_negative_count}, back={back_negative_count}")
        return 'usable_with_warning', 'Negative velocity issue resolved (front bad, back good)', 'wait_and_reassess'
    elif front_negative_count < SignalRuleConfig.NEG_VELOCITY_MAX_COUNT and back_negative_count >= SignalRuleConfig.NEG_VELOCITY_MAX_COUNT:
        print(f"[RULE_ENGINE] Negative velocity segment change: front={front_negative_count}, back={back_negative_count}")
        return 'unusable', 'Negative velocity issue developed (front good, back bad)', 'inspect_and_recalibrate'
    elif front_negative_count >= SignalRuleConfig.NEG_VELOCITY_MAX_COUNT and back_negative_count >= SignalRuleConfig.NEG_VELOCITY_MAX_COUNT:
        print(f"[RULE_ENGINE] Negative velocity check: front={front_negative_count}, back={back_negative_count} >= {SignalRuleConfig.NEG_VELOCITY_MAX_COUNT}")
        return 'unusable', 'Negative velocity count exceeds threshold', 'inspect_and_recalibrate'
    
    return None, None, None

def check_zero_velocity(df, velocity_col='velocity'):
    if velocity_col not in df:
        return None, None, None
    # Handle None values
    velocity_data = df[velocity_col].dropna()
    
    # Split data into front and back segments (20%:80%)
    front_segment, back_segment = split_data_segments(velocity_data)
    if front_segment is None or back_segment is None:
        return None, None, None
    
    # Check front segment
    front_zero_count = (front_segment == 0).sum()
    # Check back segment  
    back_zero_count = (back_segment == 0).sum()
    
    # Determine status based on segment comparison
    if front_zero_count >= SignalRuleConfig.ZERO_VELOCITY_MAX_COUNT and back_zero_count < SignalRuleConfig.ZERO_VELOCITY_MAX_COUNT:
        print(f"[RULE_ENGINE] Zero velocity segment change: front={front_zero_count}, back={back_zero_count}")
        return 'usable_with_warning', 'Zero velocity issue resolved (front bad, back good)', 'wait_and_reassess'
    elif front_zero_count < SignalRuleConfig.ZERO_VELOCITY_MAX_COUNT and back_zero_count >= SignalRuleConfig.ZERO_VELOCITY_MAX_COUNT:
        print(f"[RULE_ENGINE] Zero velocity segment change: front={front_zero_count}, back={back_zero_count}")
        return 'usable_with_warning', 'Zero velocity issue developed (front good, back bad)', 'wait_and_reassess'
    elif front_zero_count >= SignalRuleConfig.ZERO_VELOCITY_MAX_COUNT and back_zero_count >= SignalRuleConfig.ZERO_VELOCITY_MAX_COUNT:
        print(f"[RULE_ENGINE] Zero velocity check: front={front_zero_count}, back={back_zero_count} >= {SignalRuleConfig.ZERO_VELOCITY_MAX_COUNT}")
        return 'usable_with_warning', 'Zero velocity count exceeds threshold', 'wait_and_reassess'
    
    return None, None, None

def check_zero_flow(df, flow_col='flow'):
    if flow_col not in df:
        return None, None, None
    # Handle None values
    flow_data = df[flow_col].dropna()
    
    # Split data into front and back segments (20%:80%)
    front_segment, back_segment = split_data_segments(flow_data)
    if front_segment is None or back_segment is None:
        return None, None, None
    
    # Check if all values in each segment are zero
    front_all_zero = (front_segment == 0).all()
    back_all_zero = (back_segment == 0).all()
    
    
    # Determine status based on segment comparison
    if front_all_zero and not back_all_zero:
        print(f"[RULE_ENGINE] Zero flow segment change: front=all_zero, back=not_all_zero")
        return 'usable_with_warning', 'Zero flow issue resolved (front all zero, back not all zero)', 'wait_and_reassess'
    elif not front_all_zero and back_all_zero:
        print(f"[RULE_ENGINE] Zero flow segment change: front=not_all_zero, back=all_zero")
        return 'unusable', 'Zero flow issue developed (front not all zero, back all zero)', 'inspect_and_recalibrate'
    elif front_all_zero and back_all_zero:
        print(f"[RULE_ENGINE] Zero flow check: front=all_zero, back=all_zero")
        return 'unusable', 'All flow values are zero', 'inspect_and_recalibrate'
    
    return None, None, None

def check_data_constant(df, col, std_thresh):
    if col not in df:
        return None, None, None
    # Handle None values
    col_data = df[col].dropna()
    
    try:
        # ensure data is numeric and no NaN
        col_data = col_data.astype(float)
        if col_data.isna().any():
            col_data = col_data.dropna()
        
        # Split data into front and back segments (20%:80%)
        front_segment, back_segment = split_data_segments(col_data)
        if front_segment is None or back_segment is None:
            return None, None, None
        
        # Calculate std for each segment
        front_std = front_segment.std()
        back_std = back_segment.std()
        
        if pd.isna(front_std) or pd.isna(back_std):
            return None, None, None
        
        # Determine status based on segment comparison
        if front_std < std_thresh and back_std >= std_thresh:
            print(f"[RULE_ENGINE] Data constant segment change: {col} front_std={front_std:.6f}, back_std={back_std:.6f}")
            return 'usable_with_warning', f'{col} device started working (front constant, back active)', 'wait_and_reassess'
        elif front_std >= std_thresh and back_std < std_thresh:
            print(f"[RULE_ENGINE] Data constant segment change: {col} front_std={front_std:.6f}, back_std={back_std:.6f}")
            return 'unusable', f'{col} device stopped working (front active, back constant)', 'inspect_and_reboot'
        elif front_std < std_thresh and back_std < std_thresh:
            print(f"[RULE_ENGINE] Data constant check: {col} front_std={front_std:.6f}, back_std={back_std:.6f} < {std_thresh}")
            return 'unusable', f'{col} constant (std below threshold)', 'inspect_and_reboot'
    except Exception as e:
        return None, None, None
    return None, None, None

def check_zero_velocity_duration(df, velocity_col='velocity'):
    if velocity_col not in df:
        return None, None, None
    # Handle None values
    velocity_data = df[velocity_col].dropna()
    
    # Split data into front and back segments (20%:80%)
    front_segment, back_segment = split_data_segments(velocity_data)
    if front_segment is None or back_segment is None:
        return None, None, None
    
    # Calculate max zero duration for each segment
    def get_max_zero_duration(data):
        zero_seq = (data == 0).astype(int)
        max_zero = 0
        cnt = 0
        for v in zero_seq:
            if v == 1:
                cnt += 1
                max_zero = max(max_zero, cnt)
            else:
                cnt = 0
        return max_zero
    
    front_max_zero = get_max_zero_duration(front_segment)
    back_max_zero = get_max_zero_duration(back_segment)
    
    # Determine status based on segment comparison
    if front_max_zero >= SignalRuleConfig.ZERO_VELOCITY_DURATION and back_max_zero < SignalRuleConfig.ZERO_VELOCITY_DURATION:
        print(f"[RULE_ENGINE] Zero velocity duration segment change: front={front_max_zero}, back={back_max_zero}")
        return 'usable_with_warning', 'Zero velocity duration issue resolved (front bad, back good)', 'wait_and_reassess'
    elif front_max_zero < SignalRuleConfig.ZERO_VELOCITY_DURATION and back_max_zero >= SignalRuleConfig.ZERO_VELOCITY_DURATION:
        print(f"[RULE_ENGINE] Zero velocity duration segment change: front={front_max_zero}, back={back_max_zero}")
        return 'usable_with_warning', 'Zero velocity duration issue developed (front good, back bad)', 'wait_and_reassess'
    elif front_max_zero >= SignalRuleConfig.ZERO_VELOCITY_DURATION and back_max_zero >= SignalRuleConfig.ZERO_VELOCITY_DURATION:
        print(f"[RULE_ENGINE] Zero velocity duration: front={front_max_zero}, back={back_max_zero} >= {SignalRuleConfig.ZERO_VELOCITY_DURATION}")
        return 'usable_with_warning', 'Zero velocity duration exceeds threshold', 'wait_and_reassess'
    
    return None, None, None

def check_platform_signal(df, col, platform_std, platform_duration):
    if col not in df:
        return None, None, None
    # Handle None values
    col_data = df[col].dropna()
    
    try:
        # ensure data is numeric and no NaN
        col_data = col_data.astype(float)
        if col_data.isna().any():
            col_data = col_data.dropna()
        
        # Split data into front and back segments (20%:80%)
        front_segment, back_segment = split_data_segments(col_data)
        if front_segment is None or back_segment is None:
            return None, None, None
        
        # Calculate platform count for each segment
        def get_platform_count(data):
            if len(data) < platform_duration:
                return 0
            rolling_std = data.rolling(window=platform_duration, min_periods=1).std()
            rolling_std = rolling_std.dropna()
            if len(rolling_std) == 0:
                return 0
            return (rolling_std < platform_std).sum()
        
        front_platform_count = get_platform_count(front_segment)
        back_platform_count = get_platform_count(back_segment)
        
        # Determine status based on segment comparison
        if front_platform_count >= platform_duration and back_platform_count < platform_duration:
            print(f"[RULE_ENGINE] Platform signal segment change: {col} front={front_platform_count}, back={back_platform_count}")
            return 'usable_with_warning', f'{col} platform signal resolved (front bad, back good)', 'wait_and_reassess'
        elif front_platform_count < platform_duration and back_platform_count >= platform_duration:
            print(f"[RULE_ENGINE] Platform signal segment change: {col} front={front_platform_count}, back={back_platform_count}")
            return 'usable_with_warning', f'{col} platform signal developed (front good, back bad)', 'wait_and_reassess'
        elif front_platform_count >= platform_duration and back_platform_count >= platform_duration:
            print(f"[RULE_ENGINE] Platform signal: {col} front={front_platform_count}, back={back_platform_count} >= {platform_duration}")
            return 'usable_with_warning', f'{col} platform signal detected', 'wait_and_reassess'
    except Exception as e:
        return None, None, None
    return None, None, None

def check_step_jump_stable(df, depth_col='depth'):
    if depth_col not in df:
        return None, None, None
    # Handle None values
    depth_data = df[depth_col].dropna()
    
    try:
        # ensure data is numeric and no NaN
        depth_data = depth_data.astype(float)
        if depth_data.isna().any():
            depth_data = depth_data.dropna()
        
        # Split data into front and back segments (20%:80%)
        front_segment, back_segment = split_data_segments(depth_data)
        if front_segment is None or back_segment is None:
            return None, None, None
        
        # Check step jump for each segment
        def check_segment_step_jump(data):
            if len(data) < 2:
                return False
            diff = data.diff()
            diff = diff.dropna()
            if len(diff) == 0:
                return False
            diff_abs = diff.abs()
            jump_idx = diff_abs[diff_abs > SignalRuleConfig.STEP_JUMP_THRESH].index
            if not jump_idx.empty:
                for idx in jump_idx:
                    after = data.iloc[idx:idx+SignalRuleConfig.STEP_JUMP_STABLE_DURATION]
                    if len(after) >= 2 and after.std() < SignalRuleConfig.STEP_JUMP_STABLE_STD:
                        return True
            return False
        
        front_has_step_jump = check_segment_step_jump(front_segment)
        back_has_step_jump = check_segment_step_jump(back_segment)
        
        # Determine status based on segment comparison
        if front_has_step_jump and not back_has_step_jump:
            print(f"[RULE_ENGINE] Step jump stable segment change: front=True, back=False")
            return 'usable_with_warning', 'Step jump stable resolved (front bad, back good)', 'wait_and_reassess'
        elif not front_has_step_jump and back_has_step_jump:
            print(f"[RULE_ENGINE] Step jump stable segment change: front=False, back=True")
            return 'usable_with_warning', 'Step jump stable developed (front good, back bad)', 'wait_and_reassess'
        elif front_has_step_jump and back_has_step_jump:
            print(f"[RULE_ENGINE] Step jump stable: front=True, back=True")
            return 'usable_with_warning', 'Step jump stable detected', 'wait_and_reassess'
    except Exception as e:
        return None, None, None
    return None, None, None

def check_high_velocity_high_depth(df, velocity_col='velocity', depth_col='depth'):
    if velocity_col not in df or depth_col not in df:
        return None, None, None
    # Handle None values
    velocity_data = df[velocity_col].dropna()
    depth_data = df[depth_col].dropna()
    
    # Align the data by index
    common_index = velocity_data.index.intersection(depth_data.index)
    if len(common_index) < 10:
        return None, None, None
    aligned_velocity = velocity_data.loc[common_index]
    aligned_depth = depth_data.loc[common_index]
    
    # Split data into front and back segments (20%:80%)
    front_velocity, back_velocity = split_data_segments(aligned_velocity)
    front_depth, back_depth = split_data_segments(aligned_depth)
    
    if front_velocity is None or back_velocity is None or front_depth is None or back_depth is None:
        return None, None, None
    
    # Check each segment
    front_mask = (front_velocity > SignalRuleConfig.HIGH_VELOCITY_THRESH) & (front_depth > SignalRuleConfig.HIGH_DEPTH_THRESH)
    back_mask = (back_velocity > SignalRuleConfig.HIGH_VELOCITY_THRESH) & (back_depth > SignalRuleConfig.HIGH_DEPTH_THRESH)
    
    front_count = front_mask.sum()
    back_count = back_mask.sum()
    
    # Determine status based on segment comparison
    if front_count > 0 and back_count == 0:
        print(f"[RULE_ENGINE] High velocity & depth segment change: front={front_count}, back={back_count}")
        return 'usable_with_warning', 'High velocity and depth issue resolved (front bad, back good)', 'wait_and_reassess'
    elif front_count == 0 and back_count > 0:
        print(f"[RULE_ENGINE] High velocity & depth segment change: front={front_count}, back={back_count}")
        return 'usable_with_warning', 'High velocity and depth issue developed (front good, back bad)', 'wait_and_reassess'
    elif front_count > 0 and back_count > 0:
        print(f"[RULE_ENGINE] High velocity & depth: front={front_count}, back={back_count} points exceed thresholds")
        return 'usable_with_warning', 'High velocity and high depth detected', 'wait_and_reassess'
    
    return None, None, None

def check_zero_depth_high_velocity(df, velocity_col='velocity', depth_col='depth'):
    if velocity_col not in df or depth_col not in df:
        return None, None, None
    # Handle None values
    velocity_data = df[velocity_col].dropna()
    depth_data = df[depth_col].dropna()
    
    # Align the data by index
    common_index = velocity_data.index.intersection(depth_data.index)
    if len(common_index) < 10:
        return None, None, None
    aligned_velocity = velocity_data.loc[common_index]
    aligned_depth = depth_data.loc[common_index]
    
    # Split data into front and back segments (20%:80%)
    front_velocity, back_velocity = split_data_segments(aligned_velocity)
    front_depth, back_depth = split_data_segments(aligned_depth)
    
    if front_velocity is None or back_velocity is None or front_depth is None or back_depth is None:
        return None, None, None
    
    # Check each segment
    front_mask = (front_depth == 0) & (front_velocity > SignalRuleConfig.ZERO_DEPTH_HIGH_VELOCITY_THRESH)
    back_mask = (back_depth == 0) & (back_velocity > SignalRuleConfig.ZERO_DEPTH_HIGH_VELOCITY_THRESH)
    
    front_count = front_mask.sum()
    back_count = back_mask.sum()
    
    # Determine status based on segment comparison
    if front_count > 0 and back_count == 0:
        print(f"[RULE_ENGINE] Zero depth high velocity segment change: front={front_count}, back={back_count}")
        return 'usable_with_warning', 'Zero depth high velocity issue resolved (front bad, back good)', 'wait_and_reassess'
    elif front_count == 0 and back_count > 0:
        print(f"[RULE_ENGINE] Zero depth high velocity segment change: front={front_count}, back={back_count}")
        return 'usable_with_warning', 'Zero depth high velocity issue developed (front good, back bad)', 'wait_and_reassess'
    elif front_count > 0 and back_count > 0:
        print(f"[RULE_ENGINE] Zero depth high velocity: front={front_count}, back={back_count} points detected")
        return 'usable_with_warning', 'Zero depth with high velocity detected', 'inspect_and_recalibrate'
    
    return None, None, None

def check_rg_stuck(df, rainfall_col='rainfall', rg_col='rg'):
    if rainfall_col not in df or rg_col not in df:
        return None, None, None
    # Handle None values
    rainfall_data = df[rainfall_col].dropna()
    rg_data = df[rg_col].dropna()
    
    # Split data into front and back segments (20%:80%)
    front_rainfall, back_rainfall = split_data_segments(rainfall_data)
    front_rg, back_rg = split_data_segments(rg_data)
    
    if front_rainfall is None or back_rainfall is None or front_rg is None or back_rg is None:
        return None, None, None
    
    # Check each segment
    def check_segment_stuck(rainfall_seg, rg_seg):
        max_rainfall = rainfall_seg.max()
        rg_unique_count = rg_seg.nunique()
        return max_rainfall > SignalRuleConfig.RG_STUCK_RAINFALL_THRESH and rg_unique_count == 1
    
    front_stuck = check_segment_stuck(front_rainfall, front_rg)
    back_stuck = check_segment_stuck(back_rainfall, back_rg)
    
    print(f"RG stuck check: front_stuck={front_stuck}, back_stuck={back_stuck}")
    
    # Determine status based on segment comparison
    if front_stuck and not back_stuck:
        print(f"[RULE_ENGINE] RG stuck segment change: front=True, back=False")
        return 'usable_with_warning', 'RG stuck issue resolved (front bad, back good)', 'wait_and_reassess'
    elif not front_stuck and back_stuck:
        print(f"[RULE_ENGINE] RG stuck segment change: front=False, back=True")
        return 'unusable', 'RG stuck issue developed (front good, back bad)', 'inspect_and_recalibrate'
    elif front_stuck and back_stuck:
        print(f"[RULE_ENGINE] RG stuck: front=True, back=True")
        return 'unusable', 'RG tipping bucket stuck during rainfall', 'inspect_and_recalibrate'
    
    return None, None, None



def check_rg_overflow(df, rg_col='rg'):
    """Check if RG values exceed physical maximum range"""
    if rg_col not in df:
        return None, None, None
    # Handle None values
    rg_data = df[rg_col].dropna()
    
    # Split data into front and back segments (20%:80%)
    front_rg, back_rg = split_data_segments(rg_data)
    if front_rg is None or back_rg is None:
        return None, None, None
    
    # Check each segment
    front_overflow_count = (front_rg > SignalRuleConfig.RG_MAX_VALUE).sum()
    back_overflow_count = (back_rg > SignalRuleConfig.RG_MAX_VALUE).sum()
    
    print(f"RG overflow check: front_overflow={front_overflow_count}, back_overflow={back_overflow_count}, max_value={SignalRuleConfig.RG_MAX_VALUE}")
    
    # Determine status based on segment comparison
    if front_overflow_count > 0 and back_overflow_count == 0:
        print(f"[RULE_ENGINE] RG overflow segment change: front={front_overflow_count}, back={back_overflow_count}")
        return 'usable_with_warning', 'RG overflow issue resolved (front bad, back good)', 'wait_and_reassess'
    elif front_overflow_count == 0 and back_overflow_count > 0:
        print(f"[RULE_ENGINE] RG overflow segment change: front={front_overflow_count}, back={back_overflow_count}")
        return 'unusable', 'RG overflow issue developed (front good, back bad)', 'inspect_and_recalibrate'
    elif front_overflow_count > 0 and back_overflow_count > 0:
        print(f"[RULE_ENGINE] RG overflow: front={front_overflow_count}, back={back_overflow_count} values exceed {SignalRuleConfig.RG_MAX_VALUE}")
        return 'unusable', f'RG values exceed maximum range: {front_overflow_count + back_overflow_count} readings', 'inspect_and_recalibrate'
    
    return None, None, None

def check_rg_extreme_rainfall(df, rainfall_col='rainfall', rg_col='rg'):
    """Check if continuous rainfall exceeds physical limits"""
    if rainfall_col not in df or rg_col not in df:
        return None, None, None
    # Handle None values
    rainfall_data = df[rainfall_col].dropna()
    rg_data = df[rg_col].dropna()
    
    # Split data into front and back segments (20%:80%)
    front_rainfall, back_rainfall = split_data_segments(rainfall_data)
    if front_rainfall is None or back_rainfall is None:
        return None, None, None
    
    # Check each segment for extreme rainfall
    def check_segment_extreme_rainfall(data):
        continuous_extreme = 0
        max_continuous = 0
        for rain in data:
            if rain > SignalRuleConfig.RG_EXTREME_RAINFALL_THRESH:
                continuous_extreme += 1
                max_continuous = max(max_continuous, continuous_extreme)
            else:
                continuous_extreme = 0
        return max_continuous >= SignalRuleConfig.RG_EXTREME_RAINFALL_DURATION
    
    front_extreme = check_segment_extreme_rainfall(front_rainfall)
    back_extreme = check_segment_extreme_rainfall(back_rainfall)
    
    print(f"RG extreme rainfall check: front_extreme={front_extreme}, back_extreme={back_extreme}, threshold={SignalRuleConfig.RG_EXTREME_RAINFALL_THRESH}")
    
    # Determine status based on segment comparison
    if front_extreme and not back_extreme:
        print(f"[RULE_ENGINE] RG extreme rainfall segment change: front=True, back=False")
        return 'usable_with_warning', 'RG extreme rainfall issue resolved (front bad, back good)', 'wait_and_reassess'
    elif not front_extreme and back_extreme:
        print(f"[RULE_ENGINE] RG extreme rainfall segment change: front=False, back=True")
        return 'unusable', 'RG extreme rainfall issue developed (front good, back bad)', 'inspect_and_recalibrate'
    elif front_extreme and back_extreme:
        print(f"[RULE_ENGINE] RG extreme rainfall: front=True, back=True")
        return 'unusable', f'Extreme rainfall pattern detected in both segments', 'inspect_and_recalibrate'
    
    return None, None, None

def check_negative_depth(df, depth_col='depth'):
    if depth_col not in df:
        return None, None, None
    # Handle None values
    depth_data = df[depth_col].dropna()
    if len(depth_data) < 10:  # Need minimum data points for segmentation
        return None, None, None
    
    # Split data into front and back segments (20%:80%)
    front_segment, back_segment = split_data_segments(depth_data)
    
    if front_segment is None or back_segment is None:
        return None, None, None
    
    # Check each segment
    front_negative_count = (front_segment < 0).sum()
    back_negative_count = (back_segment < 0).sum()
    
    # Determine status based on segment comparison
    # First classify each segment
    def classify_segment(count):
        if count >= SignalRuleConfig.NEG_DEPTH_CRITICAL_THRESHOLD:
            return 'critical'
        elif count >= SignalRuleConfig.NEG_DEPTH_HIGH_THRESHOLD:
            return 'high'
        else:
            return 'good'
    
    front_level = classify_segment(front_negative_count)
    back_level = classify_segment(back_negative_count)
    
    print(f"[RULE_ENGINE] Negative depth analysis: front={front_negative_count}({front_level}), back={back_negative_count}({back_level})")
    
    # Determine final status based on back segment (latest data)
    if back_level == 'critical':
        final_status = 'unusable'
        final_action = 'inspect_and_recalibrate'
    elif back_level == 'high':
        final_status = 'usable_with_warning'
        final_action = 'wait_and_reassess'
    else:  # back_level == 'good'
        # If back segment is good, check if there's any issue at all
        if front_level == 'good':
            return None, None, None  # Both segments are good, no issue
        else:
            final_status = 'usable_with_warning'
            final_action = 'wait_and_reassess'
    
    # Generate descriptive message based on trend
    if front_level == back_level:
        # No change between segments
        if back_level == 'critical':
            message = f'Critical negative depth count: {front_negative_count + back_negative_count}'
        elif back_level == 'high':
            message = f'High negative depth count: {front_negative_count + back_negative_count}'
        else:  # both good, but we shouldn't reach here due to early return above
            message = f'Negative depth issue resolved: {front_negative_count + back_negative_count}'
    else:
        # Change between segments
        if front_level == 'critical' and back_level in ['high', 'good']:
            message = 'Critical negative depth issue resolved (front bad, back better)'
        elif front_level == 'high' and back_level == 'good':
            message = 'High negative depth issue resolved (front bad, back good)'
        elif front_level == 'good' and back_level == 'high':
            message = 'High negative depth issue developed (front good, back bad)'
        elif front_level == 'good' and back_level == 'critical':
            message = 'Critical negative depth issue developed (front good, back bad)'
        elif front_level == 'high' and back_level == 'critical':
            message = 'Critical negative depth issue worsened (front high, back critical)'
        else:  # front_level == 'critical' and back_level == 'high'
            message = 'Critical negative depth issue improved (front critical, back high)'
    
    return final_status, message, final_action



def check_physical_anomaly(df, depth_col='depth'):
    if depth_col not in df:
        return None, None, None
    # Handle None values
    depth_data = df[depth_col].dropna()
    if len(depth_data) < 10:  # Need minimum data points for segmentation
        return None, None, None
    
    # Split data into front and back segments (20%:80%)
    front_segment, back_segment = split_data_segments(depth_data)
    
    if front_segment is None or back_segment is None:
        return None, None, None
    
    # Check each segment for physical anomalies
    front_above_max = (front_segment > SignalRuleConfig.DEPTH_MAX).sum()
    front_below_min = (front_segment < SignalRuleConfig.DEPTH_MIN).sum()
    back_above_max = (back_segment > SignalRuleConfig.DEPTH_MAX).sum()
    back_below_min = (back_segment < SignalRuleConfig.DEPTH_MIN).sum()
    
    front_anomaly = front_above_max > 0 or front_below_min > 0
    back_anomaly = back_above_max > 0 or back_below_min > 0
    
    # Determine status based on segment comparison
    if front_anomaly and not back_anomaly:
        print(f"[RULE_ENGINE] Physical anomaly segment change: front=True (above_max={front_above_max}, below_min={front_below_min}), back=False")
        return 'usable_with_warning', 'Physical anomaly issue resolved (front bad, back good)', 'wait_and_reassess'
    elif not front_anomaly and back_anomaly:
        print(f"[RULE_ENGINE] Physical anomaly segment change: front=False, back=True (above_max={back_above_max}, below_min={back_below_min})")
        return 'usable_with_warning', 'Physical anomaly issue developed (front good, back bad)', 'wait_and_reassess'
    elif front_anomaly and back_anomaly:
        print(f"[RULE_ENGINE] Physical anomaly: front=True, back=True (above_max={front_above_max + back_above_max}, below_min={front_below_min + back_below_min})")
        return 'unusable', 'Depth physical anomaly detected', 'inspect_and_reposition'
    
    return None, None, None

def process_all_rules(df, velocity_col='velocity', depth_col='depth', rainfall_col='rainfall', rg_col='rg', monitor_name=None):
    """
    Unified rule engine, output device_status/device_status_reason/device_action, priority unusable>usable_with_warning>usable
    For DM devices (monitor_name starts with 'DM'): depth issues -> unusable, velocity issues -> usable_with_warning
    """
    try:
        # Log start of rule processing
        print(f"[RULE_ENGINE] Starting rule processing for {monitor_name}")
        print(f"[RULE_ENGINE] Data shape: {df.shape}, columns: {list(df.columns)}")
        
        # Check if this is a RG device
        is_rg_device = monitor_name and monitor_name.upper().startswith('RG')
        # Check if this is a DM device
        is_dm_device = monitor_name and monitor_name.upper().startswith('DM')
        
        rule_results = []
        
        # For RG devices, only perform RG-related checks
        if is_rg_device:
            print(f"[RULE_ENGINE] RG device {monitor_name}: performing RG-only checks")
            # RG checks only
            rule_results.append(check_rg_stuck(df, rainfall_col, rg_col))
            rule_results.append(check_rg_overflow(df, rg_col))
            rule_results.append(check_rg_extreme_rainfall(df, rainfall_col, rg_col))
            
        # For DM devices, velocity checks should only result in warnings
        elif is_dm_device:
            # Velocity checks - force to usable_with_warning for DM devices
            neg_vel_result = check_negative_velocity(df, velocity_col)
            if neg_vel_result[0] == 'unusable':
                print(f"DM device {monitor_name}: velocity negative -> downgraded to warning")
                rule_results.append(('usable_with_warning', neg_vel_result[1], 'wait_and_reassess'))
            else:
                rule_results.append(neg_vel_result)
                
            zero_vel_result = check_zero_velocity(df, velocity_col)
            if zero_vel_result[0] == 'unusable':
                print(f"DM device {monitor_name}: velocity zero -> downgraded to warning")
                rule_results.append(('usable_with_warning', zero_vel_result[1], 'wait_and_reassess'))
            else:
                rule_results.append(zero_vel_result)
                
            vel_const_result = check_data_constant(df, velocity_col, SignalRuleConfig.VELOCITY_CONST_STD_THRESH)
            if vel_const_result[0] == 'unusable':
                print(f"DM device {monitor_name}: velocity constant -> downgraded to warning")
                rule_results.append(('usable_with_warning', vel_const_result[1], 'wait_and_reassess'))
            else:
                rule_results.append(vel_const_result)
                
            zero_vel_dur_result = check_zero_velocity_duration(df, velocity_col)
            rule_results.append(zero_vel_dur_result)  # Already usable_with_warning
            
            vel_platform_result = check_platform_signal(df, velocity_col, SignalRuleConfig.VELOCITY_PLATFORM_STD, SignalRuleConfig.VELOCITY_PLATFORM_DURATION)
            rule_results.append(vel_platform_result)  # Already usable_with_warning
            
            # Depth checks - keep original severity for DM devices (unusable if issues)
            depth_const_result = check_data_constant(df, depth_col, SignalRuleConfig.DEPTH_CONST_STD_THRESH)
            if depth_const_result[0] == 'unusable':
                print(f"DM device {monitor_name}: depth constant -> unusable (strict)")
            rule_results.append(depth_const_result)
            
            depth_platform_result = check_platform_signal(df, depth_col, SignalRuleConfig.DEPTH_PLATFORM_STD, SignalRuleConfig.DEPTH_PLATFORM_DURATION)
            if depth_platform_result[0] == 'usable_with_warning':
                print(f"DM device {monitor_name}: depth platform -> warning")
            rule_results.append(depth_platform_result)
            
            step_jump_result = check_step_jump_stable(df, depth_col)
            if step_jump_result[0] == 'usable_with_warning':
                print(f"DM device {monitor_name}: step jump -> warning")
            rule_results.append(step_jump_result)
            
            neg_depth_result = check_negative_depth(df, depth_col)
            if neg_depth_result[0] == 'unusable':
                print(f"DM device {monitor_name}: negative depth -> unusable (strict)")
            elif neg_depth_result[0] == 'usable_with_warning':
                print(f"DM device {monitor_name}: negative depth -> warning ({neg_depth_result[1]})")
            rule_results.append(neg_depth_result)
            
            phys_anomaly_result = check_physical_anomaly(df, depth_col)
            if phys_anomaly_result[0] == 'unusable':
                print(f"DM device {monitor_name}: physical anomaly -> unusable (strict)")
            rule_results.append(phys_anomaly_result)
            
            # Combined checks - keep original logic
            rule_results.append(check_high_velocity_high_depth(df, velocity_col, depth_col))
            rule_results.append(check_zero_depth_high_velocity(df, velocity_col, depth_col))
            

            
        else:
            # For non-DM devices, categorize checks into 3 classes
            # Velocity class checks
            velocity_results = []
            velocity_results.append(check_negative_velocity(df, velocity_col))
            velocity_results.append(check_zero_velocity(df, velocity_col))
            velocity_results.append(check_data_constant(df, velocity_col, SignalRuleConfig.VELOCITY_CONST_STD_THRESH))
            velocity_results.append(check_zero_velocity_duration(df, velocity_col))
            velocity_results.append(check_platform_signal(df, velocity_col, SignalRuleConfig.VELOCITY_PLATFORM_STD, SignalRuleConfig.VELOCITY_PLATFORM_DURATION))
            
            # Depth class checks
            depth_results = []
            depth_results.append(check_data_constant(df, depth_col, SignalRuleConfig.DEPTH_CONST_STD_THRESH))
            depth_results.append(check_platform_signal(df, depth_col, SignalRuleConfig.DEPTH_PLATFORM_STD, SignalRuleConfig.DEPTH_PLATFORM_DURATION))
            depth_results.append(check_step_jump_stable(df, depth_col))
            depth_results.append(check_negative_depth(df, depth_col))
            depth_results.append(check_physical_anomaly(df, depth_col))
            
            # Flow class checks
            flow_results = []
            flow_results.append(check_zero_flow(df, 'flow'))
            
            # Combined checks (affect both velocity and depth)
            combined_results = []
            combined_results.append(check_high_velocity_high_depth(df, velocity_col, depth_col))
            combined_results.append(check_zero_depth_high_velocity(df, velocity_col, depth_col))
            
            
            # Determine class statuses
            velocity_has_unusable = any(result[0] == 'unusable' for result in velocity_results if result[0] is not None)
            depth_has_unusable = any(result[0] == 'unusable' for result in depth_results if result[0] is not None)
            flow_has_unusable = any(result[0] == 'unusable' for result in flow_results if result[0] is not None)
            
            print(f"[RULE_ENGINE] Class status for {monitor_name}: velocity_unusable={velocity_has_unusable}, depth_unusable={depth_has_unusable}, flow_unusable={flow_has_unusable}")
            
            # Collect all results for final processing
            rule_results = []
            rule_results.extend(velocity_results)
            rule_results.extend(depth_results)
            rule_results.extend(flow_results)
            rule_results.extend(combined_results)
            
            
            # Apply class-based logic: unusable if 2 or more classes are unusable
            unusable_class_count = sum([velocity_has_unusable, depth_has_unusable, flow_has_unusable])
            
            if unusable_class_count >= 2:
                print(f"[RULE_ENGINE] {unusable_class_count}/3 classes unusable for {monitor_name} - keeping unusable status")
                # Keep original logic for final status determination
            else:
                print(f"[RULE_ENGINE] Only {unusable_class_count}/3 classes unusable for {monitor_name} - downgrading unusable to warning")
                # Downgrade any unusable results to warning
                for i, result in enumerate(rule_results):
                    if result is not None and len(result) == 3 and result[0] == 'unusable':
                        rule_results[i] = ('usable_with_warning', result[1], 'wait_and_reassess')
        
        # Priority processing
        final_status = 'usable'
        final_action = ''
        reasons_unusable = []
        reasons_warning = []
        actions_unusable = []
        actions_warning = []
        
        for result in rule_results:
            if result is None or len(result) != 3:
                continue  # Skip invalid results
            status, reason, action = result
            if status == 'unusable' and reason:
                reasons_unusable.append(reason)
                actions_unusable.append(action)
            elif status == 'usable_with_warning' and reason:
                reasons_warning.append(reason)
                actions_warning.append(action)
        
        if reasons_unusable:
            final_status = 'unusable'
            final_reason = '; '.join(reasons_unusable)
            final_action = actions_unusable[0] if actions_unusable else ''
            if is_rg_device:
                print(f"RG device {monitor_name}: final status = unusable - {final_reason}")
            elif is_dm_device:
                print(f"DM device {monitor_name}: final status = unusable - {final_reason}")
        elif reasons_warning:
            final_status = 'usable_with_warning'
            final_reason = '; '.join(reasons_warning)
            final_action = actions_warning[0] if actions_warning else ''
            if is_rg_device:
                print(f"RG device {monitor_name}: final status = warning - {final_reason}")
            elif is_dm_device:
                print(f"DM device {monitor_name}: final status = warning - {final_reason}")
        else:
            final_status = 'usable'
            final_reason = ''
            final_action = ''
            if is_rg_device:
                print(f"RG device {monitor_name}: final status = usable")
            elif is_dm_device:
                print(f"DM device {monitor_name}: final status = usable")
        
        result = df.copy()
        result['device_status'] = final_status
        result['device_status_reason'] = final_reason
        result['device_action'] = final_action
        return result
    except Exception as e:
        print(f"[RULE_ENGINE] ERROR processing rules for {monitor_name}: {str(e)}")
        result = df.copy()
        result['device_status'] = 'system_error'  
        result['device_status_reason'] = f'Rule engine error: {str(e)}'
        result['device_action'] = 'seek_it_support' 
        return result 