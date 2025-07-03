# Configuration class for signal rule engine thresholds
# All threshold values and rule parameters are managed here for easy maintenance and extension.

class SignalRuleConfig:
    # Depth signal thresholds
    DEPTH_NOISE_THRESHOLD = 3  # mm, small negative values within [-3, 0) are considered minor noise
    DEPTH_BAD_SEGMENT_MIN_POINTS = 3  # minimum consecutive points to consider as a bad segment (for future use)
    DEPTH_MIN = 0  # mm, physical lower bound
    DEPTH_MAX = 5000  # mm, physical upper bound (adjust as needed)

    # Velocity signal thresholds
    VELOCITY_MIN = 0  # m/s, physical lower bound
    VELOCITY_MAX = 10  # m/s, physical upper bound
    VELOCITY_FLAT_STD = 0.01  # m/s, std below this means flat signal

    # Flow signal thresholds
    FLOW_MIN = 0  # L/s, physical lower bound
    FLOW_MAX = 300  # L/s, physical upper bound
    FLOW_NO_DWF = 0.01  # L/s, below this means no dry weather flow

    # No storm response rule
    NO_STORM_STD_THRESH = 0.005  # depth std below this means no response
    NO_STORM_RAIN_THRESH = 10    # rainfall above this triggers storm response check

    # Scatter pattern rule
    SCATTER_IQR_THRESH = 0.5     # IQR/mean above this means scatter pattern

    # Sensor drift rule
    DRIFT_WINDOW = 48            # rolling window size for drift detection
    DRIFT_THRESH = 0.2           # max rolling mean diff above this means drift

    # Step change rule
    STEP_CHANGE_THRESH = 0.1     # max diff between consecutive points

    # No DWF + scatter rule
    NO_DWF_SCATTER_STD_THRESH = 0.05  # std above this with low flow means scatter

    # Steep pipe rule
    STEEP_PIPE_VEL_THRESH = 1.0  # mean velocity above this
    STEEP_PIPE_FLOW_THRESH = 0.1 # mean flow below this

    # No clear profile rule
    NO_PROFILE_STD_THRESH = 0.01  # std below this means no clear diurnal/profile

    # Add more thresholds and parameters as needed for other rules 