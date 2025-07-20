# Configuration for signal-based fault detection in sewer monitoring
# All thresholds are calibrated based on 2-minute interval data (30 points per hour)

class SignalRuleConfig:
    # Maximum allowed count of negative velocity readings before flagging as faulty
    NEG_VELOCITY_MAX_COUNT = 70

    # Maximum allowed count of zero velocity readings before flagging
    ZERO_VELOCITY_MAX_COUNT = 5000

    # Thresholds for identifying flat-line (platform) behavior in data
    VELOCITY_CONST_STD_THRESH = 0.005  # m/s
    DEPTH_CONST_STD_THRESH = 0.5       # mm

    # If velocity stays at zero for more than this count, mark as fault
    ZERO_VELOCITY_DURATION = 60  # = 120 minutes

    # Platform behavior thresholds
    VELOCITY_PLATFORM_STD = 0.005
    VELOCITY_PLATFORM_DURATION = 60  # 120 minutes = 60 × 2min
    DEPTH_PLATFORM_STD = 0.5
    DEPTH_PLATFORM_DURATION = 60     # 120 minutes

    # Step jump detection: sudden change followed by a stable state
    STEP_JUMP_THRESH = 10.0  # mm or mm/s
    STEP_JUMP_STABLE_DURATION = 60  # = 120 minutes
    STEP_JUMP_STABLE_STD = 0.5

    # Extremely high velocity or depth detection
    HIGH_VELOCITY_THRESH = 2.5  # m/s
    HIGH_DEPTH_THRESH = 1200.0  # mm

    # Inconsistent data: zero depth with high velocity
    ZERO_DEPTH_HIGH_VELOCITY_THRESH = 0.3  # m/s

    # Rain gauge tipping bucket stuck detection (e.g., rainfall fixed at 5mm)
    RG_STUCK_RAINFALL_THRESH = 3.0  # mm


    
    # RG physical maximum value (overflow detection)
    RG_MAX_VALUE = 1000.0  # mm
    
    # Extreme rainfall detection thresholds
    RG_EXTREME_RAINFALL_THRESH = 50.0  # mm per 2-minute interval
    RG_EXTREME_RAINFALL_DURATION = 5    # = 6 minutes of extreme rainfall

    # Negative depth graded thresholds
    # >= CRITICAL_THRESHOLD: unusable, < HIGH_THRESHOLD: usable
    NEG_DEPTH_CRITICAL_THRESHOLD = 5000  # Critical level - unusable
    NEG_DEPTH_HIGH_THRESHOLD = 500      # High level - usable_with_warning

    # Maximum expected valid depth in mm
    DEPTH_MAX = 2000.0

    # Minimum expected valid depth in mm
    DEPTH_MIN = -20  # Minimum allowed depth (mm), set according to business/physical requirements

    
    # Standard deviation thresholds for considering data as 'unstable'
    VELOCITY_QUALITY_STD_THRESHOLD = 0.1  # m/s - threshold for velocity instability
    DEPTH_QUALITY_STD_THRESHOLD = 0.05    # m - threshold for depth instability
    
    # Back segment stability factor (back_std must be < threshold * this factor)
    BACK_SEGMENT_STABILITY_FACTOR = 0.5   # Back segment must be twice as stable
