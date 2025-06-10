import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import random


def generate_temperature_data(n_samples, start_time, sampling_rate_seconds=1, base_temp=25.0):
    """
    Generate realistic temperature sensor data with daily patterns and random noise
    
    Args:
        n_samples: Number of data points to generate
        start_time: Starting timestamp
        sampling_rate_seconds: Time between samples in seconds
        base_temp: Base temperature in Celsius
        
    Returns:
        DataFrame with timestamps and temperature values
    """
    # Create time index
    timestamps = [start_time + timedelta(seconds=i*sampling_rate_seconds) for i in range(n_samples)]
    
    # Generate daily pattern (24-hour cycle)
    time_of_day = np.array([(t.hour * 3600 + t.minute * 60 + t.second) / 86400 for t in timestamps])
    daily_pattern = 3 * np.sin(2 * np.pi * time_of_day)
    
    # Add random noise
    noise = np.random.normal(0, 0.5, n_samples)
    
    # Add slow random walk to simulate environmental changes
    random_walk = np.cumsum(np.random.normal(0, 0.01, n_samples))
    
    # Combine components
    temperature = base_temp + daily_pattern + noise + random_walk
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature
    })


def generate_motion_data(n_samples, start_time, sampling_rate_seconds=1, activity_periods=None):
    """
    Generate motion sensor data with periods of activity
    
    Args:
        n_samples: Number of data points to generate
        start_time: Starting timestamp
        sampling_rate_seconds: Time between samples in seconds
        activity_periods: List of (start_hour, end_hour) tuples for activity periods
        
    Returns:
        DataFrame with timestamps and motion values (0 or 1)
    """
    # Create time index
    timestamps = [start_time + timedelta(seconds=i*sampling_rate_seconds) for i in range(n_samples)]
    
    # Default activity periods if none provided (morning and evening activity)
    if activity_periods is None:
        activity_periods = [(7, 9), (12, 13), (17, 22)]
    
    # Initialize motion data with zeros
    motion = np.zeros(n_samples)
    
    # Set motion to 1 during activity periods with some randomness
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        
        # Check if current hour is in any activity period
        is_active_period = any(start <= hour < end for start, end in activity_periods)
        
        if is_active_period:
            # Higher probability of motion during active periods
            motion[i] = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            # Lower probability of motion during inactive periods
            motion[i] = np.random.choice([0, 1], p=[0.95, 0.05])
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'motion': motion.astype(int)
    })


def generate_pulse_data(n_samples, start_time, sampling_rate_seconds=1, base_pulse=70):
    """
    Generate pulse sensor data with activity correlation
    
    Args:
        n_samples: Number of data points to generate
        start_time: Starting timestamp
        sampling_rate_seconds: Time between samples in seconds
        base_pulse: Base pulse rate in BPM
        
    Returns:
        DataFrame with timestamps and pulse values
    """
    # Create time index
    timestamps = [start_time + timedelta(seconds=i*sampling_rate_seconds) for i in range(n_samples)]
    
    # Generate daily pattern (higher during day, lower at night)
    time_of_day = np.array([(t.hour * 3600 + t.minute * 60 + t.second) / 86400 for t in timestamps])
    
    # Create a pattern that's higher during waking hours (6am-10pm)
    awake_pattern = np.zeros(n_samples)
    for i, t in enumerate(timestamps):
        if 6 <= t.hour < 22:  # Awake hours
            awake_pattern[i] = 10  # Higher pulse when awake
    
    # Add activity spikes
    activity_spikes = np.zeros(n_samples)
    activity_periods = [(7, 8), (12, 13), (17, 19)]  # Exercise/activity times
    
    for i, t in enumerate(timestamps):
        hour = t.hour
        if any(start <= hour < end for start, end in activity_periods):
            # Random spikes during activity periods
            if random.random() < 0.3:  # 30% chance of activity spike
                activity_spikes[i] = random.uniform(20, 40)  # Significant pulse increase
    
    # Add random noise
    noise = np.random.normal(0, 2, n_samples)
    
    # Combine components
    pulse = base_pulse + awake_pattern + activity_spikes + noise
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'pulse': pulse
    })


def merge_sensor_data(temp_df, motion_df, pulse_df):
    """
    Merge data from multiple sensors into a single DataFrame
    
    Args:
        temp_df: Temperature sensor DataFrame
        motion_df: Motion sensor DataFrame
        pulse_df: Pulse sensor DataFrame
        
    Returns:
        Merged DataFrame with all sensor data
    """
    # Merge on timestamp
    merged_df = pd.merge(temp_df, motion_df, on='timestamp')
    merged_df = pd.merge(merged_df, pulse_df, on='timestamp')
    
    # Add a sensor_id column for identification
    merged_df['sensor_id'] = 'WSN001'
    
    # Add a normal/anomaly label column (all normal by default)
    merged_df['label'] = 'normal'
    
    return merged_df


def simulate_sensor_network(n_days=7, sampling_rate_seconds=60, output_file=None):
    """
    Simulate a complete wireless sensor network dataset
    
    Args:
        n_days: Number of days to simulate
        sampling_rate_seconds: Time between samples in seconds
        output_file: Path to save the CSV output (optional)
        
    Returns:
        DataFrame with simulated sensor data
    """
    # Calculate number of samples
    n_samples = int((n_days * 24 * 60 * 60) / sampling_rate_seconds)
    
    # Set start time to beginning of current day
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=n_days)
    
    # Generate data for each sensor type
    print(f"Generating {n_samples} data points for each sensor type...")
    temp_df = generate_temperature_data(n_samples, start_time, sampling_rate_seconds)
    motion_df = generate_motion_data(n_samples, start_time, sampling_rate_seconds)
    pulse_df = generate_pulse_data(n_samples, start_time, sampling_rate_seconds)
    
    # Merge sensor data
    print("Merging sensor data...")
    merged_df = merge_sensor_data(temp_df, motion_df, pulse_df)
    
    # Save to CSV if output file is specified
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    
    return merged_df


if __name__ == "__main__":
    # Simulate 7 days of sensor data with 1-minute sampling rate
    output_path = "../data/simulated_sensor_data.csv"
    df = simulate_sensor_network(n_days=7, sampling_rate_seconds=60, output_file=output_path)
    
    # Display sample of the data
    print("\nSample of generated data:")
    print(df.head())
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())