import numpy as np
import pandas as pd
import random
from datetime import timedelta
import os


def inject_dos_attack(df, attack_duration_minutes=30, start_offset_days=None):
    """
    Inject Denial of Service (DoS) attack anomalies
    Simulates packet loss by nullifying or dropping data points
    
    Args:
        df: DataFrame with sensor data
        attack_duration_minutes: Duration of attack in minutes
        start_offset_days: Days from start to begin attack (random if None)
        
    Returns:
        DataFrame with injected DoS anomalies
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Get timestamps
    timestamps = df_copy['timestamp'].unique()
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    
    # Determine attack start time
    if start_offset_days is None:
        # Random start between day 1 and day (total_days - 1)
        start_offset_days = random.uniform(1, max(1.5, total_days - 1))
    
    attack_start = timestamps[0] + timedelta(days=start_offset_days)
    attack_end = attack_start + timedelta(minutes=attack_duration_minutes)
    
    # Create attack mask
    attack_mask = (df_copy['timestamp'] >= attack_start) & (df_copy['timestamp'] <= attack_end)
    attack_indices = df_copy[attack_mask].index
    
    if len(attack_indices) == 0:
        print("No data points found in the specified attack window.")
        return df_copy
    
    # Apply DoS effects: nullify data or drop packets
    for idx in attack_indices:
        attack_type = random.choice(['nullify', 'drop'])
        
        if attack_type == 'nullify':
            # Set sensor values to 0 or NaN
            if random.random() < 0.7:  # 70% chance of nullifying
                df_copy.loc[idx, 'temperature'] = 0
                df_copy.loc[idx, 'motion'] = 0
                df_copy.loc[idx, 'pulse'] = 0
        else:  # drop
            # We'll simulate dropped packets by setting to NaN
            # In a real system, these would be missing entirely
            if random.random() < 0.5:  # 50% chance of dropping
                df_copy.loc[idx, ['temperature', 'motion', 'pulse']] = np.nan
    
    # Label the anomalies
    df_copy.loc[attack_mask, 'label'] = 'DoS'
    
    print(f"Injected DoS attack from {attack_start} to {attack_end} affecting {len(attack_indices)} data points")
    return df_copy


def inject_jamming_attack(df, attack_duration_minutes=45, start_offset_days=None):
    """
    Inject jamming attack anomalies
    Adds random high-frequency noise to sensor readings
    
    Args:
        df: DataFrame with sensor data
        attack_duration_minutes: Duration of attack in minutes
        start_offset_days: Days from start to begin attack (random if None)
        
    Returns:
        DataFrame with injected jamming anomalies
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Get timestamps
    timestamps = df_copy['timestamp'].unique()
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    
    # Determine attack start time
    if start_offset_days is None:
        # Random start between day 2 and day (total_days - 1)
        start_offset_days = random.uniform(2, max(2.5, total_days - 1))
    
    attack_start = timestamps[0] + timedelta(days=start_offset_days)
    attack_end = attack_start + timedelta(minutes=attack_duration_minutes)
    
    # Create attack mask
    attack_mask = (df_copy['timestamp'] >= attack_start) & (df_copy['timestamp'] <= attack_end)
    attack_indices = df_copy[attack_mask].index
    
    if len(attack_indices) == 0:
        print("No data points found in the specified attack window.")
        return df_copy
    
    # Apply jamming effects: add high-frequency noise
    # Get standard deviations of normal data for scaling the noise
    temp_std = df_copy[df_copy['label'] == 'normal']['temperature'].std()
    pulse_std = df_copy[df_copy['label'] == 'normal']['pulse'].std()
    
    for idx in attack_indices:
        # Add strong random noise to temperature and pulse
        if random.random() < 0.9:  # 90% of points affected
            noise_temp = random.uniform(-5 * temp_std, 5 * temp_std)
            df_copy.loc[idx, 'temperature'] += noise_temp
            
            noise_pulse = random.uniform(-5 * pulse_std, 5 * pulse_std)
            df_copy.loc[idx, 'pulse'] += noise_pulse
            
            # Motion is binary, so randomly flip it with 50% probability
            if random.random() < 0.5:
                df_copy.loc[idx, 'motion'] = 1 - df_copy.loc[idx, 'motion']
    
    # Label the anomalies
    df_copy.loc[attack_mask, 'label'] = 'Jamming'
    
    print(f"Injected jamming attack from {attack_start} to {attack_end} affecting {len(attack_indices)} data points")
    return df_copy


def inject_tampering_attack(df, attack_duration_minutes=60, start_offset_days=None):
    """
    Inject tampering attack anomalies
    Creates sudden drifts or step changes in sensor values
    
    Args:
        df: DataFrame with sensor data
        attack_duration_minutes: Duration of attack in minutes
        start_offset_days: Days from start to begin attack (random if None)
        
    Returns:
        DataFrame with injected tampering anomalies
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Get timestamps
    timestamps = df_copy['timestamp'].unique()
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    
    # Determine attack start time
    if start_offset_days is None:
        # Random start between day 3 and day (total_days - 1)
        start_offset_days = random.uniform(3, max(3.5, total_days - 1))
    
    attack_start = timestamps[0] + timedelta(days=start_offset_days)
    attack_end = attack_start + timedelta(minutes=attack_duration_minutes)
    
    # Create attack mask
    attack_mask = (df_copy['timestamp'] >= attack_start) & (df_copy['timestamp'] <= attack_end)
    attack_indices = df_copy[attack_mask].index
    
    if len(attack_indices) == 0:
        print("No data points found in the specified attack window.")
        return df_copy
    
    # Choose tampering type
    tampering_type = random.choice(['drift', 'step'])
    
    if tampering_type == 'drift':
        # Create a gradual drift in temperature
        drift_magnitude = random.uniform(5, 15)  # Significant drift
        drift_direction = random.choice([-1, 1])  # Up or down
        
        # Calculate drift for each point based on position in attack window
        for i, idx in enumerate(attack_indices):
            progress = i / len(attack_indices)  # 0 to 1 progress through attack
            drift_amount = drift_magnitude * progress * drift_direction
            df_copy.loc[idx, 'temperature'] += drift_amount
            
    else:  # step change
        # Create a sudden step change in temperature
        step_magnitude = random.uniform(8, 20)  # Significant step
        step_direction = random.choice([-1, 1])  # Up or down
        
        # Apply step change to all points in attack window
        df_copy.loc[attack_indices, 'temperature'] += step_magnitude * step_direction
    
    # Label the anomalies
    df_copy.loc[attack_mask, 'label'] = 'Tampering'
    
    print(f"Injected {tampering_type} tampering attack from {attack_start} to {attack_end} affecting {len(attack_indices)} data points")
    return df_copy


def inject_hardware_fault(df, attack_duration_minutes=120, start_offset_days=None):
    """
    Inject hardware fault anomalies
    Creates flat lines or repeated constant values
    
    Args:
        df: DataFrame with sensor data
        attack_duration_minutes: Duration of fault in minutes
        start_offset_days: Days from start to begin fault (random if None)
        
    Returns:
        DataFrame with injected hardware fault anomalies
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Get timestamps
    timestamps = df_copy['timestamp'].unique()
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    
    # Determine fault start time
    if start_offset_days is None:
        # Random start between day 4 and day (total_days - 1)
        start_offset_days = random.uniform(4, max(4.5, total_days - 1))
    
    fault_start = timestamps[0] + timedelta(days=start_offset_days)
    fault_end = fault_start + timedelta(minutes=attack_duration_minutes)
    
    # Create fault mask
    fault_mask = (df_copy['timestamp'] >= fault_start) & (df_copy['timestamp'] <= fault_end)
    fault_indices = df_copy[fault_mask].index
    
    if len(fault_indices) == 0:
        print("No data points found in the specified fault window.")
        return df_copy
    
    # Choose fault type and affected sensor
    fault_type = random.choice(['flatline', 'stuck_value'])
    affected_sensor = random.choice(['temperature', 'pulse'])  # Motion is binary, so less interesting
    
    if fault_type == 'flatline':
        # Set sensor to a constant value (0 or last value before fault)
        if random.random() < 0.5:  # 50% chance of zero value
            df_copy.loc[fault_indices, affected_sensor] = 0
        else:
            # Use last value before fault
            if fault_indices[0] > 0:
                last_value = df_copy.loc[fault_indices[0] - 1, affected_sensor]
                df_copy.loc[fault_indices, affected_sensor] = last_value
            else:
                # If fault starts at beginning, use a reasonable constant
                if affected_sensor == 'temperature':
                    df_copy.loc[fault_indices, affected_sensor] = 25.0  # Room temperature
                else:  # pulse
                    df_copy.loc[fault_indices, affected_sensor] = 70.0  # Resting heart rate
    
    else:  # stuck_value
        # Repeat a small set of values
        if affected_sensor == 'temperature':
            stuck_value = random.uniform(15, 35)  # Reasonable temperature range
        else:  # pulse
            stuck_value = random.uniform(60, 100)  # Reasonable pulse range
            
        # Add tiny variations to make it look like a stuck sensor with noise
        for idx in fault_indices:
            tiny_noise = random.uniform(-0.1, 0.1)
            df_copy.loc[idx, affected_sensor] = stuck_value + tiny_noise
    
    # Label the anomalies
    df_copy.loc[fault_mask, 'label'] = 'HardwareFault'
    
    print(f"Injected {fault_type} hardware fault on {affected_sensor} sensor from {fault_start} to {fault_end} affecting {len(fault_indices)} data points")
    return df_copy


def inject_environmental_noise(df, attack_duration_minutes=40, start_offset_days=None):
    """
    Inject environmental noise anomalies
    Adds Gaussian or salt-and-pepper noise bursts
    
    Args:
        df: DataFrame with sensor data
        attack_duration_minutes: Duration of noise in minutes
        start_offset_days: Days from start to begin noise (random if None)
        
    Returns:
        DataFrame with injected environmental noise anomalies
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Get timestamps
    timestamps = df_copy['timestamp'].unique()
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    
    # Determine noise start time
    if start_offset_days is None:
        # Random start between day 5 and day (total_days - 1)
        start_offset_days = random.uniform(5, max(5.5, total_days - 1))
    
    noise_start = timestamps[0] + timedelta(days=start_offset_days)
    noise_end = noise_start + timedelta(minutes=attack_duration_minutes)
    
    # Create noise mask
    noise_mask = (df_copy['timestamp'] >= noise_start) & (df_copy['timestamp'] <= noise_end)
    noise_indices = df_copy[noise_mask].index
    
    if len(noise_indices) == 0:
        print("No data points found in the specified noise window.")
        return df_copy
    
    # Choose noise type
    noise_type = random.choice(['gaussian', 'salt_pepper'])
    
    # Get standard deviations of normal data for scaling the noise
    temp_std = df_copy[df_copy['label'] == 'normal']['temperature'].std()
    pulse_std = df_copy[df_copy['label'] == 'normal']['pulse'].std()
    
    if noise_type == 'gaussian':
        # Add Gaussian noise to temperature and pulse
        for idx in noise_indices:
            # Stronger noise than normal variations
            noise_temp = np.random.normal(0, 3 * temp_std)
            noise_pulse = np.random.normal(0, 3 * pulse_std)
            
            df_copy.loc[idx, 'temperature'] += noise_temp
            df_copy.loc[idx, 'pulse'] += noise_pulse
    
    else:  # salt_pepper
        # Add occasional extreme values (salt and pepper noise)
        for idx in noise_indices:
            if random.random() < 0.2:  # 20% of points affected with extreme values
                # Decide whether to add a high or low extreme value
                if random.random() < 0.5:  # high value (salt)
                    df_copy.loc[idx, 'temperature'] += random.uniform(10, 20)
                    df_copy.loc[idx, 'pulse'] += random.uniform(30, 50)
                else:  # low value (pepper)
                    df_copy.loc[idx, 'temperature'] -= random.uniform(10, 20)
                    df_copy.loc[idx, 'pulse'] -= random.uniform(30, 50)
    
    # Label the anomalies
    df_copy.loc[noise_mask, 'label'] = 'EnvironmentalNoise'
    
    print(f"Injected {noise_type} environmental noise from {noise_start} to {noise_end} affecting {len(noise_indices)} data points")
    return df_copy


def inject_all_anomalies(input_file, output_file=None):
    """
    Inject all types of anomalies into a dataset
    
    Args:
        input_file: Path to input CSV file with normal sensor data
        output_file: Path to save the CSV output with anomalies (optional)
        
    Returns:
        DataFrame with all injected anomalies
    """
    # Load the data
    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    
    # Make sure all data is initially labeled as normal
    df['label'] = 'normal'
    
    print(f"Loaded {len(df)} data points from {input_file}")
    
    # Calculate total days in dataset for spacing anomalies
    timestamps = df['timestamp'].unique()
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    
    print(f"Dataset spans {total_days:.1f} days from {timestamps[0]} to {timestamps[-1]}")
    
    # Space out anomalies across the dataset
    day_spacing = max(1, total_days / 6)  # Divide dataset into 6 segments
    
    # Inject each type of anomaly with appropriate spacing
    df = inject_dos_attack(df, start_offset_days=day_spacing * 1)
    df = inject_jamming_attack(df, start_offset_days=day_spacing * 2)
    df = inject_tampering_attack(df, start_offset_days=day_spacing * 3)
    df = inject_hardware_fault(df, start_offset_days=day_spacing * 4)
    df = inject_environmental_noise(df, start_offset_days=day_spacing * 5)
    
    # Save to CSV if output file is specified
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Data with anomalies saved to {output_file}")
    
    # Print summary of anomalies
    anomaly_counts = df['label'].value_counts()
    print("\nAnomaly distribution:")
    for label, count in anomaly_counts.items():
        percentage = 100 * count / len(df)
        print(f"{label}: {count} data points ({percentage:.2f}%)")
    
    return df


if __name__ == "__main__":
    # Inject anomalies into the simulated data
    input_path = "../data/simulated_sensor_data.csv"
    output_path = "../data/simulated_sensor_data_with_anomalies.csv"
    
    # Check if input file exists, if not, generate it
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found. Generating normal data first...")
        from simulate_data import simulate_sensor_network
        simulate_sensor_network(n_days=7, sampling_rate_seconds=60, output_file=input_path)
    
    # Inject anomalies
    df_with_anomalies = inject_all_anomalies(input_path, output_path)