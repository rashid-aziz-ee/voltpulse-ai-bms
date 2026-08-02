import pandas as pd
import numpy as np
import datetime
import os

def generate_telemetry_data(num_timestamps=200):
    print("Generating synthetic battery telemetry data (55-Cell 132kV Grid)...")
    
    cells = [f"Cell_{i}" for i in range(1, 56)]
    base_time = datetime.datetime.now() - datetime.timedelta(days=10)
    
    data = []
    
    # Track cycles for each cell to simulate progression
    cycle_counts = {cell: np.random.randint(50, 200) for cell in cells}
    
    for i in range(num_timestamps):
        timestamp = base_time + datetime.timedelta(minutes=i*2)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate data for ALL 55 cells at this timestamp
        for cell_id in cells:
            
            if np.random.random() < 0.05:
                cycle_counts[cell_id] += 1
                
            cycle = cycle_counts[cell_id]
            if cycle > 1200:
                cycle = 1200
                
            base_voltage = 2.25 - (cycle / 1200) * 0.4
            voltage = np.clip(np.random.normal(base_voltage, 0.05), 1.7, 2.3)
            
            current = np.clip(np.random.normal(25, 10), 0.0, 50.0)
            
            base_temp = 25.0 + (current / 50.0) * 20.0 + (cycle / 1200) * 10.0
            
            rand_event = np.random.random()
            if rand_event < 0.05:
                temperature = np.clip(np.random.normal(65, 5), 60, 75)
                voltage = np.clip(voltage - 0.1, 1.7, 2.3)
            elif rand_event < 0.07:
                temperature = np.clip(np.random.normal(80, 5), 75, 90)
                voltage = np.clip(voltage - 0.3, 1.5, 1.9)
                current = np.clip(current + 15, 0.0, 50.0)
            else:
                temperature = np.clip(np.random.normal(base_temp, 3), 20, 60)
                
            if temperature >= 75.0 or voltage < 1.7:
                thermal_risk = 2  # Critical
            elif temperature >= 60.0 or voltage < 1.9:
                thermal_risk = 1  # Warning
            else:
                thermal_risk = 0  # Normal
                
            degradation_factor = 1.0 + (temperature > 60) * 0.5
            rul_cycles = int(max(0, 1200 - cycle * degradation_factor - np.random.randint(0, 20)))
            
            data.append({
                "Timestamp": timestamp_str,
                "Cell_ID": cell_id,
                "Voltage_V": round(voltage, 3),
                "Current_A": round(current, 3),
                "Temperature_C": round(temperature, 2),
                "Cycle_Count": cycle,
                "Thermal_Risk": thermal_risk,
                "RUL_Cycles": rul_cycles
            })
        
    df = pd.DataFrame(data)
    
    # Sort by timestamp
    df = df.sort_values(by=["Timestamp", "Cell_ID"]).reset_index(drop=True)
    
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", "battery_telemetry.csv")
    df.to_csv(file_path, index=False)
    
    print(f"✅ Generated {len(df)} rows of battery telemetry data for 55 cells.")
    print(f"✅ Saved to: {file_path}")
    
    return df

if __name__ == "__main__":
    generate_telemetry_data()
