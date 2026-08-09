import sqlite3
import numpy as np
import datetime
import os
import time

def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            Cell_ID TEXT,
            Voltage_V REAL,
            Current_A REAL,
            Temperature_C REAL,
            Cycle_Count INTEGER,
            Thermal_Risk INTEGER,
            RUL_Cycles INTEGER
        )
    ''')
    # Clear old data for a fresh run
    cursor.execute('DELETE FROM telemetry')
    conn.commit()
    return conn

def stream_iot_data():
    print("Starting IoT Telemetry Streamer (55-Cell 132kV Grid)...")
    
    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", "telemetry.db")
    conn = setup_db(db_path)
    cursor = conn.cursor()
    
    cells = [f"Cell_{i}" for i in range(1, 56)]
    cycle_counts = {cell: np.random.randint(50, 200) for cell in cells}
    
    # State tracking so faults persist long enough to be inspected in the UI
    faulty_cells = {}
    
    print("Streaming Live Data to SQLite DB... (Press Ctrl+C to stop)")
    
    while True:
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        data_batch = []
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
            
            # Manage persistent faults
            if cell_id in faulty_cells:
                faulty_cells[cell_id]['time'] -= 1
                if faulty_cells[cell_id]['time'] <= 0:
                    del faulty_cells[cell_id]
            
            # 2% chance to start a new fault if not already faulty
            if cell_id not in faulty_cells and np.random.random() < 0.02:
                severity = 'warning' if np.random.random() < 0.6 else 'critical'
                faulty_cells[cell_id] = {'time': np.random.randint(15, 30), 'severity': severity}
                
            if cell_id in faulty_cells:
                if faulty_cells[cell_id]['severity'] == 'critical':
                    # Force critical state (RED)
                    temperature = np.clip(np.random.normal(80, 5), 75, 90)
                    voltage = np.clip(voltage - 0.3, 1.5, 1.9)
                    current = np.clip(current + 15, 0.0, 50.0)
                else:
                    # Force warning state (YELLOW)
                    temperature = np.clip(np.random.normal(68, 3), 60, 74)
                    voltage = np.clip(voltage - 0.1, 1.7, 2.0)
            else:
                # Normal state (GREEN)
                temperature = np.clip(np.random.normal(base_temp, 3), 20, 55)
                
            if temperature >= 75.0 or voltage < 1.7:
                thermal_risk = 2
            elif temperature >= 60.0 or voltage < 1.9:
                thermal_risk = 1
            else:
                thermal_risk = 0
                
            degradation_factor = 1.0 + (temperature > 60) * 0.5
            rul_cycles = int(max(0, 1200 - cycle * degradation_factor - np.random.randint(0, 20)))
            
            data_batch.append((
                timestamp_str, cell_id, round(voltage, 3), round(current, 3),
                round(temperature, 2), cycle, thermal_risk, rul_cycles
            ))
            
        cursor.executemany('''
            INSERT INTO telemetry 
            (Timestamp, Cell_ID, Voltage_V, Current_A, Temperature_C, Cycle_Count, Thermal_Risk, RUL_Cycles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_batch)
        
        conn.commit()
        print(f"[{timestamp_str}] Inserted 55 rows into database.")
        time.sleep(2) # Stream every 2 seconds

if __name__ == "__main__":
    try:
        stream_iot_data()
    except KeyboardInterrupt:
        print("\nStreaming stopped.")
