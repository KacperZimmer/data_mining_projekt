import pandas as pd
import numpy as np
import os


def load_data(data_dir):
    print("Wczytywanie danych...")
    results = pd.read_csv(os.path.join(data_dir, 'results.csv'))
    races = pd.read_csv(os.path.join(data_dir, 'races.csv'))
    drivers = pd.read_csv(os.path.join(data_dir, 'drivers.csv'))
    constructors = pd.read_csv(os.path.join(data_dir, 'constructors.csv'))
    qualifying = pd.read_csv(os.path.join(data_dir, 'qualifying.csv'))
    
    for df in [results, races, drivers, constructors, qualifying]:
        df.replace(r'\N', np.nan, inplace=True)
        
    return results, races, drivers, constructors, qualifying


def clean_and_feature_engineer(data_dir):
    results, races, drivers, constructors, qualifying = load_data(data_dir)
    print("Czyszczenie i Feature Engineering...")
    
    results['top3'] = results['positionOrder'].apply(lambda x: 1 if int(x) <= 3 else 0)
    
    results = results[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder', 'top3', 'statusId']]
    results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
    
    races = races[['raceId', 'year', 'round', 'circuitId', 'date']]
    races['date'] = pd.to_datetime(races['date'])
    races['year'] = pd.to_numeric(races['year'])
    df = pd.merge(results, races, on='raceId', how='left')
    
    drivers = drivers[['driverId', 'dob', 'nationality']]
    drivers.rename(columns={'nationality': 'driver_nationality'}, inplace=True)
    drivers['dob'] = pd.to_datetime(drivers['dob'])
    df = pd.merge(df, drivers, on='driverId', how='left')
    
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
    
    constructors = constructors[['constructorId', 'nationality']]
    constructors.rename(columns={'nationality': 'constructor_nationality'}, inplace=True)
    df = pd.merge(df, constructors, on='constructorId', how='left')
    
    qualifying = qualifying[['raceId', 'driverId', 'position']]
    qualifying.rename(columns={'position': 'quali_position'}, inplace=True)
    qualifying['quali_position'] = pd.to_numeric(qualifying['quali_position'], errors='coerce')
    
    df = pd.merge(df, qualifying, on=['raceId', 'driverId'], how='left')
    
    df['quali_position'] = df['quali_position'].fillna(df['grid'])
    
    df.drop(columns=['dob', 'date'], inplace=True)
    df.dropna(inplace=True)
    
    return df


if __name__ == "__main__":
    DATA_DIR = 'f1_data_raw'
    OUTPUT_DIR = 'f1_data_cleaned'
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'f1_processed_dataset.csv')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_final = clean_and_feature_engineer(DATA_DIR)
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Dane wyczyszczone i zapisane do: {OUTPUT_FILE}")
    print(f"Kształt danych: {df_final.shape}")
    print("\nPodsumowanie zmiennej celu (top3):")
    print(df_final['top3'].value_counts(normalize=True).round(3))
