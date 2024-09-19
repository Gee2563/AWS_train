import pandas as pd
def load_data(filename):
    try:
        print(f"Loading data from {filename}")
        return pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()