import json
import pandas as pd

# utils
        
def save_file(data, filename: str):
    with open(f'{filename}.json', "w") as file:
            json.dump(data, file)

def load_file(filename: str):
        try:
            with open(f"{filename}.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return None
        
def save_to_csv(df, filename):
    try:
        df.to_csv(filename, index=False)  # index=False para não salvar o índice como uma coluna no CSV
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

def load_csv_to_df(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return None
