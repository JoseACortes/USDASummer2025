import pandas as pd

def readcsv(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please make sure it exists in the specified path.")
        raise
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        raise
