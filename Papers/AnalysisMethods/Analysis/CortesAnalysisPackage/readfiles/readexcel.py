import pandas as pd

def readexcel(file_path):
    """
    Reads an Excel file and processes it into a DataFrame.
    
    Parameters:
    file_path (str): The path to the Excel file.
    
    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    df = df[7:]
    df = df[df.columns[3:]]
    df.columns = df.iloc[1]
    df = df[2:]
    df = df.reset_index(drop=True)
    return df
