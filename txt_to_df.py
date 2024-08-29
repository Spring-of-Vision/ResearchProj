import csv

import pandas as pd
import os
import warnings

# Convert elements to appropriate types (str or float)
def convert_element(element):
    try:
        # Attempt to convert to float
        return float(element)
    except ValueError:
        # If conversion fails, keep as string
        return element

def txt_to_series(txt_file_path):
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()
        
    # Create an initial empty Series
    series_of_lists = pd.Series(dtype=object)        

    for line in lines:
        # Split the line into words and numbers
        elements = line.split()
        #converted_elements = [convert_element(e) for e in elements]

        series = pd.Series([elements])
        
        # Append the list to the Series (turning the list into a series first)
        series_of_lists = series_of_lists.append(series, ignore_index=True)
        #print(series)
    
    return series_of_lists


def txt_to_list(txt_file_path):
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()

    # Create an initial empty list
    list_of_lists = []

    for line in lines:
        # Strip the newline character and split the line into words
        words = line.strip().split()
        # Append the list of words to the overarching list
        list_of_lists.append(words)

    return list_of_lists

def txt_to_df(folder_path):
    # Define the column labels
    columns = ['Label', 'Data']
    # Create an empty DataFrame with these column labels
    df = pd.DataFrame(columns=columns)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(folder_path, filename)
            # Create a new row to add
            new_row = pd.DataFrame({
                'Label': [folder_path],
                'Data': [txt_to_list(txt_file_path)]
            })
            # Concatenate the new row to the original DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
            print(f'finished with file {filename}')
            
    return df

def create_parquet_csv():
    folders = ['Youtube', 'Google Doc', 'Google Drive']

    # Define the column labels
    columns = ['Label', 'Data']
    # Create an empty DataFrame with these column labels
    df = pd.DataFrame(columns=columns)

    for folder in folders:
        df = pd.concat([df, txt_to_df(folder)], ignore_index=True)  # ignore_index?

    print(df)
    # Save DataFrame to a Parquet file using pyarrow
    df.to_parquet('quic_text.parquet', engine='pyarrow')

    # Save to csv
    df.to_csv("quic_text.csv", index=False)
    return df

def count_lines_in_file(filename):
    with open(filename, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count

def validations(df):
    folders = ['Youtube', 'Google Doc', 'Google Drive']

    i = 0  # current row
    problems = 0  # rows that have mismatched line numbers
    not_lists = 0  # rows that aren't lists

    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                txt_file_path = os.path.join(folder, filename)
                outer_list = df.iloc[i, 1]
                if not isinstance(outer_list, list) and all(isinstance(inner_list, list) for inner_list in outer_list):
                    not_lists += 1
                    # print(f"Row {i} is not all list")

                # Different line check if it's a string, otherwise count the length of the list...
                if isinstance(outer_list, str):
                    if count_lines_in_file(txt_file_path) != (outer_list.count('[') - 1):
                        print(f"Row {i} has {(outer_list.count('[') - 1)} brackets but the file has {count_lines_in_file(txt_file_path)} lines.")
                        problems += 1
                elif count_lines_in_file(txt_file_path) != len(outer_list):
                    print(f"Row {i} has {len(outer_list)} lines but the file has {count_lines_in_file(txt_file_path)} lines.")
                    problems += 1

                i += 1

    print(f"There are {not_lists} rows that are not lists and {problems} mistmatched line numbers out of {i} rows.")

def add_column(df):
    # New column values
    new_column = []

    for i in range(len(df)):
        new_column.append(len(df.iloc[i, 1]))

    # Add the new column
    df["num_of_packets"] = new_column

    print("DataFrame after adding a new column with a list:")
    print(df)
    return df

def main():
    df = create_parquet_csv()

    # Load the DataFrame from the Parquet file
    parq = pd.read_parquet('quic_text.parquet')

    csv = pd.read_csv("quic_text.csv")
    #csv = csv.drop(df.columns[0], axis=1)
    #print(df)
    validations(df)

    cell = csv.iloc[0,1]

    #print(cell)
    print(cell.count('[')-1)

    validations(csv)
    validations(parq)

if __name__ == '__main__':
    main()