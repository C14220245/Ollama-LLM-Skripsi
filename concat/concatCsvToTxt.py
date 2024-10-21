import requests
import pandas as pd
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--abstract", type=int, default=1, help="Do you want to include the journal's abstract?")
args = vars(ap.parse_args())

def get_google_sheet_data(spreadsheet_id, sheet_name, api_key):
    # Construct the URL for the Google Sheets API
    url = f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{sheet_name}!A1:Z?alt=json&key={api_key}'
    try:
        # Make a GET request to retrieve data from the Google Sheets API
        response = requests.get(url)

        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()
        return data.get('values')
    except requests.exceptions.RequestException as e:
        # Handle any errors that occur during the request
        print(f"An error occurred: {e}")
        return None

def json_value_list_to_dataframe(data):
    # The first sublist contains the column titles
    columns = data[0]
    # The rest of the sublists contain the data
    rows = data[1:]
    # Create the DataFrame
    df = pd.DataFrame(rows, columns=columns)
    return df



# Write all files in folder docs_source to docs.txt

folder_path = '../docs_source'  # Update this to your folder path

# Open the output file in write mode
with open('../datum_RAG/docs.txt', 'w') as outfile:
    # Loop through all the files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as infile:
                # Write the content of each file to the output file
                outfile.write(infile.read())
                outfile.write('\n\n')  # Add newline for separation


# api key dari google cloud
api_google_sheets = 'AIzaSyDKMiCCUkImuBeDyZzMJtPk0ES8-MpIRjk'


# write file docs_dosen
spreadsheet_id_data_dosen = '1oS_bCqdYBJTteCyKJfDca27LBg68ABFcCA4vC_4ACbc'
sheet_name_docs_dosen = "Sheet1"

sheet_data_dosen = get_google_sheet_data(spreadsheet_id_data_dosen, sheet_name_docs_dosen, api_google_sheets)

if sheet_data_dosen:
    sheet_data_dosen = json_value_list_to_dataframe(sheet_data_dosen)
    # sheet_data_dosen.to_csv('../docs_dosen_30sept/docs_dosen.csv', index = False)
else:
    print("Failed to fetch data from Google Sheets API.")


# read & concat data_jurnal_dosen to docs_dosen
spreadsheet_id_data_jurnal_dosen = '1WTwZonmTbxW1gMHxJAHd3ZpPAn1JeWluXtgEBb9XZ4Y'
sheet_name_docs_jurnal_dosen = "Sheet1"

sheet_data_jurnal_dosen = get_google_sheet_data(spreadsheet_id_data_jurnal_dosen, sheet_name_docs_jurnal_dosen, api_google_sheets)


if sheet_data_jurnal_dosen:
        sheet_data_jurnal_dosen = json_value_list_to_dataframe(sheet_data_jurnal_dosen)
    
        index_data_dosen = 0
        last_paper_index = 0
    
        indexPaperDosen = [0 for _ in range(len(sheet_data_dosen))]
        
        for i in range(len(sheet_data_jurnal_dosen)):
            # print(i)
            if(sheet_data_jurnal_dosen.at[i, 'Nama Dosen'].strip() != ""):
                # Find the index in sheet_data_dosen where the 'nama' column matches
                index_data_dosen = sheet_data_dosen[sheet_data_dosen['Nama Dosen Pembimbing UK Petra'] == sheet_data_jurnal_dosen.at[i, 'Nama Dosen']].index
    
            indexPaperDosen[index_data_dosen.tolist()[0]] += 1
    
            column_name = f'Paper {indexPaperDosen[index_data_dosen.tolist()[0]]}'
            if column_name not in sheet_data_dosen.columns:
                sheet_data_dosen[column_name] = None  # Create the column with None values
    
            
            if(args["abstract"] > 0):
                sheet_data_dosen.at[index_data_dosen.tolist()[0], column_name] = f"Judul: {sheet_data_jurnal_dosen.at[i, 'Judul Paper']}, Abstrak: {sheet_data_jurnal_dosen.at[i, 'Abstrak']}, Jenis: {sheet_data_jurnal_dosen.at[i, 'Jenis']}, Tahun: {sheet_data_jurnal_dosen.at[i, 'Tahun']}, Tags: {sheet_data_jurnal_dosen.at[i, 'Tag']}"
            else:
                sheet_data_dosen.at[index_data_dosen.tolist()[0], column_name] = f"Judul: {sheet_data_jurnal_dosen.at[i, 'Judul Paper']}, Jenis: {sheet_data_jurnal_dosen.at[i, 'Jenis']}, Tahun: {sheet_data_jurnal_dosen.at[i, 'Tahun']}, Tags: {sheet_data_jurnal_dosen.at[i, 'Tag']}"
            

else:
    print("Failed to fetch data from Google Sheets API.")


with open('../datum_RAG/docs.txt', 'a') as docs_txt:
    docs_txt.write("\n\nDi bawah ini adalah daftar semua dosen pembimbing Informatika UK Petra yang terdiri dari 29 orang:  \n")
    # for i in range(len(sheet_data_dosen)):
    #     docs_txt.write("\n")
    #     for col in sheet_data_dosen:
    #         if sheet_data_dosen.at[i, col] is not None:
    #             if col == "Nama":
    #                 docs_txt.write(f'\n{i + 1}. {col}: {sheet_data_dosen.at[i, col]}')
    #                 # docs_txt.write(f'{sheet_data_dosen.at[i, col]}; ')
    #             # else:
    #             #     docs_txt.write(f'\n   {col}: {sheet_data_dosen.at[i, col]}')

    for i in range(len(sheet_data_dosen)):
        # docs_txt.write("\n")
        for col in sheet_data_dosen:
            if sheet_data_dosen.at[i, col] is not None:
                if col == "Nama Dosen Pembimbing UK Petra":
                    docs_txt.write(f'\n{i + 1}. {col}: {sheet_data_dosen.at[i, col]}')
                else:
                    docs_txt.write(f'\n   {col}: {sheet_data_dosen.at[i, col]}')
            
print(sheet_data_dosen)
















# # Reading a CSV file
# df = pd.read_csv('data_dosen.csv')
# print(df)

# # Writing to a CSV file
# df.to_csv('example.csv', index=False)





# with open("untitled.txt", "r") as file:
#     content = file.read()
#     print(content)

# import csv

# # Reading a CSV file
# with open('data_dosen.csv', mode='r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row)

# # Writing to a CSV file
# with open('data_dosen.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Name', 'Age', 'City'])
#     writer.writerow(['Alice', '30', 'New York'])