import requests
import pandas as pd
import time
from io import StringIO
import re
import urllib3

# Specify your API key
API_KEY = "b5d672e8-f674-442c-ab41-2ea8991dd076"

# Function to retrieve disease identifiers
def get_disease_identifiers(disease_free_text_search_string):
    # Initialize an empty list to collect CSV data from all pages
    all_csv_data = []

    for page_number in range(100):  # Request from page 0 to 100
        # Specify query parameters
        params = {
            "disease_free_text_search_string": disease_free_text_search_string,
            "type": "disease",
            "page_number": page_number
        }

        # HTTP headers
        headers = {
            "Authorization": API_KEY,
            "accept": "application/csv"
        }

        # API endpoint
        url = "https://api.disgenet.com/api/v1/entity/disease"

        while True:
            response = requests.get(url, params=params, headers=headers, verify=False)
            # Successful response
            if response.status_code == 200:
                # Check if the response contains "-1"
                if "-1" in response.text:
                    print(f"Received undesired content on page {page_number}, stopping further requests.")
                    return all_csv_data  # Return collected data
                else:
                    all_csv_data.append(response.text)
                    break  # Move to the next page

            # Rate limiting handling
            elif response.status_code == 429:
                retry_after = int(response.headers.get('x-rate-limit-retry-after-seconds', '60'))
                print(f"Rate limit reached. Waiting for {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                print(f"Request failed on page {page_number} with status code {response.status_code}: {response.text}")
                return None

    return all_csv_data  # Return collected data if the loop completes

# Sample data for demonstration (replace this with your actual DataFrame)
    # df = pd.DataFrame({
    #     'diseaseCodes': [
    #         "[DiseaseCodeDTO(vocabulary=UMLS, code=C4733092)]",
    #         "[DiseaseCodeDTO(vocabulary=MONDO, code=0000615), DiseaseCodeDTO(vocabulary=UMLS, code=C4733094)]",
    #         # Add more rows as needed
    #     ]
    # })

    # Function to extract disease identifiers
    def extract_disease_identifiers(disease_code_str):
        # Regular expression pattern to match 'DiseaseCodeDTO(vocabulary=..., code=...)'
        pattern = r"DiseaseCodeDTO\(vocabulary=(.*?), code=(.*?)\)"

        # Find all matches in the string
        matches = re.findall(pattern, disease_code_str)

        # Combine vocabulary and code to form the identifiers
        disease_identifiers = []
        for vocab, code in matches:
            # Clean up any extra whitespace or quotes
            vocab = vocab.strip().strip('"').strip("'")
            code = code.strip().strip('"').strip("'")
            identifier = f"{vocab}_{code}"
            disease_identifiers.append(identifier)

        return disease_identifiers

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Your API key
API_KEY = "b5d672e8-f674-442c-ab41-2ea8991dd076"

# Assume 'disease_identifiers_array' is already defined
# For example:
# disease_identifiers_array = ['UMLS_C4733092', 'MONDO_0000615', 'UMLS_C4733094']

# Function to retrieve data for a disease identifier
def get_gda_summary(disease_identifier):
    url = "https://api.disgenet.com/api/v1/gda/summary"
    all_csv_data = []

    for page_number in range(100):  # Request from page 0 to 99
        # Prepare parameters
        params = {
            'disease': disease_identifier,  # Disease code
            'type': 'disease',              # Type parameter set to 'disease'
            'page_number': page_number      # Current page number
        }

        # HTTP headers
        headers = {
            'Authorization': API_KEY,
            'accept': 'application/csv'     # Response format set to CSV
        }

        while True:
            response = requests.get(url, params=params, headers=headers, verify=False)
            # Successful response
            if response.status_code == 200:
                if "-1" in response.text:
                    print(f"Received undesired content for {disease_identifier} on page {page_number}, skipping to next identifier.")
                    return all_csv_data if all_csv_data else None
                else:
                    all_csv_data.append(response.text)
                    break

            # Rate limiting handling
            elif response.status_code == 429:
                retry_after = int(response.headers.get('x-rate-limit-retry-after-seconds', '60'))
                print(f"Rate limit reached. Waiting for {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                print(f"Request failed for {disease_identifier} on page {page_number} with status code {response.status_code}: {response.text}")
                return None

    return all_csv_data if all_csv_data else None

# Main execution
if __name__ == "__main__":
    # Replace 'cancer' with your desired disease search term
    disease_free_text_search_string = "cancer"

    # Retrieve disease identifiers
    csv_data_list = get_disease_identifiers(disease_free_text_search_string)
    print(csv_data_list)

    if csv_data_list:
        # Combine all CSV data into a single DataFrame
        combined_csv_data = "\n".join(csv_data_list)
        df = pd.read_csv(StringIO(combined_csv_data), sep='\t')
        # Save the DataFrame to a CSV file
        df.to_csv('./data/disease_identifiers.csv', index=False)
        print("Data has been saved to './data/disease_identifiers.csv'.")
    else:
        print("Failed to retrieve disease identifiers.")

    print(f'Sor*Oszlop: {df.shape[0]} * {df.shape[1]}')
    print(df.head())

    # Apply the function to the 'diseaseCodes' column
    df['disease_identifiers'] = df['diseaseCodes'].apply(extract_disease_identifiers)

    # Put all disease identifiers into an array
    disease_identifiers_array = df['disease_identifiers'].explode().tolist()

    # Display the array of disease identifiers
    print(disease_identifiers_array)
    print(len(disease_identifiers_array))

    # Ensure 'disease_identifiers_array' is defined
    # For example, you can extract it from your DataFrame as follows:
    # disease_identifiers_array = df['disease_identifiers'].explode().unique().tolist()

    # Example:
    # disease_identifiers_array = ['UMLS_C4733092', 'MONDO_0000615', 'UMLS_C4733094']

    # Initialize a list to collect DataFrames
    all_data = []
    #disease_identifiers_arrayNEW = disease_identifiers_array[:25]

    for disease_identifier in disease_identifiers_array:
        print(f"Processing disease identifier: {disease_identifier}")
        csv_data_list = get_gda_summary(disease_identifier)

        if csv_data_list:
            # Combine all CSV data into a single DataFrame
            combined_csv_data = "\n".join(csv_data_list)
            df = pd.read_csv(StringIO(combined_csv_data), sep='\t')

            # Add the disease identifier to the DataFrame
            df['disease_identifier'] = disease_identifier

            # Append the DataFrame to the list
            all_data.append(df)
        else:
            print(f"No data retrieved for {disease_identifier}")

    if all_data:
        # Concatenate all DataFrames
        final_df = pd.concat(all_data, ignore_index=True)

        # Save the combined data to a CSV file
        final_df.to_csv('./data/gda_summary_data.csv', index=False)
        print("All data has been saved to './data/gda_summary_data.csv'.")
        print(final_df)
    else:
        print("No data was retrieved.")

print(f'Sor*Oszlop: {df.shape[0]} * {df.shape[1]}')
print(df.head())
