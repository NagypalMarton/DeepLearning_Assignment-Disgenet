import requests
import time
import pandas as pd
import torch
from torch_geometric.data import Data

API_KEY = "c89e2d9e-94b2-4b84-8d22-bb525e63b73b"

params = {
    "page_number": 0,
    "type": "disease"
}

# Create a dictionary with HTTP headers
headers = {
    'Authorization': API_KEY,
    'accept': 'application/json'
}

# API endpoints
url_gda = "https://api.disgenet.com/api/v1/gda/summary"
url_disease = "https://api.disgenet.com/api/v1/entity/disease"
# Function to handle API requests with rate-limiting handling
def make_request(url, params, headers):
    retries = 0
    while retries < 5:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            # If rate-limited (HTTP 429), retry after waiting
            if response.status_code == 429:
                wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 60))
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                return response  # Return response if successful or error other than 429

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            retries += 1
            time.sleep(2)  # Wait before retrying

    return None  # Return None if retries are exhausted

def get_max_pages(url, params=params, headers=headers):
  response = make_request(url, params=params, headers=headers)
  if response.ok:
      response_json = response.json()
      total_results = response_json.get("paging", {}).get("totalElements", 0)
      results_in_page = response_json.get("paging", {}).get("totalElementsInPage", 0)
      max_pages = min((total_results + results_in_page - 1) // results_in_page, 100)
  else:
      max_pages = 100
      print("Request failed, returned max_pages=100")
  return max_pages

def get_disease_ids(disease_type):
    disease_ids = []
    params['disease_free_text_search_string'] = disease_type

    for page in range(100):
      params['page_number'] = str(page)
      response_disease = make_request(url_disease, params, headers)
      if response_disease and response_disease.ok:
          response_disease_json = response_disease.json()
          data = response_disease_json.get("payload", [])
          for item in data:
              for code_info in item.get("diseaseCodes", []):
                if code_info.get("vocabulary") == "MONDO":
                  disease_ids.append(f'MONDO_{code_info.get("code")}')
      else:
          print(f"Failed to fetch data for page {page}. Status code: {response_disease_json.status_code}")
          break
    return list(set(disease_ids))

def download_gda(disease_ids):
    gda_data = []
    params['disease'] = disease_ids

    for page in range(100):
        params['page_number'] = str(page)  # Különböző oldalak lekérése
        response_gda = make_request(url_gda, params, headers)
        if response_gda and response_gda.ok:
            response_json = response_gda.json()
            data = response_json.get("payload", [])
            gda_data.extend(data)
        else:
            print(f"Failed to fetch data for page {page}. Status code: {response_json.status_code}")
            break  # Ha nincs több oldal vagy hiba történik, kilépünk a ciklusból

    return gda_data

def download_all_gda(ids, chunk_size=100):
    all_data = []
    for i in range(0, len(ids), chunk_size):
        ids_chunk = ids[i:i + chunk_size]
        ids_string = '"' + ', '.join(ids_chunk) + '"'
        chunk_data = download_gda(ids_string)
        all_data.extend(chunk_data)
    df_gda = pd.DataFrame(all_data)
    df_gda.to_csv('GDA_df_raw.csv', index=False)
    print(f"All data saved to GDA_df_raw.csv")
    
disease_ids = get_disease_ids("cancer")

download_all_gda(disease_ids)