import requests
from bs4 import BeautifulSoup
import re
import json
import csv

def scrape_nigeria_total_cases():
    url = "https://www.worldometers.info/coronavirus/country/nigeria/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/90.0.4430.93 Safari/537.36"
    }
    
    # Fetch the page content
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate the <script> tag that contains the Highcharts configuration for total cases.
    target_script = None
    for script in soup.find_all("script"):
        if script.string and "Highcharts.chart('coronavirus-cases-linear'" in script.string:
            target_script = script.string
            break

    if not target_script:
        print("Could not locate the total cases chart script on the page.")
        return

    # Use regex to extract the 'categories' (dates) and 'data' (total cases) arrays from the script.
    categories_match = re.search(r"categories:\s*(\[[^\]]+\])", target_script, re.DOTALL)
    data_match = re.search(r"data:\s*(\[[^\]]+\])", target_script, re.DOTALL)

    if not categories_match or not data_match:
        print("Failed to extract categories or data arrays from the script.")
        return

    categories_str = categories_match.group(1)
    data_str = data_match.group(1)

    # Convert extracted string data into Python lists.
    try:
        # Replace single quotes with double quotes for valid JSON format.
        categories = json.loads(categories_str.replace("'", '"'))
        total_cases = json.loads(data_str.replace("'", '"'))
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return

    # Write the extracted data to a CSV file.
    filename = "nigeria_covid_total_cases.csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date", "cases"])  # CSV header
        for date, cases in zip(categories, total_cases):
            writer.writerow([date, cases])
    
    print(f"Data has been written to {filename}")

if __name__ == "__main__":
    scrape_nigeria_total_cases()
