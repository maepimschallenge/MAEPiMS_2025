import requests
from bs4 import BeautifulSoup
import re
import json
import csv

def scrape_nigeria_daily_cases():
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
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate the <script> tag that contains the Highcharts configuration for daily cases.
    target_script = None
    for script in soup.find_all("script"):
        if script.string and "Highcharts.chart('graph-cases-daily'" in script.string:
            target_script = script.string
            break

    if not target_script:
        print("Could not locate the daily cases chart script on the page.")
        return

    # Extract the categories (dates) and data (daily cases) arrays using regex.
    categories_match = re.search(r"categories:\s*(\[[^\]]+\])", target_script, re.DOTALL)
    data_match = re.search(r"data:\s*(\[[^\]]+\])", target_script, re.DOTALL)

    if not categories_match or not data_match:
        print("Failed to extract categories or data from the script.")
        return

    categories_str = categories_match.group(1)
    data_str = data_match.group(1)

    # Convert the extracted string data into Python objects.
    try:
        categories = json.loads(categories_str.replace("'", '"'))
        daily_cases = json.loads(data_str.replace("'", '"'))
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return

    # Write the data to a CSV file
    filename = "nigeria_covid_daily_cases.csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date", "Daily Cases"])  # Write header row
        for date, cases in zip(categories, daily_cases):
            writer.writerow([date, cases])
    
    print(f"Data has been written to {filename}")

if __name__ == "__main__":
    scrape_nigeria_daily_cases()

