import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Get the Billboard Year-End Hot 100 singles for a given year.')
    parser.add_argument('year', type=int, help='The year to get the chart for (must be 1946 or later).')
    args = parser.parse_args()

    year = args.year
    if year < 1946:
        print("Error: Please provide a year of 1946 or later.")
        return
    url = f'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}'
    print(f"Fetching data for the year: {year}")
    print(f"URL: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Error: Could not find a page for the year {year}. Please check the year and try again.")
        else:
            print(f"Error fetching URL: {e}")
        return
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table with the song list
    table = soup.find('table', {'class': 'wikitable'})

    if not table:
        print("Could not find the table.")
        return

    # Extract data from the table
    data = []
    rows = table.find_all('tr')
    for row in rows[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) >= 3:
            no = cols[0].text.strip()
            title = cols[1].text.strip().replace('"', '')
            artists = cols[2].text.strip()
            data.append([no, title, artists])

    # Save data to a CSV file
    df = pd.DataFrame(data, columns=['No.', 'Title', 'Artist(s)'])
    csv_filename = f'{year}_Year_End_Hot_100.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")


if __name__ == '__main__':
    main()