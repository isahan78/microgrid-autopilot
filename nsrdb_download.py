import requests

API_KEY = "Udgmshu0LG1trcsRqhLfpQgdIOarMyM8Q4uj9WIu"          # <-- replace
EMAIL = "isahankhan.mlengineer@gmail.com"    # <-- replace
FULL_NAME = "Isahan"           # <-- replace

url = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

params = {
    "api_key": API_KEY,
    "wkt": "POINT(-117.05 32.55)",              # PV site
    "attributes": "ghi,dni,air_temperature",    # choose attributes you need
    "names": "2018",                            # any year 1998-2024
    "interval": "30",                           # allowed: 30 or 60
    "utc": "false",
    "leap_day": "false",
    "email": EMAIL,
    "full_name": FULL_NAME,
    "affiliation": "None",
    "mailing_list": "false"
}

print("Requesting GOES Aggregated PSM v4 from NSRDB...")
response = requests.get(url, params=params, stream=True)

if response.status_code != 200:
    print("❌ Error:", response.status_code)
    print(response.text)
    raise SystemExit

# Save CSV
filename = "weather_goes_psm4.csv"
with open(filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"✅ Download complete: {filename}")