import subprocess
from datetime import datetime

filename = "mbg.csv"
search_keyword = "mbg"
since_date = "2025-11-01"
until_date = "2025-11-17"
limit = "700"
auth_token = "ini diisi token auth cookies login twitter akun kalian"

query = f"{search_keyword} since:{since_date} until:{until_date} lang:id"
command = [
    "npx", "-y", "tweet-harvest@2.6.1",
    "-o", filename,
    "-s", query,
    "--tab", "LATEST",
    "-l", limit,
    "--token", auth_token
]

print("\nSedang melakukan scraping data dari Twitter/X...\n")
try:
    subprocess.run(command, check=True)
    print(f"[{datetime.now()}] Selesai! Hasil tersimpan di: {filename}")
except subprocess.CalledProcessError as e:
    print(f"[{datetime.now()}] Terjadi kesalahan saat scraping.")

from google.colab import files
files.download("tweets-data/hasil_ledakan.csv")