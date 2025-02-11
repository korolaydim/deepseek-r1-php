import requests
import tarfile
import os

# Dosyanın URL'si
url = "https://www.php.net/distributions/manual/php_manual_en.tar.gz"

# Dosyanın adı
file_name = "php_manual_en.tar.gz"

# Dosyayı indir
response = requests.get(url)

# Dosyayı kaydet
with open(file_name, 'wb') as f:
    f.write(response.content)

# Dosyayı çıkart
if file_name.endswith(".tar.gz"):
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=os.getcwd())  # Mevcut çalışma dizinine çıkartır

print("Dosya indirildi ve çıkarıldı.")
