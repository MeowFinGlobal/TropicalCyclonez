from selenium import webdriver
import re
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import numpy as np

# 设置Chrome选项
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 启动无头模式
options.add_argument('--disable-gpu')

# 设置更真实的 User-Agent
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# 初始化 Selenium 驱动
s = Service("/Users/dogepro/Desktop/chromedriver-mac-x64/chromedriver")
driver = webdriver.Chrome(service=s)
random_int = np.random.randint(114514, 1919811)
url = f"https://science.nrlmry.navy.mil/geoips/tcdat/sectors/atcf_sector_file?t={random_int}"

try:
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")
    text = soup.find("pre").get_text()
    pattern = r"(\w{3,4})\s+(\w+)\s+(\d{6})\s+(\d{4})\s+([+-]?\d+\.\d+[NS])\s+([+-]?\d+\.\d+[EW])\s+(\w+)\s+(\d+)\s+(\d+)"

    for line in text.splitlines():
        match = re.match(pattern, line)
        if match:
            name = match.group(1)
            date = f"{match.group(3)} {match.group(4)}"
            location = f"{match.group(5)} {match.group(6)}"
            intensity = f"{match.group(8)}KT {match.group(9)}HPA"
            print(f"Name: {name}")
            print(f"Date: {date}")
            print(f"Location: {location}")
            print(f"Intensity: {intensity}")
            print("-" * 40)
        else:
            print("No match found in line:", line)
except Exception as e:
    print("Error fetching data from NRL:", e)

driver.quit()
