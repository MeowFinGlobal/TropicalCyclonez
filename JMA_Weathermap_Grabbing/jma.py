import requests
from pdf2image import convert_from_path
import datetime
import os
import numpy as np

def download_weather_chart():
    rand = np.random.randint(114514, 1919811)
    url = f"https://www.data.jma.go.jp/yoho/data/wxchart/quick/FSAS48_COLOR_ASIA.pdf?t={rand}"
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    pdf_path = f"FSAS24_{today}.pdf"
    png_folder = "charts"

    os.makedirs(png_folder, exist_ok=True)

    # 下载 PDF
    response = requests.get(url)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print(f"已下载: {pdf_path}")

    # 转换为 PNG（默认第一页）
    images = convert_from_path(pdf_path, dpi=200)
    for i, img in enumerate(images):
        png_path = os.path.join(png_folder, f"{today}_page{i+1}.png")
        img.save(png_path, "PNG")
        print(f"已保存 PNG: {png_path}")

    # 可选：下载完后删除 PDF
    os.remove(pdf_path)

download_weather_chart()