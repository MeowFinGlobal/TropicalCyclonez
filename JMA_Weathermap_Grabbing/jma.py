import requests
from pdf2image import convert_from_path
import datetime
import os
import numpy as np
import pytesseract
from PIL import Image

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
    td_found = False
    for i, img in enumerate(images):
        png_path = os.path.join(png_folder, f"{today}_page{i+1}.png")
        img.save(png_path, "PNG")
        print(f"已保存 PNG: {png_path}")

        # 进行OCR识别
        text = pytesseract.image_to_string(img)
        print(f"OCR识别内容:\n{text}")
        if 'TD' in text:
            print(">>> 检测到“TD”字样！")
            td_found = True
        else:
            print("未检测到“TD”。")

    # 可选：下载完后删除 PDF
    os.remove(pdf_path)

    if td_found:
        print("结果：本次抓取的天气图含有“TD”。")
    else:
        print("结果：本次抓取的天气图未检测到“TD”。")

download_weather_chart()
