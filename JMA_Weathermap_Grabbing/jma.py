import requests
from pdf2image import convert_from_path
import datetime
import os
import numpy as np
import pytesseract
from PIL import Image
import sys

def filter_red_text(img):
    """仅保留图片中的红色部分，其他变白，用于提升红字OCR准确度"""
    img = img.convert("RGB")
    arr = np.array(img)
    # 红色像素判定条件（可适当调整）
    mask = (arr[:,:,0] > 180) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
    arr_out = np.ones_like(arr) * 255
    arr_out[mask] = arr[mask]
    return Image.fromarray(arr_out)

def ocr_with_rotation(img, psm=11, angle_list=[-20, -10, 0, 10, 20]):
    """多角度旋转OCR，返回所有结果，避免漏检倾斜字"""
    results = []
    config = f'--psm {psm}'
    for angle in angle_list:
        rotated = img.rotate(angle, expand=1, fillcolor=(255,255,255))
        text = pytesseract.image_to_string(rotated, config=config)
        results.append((angle, text))
    return results

def download_weather_chart(forecast_time):
    rand = np.random.randint(114514, 1919811)
    url = f"https://www.data.jma.go.jp/yoho/data/wxchart/quick/FSAS{forecast_time}_COLOR_ASIA.pdf?t={rand}"
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    pdf_path = f"FSAS{forecast_time}_{today}.pdf"
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

        # ======= 只检测红色区域 =======
        red_img = filter_red_text(img)
        # red_img.save(os.path.join(png_folder, f"{today}_page{i+1}_red.png"))  # 如需调试可保存
        # ======= 倾斜多角度OCR =======
        ocr_results = ocr_with_rotation(red_img, psm=11, angle_list=[-20, -10, 0, 10, 20])
        found_this_page = False
        for angle, text in ocr_results:
            if 'TD' in text.upper():  # 防止小写/识别错误
                print(f">>> 检测到“TD”字样！")
                td_found = True
                found_this_page = True
                break
        if not found_this_page:
            print("未检测到红色倾斜‘TD’。")

    os.remove(pdf_path)

    if td_found:
        print("结果：本次抓取的天气图含有‘TD’。")
    else:
        print("结果：本次抓取的天气图未检测到‘TD’。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python jma.py 24 或 48（即FSAS24、FSAS48等）")
    else:
        forecast_time = sys.argv[1]
        download_weather_chart(forecast_time)
