from PIL import Image, ImageSequence
import sys
import os

def gif_to_lossless_webp(input_path, output_path=None):
    # 打开 GIF 图像
    im = Image.open(input_path)

    # 如果没有提供输出路径，则使用默认名称
    if not output_path:
        output_path = os.path.splitext(input_path)[0] + ".webp"

    # 确保是动画 GIF
    if getattr(im, "is_animated", False):
        im.save(output_path, format="WEBP", lossless=True, save_all=True, duration=im.info['duration'], loop=0)
    else:
        im.save(output_path, format="WEBP", lossless=True)

    print(f"转换完成：{output_path}")

# 示例使用
if __name__ == "__main__":
    input_gif = "example.gif"  # 你的GIF文件路径
    gif_to_lossless_webp(input_gif)

from PIL import Image
import os

def webp_to_gif(input_path, output_path=None):
    im = Image.open(input_path)

    # 如果没有提供输出路径，则自动命名
    if not output_path:
        output_path = os.path.splitext(input_path)[0] + "_converted.gif"

    if getattr(im, "is_animated", False):
        im.save(output_path, format="GIF", save_all=True, duration=im.info.get("duration", 100), loop=0)
    else:
        im.save(output_path, format="GIF")

    print(f"转换完成：{output_path}")

# 示例使用
if __name__ == "__main__":
    input_webp = "example.webp"  # 你的WebP文件路径
    webp_to_gif(input_webp)