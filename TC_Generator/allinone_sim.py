import numpy as np
from geopy import Point
from geopy.distance import distance
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import requests
import re

# ========== 生成位置种子，用BTC波幅实现伪随机 ==========
url = "https://stooq.com/q/?s=btcusd"
headers = {'User-Agent': 'Mozilla/5.0'}
resp = requests.get(url, headers=headers, timeout=10)
sta = 0

title_match = re.search(r'<title>(.*?)</title>', resp.text, re.IGNORECASE)
if title_match:
    title = title_match.group(1)
    pct_match = re.search(r'\(([+-]?[0-9.,]+)%\)', title)
    if pct_match:
        pct_str = pct_match.group(1).replace(',', '')  # 防止小数点为逗号
        try:
            pct_val = float(pct_str)
            sta = abs(pct_val ** 0.5 * 10) #将sta改为=0停用随机生成器
            print(f"sta = {sta}")
        except Exception as e:
            print(f"转换为数字出错: {e}")
    else:
        print("未能在<title>中找到涨幅百分比")
else:
    print("未找到<title>标签")

# ========== 路径生成 ==========
def get_point_at_distance(start_lat, start_lon, distance_km, bearing_deg):
    start_point = Point(start_lat, start_lon)
    destination = distance(kilometers=distance_km).destination(point=start_point, bearing=bearing_deg)
    return destination.latitude, destination.longitude

def generate_realistic_northbound_path(start_lat, start_lon, num_points=74, point_distance_km=80): #num_points为步数
    lats = [start_lat]
    lons = [start_lon]
    luck = random.randint(0, 10) #转向速度
    dir = 0.1 * random.randint(0, 30) #方向参数，上限越高越容易产生奇特路径
    mov = 0.1 * random.randint(-3, 6) #初始北向量
    tilt = random.randint(0, 0) #东行或西行，0默认西行
    if (tilt==0):
        tilt = -1
    else:
        tilt = 1

    for i in range(1, num_points):
        north_distance = point_distance_km * (mov + 0.0002 * luck * i**2)
        east_distance = tilt * point_distance_km * np.cos(i * dir / num_points * np.pi) * 1.4
        east_distance += np.random.normal(-36, 36) #随机运动
        total_distance = np.sqrt(north_distance**2 + east_distance**2)
        bearing = (np.degrees(np.arctan2(east_distance, north_distance)) + 360) % 360
        next_lat, next_lon = get_point_at_distance(lats[-1], lons[-1], total_distance, bearing)
        if next_lat > 41: #变性的纬度
            break
        lats.append(next_lat)
        lons.append(next_lon)
    return np.array(lats), np.array(lons)

# ========== MPI/强度相关 ==========
def calculate_mpi(sst):
    if (sst > 25.30):
        return (0.9 * ((sst + 273 - 195) / 195) * (1.2 - (sst + 273 - 283) * 0.006) * 2.27e6 * 1.67e-3 * (sst - 25.30)) ** 0.5 / 0.51444
    elif (sst > 15):
        return 25.30
    else:
        return 22

def intensity_simulation(lats, lons, nc_filename):
    # SST 数据读取
    ds = xr.open_dataset(nc_filename)
    sst = ds['analysed_sst'].isel(time=0) - 273.15

    # 用户输入初始参数（可修改）
    current_intensity = 22
    convergence = 20
    divergence = 30
    vertical_shear = 10
    westerlies = 33.0
    ERCD = 0

    intensity_array = []

    for i in range(1, len(lats)):
        if i == 0:
            lat1, lon1 = lats[0], lons[0]
        else:
            lat1, lon1 = lats[i-1], lons[i-1]
        lat2, lon2 = lats[i], lons[i]
        dx = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))
        dy = (lat2 - lat1) * 111
        sp = (dx**2 + dy**2)**0.5

        # ============ 取1度x1度均值 =============
        lat_min = lat2 - 0.5
        lat_max = lat2 + 0.5
        lon_min = lon2 - 0.5
        lon_max = lon2 + 0.5
        sst_region = sst.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        # 排除缺测点，取均值
        sst_point = float(np.nanmean(sst_region.values))
        # ============ END =============

        mpi = calculate_mpi(sst_point)

        # 西风带处理
        if (lat2 > westerlies - 1):
            vertical_shear += max(0, 0.1*(lat2 - (westerlies-1)))
        # 眼壁置换过程
        if (current_intensity > 120):
            ERC = random.randint(0, 100)
            if (ERC > 85):
                for a in range(0, random.randint(1, 4)):
                    ERCD = random.randint(int(0.03*current_intensity), int(0.06*current_intensity))
            else:
                ERCD = 0

        # 强度更新公式
        current_intensity += ((mpi - vertical_shear * 3 - current_intensity) / 6) * ((min(current_intensity, 70) / 70)**2) - (20 / (sp + 0.2)) - ERCD
        intensity_array.append(current_intensity)
    return np.array(intensity_array)

# ========== 主流程 ==========
if __name__ == "__main__":
    # 路径生成（可更改起点）
    if(sta>0):
        lats, lons = generate_realistic_northbound_path(sta, 139.0)
    else:
        lats, lons = generate_realistic_northbound_path(13.0, 129.0)
    print("[INFO] 路径生成完毕，点数：", len(lats))

    # 强度模拟，读取海温文件
    nc_filename = "20090817120000-REMSS-L4_GHRSST-SSTfnd-MW_IR_OI-GLOB-v02.0-fv05.1.nc" # REMSS的SST文件，模拟真实下垫面
    intensity_array = intensity_simulation(lats, lons, nc_filename)

    # 四舍五入到最接近的5倍，筛掉低于阈值后的所有点
    intensity_rounded = []
    filtered_lats = []
    filtered_lons = []
    mslp_array = []

    threshold = 22  # 阈值

    for i, raw_intensity in enumerate(intensity_array):
        print(f"点{i} 纬度{lats[i]:.2f} 经度{lons[i]:.2f} 强度{raw_intensity:.2f}")
        rounded_intensity = int(round(raw_intensity / 5) * 5)
        if rounded_intensity < threshold:
            break
        intensity_rounded.append(rounded_intensity)
        filtered_lats.append(lats[i])
        filtered_lons.append(lons[i])
        base = max(rounded_intensity / 7.4, 0.01)
        mslp = 1011 - base ** (1 / 0.648)
        mslp_array.append(int(round(mslp)))

    if len(filtered_lats) == 0:
        print("所有路径点都被屏蔽，请调整起点、路径参数或阈值！")
        exit()

    # 输出 NumPy 数组格式
    print("lats = np.array([", ", ".join(f"{i:.2f}" for i in filtered_lats), "])")
    print("lons = np.array([", ", ".join(f"{i:.2f}" for i in filtered_lons), "])")
    print("Intens = np.array([", ", ".join(str(i) for i in intensity_rounded), "])")
    print("MSLP = np.array([", ", ".join(str(p) for p in mslp_array), "])")

# 定义颜色区间与对应颜色
def get_color(val):
    if val <= 20:
        return 'gray'
    elif 25 <= val <= 30:
        return 'green'
    elif 35 <= val <= 60:
        return 'blue'
    elif 65 <= val <= 80:
        return 'yellow'
    elif 85 <= val <= 95:
        return 'orange'
    elif 100 <= val <= 110:
        return 'red'
    elif 115 <= val <= 135:
        return 'pink'
    elif val >= 140:
        return 'purple'
    else:
        return 'gray'

# 每点颜色
point_colors = [get_color(val) for val in intensity_rounded]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([min(filtered_lons)-10, max(filtered_lons)+10, min(filtered_lats)-5, max(filtered_lats)+5], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True)

# 按顺序plot整条轨迹线（统一色，或可按最高色分段连线，简单起见先统一黑色）
ax.plot(filtered_lons, filtered_lats, '-k', linewidth=1.2, label='Path')

# 彩色散点
for x, y, c in zip(filtered_lons, filtered_lats, point_colors):
    ax.scatter(x, y, color=c, s=60, marker='o', edgecolor='k', zorder=10)

# 标记起止
ax.scatter(filtered_lons[0], filtered_lats[0], color='green', label='Start', s=70, zorder=12)
ax.scatter(filtered_lons[-1], filtered_lats[-1], color='red', label='End', s=70, zorder=12)

# 自定义legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', edgecolor='k', label='DB'),
    Patch(facecolor='green', edgecolor='k', label='TD'),
    Patch(facecolor='blue', edgecolor='k', label='TS'),
    Patch(facecolor='yellow', edgecolor='k', label='C1'),
    Patch(facecolor='orange', edgecolor='k', label='C2'),
    Patch(facecolor='red', edgecolor='k', label='C3'),
    Patch(facecolor='pink', edgecolor='k', label='C4'),
    Patch(facecolor='purple', edgecolor='k', label='C5'),
    Patch(facecolor='white', edgecolor='green', label='Start'),
    Patch(facecolor='white', edgecolor='red', label='End'),
]
ax.legend(handles=legend_elements, loc='best', title="Intensity Range")

plt.title("Simulated Cyclone Path & Intensity (Discrete Colors)")
plt.show()
