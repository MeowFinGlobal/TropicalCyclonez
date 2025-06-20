import numpy as np
from geopy import Point
from geopy.distance import distance
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import requests
import re
from matplotlib.collections import LineCollection

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
            sta = abs(pct_val ** 0.5 * 0) #将sta改为=0停用随机生成器
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

def generate_realistic_northbound_path(start_lat, start_lon, num_points=74, point_distance_km=80, westerlies=30): #num_points为步数，westerlies为西风槽纬度
    lats = [start_lat]
    lons = [start_lon]
    luck = random.randint(0, 10) #转向速度
    dir = 0.1 * random.randint(0, 30) #方向参数，上限越高越容易产生奇特路径
    mov = 0.1 * random.randint(-3, 0) #初始北向量
    c = 0.1 * random.randint(0, 0) #初始西向量
    tilt = random.randint(0, 0) #东行或西行，0默认西行
    if (tilt==0):
        tilt = -1
    else:
        tilt = 1

    for i in range(1, num_points):
        prev_lat = lats[-1]
        # ====== 北向量设计：westerlies附近北向陡增 ======
        # 低纬慢慢向北，高纬靠近westerlies快速加大
        if prev_lat < westerlies - 3:
            north_factor = mov + 0.04 * i  # 初期缓增
        else:
            north_factor = mov + 0.04 * i + 0.02 * ((prev_lat - (westerlies-3)) ** 1.6)
        north_distance = point_distance_km * north_factor

        # ====== 东西向设计：转向期才剧烈（如高纬才有大偏转/波动）======
        if prev_lat < westerlies - 2:
            east_noise = np.random.normal(-24, 24)
        else:
            east_noise = np.random.normal(-60, 60)  # 副热带更剧烈
        east_distance = tilt * point_distance_km * np.cos(i * dir / num_points * np.pi + c * np.pi) * 1.4 + east_noise

        total_distance = np.sqrt(north_distance**2 + east_distance**2)
        bearing = (np.degrees(np.arctan2(east_distance, north_distance)) + 360) % 360

        next_lat, next_lon = get_point_at_distance(lats[-1], lons[-1], total_distance, bearing)
        if next_lat > westerlies+10: #变性的纬度
            break
        lats.append(next_lat)
        lons.append(next_lon)
    return np.array(lats), np.array(lons)

# ========== MPI/强度相关 ==========
def calculate_mpi(sst):
    if (sst > 25.30):
        return (0.9 * ((sst + 273 - 195) / 195) * (1.2 - (sst + 273 - 283) * 0.006) * 2.27e6 * 1.67e-3 * (sst - 25.30)) ** 0.5 / 0.51444
    elif (sst > 24.30):
        return 25
    else:
        return 0

def intensity_simulation(lats, lons, nc_filename):
    # SST 数据读取
    ds = xr.open_dataset(nc_filename)
    sst = ds['analysed_sst'].isel(time=0) - 273.15

    # 用户输入初始参数（可修改）
    current_intensity = 22
    convergence = 20
    divergence = 30
    vertical_shear = 5
    westerlies = 12
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
        lat_min = lat2 - 0.25
        lat_max = lat2 + 0.25
        lon_min = lon2 - 0.25
        lon_max = lon2 + 0.25
        sst_region = sst.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        # 缺测点视为陆地设为冷水，取均值
        sst_point = float(np.mean(np.nan_to_num(sst_region.values, nan=18)))
        # ============ END =============

        mpi = calculate_mpi(sst_point)

        # 西风带处理
        if (lat2 > westerlies - 1):
            vertical_shear += max(0.2*(lat2 - (westerlies-1)), 0.4*(lat2 - (westerlies-1)))
        # 眼壁置换过程
        if (current_intensity > 120):
            ERC = random.randint(0, 100)
            if (ERC > 85):
                for a in range(0, random.randint(1, 4)):
                    ERCD = random.randint(int(0.03*current_intensity), int(0.06*current_intensity))
            else:
                ERCD = 0
        # 强度更新公式
        current_intensity += ((mpi - vertical_shear * 3 - current_intensity) / 6) * ((min(current_intensity, 70) / 70)**1.5) - (20 / (sp + 0.2)) - ERCD
        intensity_array.append(current_intensity)
    return np.array(intensity_array)

# ========== 主流程 ==========
if __name__ == "__main__":
    use_manual = input("是否手动输入lats/lons？(y/n): ").strip().lower() == "y"
    if use_manual:
        # 输入格式：以逗号分隔
        lats_str = input("请输入纬度数组（如13.0,13.5,14.0,...）: ")
        lons_str = input("请输入经度数组（如129.0,129.3,129.6,...）: ")
        # 处理输入
        lats = np.array([float(x) for x in lats_str.split(",")])
        lons = np.array([float(x) for x in lons_str.split(",")])
    else:
        if sta > 0:
            lats, lons = generate_realistic_northbound_path(sta, 139.0)
        else:
            lats, lons = generate_realistic_northbound_path(0.1*random.randint(45,96), random.randint(108,146))
    print("[INFO] 路径生成完毕，点数：", len(lats))

    # 强度模拟，读取海温文件
    nc_filename = "20191222120000-REMSS-L4_GHRSST-SSTfnd-MW_IR_OI-GLOB-v02.0-fv05.1.nc" # REMSS的SST文件，模拟真实下垫面
    intensity_array = intensity_simulation(lats, lons, nc_filename)

    # 四舍五入到最接近的5倍，筛掉低于阈值后的所有点
    intensity_rounded = []
    filtered_lats = []
    filtered_lons = []
    mslp_array = []

    threshold = 22  # 阈值

    for i, raw_intensity in enumerate(intensity_array):
        start_time = datetime.strptime("2065061200", "%Y%m%d%H") # 设置起始时间，举例2065081700
        now_time = start_time + timedelta(hours=6*i)
        ts = now_time.strftime("%Y%m%d%H")
        rounded_intensity = int(round(raw_intensity / 5) * 5)
        if rounded_intensity < threshold:
            break
        intensity_rounded.append(rounded_intensity)
        filtered_lats.append(lats[i])
        filtered_lons.append(lons[i])
        base = max(rounded_intensity / 7.4, 0.01)
        mslp = 1011 - base ** (1 / 0.648)
        mslp_array.append(int(round(mslp)))
        print(f"WP, 01, {ts},   , BEST,   0,  {lats[i]:.1f}N, {lons[i]:.1f}E,  {rounded_intensity}, {mslp_array[i]}, TD,")

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

# 生成折线的每一段
points = np.array([filtered_lons, filtered_lats]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 每段的强度，取前一节点（可用前后均值、后节点等，自定）
segment_colors = [get_color(val) for val in intensity_rounded[:-1]]

# 创建LineCollection
lc = LineCollection(segments, colors=segment_colors, linewidths=2.5, zorder=10, alpha=1, transform=ccrs.PlateCarree())

fig = plt.figure(figsize=(10, 6)) #地图
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([min(filtered_lons)-10, max(filtered_lons)+10, min(filtered_lats)-5, max(filtered_lats)+5], crs=ccrs.PlateCarree()) #范围
ax.coastlines() #海岸线
ax.add_feature(cfeature.LAND, facecolor='#e5faec', edgecolor='none') #陆地
ax.add_feature(cfeature.BORDERS, linestyle=':') #国界线
ax.gridlines(draw_labels=True, color='#666666', linestyle='--', alpha=0.1) #经纬度

ax.add_collection(lc)

# 每4个点标记一次，颜色对应强度
for i in range(0, len(filtered_lats), 4):
    ax.scatter(filtered_lons[i], filtered_lats[i], 
               color=get_color(intensity_rounded[i]), 
               s=40, zorder=12, transform=ccrs.PlateCarree())

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
]
ax.legend(handles=legend_elements, loc='best', title="Intensity Range")

# 随机命名表
names = [
    "Andromeda", "Antlia", "Apus", "Aquarius", "Aquila", "Ara", "Aries", "Auriga",
    "Boötes", "Caelum", "Camelopardalis", "Cancer", "Canes Venatici", "Canis Major", "Canis Minor",
    "Capricornus", "Carina", "Cassiopeia", "Centaurus", "Cepheus", "Cetus", "Chamaeleon",
    "Circinus", "Columba", "Coma Berenices", "Corona Australis", "Corona Borealis",
    "Corvus", "Crater", "Crux", "Cygnus", "Delphinus", "Dorado", "Draco", "Equuleus",
    "Eridanus", "Fornax", "Gemini", "Grus", "Hercules", "Horologium", "Hydra", "Hydrus",
    "Indus", "Lacerta", "Leo", "Leo Minor", "Lepus", "Libra", "Lupus", "Lynx", "Lyra",
    "Mensa", "Microscopium", "Monoceros", "Musca", "Norma", "Octans", "Ophiuchus",
    "Orion", "Pavo", "Pegasus", "Perseus", "Phoenix", "Pictor", "Pisces", "Piscis Austrinus",
    "Puppis", "Pyxis", "Reticulum", "Sagitta", "Sagittarius", "Scorpius", "Sculptor",
    "Scutum", "Serpens", "Sextans", "Taurus", "Telescopium", "Triangulum", "Triangulum Australe",
    "Tucana", "Ursa Major", "Ursa Minor", "Vela", "Virgo", "Volans", "Vulpecula"
]
name = random.choice(names)

# 计算ACE
ACE = 0.0
for w in intensity_rounded:
    if w >= 34:  # 34节（TS标准），或你可选35节
        ACE += (w ** 2) / 10000

plt.title(f"Simulated Cyclone Path & Intensity ({name}) ACE: {ACE:.4f}")
plt.show()
