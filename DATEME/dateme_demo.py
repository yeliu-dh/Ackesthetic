# dateme_demo.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageEnhance, UnidentifiedImageError
import numpy as np
import calendar, random, os
from datetime import datetime
from sklearn.cluster import KMeans
import colorsys
import argparse
import time
# import webcolors
import matplotlib
from matplotlib import colors
import shutil


#-------rename folder files!-------

def rename_folder_files(folder, prefix_file='image', exts=('.jpg', '.jpeg', '.png', '.webp')):

    # 只保留目标格式exts中的文件
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(exts)
        and os.path.isfile(os.path.join(folder, f))
    ]

    files.sort()  # 保证顺序稳定
    
    # 例如按序号重命名所有图片
    for i, filename in enumerate(files, start=1):
        # 获取文件扩展名,保持不变
        ext = os.path.splitext(filename)[1]  # 包含点，如 '.jpg'
        # 构造新文件名
        new_name = f"{prefix_file}_{i:03d}{ext}"  # image_001.jpg, image_002.jpg ...
        # 拼接完整路径
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        # 重命名
        os.rename(old_path, new_path)
    print(f'[SUCCES] files in folder renamed: {prefix_file}_idx{ext}')
    return 



# ===== HELPERS =====
def load_font(path, size, fallback_names=("DejaVuSans","Arial")):
    for p in ([path] if path else []) + list(fallback_names):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


##--------------------------------------------------------------------##
def estimate_img_saturation(img):
    # 计算饱和度
    arr = np.array(img.convert("HSV"))/255.0
    return arr[:,:,1].mean()
    
def get_rgb_saturation(rgb_or_rgba):
    r, g, b = rgb_or_rgba[:3]  # 不管是不是 4 元组，只取前 3 个
    # r, g, b = rgb
    # 归一化到 0~1
    r_f, g_f, b_f = r/255, g/255, b/255
    h, l, s = colorsys.rgb_to_hls(r_f, g_f, b_f)
    return s  # 0~1


def get_dominant_color(img, k=4):
    # 缩小图像加速
    small = img.resize((100, 100))
    data = np.array(small).reshape(-1, 3)

    # 聚类找出 K 个中心色
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    colors = kmeans.cluster_centers_  # RGB 中心
    counts = np.bincount(kmeans.labels_)

    # 最大的 label 对应主色
    dom = colors[counts.argmax()]
    return tuple(map(int, dom))



def get_main_colors(img, k=4):
    small = img.resize((100, 100))
    data = np.array(small).reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]  # 按出现频率排序（从多到少）

    main_colors = [tuple(map(int, colors[i])) for i in order]
    weights = counts[order] / counts.sum()

    return main_colors


def get_complementary_color(color):
    """
    互补色定义：在色轮（Hue Circle）上相对 180° 的颜色。
    效果：视觉对比强烈、最醒目。

    input_color : hex, rgb, rgba

    1.color=> rgba
    
    互补色原理:
    2. RGB → HLS（colorsys.rgb_to_hls）(0~1)
    3. 色相 + 0.5（即 180°） → 互补色
    4. HLS → RGB（colorsys.hls_to_rgb），输出整数 RGB (0~255)
    
    option : 返回color_to_rgba函数，设定alpha 
    """
    
    # 先统一转换成 RGBA，保证 r,g,b 都是整数
    r, g, b, _ = color_to_rgba(color, alpha=225)

    # RGB -> 0~1 范围
    r_f, g_f, b_f = r/255, g/255, b/255

    # 转 HLS
    h, l, s = colorsys.rgb_to_hls(r_f, g_f, b_f)

    # 互补色：色相 + 0.5
    h2 = (h + 0.5) % 1.0
    # l2 = 1.5 - l     # ⭐ 关键：亮度反转，否则极浅的颜色色相翻转还是很浅，因为亮度很高！
    r2, g2, b2 = colorsys.hls_to_rgb(h2, l, s)

    # 互补色算法只适合“有色彩”的颜色，不适合白 / 灰 / 近白
    
    
    # 转回整数 0~255
    r2_i, g2_i, b2_i = int(r2*255), int(g2*255), int(b2*255)
    return (r2_i, g2_i, b2_i, 225)




def color_to_rgba(color, alpha=255):
    """
    将各种格式的颜色统一转换成 RGBA tuple
    :param color: 支持格式：
        - HEX 字符串，如 "#123456" 或 "123456"
        - RGB tuple/list (R,G,B)
        - RGBA tuple/list (R,G,B,A)
    :param alpha: 透明度 0~255（RGB 或 HEX 输入时使用）
    :return: RGBA tuple (R,G,B,A)

    """
    if isinstance(color, str):
        # hex
        hex_color = color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color: {color}")
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (*rgb, alpha)

    elif isinstance(color, (tuple, list)):
        if len(color) == 3:  # RGB
            return (*color, alpha)
        elif len(color) == 4:  # RGBA
            # 可以选择覆盖 alpha，也可以保留原 alpha
            # 没有输入则覆盖，统一为完全不透明，不保留当前alpha
            return (*color[:3], alpha)
        
        else:
            raise ValueError(f"Invalid color tuple/list length: {len(color)}")
    else:
        raise TypeError(f"Unsupported color type: {type(color)}")




#---------------------------避免颜色过于接近--------------------------------
import colorsys
import math

##取cell内的rgb
def get_cell_mean_color(img, cell, margin_ratio=0.15):
    """
    img: PIL Image (RGB)
    cell: (x0, y0, x1, y1)
    margin_ratio: 内缩比例，避免取到边框
    """
    x0, y0, x1, y1 = cell
    w = x1 - x0
    h = y1 - y0

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    crop = img.crop((
        x0 + mx,
        y0 + my,
        x1 - mx,
        y1 - my
    ))

    arr = np.array(crop)
    mean_rgb = arr.mean(axis=(0, 1))[:3]

    return tuple(map(int, mean_rgb))

def rgb_to_hls(rgb):
    r, g, b = rgb
    return colorsys.rgb_to_hls(r/255, g/255, b/255)

def color_distance(rgb1, rgb2):
    # 差平方
    # 欧氏距离（RGB 空间）
    return math.sqrt(sum((a-b)**2 for a, b in zip(rgb1, rgb2)))


# def perceived_brightness(rgb):
#     r, g, b = rgb
#     # CIE / ITU-R BT.709 的相对亮度（luminance）公式:
#     return 0.2126*r + 0.7152*g + 0.0722*b


def perceived_brightness(rgb_or_rgba):
    r, g, b = rgb_or_rgba[:3]  # 不管是不是 4 元组，只取前 3 个
    # CIE / ITU-R BT.709 的相对亮度（luminance）公式, 模拟肉眼所见亮度:
    return 0.2126*r + 0.7152*g + 0.0722*b  #不能在这里除以225，阈值难改


def is_grayish(rgb, sat_thresh=0.20):
    
    """
    Docstring for is_grayish
    sat 值	含义
    < 0.08	几乎是灰 / 黑白
    < 0.12	偏灰
    > 0.25	明显有颜色

    :param rgb: Description
    :param sat_thresh: Description
    """
    s = get_rgb_saturation(rgb)
    return s < sat_thresh


def is_too_light(rgb_or_rgba, l_thresh=0.8):
    """
    brightness	视觉
    < 0.25	非常暗
    < 0.35	偏暗
    > 0.6	偏亮
        
    :param rgb: Description
    :param l_thresh: Description
    """
    r, g, b = rgb_or_rgba[:3]  # 不管是不是 4 元组，只取前 3 个
    # r, g, b = rgb
    brightness = perceived_brightness((r, g, b)) / 255.0
    return brightness > l_thresh




def brightness_contrast(rgb1, rgb2):
    return abs(
        perceived_brightness(rgb1)
        - perceived_brightness(rgb2)
    )



def is_too_similar(c1, c2,
                   dist_thresh=80,
                   bright_thresh=50):
    # 统一转rgba
    r1,g1,b1,_ = color_to_rgba(c1)
    r2,g2,b2,_ = color_to_rgba(c2)

    dist = color_distance(c1, c2)
    b_diff = abs(perceived_brightness((r1,g1,b1)) -
                 perceived_brightness((r2,g2,b2)))

    return dist < dist_thresh or b_diff < bright_thresh


def lower_sat(rgb_or_rgba, strength=0.5):
    """
    RGB → HLS
    降低 s
    HLS → RGB
   
    :param rgb: Description
    """
    # r, g, b = rgb
    r, g, b = rgb_or_rgba[:3]  # 不管是不是 4 元组，只取前 3 个
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)# ==rgb_to_hls

    s_new = s * strength

    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s_new)
    return (int(r2*255), int(g2*255), int(b2*255))


def adjust_color_by_bg_brightness(color, bg_mean_brightness, min_contrast=0.2, strength=0.5):
    """
    根据背景平均亮度调整 grid 颜色，保证亮度差 >= min_contrast（0~1）
    只降低？避免grid过亮
    :param grid_color: RGBA tuple
    :param bg_mean_brightness: 背景平均亮度，0~1
    :param min_contrast: 最小亮度差，0~1，推荐 0.3~0.5
    :return: 调整后的 RGBA tuple
    """
    # 拆解 RGBA
    r, g, b, a = color_to_rgba(color)
    r_f, g_f, b_f = r/255, g/255, b/255  # 归一化到 0~1

    # 当前 grid 亮度
    fg_brightness = perceived_brightness((r, g, b)) / 255.0

    contrast = abs(fg_brightness - bg_mean_brightness)
    if contrast >= min_contrast:
        print("[bg BR OK]ENOUGH br contrast")
        return color  # 已够对比，不动

    # 根据背景亮暗调整 grid
    if bg_mean_brightness < 0.4:
        # 背景暗 → 提亮 grid
        print(f'[TOO DARK bg] go uplight grid color!')
        target_brightness = min(bg_mean_brightness + min_contrast, 1.0)
    else:
        # 背景亮 → 降低 grid 亮度
        print(f'[TOO LIGHT bg] go dim grid color!')
        target_brightness = max(bg_mean_brightness - min_contrast, 0.0)

    # 调整比例
    # factor = target_brightness / max(fg_brightness, 1e-5)

    # r_new = min(max(int(r * factor), 0), 255)
    # g_new = min(max(int(g * factor), 0), 255)
    # b_new = min(max(int(b * factor), 0), 255)
    
    # 温和调整亮度（线性插值）
    new_brightness = fg_brightness + (target_brightness - fg_brightness) * strength
    factor = new_brightness / max(fg_brightness, 1e-5)

    r_new = min(max(int(r * factor), 0), 255)
    g_new = min(max(int(g * factor), 0), 255)
    b_new = min(max(int(b * factor), 0), 255)

    return (r_new, g_new, b_new, a)



def ensure_title_brightness(
    color,
    min_brightness=0.45,
    strength=0.6
):
    """
    保证标题颜色不至于太暗（温和提亮）
    strength = 每一次亮度修正「靠近目标亮度的力度」
     = 1.0 → 直接跳到目标亮度（很硬、很亮）
     = 0.5 → 走一半（比较自然）
     = 0.2 → 轻轻推一下（非常温和）
    如果你觉得“一提就太亮” → 就是 strength 太大

    """
    #==is dark
    
    r, g, b, a = color_to_rgba(color)
    br = perceived_brightness((r, g, b)) / 255.0

    if br >= min_brightness:
        return (r, g, b, a)

    # 目标亮度（不要拉太高）
    target_br = min(min_brightness + 0.1, 0.65)

    # 线性插值（温和）
    new_br = br + (target_br - br) * strength
    factor = new_br / max(br, 1e-5)

    r_new = min(int(r * factor), 255)
    g_new = min(int(g * factor), 255)
    b_new = min(int(b * factor), 255)

    return (r_new, g_new, b_new, a)


# def adjust_grid_color_by_bg_brightness(grid_color, bg_mean_brightness, min_contrast=0.15, max_contrast=0.4, strength=0.5):
#     """
#     根据背景平均亮度调整 grid 颜色，保证亮度差在 [min_contrast, max_contrast]（0~1）

#     :param grid_color: RGBA tuple
#     :param bg_mean_brightness: 背景平均亮度，0~1
#     :param min_contrast: 最小亮度差，0~1，推荐 0.15~0.3
#     :param max_contrast: 最大亮度差，0~1，推荐 0.5~0.7
#     :return: 调整后的 RGBA tuple
#     """
#     r, g, b, a = color_to_rgba(grid_color)
#     fg_brightness = perceived_brightness((r, g, b)) / 255.0  # 0~1

#     contrast = abs(fg_brightness - bg_mean_brightness)
#     print(f"[CONTRAST BR] grid & bg:{contrast}")
    
#     # 已够对比，不动
#     if min_contrast <= contrast <= max_contrast:
#         print("ENOUGH br contrast")
#         return grid_color

#     # 根据背景亮暗调整 grid
#     if bg_mean_brightness < 0.4:
#         # 背景暗 → 提亮 grid
#         target_brightness = min(bg_mean_brightness + min_contrast, bg_mean_brightness + max_contrast, 1.0)
#         print(f'TOO DARK bg, uplight grid color!')
#     else:
#         # 背景亮 → 降低 grid 亮度
#         target_brightness = max(bg_mean_brightness - min_contrast, bg_mean_brightness - max_contrast, 0.0)
#         print(f'TOO LIGHT bg, dim grid color!')

#     # 调整比例
#     # factor = target_brightness / max(fg_brightness, 1e-5)
#     # r_new = min(max(int(r * factor), 0), 255)
#     # g_new = min(max(int(g * factor), 0), 255)
#     # b_new = min(max(int(b * factor), 0), 255)

#     # 温和调整亮度（线性插值）
#     new_brightness = fg_brightness + (target_brightness - fg_brightness) * strength
#     factor = new_brightness / max(fg_brightness, 1e-5)

#     r_new = min(max(int(r * factor), 0), 255)
#     g_new = min(max(int(g * factor), 0), 255)
#     b_new = min(max(int(b * factor), 0), 255)
    
#     return (r_new, g_new, b_new, a)





## --------------------------------------sandbox----------------------------------------------------
    
# def pick_typography_color(dominant_rgb, sat, mean_brightness):
#     """
#     sat, mean_brightness是整张图片的饱和度和亮度
    
#     条件排在越前面，优先级越高
    
    
#     colorsys.rgb_to_hls要求0-1的输入
    
#     # r, g, b, h, l, s是!dominant color!的参数
#     h # 色相 0~1
#     l # 亮度 0~1
#     s # 饱和度 0~1

#     """
#     r, g, b = dominant_rgb
#     h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    
#     # ------------------------------------------------------
#     # CASE A：背景颜色非常灰、黑白（低饱和度）
#     # ------------------------------------------------------
#     if sat < 0.12:
#         # 黑白背景 → 使用三原色点缀
#         typ_color = random.choice(PRIMARY_ACCENTS)
#         return typ_color

#     # ------------------------------------------------------
#     # CASE B：背景偏灰，低饱和，但不是黑白
#     # ------------------------------------------------------
#     if sat < 0.25:
#         # 使用柔和亮色，让画面活起来（不刺眼）
#         typ_color = random.choice(PASTEL)
#         return typ_color

#     # ------------------------------------------------------
#     # CASE C：背景色饱和度适中～高 → compute complementary color
#     # ------------------------------------------------------
#     if s > 0.3:
#         # 获取互补色（饱和背景最稳）
#         return get_complementary_color(dominant_rgb)
            
#         # h2 = (h + 0.5) % 1.0  # 色相加 180° → 互补色
#         # r2, g2, b2 = colorsys.hls_to_rgb(h2, 0.55, 0.7)
#         # typ_color = (int(r2*255), int(g2*255), int(b2*255))
#         # return typ_color

#     # ------------------------------------------------------
#     # CASE D：背景偏亮 → 使用深色
#     # ------------------------------------------------------
#     if mean_brightness > 0.55:
#         typ_color = random.choice(DEEP)
#         return typ_color

#     # ------------------------------------------------------
#     # CASE E：背景偏暗 → 使用浅色
#     # ------------------------------------------------------
#     if mean_brightness < 0.4:
#         typ_color = random.choice(LIGHT)
#         return typ_color
#     # ------------------------------------------------------
#     # fallback：中性背景 → 使用深色
#     # ------------------------------------------------------
#     typ_color = "#222831"

#     return typ_color



## 十六进制颜色值（Hex color），Hex 本身 没有透明度信息！
PRIMARY_ACCENTS = ["#E84545", "#FFCE00", "#2F6FED"]
# PASTEL = ["#7BC8F6", "#F6D25A", "#F4A38C", "#C9E4DE", "#F1F7B5"]

DEEP = ["#0D0B0E", "#141936"]
LIGHT = ["#F3F5F7"]

def pick_typography_color(sat, mean_brightness):
    # CASE A: 黑白或极低饱和
    if sat < 0.20:
        print(f'PALE bg => strong color')
        return random.choice(PRIMARY_ACCENTS)

    if sat >= 0.25:
        if mean_brightness > 0.55:
            # 亮背景 → 深色文字
            print(f'LIGHT bg => deep typo')
            return random.choice(DEEP)
        else :        
            # 暗背景 → 浅色文字
            print(f'DEEP bg => light typo')
            return random.choice(LIGHT)
    # fallback
    return "#F3F5F7"


def find_color_name (color):
    rgb = colors.to_rgb(color)  # 返回 0~1 范围的浮点数
    # 找 CSS 颜色名中最接近的
    min_dist = float("inf")
    closest_name = None
    for name, hex_val in matplotlib.colors.cnames.items():
        r_c, g_c, b_c = colors.to_rgb(hex_val)
        dist = (r_c - rgb[0])**2 + (g_c - rgb[1])**2 + (b_c - rgb[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name







#===========================================================================================================================================================


# 大写变量 ≈ 逻辑常量（configuration / constant）
# “语义/习惯”上，通常不把 input() 的直接结果写成大写。
# 小写为临时产生的常量。比如weeks=calendar.calendar(YEAR,MONTH)




def generate_calender(index, YEAR, MONTH,OUTPUT_FOLDER, 
                      ALPHA, UNICOLOR,
                      input_path,  #sanboxACK/images_16,sanboxACK/output_dateme  
                      HIGHLIGHTS, TEXT, # 可以为none
                      SEED # 默认none
                      ):


    
    # ============================PARAMETERS=================================
    # OUTPUT
    # 重新创建空文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    output_filename=f"{index}_calender_{YEAR}_{MONTH}.png"
    outpath=os.path.join(OUTPUT_FOLDER, output_filename)


    # format
    weeks = calendar.monthcalendar(YEAR, MONTH)
    WIDTH, HEIGHT = 3840, 2160  # 16:9
    ROWS, COLS = len(weeks), 7 #日历格式   
    # MARGIN = int(WIDTH * 0.02)# 动态bleed margin/页边距, 只控制表格的页边距！
    MARGIN=0
    # LINE_WIDTH = 3
 
 
    # 可选的重要日期与背景文本
    # HIGHLIGHTS
    # TEXT
    
    
    # SEED
    if SEED !=None:
        random.seed(SEED)
    else : 
        random.seed(None)
        
    # FONT
    # 路径不是相对于script(dateme_demo)！而是相对于始发folder(Ackesthetic)
    ZAPFINO_PATH = "sandboxACK/font/Zapfino.ttf"   # 替换为本地路径（可选）
    FUTURA_PATH = "sandboxACK/font/futura_family\FuturaCyrillicBold.ttf"





    # ===============================打开背景图片 ===============================    
    if os.path.exists(input_path):#存在不一定能成功打开！
        print(f"[INPUT] path found: {input_path}!")
        try: # 若能打开=> resizeit
            bg = Image.open(input_path).convert("RGB")
            bg_ratio = bg.width / bg.height
            # orientation = "portrait" if bg.height > bg.width else "landscape"
            target_ratio = WIDTH / HEIGHT
            if bg_ratio > target_ratio:
                # 图片太宽 → 左右裁切
                new_w = int(bg.height * target_ratio)
                left = (bg.width - new_w)//2
                bg = bg.crop((left,0,left+new_w,bg.height))
            else:
                # 图片太高 → 上下裁切
                new_h = int(bg.width / target_ratio)
                top = (bg.height - new_h)//2
                bg = bg.crop((0,top,bg.width,top+new_h))
            bg = bg.resize((WIDTH, HEIGHT), Image.LANCZOS)
            print(f"[CUT] images opened and resized!")
            
        except (FileNotFoundError, UnidentifiedImageError):
            print("[ERROR] openning failure, generating gradient image!")# 打开失败或路径不存在 → 创建渐变背景
            bg = Image.new("RGB", (WIDTH, HEIGHT))
            draw = ImageDraw.Draw(bg)
            for y in range(HEIGHT):
                t = y/HEIGHT
                r = int(20 + 150*t)
                g = int(60 + 80*t)
                b = int(120 + 100*t)
                draw.line([(0,y),(WIDTH,y)], fill=(r,g,b))
    else :
        print(f"[WARNING] image path doesn't exists!")



    if bg:
        #--------------------------------调整背景-------------------------------
        #计算饱和度
        sat = estimate_img_saturation(bg)
        print(f"[BG SAT] {sat}")
        bg_mean_brightness = np.array(bg).mean()/255.0
        print(f"[BG MEAN BR] {bg_mean_brightness}")
        
                
        # #饱和度> 0.45时降低饱和度，亮度 > 0.7时降低对比度
        # if sat > 0.45 or bg_mean_brightness > 0.70:
        #     enhancer = ImageEnhance.Contrast(bg)
        #     bg = enhancer.enhance(0.85)  # 仅轻微降低
        
        # # 稍微增加模糊，突出文字
        # bg_proc = bg.filter(ImageFilter.GaussianBlur(radius=2))#高斯模糊
        bg_proc=bg
        
            
        # ------------------------------ GRID --------------------------------- 
        #直接规划好四周页边距，中间按照ROWS，COLS切分， 页边距可以为0
        grid_x0 = MARGIN
        grid_y0 = MARGIN
        grid_x1 = WIDTH - MARGIN
        grid_y1 = HEIGHT - MARGIN

        grid_w = grid_x1 - grid_x0
        grid_h = grid_y1 - grid_y0

        cell_w = grid_w / COLS #fixed
        cell_h = grid_h / ROWS #fixed

        cells = []
        for r in range(ROWS):
            for c in range(COLS):
                x0 = grid_x0 + c * cell_w
                y0 = grid_y0 + r * cell_h
                x1 = grid_x0 + (c+1) * cell_w
                y1 = grid_y0 + (r+1) * cell_h
                cells.append((int(x0), int(y0), int(x1), int(y1)))
        
        
        # ===============================选取字体颜色===============================
        # opt1:
        k=4
        dominant_color = get_dominant_color(bg_proc,k=k)
        print(f"[DOMINANT]{dominant_color}")

        # opt2:
        main_colors=get_main_colors(bg_proc)
        print(f"[MAIN] {main_colors}")
        
        # opt2:
        # typ_color = pick_typography_color(sat, bg_mean_brightness)
        # typ_color=color_to_rgba(typ_color)
        # print(f"[TYPO] {typ_color}")#已弃用
        print()
        

        
        ### ==============================init RBGA canvas================================
        canvas = Image.new("RGBA", (WIDTH, HEIGHT))  # RGBA 模式
        canvas.paste(bg_proc.convert("RGBA"), (0,0)) # 背景也需要 RGBA
        draw = ImageDraw.Draw(canvas)


        # ================================ 框线，日期字体，月份字体PARAMETRES============================
        #-----------------------------------grid----------------------------------------

        # typ_color #grid还是不能选主色，没有对比度
        
        # grid：降低透明度，重置ALPHA
        # grid_color=color_to_rgba(typ_color, alpha=ALPHA)
        # print(f"[ALPHA?] {grid_color}")
        
        grid_color=color_to_rgba(random.choice(main_colors),alpha=ALPHA) ###效果很好！！！
        print(f"[RESET ALPHA] {grid_color}")
        
        print(f"[old GRID BR] {perceived_brightness(grid_color)/225.0}\t [old GRID SAT] {get_rgb_saturation(grid_color)}")
        
        # 微调1： 若bg颜色饱和度过高，直接降低它的饱和度 和bg_sat形成反差？（略有用）

        if sat> 0.5:# SAT高也应该是提高亮度？不是减少grid对比度？
            grid_color=lower_sat(grid_color, strength=0.7)
            print(f"[TOO DENSE BG SAT] lower grid color sat :{get_rgb_saturation(grid_color)}")
        
        
        # 微调2：好像不是sat的问题,而是亮度!（）grid的亮度需要谨慎调节！不然很容易too light，随机一个
        # grid_color=adjust_grid_color_by_bg_brightness(grid_color=grid_color,
        #                                               bg_mean_brightness=bg_mean_brightness,
        #                                               min_contrast=0.15)
        # grid_color=adjust_grid_color_by_bg_brightness(grid_color, bg_mean_brightness, min_contrast=0.15, max_contrast=0.3)
        # grid_color=adjust_grid_color_by_bg_brightness(grid_color, bg_mean_brightness, min_contrast=0.15, max_contrast=0.4, strength=0.5)
        
        
        #非常有必要，否则仅降低sat+alpha？，grid还是很死很生硬!
        grid_color = adjust_color_by_bg_brightness(grid_color, bg_mean_brightness, min_contrast=0.2, strength=0.5)
        # print(f"[final GRID BR] {perceived_brightness(grid_color)/225.0}")
        print(f"[final GRID BR] {perceived_brightness(grid_color)/225.0}\t [final GRID SAT] {get_rgb_saturation(grid_color)}\n")

        
        grid_with=3 #必须是整数！
        
        
        # 统一字体线边距
        margin_inner = int(cell_w * 0.05)      
        
        
        #-----------------------------------date----------------------------------------
        # date: date & grid一体
        date_color=grid_color  
        # date_size=45
        # date_font = load_font(ZAPFINO_PATH, size=date_size)
        date_size=55
        date_font = load_font(FUTURA_PATH, size=date_size)
        
        #-----------------------------------month----------------------------------------
        # def is_light_rgba(color):
        #     #与绝对值比
        #     rgba=color_to_rgba(color)    
        #     if perceived_brightness(rgba)/225 > 0.8 :#==is_dark l_thresh=0.2
        #         #perceived br 的范围是0-225!!!
        #         return True
        #     else :
        #         return False 
            
            
        # month:   
        # 若不同色且dominant不为浅色：
        if UNICOLOR==False :
            # 和grid亮度比：
            
            if is_too_light(grid_color, l_thresh=0.6)==False : #非浅色！根据grid color调整，如果grid为浅，取撞色
                print("[NORMAL grid color] => compl month color !") 
                month_color=get_complementary_color(grid_color)#取和主色的互补色几乎不会撞

                # COMPL 若灰，取dominant
                if is_grayish(month_color, sat_thresh=0.2):
                    print(f"[PALE COML] take dominant color!")
                    month_color=dominant_color

                    if is_too_similar(month_color, grid_color):# 主导色和grid太像，直接取三原色
                        print(f"[TOO sim] grid & dominant => random a primary color")
                        month_color=random.choice(PRIMARY_ACCENTS)                    
                    

            else :# 若为浅色，取主色。
                # 在mains中取的grid color几乎没有过亮的！但是为了突出于背景的区别做了适当提亮？
                print("too LIGHT grid color => random a new month color!")                
                month_color=random.choice(main_colors) # 在主色里随机，但有可能和第一个格子相撞！
                # 如果颜色过近，rerandom
                cell_bg_rgb = get_cell_mean_color(bg_proc, cells[0])                
                tries = 0
                while True:
                    month_color = random.choice(main_colors)
                    print(f'[TOO sim] month & bg & grid (try{tries}) random AGAIN month color')
                    if not is_too_similar(month_color, cell_bg_rgb) and not is_too_similar(month_color, grid_color):
                        print(f"ENOUGH color contrast")
                        break
                    tries += 1
                    if tries > k+1:  # 防止死循环
                        month_color = get_complementary_color(grid_color)
                        break
            
    
        else:#统一颜色，取type
            month_color=grid_color#书接上文              
        
        month_color=ensure_title_brightness(month_color, min_brightness=0.5, strength=0.7)
        print(f'[AJUSTED month br] {perceived_brightness(month_color)/225.0} ')
        
                                  
        max_width = cell_w - 2 * margin_inner
        month_size = int(cell_h * 0.4)  # 初始字体高度     
        month_font = load_font(FUTURA_PATH, month_size)#*font_type
        
        month=calendar.month_name[MONTH].upper()# / month_abbr
        
        while True:
            text_bbox = draw.textbbox((0,0), month, font=month_font)
            text_width = text_bbox[2] - text_bbox[0]
            
            if text_width <= max_width or month_size <= 5:
                break
            # 如果文字太宽，递减字体直到合适
            month_size -= 1
            month_font = load_font(FUTURA_PATH, month_size)#* font_type


        # -----------------------------big_letter----------------------------
        
        big_letter_color=month_color # month & big_letter一体
        
        



        ##================================NO TOUCHY===================================    
        # ----------------------------- 绘制无缝格子表格 -------------------------------
        draw = ImageDraw.Draw(canvas)
    
        # 外框
        if MARGIN!=0:
            draw.rectangle([grid_x0, grid_y0, grid_x1, grid_y1],
                    outline=grid_color, width=grid_with) #线的粗细!

        # 垂直分割线
        for c in range(1, COLS):
            x = int(grid_x0 + c * cell_w)
            draw.line([(x, grid_y0), (x, grid_y1)], fill=grid_color, width=grid_with)

        # 水平分割线
        for r in range(1, ROWS):
            y = int(grid_y0 + r * cell_h)
            draw.line([(grid_x0, y), (grid_x1, y)], fill=grid_color, width=grid_with)
        

        # ---------------------------- 格子在左上角标注日期 ----------------------------
        for r, week in enumerate(weeks):
            for c, day in enumerate(week):
                if day == 0:
                    continue  # 空白格子不画日期
                # 计算当前格子在 cells 中的索引
                cell_idx = r * COLS + c
                x0, y0, x1, y1 = cells[cell_idx]#左上右下角

                # 左上角稍微内缩 margin_inner（字体离grid的距离）
                # margin_inner = int(min(x1-x0, y1-y0) * 0.08)  # 8% 内边距
                date_x = x0 + margin_inner # 向右
                date_y = y0 + margin_inner # 向下

                # 绘制日期
                draw.text((date_x, date_y), str(day), fill=date_color, font=date_font)
        
        
        # ------------------------- 左上第一个标注月份标题 --------------------------------
        top_left_cell = cells[0]  # 左上角格子
        x0, y0, x1, y1 = top_left_cell    
        
        # draw.text((x0 + margin_inner, y1-month_size-margin_inner*2),#贴着下边框
        #       calendar.month_name[MONTH].upper(),
        #       font=month_font,
        #       fill=month_color)    

        cell_center_x = (x0 + x1) / 2
        text_x = cell_center_x - text_width / 2
        text_y = y1 - month_size - margin_inner*2  # 保持贴下边
        draw.text((text_x, text_y), month, font=month_font, fill=month_color)



        #-----------------------------HIGHLOGHTS------------------------------
        # if HIGHLIGHTS:



        #------------------------------自定义文本-----------------------------
        if TEXT:
            #  info: weeks = calendar.monthcalendar(year, month)
            
            letters = list(TEXT.replace(" ", ""))#连成无空格str
            n = len(letters)
            
            #只分布于两行
            row1 = letters[: n//2]
            row2 = letters[n//2 :]

            #选row：row1 放在几行星期中 #在前几行/weeks选;小心上下不能越界
            # row1_week = random.choice(range(len(weeks)))
            row1_week = random.choice([1, 2, 3])
            row2_week = min(max(0, row1_week + random.choice([-1,1])), len(weeks)-1)
            # row1_week = random.choice([1, 2, 3])
            # row2_week = row1_week + random.choice([-1,1])
            
            
            # 在row上随机性选择填充的格子，但日期不可为0
            valid_cells_row1 = [d for d in weeks[row1_week] if d != 0]
            valid_cells_row2 = [d for d in weeks[row2_week] if d != 0]

            # row1_cells = random.sample(weeks[row1_week], len(row1))
            # row2_cells = random.sample(weeks[row2_week], len(row2))

            # 统计cells_letters
            cells_letters = []

            for day in valid_cells_row1:
                # 找到 day 在 weeks[row1_week] 的列索引
                col_idx = weeks[row1_week].index(day)
                cell_idx = row1_week * COLS + col_idx
                cells_letters.append(cells[cell_idx])

            for day in valid_cells_row2:
                col_idx = weeks[row2_week].index(day)
                cell_idx = row2_week * COLS + col_idx
                cells_letters.append(cells[cell_idx])


            for idx, ch in enumerate(letters[:n]):
                x0, y0, x1, y1 = cells[idx]
                cell_w = x1 - x0
                cell_h = y1 - y0

                # 字体大小
                big_letter_font_size = int(min(cell_w, cell_h) * 0.9)
                # use_zapf = random.random() < 0.5
                big_letter_font = load_font(FUTURA_PATH, big_letter_font_size)# 根据cell大小，自动调整fontsize
                
                
                # 超出的部分消失
                
                
                # 中心 + 偏移
                cx = x0 + cell_w // 2
                cy = y0 + cell_h // 2
                ox = int((random.random() - 0.5) * cell_w * 0.24)
                oy = int((random.random() - 0.5) * cell_h * 0.24)
                pos = (cx + ox, cy + oy)

                # 绘制图层
                txt_layer = Image.new("RGBA", canvas.size, (255,255,255,0))
                td = ImageDraw.Draw(txt_layer)
                td.text(pos, ch, font=big_letter_font, fill=month_color)

                # 可旋转
                angle = random.uniform(-8,8)
                txt_layer = txt_layer.rotate(angle, resample=Image.BICUBIC, center=pos)

                # 合成
                canvas = Image.alpha_composite(canvas.convert("RGBA"), txt_layer).convert("RGB")
                draw = ImageDraw.Draw(canvas, "RGBA")



    # ================================ SAVE =================================
    canvas.save(outpath, quality=95)
    print(f"[OUTPUT] image saved to {outpath}!")
    print("-"*80)

    pass



    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="generate canlender")
        
    # ================== NLI INPUT ==================

    #ps.在nli中的- 会被自动转换成_
    ##require=True会忽略default
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument('--alpha', type=int, default=120)
    parser.add_argument("--unicolor", type=bool, default=False)
    
    parser.add_argument("--input-folder", type=str, default="sandboxACK/images_16_2")
    parser.add_argument("--output-folder", type=str, default=None)
    
    parser.add_argument("--highlights", type=list, default=None)
    parser.add_argument("--text", type=str)    
    
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    
        
    # ================== CONFIG ==================
    YEAR = args.year
    # MONTH = args.month
    START_MONTH=args.start_month
    
    ALPHA = args.alpha
    UNICOLOR= args.unicolor
     
    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder
    if OUTPUT_FOLDER==None:
            OUTPUT_FOLDER=os.path.join(os.path.dirname(INPUT_FOLDER), 'output_'+os.path.basename(INPUT_FOLDER))
    
    # 如果文件夹存在，先删除
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    print(f"[CLEAN] clean output folder :{OUTPUT_FOLDER}")
    
    
    HIGHLIGHTS=args.highlights
    TEXT=args.text
    
    SHUFFLE=args.shuffle
    SEED=args.seed
    
    # GRID_ALPHA = args.alpha



    #--------------------INPUT_FILES-----------------
    start_time=time.time()
    #给input_folder中的文件重新按顺序命名，不改变文件格式

    rename_folder_files(INPUT_FOLDER, prefix_file='image')
    
    files=os.listdir(INPUT_FOLDER)
    if SHUFFLE==True:
        random.shuffle(files)   # 原地打乱顺序

    for i, f in enumerate(files):
    
        input_path=os.path.join(INPUT_FOLDER, f)    
        
        MONTH=START_MONTH+i
        if MONTH>12:
            MONTH=MONTH-12
            
        print(f"\n[IDX] calendar {MONTH}/{YEAR}:")
        generate_calender(index=i,YEAR=YEAR,MONTH=MONTH, OUTPUT_FOLDER=OUTPUT_FOLDER,
                    ALPHA=ALPHA, UNICOLOR=UNICOLOR,
                    input_path=input_path,
                    HIGHLIGHTS=HIGHLIGHTS, TEXT=TEXT,
                    SEED=SEED)
        # if i > 2:
        #     break # try the first ones
        
          
    end_time=time.time()

    print(f"[SUCCES] {len(files)} calendar generation done in {end_time-start_time:.2f} sec!")    
    # except Exception as e :
    #     print(f"[ERROR] {e}!")
    
if __name__ == "__main__":
    main()






