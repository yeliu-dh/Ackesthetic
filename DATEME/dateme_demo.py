# dateme_demo.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageEnhance
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


# ===== HELPERS =====
def load_font(path, size, fallback_names=("DejaVuSans","Arial")):
    for p in ([path] if path else []) + list(fallback_names):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

def estimate_saturation(img):
    # 计算饱和度
    arr = np.array(img.convert("HSV"))/255.0
    return arr[:,:,1].mean()
    


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

PRIMARY_ACCENTS = ["#E84545", "#FFCE00", "#2F6FED"]
PASTEL = ["#7BC8F6", "#F6D25A", "#F4A38C", "#C9E4DE", "#F1F7B5"]
DEEP = ["#020B16", "#141936"]
LIGHT = ["#E1E6E7", "#F3F5F7"]




def pick_typography_color(dominant_rgb, sat, mean_brightness):
    """
    自动选择适合背景的 typographic 主色
    越前面，优先级越高
    
    """
    r, g, b = dominant_rgb
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)

    # ------------------------------------------------------
    # CASE A：背景颜色非常灰、黑白（低饱和度）
    # ------------------------------------------------------
    if sat < 0.12:
        # 黑白背景 → 使用三原色点缀
        typ_color = random.choice(PRIMARY_ACCENTS)
        return typ_color

    # ------------------------------------------------------
    # CASE B：背景偏灰，低饱和，但不是黑白
    # ------------------------------------------------------
    if sat < 0.25:
        # 使用柔和亮色，让画面活起来（不刺眼）
        typ_color = random.choice(PASTEL)
        return typ_color

    # ------------------------------------------------------
    # CASE C：背景色饱和度适中～高 → compute complementary color
    # ------------------------------------------------------
    if s > 0.3:
        # 互补色（饱和背景最稳）
        h2 = (h + 0.5) % 1.0
        r2, g2, b2 = colorsys.hls_to_rgb(h2, 0.55, 0.7)
        typ_color = (int(r2*255), int(g2*255), int(b2*255))
        return typ_color

    # ------------------------------------------------------
    # CASE D：背景偏亮 → 使用深色
    # ------------------------------------------------------
    if mean_brightness > 0.55:
        typ_color = random.choice(DEEP)
        return typ_color

    # ------------------------------------------------------
    # CASE E：背景偏暗 → 使用浅色
    # ------------------------------------------------------
    if mean_brightness < 0.4:
        typ_color = random.choice(LIGHT)
        return typ_color
    # ------------------------------------------------------
    # fallback：中性背景 → 使用深色
    # ------------------------------------------------------
    typ_color = "#222831"

    return typ_color

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


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




def pick_color_system(dominant_rgb, sat, mean_brightness, unicolor=False):
    """
    返回一套颜色体系:
    - typ_color : 主字体颜色
    - grid_color : 网格线颜色
    - big_letter_color : 大字母点缀颜色
    如果 unicolor=True，全局统一一个颜色
    """
    # ----------------- unicolor模式 -----------------
    if unicolor:
        # 可以选一个深色或亮色作为统一颜色
        if mean_brightness > 0.5:
            color_hex = "#141936"  # 亮背景用深色
        else:
            color_hex = "#E1E6E7"  # 暗背景用浅色
        return color_hex, color_hex, color_hex

    # ----------------- 动态配色模式 -----------------
    r, g, b = dominant_rgb
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)

    # 1️⃣ 主字体颜色 typ_color
    if sat < 0.12:
        typ_color = random.choice(PRIMARY_ACCENTS)
    elif sat < 0.25:
        typ_color = random.choice(PASTEL)
    elif s > 0.3:
        h2 = (h + 0.5) % 1.0  # 互补色
        r2, g2, b2 = colorsys.hls_to_rgb(h2, 0.55, 0.7)
        typ_color = (int(r2*255), int(g2*255), int(b2*255))
    elif mean_brightness > 0.55:
        typ_color = random.choice(DEEP)
    elif mean_brightness < 0.4:
        typ_color = random.choice(LIGHT)
    else:
        typ_color = "#222831"

    # 2️⃣ 网格线颜色 grid_color (半透明)
    # 判断typ_color亮度
    if isinstance(typ_color, tuple):
        brightness = sum(typ_color)/3 / 255
    else:
        typ_rgb = tuple(int(typ_color[i:i+2],16) for i in (1,3,5))
        brightness = sum(typ_rgb)/3 / 255

    if brightness > 0.6:
        grid_color = (0,0,0,60)   # 黑半透明
    else:
        grid_color = (255,255,255,60)  # 白半透明

    # 3️⃣ 大字母点缀颜色 big_letter_color
    # 比主字体颜色稍微调亮或透明
    if isinstance(typ_color, tuple):
        big_letter_color = tuple(min(255,int(c*1.1)) for c in typ_color)
    else:
        # hex to RGB
        r0, g0, b0 = tuple(int(typ_color[i:i+2],16) for i in (1,3,5))
        big_letter_color = (min(255,int(r0*1.1)), min(255,int(g0*1.1)), min(255,int(b0*1.1)))

    return typ_color, grid_color, big_letter_color










def generate_calender(year, month, 
                      bg_path, output_folder, # 可默认
                      highlights, bg_text, # 可以为none
                      seed # 默认none
                      ):
    
    # ======================PARAMETERS=============================
    ## 日历最重要的年月
    YEAR = year
    MONTH = month
    weeks = calendar.monthcalendar(year, month)

    # input & output 
    if bg_path==None:    
        BG_PATH = "DATEME\sandboxDATE\images_16\image_001.png" # image_demo


    if output_folder==None:
        OUTPUT_FOLDER=f"DATEME/sandboxDATE/output"
    else :#有输入
        OUTPUT_FOLDER=output_folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                
    filename=f"calender_{year}_{month}.png"
    OUT_PATH=os.path.join(OUTPUT_FOLDER, filename)

    # os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # format
    WIDTH, HEIGHT = 3840, 2160  # 16:9
    ROWS, COLS = len(weeks), 7 #日历格式   
    # MARGIN = int(WIDTH * 0.02)# bleed margin/页边距
    MARGIN=0
    LINE_WIDTH = 3
 
    # 可选的重要日期与背景文本
    HIGHLIGHTS = highlights
    BG_TEXT = bg_text
   
    # 文本位置的随机性
    if seed !=None:
        random.seed(seed)
    else : 
        random.seed(None)
        
    # font
    ZAPFINO_PATH = "../sandboxACK/font/Zapfino.ttf"   # 替换为本地路径（可选）
    FUTURA_PATH = "sandboxACK/font/futura_family\FuturaCyrillicBold.ttf"

    # zapfino = load_font(ZAPFINO_PATH, size=220)
    # futura = load_font(FUTURA_PATH, size=50)


    # --------------------- CROP BACKGROUND -----------------------
    if os.path.exists(BG_PATH):
        bg = Image.open(BG_PATH).convert("RGB")
        # 打开图片并统一转换成 RGB 模式（确保后续处理不会报错）
        
        # 左右裁切成16:9保留中间区域
        bg_ratio = bg.width / bg.height
        target_ratio = WIDTH / HEIGHT
        if bg_ratio > target_ratio:
            new_w = int(bg.height * target_ratio)
            left = (bg.width - new_w)//2
            bg = bg.crop((left,0,left+new_w,bg.height))
         
        # 如果没有背景图，生成渐变背景
        else:
            new_h = int(bg.width / target_ratio)
            top = (bg.height - new_h)//2
            bg = bg.crop((0,top,bg.width,top+new_h))
        bg = bg.resize((WIDTH, HEIGHT), Image.LANCZOS)
    
    else:
        # create demo gradient background
        bg = Image.new("RGB", (WIDTH, HEIGHT))
        draw = ImageDraw.Draw(bg)
        for y in range(HEIGHT):
            t = y/HEIGHT
            r = int(20 + 150*t)
            g = int(60 + 80*t)
            b = int(120 + 100*t)
            draw.line([(0,y),(WIDTH,y)], fill=(r,g,b))
            
    # orientation
    orientation = "portrait" if bg.height > bg.width else "landscape"



    #计算饱和度
    #-----------------------调整背景----------------------
    sat = estimate_saturation(bg)
    mean_brightness = np.array(bg).mean()/255.0
    
    #饱和度> 0.45时降低饱和度，亮度 > 0.7时降低对比度
    if sat > 0.45 or mean_brightness > 0.70:
        enhancer = ImageEnhance.Contrast(bg)
        bg = enhancer.enhance(0.85)  # 仅轻微降低
    
    # 稍微增加模糊，突出文字
    bg_proc = bg.filter(ImageFilter.GaussianBlur(radius=4))#高斯模糊


    #-------------------选取字体颜色-----------------------
    dominant = get_dominant_color(bg_proc)
    typ_color = pick_typography_color(dominant, sat, mean_brightness)
    # color_name = find_color_name(typ_color)
    print(f"🌈[COLOR] {typ_color}")
    
    
    # --------------------- GRID ----------------------- 
    #直接规划好四周页边距，中间按照ROWS，COLS切分， 页边距可以为0
    grid_x0 = MARGIN
    grid_y0 = MARGIN
    grid_x1 = WIDTH - MARGIN
    grid_y1 = HEIGHT - MARGIN

    grid_w = grid_x1 - grid_x0
    grid_h = grid_y1 - grid_y0

    cell_w = grid_w / COLS
    cell_h = grid_h / ROWS

    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            x0 = grid_x0 + c * cell_w
            y0 = grid_y0 + r * cell_h
            x1 = grid_x0 + (c+1) * cell_w
            y1 = grid_y0 + (r+1) * cell_h
            cells.append((int(x0), int(y0), int(x1), int(y1)))
    
    
    
    ### ==========================init canvas=============================
    canvas = Image.new("RGB", (WIDTH, HEIGHT))  # 新建画布
    canvas.paste(bg_proc, (0,0))               # 把背景贴上去
    draw = ImageDraw.Draw(canvas, "RGBA")

    
    # -------------------------- 绘制无缝格子表格 --------------------------
    draw = ImageDraw.Draw(canvas)
    # if typ_color.startswith("#"):
    #     typ_color_rgb = hex_to_rgb(typ_color)  # (34, 40, 49)
    # else :
    #     typ_color_rgb=typ_color
    # alpha=100#降低grid的透明度
    # grid_color=(*typ_color_rgb, alpha)  
    grid_color=typ_color
    grid_with=3 #必须是整数！
 
    # # 外框
    # draw.rectangle([grid_x0, grid_y0, grid_x1, grid_y1],
    #             outline=grid_color, width=grid_with) #线的粗细!

    # 垂直分割线
    for c in range(1, COLS):
        x = int(grid_x0 + c * cell_w)
        draw.line([(x, grid_y0), (x, grid_y1)], fill=grid_color, width=grid_with)

    # 水平分割线
    for r in range(1, ROWS):
        y = int(grid_y0 + r * cell_h)
        draw.line([(grid_x0, y), (grid_x1, y)], fill=grid_color, width=grid_with)
    

    # ------------------------ 每个格子在左上角标注日期 ----------------------
    for r, week in enumerate(weeks):
        for c, day in enumerate(week):
            if day == 0:
                continue  # 空白格子不画日期
            # 计算当前格子在 cells 中的索引
            cell_idx = r * COLS + c
            x0, y0, x1, y1 = cells[cell_idx]

            # 左上角稍微内缩 margin
            margin_inner = int(min(x1-x0, y1-y0) * 0.08)  # 8% 内边距
            date_x = x0 + margin_inner
            date_y = y0 + margin_inner

            # 绘制日期
            date_font = load_font(FUTURA_PATH, size=80)# 用ZAP不行？
            draw.text((date_x, date_y), str(day), fill=typ_color, font=date_font)
    
    
    
    # --------------------------- MONTH TITLE --------------------------------
    top_left_cell = cells[0]  # 左上角格子
    x0, y0, x1, y1 = top_left_cell
    cell_w = x1 - x0
    cell_h = y1 - y0
    
    month_size = int(cell_h * 0.4)  # 高度的60%作为字体大小
    month_font = load_font(FUTURA_PATH, month_size)
    
    margin_inner = int(cell_w * 0.05)
    draw.text((x0 + margin_inner, y1-month_size-margin_inner*2),#贴着下边框
          calendar.month_name[MONTH].upper(),
          font=month_font,
        #   fill=(255,0,0,255))  # 红色+完全不透明
          fill=typ_color)    

    #----------------------------------BG_TEXT-----------------------------
    if BG_TEXT:
        #  info: weeks = calendar.monthcalendar(year, month)
        
        letters = list(BG_TEXT.replace(" ", ""))#连成无空格str
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
            td.text(pos, ch, font=big_letter_font, fill=typ_color)

            # 可旋转
            angle = random.uniform(-8,8)
            txt_layer = txt_layer.rotate(angle, resample=Image.BICUBIC, center=pos)

            # 合成
            canvas = Image.alpha_composite(canvas.convert("RGBA"), txt_layer).convert("RGB")
            draw = ImageDraw.Draw(canvas, "RGBA")



    # ===== SAVE =====
    canvas.save(OUT_PATH, quality=95)
    print(f"🗓️[SAVE] image saved to {OUT_PATH}!")

    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="generate canlender")
    #在nli中的- 会被自动转换成_
    parser.add_argument("--year", type=int, default=2026, help="年份")
    parser.add_argument("--month", type=int, default=3, help="月份")

    parser.add_argument("--bg-path", type=str, default=None, help="背景图片路径")
    parser.add_argument("--output-folder", type=str, default=None , help="输出路径")

    parser.add_argument("--highlights", type=list, default="重要日期")
    parser.add_argument("--bg-text", type=str, default=None, help="大字文本")    

    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    
    args = parser.parse_args()
    
    #-----------------------------------------------------------------------------
    start_time=time.time()
    try :
        generate_calender(year=args.year, month=args.month,
                    bg_path=args.bg_path, output_folder=args.output_folder,
                    highlights=args.highlights, bg_text=args.bg_text,
                    seed=args.seed)  
        end_time=time.time()
        print(f"✅ [SUCCES] calendar generation done in {end_time-start_time:.2f} sec!")
    except Exception as e :
        print(f"[ERROR] {e}!")

    
if __name__ == "__main__":
    main()






