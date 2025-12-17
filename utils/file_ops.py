
import os, sys
import re

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
    print(f"[INFO]3 firsts: {os.listdir(folder)[:3]}")
    print(f'[SUCCES] files in folder renamed: {prefix_file}_idx{ext}')
    return 
