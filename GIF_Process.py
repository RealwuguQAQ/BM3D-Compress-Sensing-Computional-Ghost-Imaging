import imageio.v2 as imageio
import glob
import os

# 存放 PNG 的文件夹
img_dir = "reconstruction_frames"
# 输出 GIF 的文件名
gif_path = "reconstruction.gif"

# 找到所有符合条件的 PNG，按文件名排序
file_list = sorted(glob.glob(os.path.join(img_dir, "iter_*.png")))

# 读取所有帧
frames = [imageio.imread(fname) for fname in file_list]

# 保存为 GIF
imageio.mimsave(gif_path, frames, fps=5)  # fps=5 表示每秒 5 帧

print(f"GIF 已保存到 {gif_path}, 共 {len(frames)} 帧")
