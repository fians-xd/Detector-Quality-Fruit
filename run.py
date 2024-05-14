import roboflow as rf
import time
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.font_manager as fm

# Warna
ht = ('\x1b[38;5;40m') # hijau terang
b  = ('\x1b[0;36m') # biru gelap
bt = ('\x1b[36;1m') # biru terang
m  = ('\x1b[31;1m') # merah
p  = ('\x1b[37;1m') # putih
h  = ('\x1b[30;1m') # hitam
o  = ('\x1b[33;1m') # oren
kt  = ('\x1b[1;33m') # kuning terang
c  = ('\x1b[38;5;172m') # Coklat terang
b  = ('\x1b[0;34m') # biru tua
u  = ('\x1b[38;5;135m') # ungu
n  = ('\x1b[0;0m') # normal
mc = ('\x1b[38;5;52m') # Merah Coklat
pk = ('\x1b[38;5;207m') # pink
pn = ('\x1b[38;5;86m') # pesan

# Logoku
# Cetak Logo
speed_logo = 0.0010
def berjalan_teks(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(speed_logo)

logoku = (f"""
                (`.        ,-,
  {bt}1.0 version{n}   ` `.    ,;' /
                 `.  ,'/ .'
                  `. X /.'
        .-;--''--.._` ` (
      .'            /   `
     ,           ` '   {m}@{n} '
     ,         ,   `._    )      {ht}\/'{n}
  ,.|         '     `-.;_'       {ht}\/'{n}
  :  . `  ;    `  ` --,.._;     {ht}_/|{n}
   ' `    ,   )   .'           {o}(,;){n}
      `._ ,  '   /_            {o}(,.){n}
         ; ,''-,;' ``-         {o}(,/{n}
          ``-..__``--`         {o}|/{n}

{bt}+ -- --={m}[{b}      Auth Script: By Yan-xd        {m}]{n}
{bt}+ -- --={m}[{b} Program Detect&Check Fruit Quality {m}]{n}
\n""")

# Replace with your actual Roboflow API key
api_key = "anvuqAgs3o2rBZURMdRf"

# Connect to Roboflow
rf = rf.Roboflow(api_key=api_key)

# Model-Model yang bagus
# 1. fruit-freshness-detection-08shj Gunakan versi (8)
# 2. (freshness-fruits-and-vegetables) Gunakan versi (7)
# 3. Lainya dicoba sampai nemu yang bagus
# 4. webnya (https://universe.roboflow.com/)

# Access project and model (assuming project name and version)
project = rf.workspace().project("rotten-fruit-detector-ver-2") # Masukan model yang bagus disini
model = project.version(3).model # Masukan angka versi nya disini

# Cetak Logo
os.system("clear")
berjalan_teks(logoku)

# Get image path from user
foto = input(f"{pn}Poto Ges Lebokno:{n} ")

# Read the image
image = cv2.imread(foto)

# Maximize the size of the image
max_size = 2048
height, width, _ = image.shape
if max(height, width) > max_size:
    scale_factor = max_size / max(height, width)
    image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

# Make prediction with confidence and overlap thresholds
result = model.predict(image, confidence=45, overlap=35).json()

# Extract labels and bounding boxes
detections = result["predictions"]

# Create a figure and axis with black background
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('black')

# Add stars background
stars = np.random.randint(0, 256, size=(1000, 1000, 3))
ax.imshow(stars, extent=[-0.5, 9.5, -0.5, 9.5], alpha=0.3)

# Display the image
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Add title with custom font
font_path = "/usr/share/fonts/Super Creamy Personal Use.ttf"
custom_font = fm.FontProperties(fname=font_path)

# Specify the font size
font_size = 15

# Add title
plt.title("=>> By Yan-xd <<=\n=>> Detected Objects With Image <<=\n", color='green', fontproperties=custom_font, fontsize=font_size)

# Add bounding boxes
for detection in detections:
    x, y, w, h = detection["x"], detection["y"], detection["width"], detection["height"]
    class_label = detection["class"]
    
    # Adjust the position and size of the bounding box to center it around the detected object
    x_new = x - w / 2
    y_new = y - h / 2
    ax.text(x_new, y_new - 5, class_label, color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5), fontweight='bold', fontproperties=custom_font)
    ax.add_patch(Rectangle((x_new, y_new), w, h, linewidth=2, edgecolor='blue', facecolor='none'))

# Hide axis
ax.axis('off')

# Show plot
plt.show()
print("")
