

import pandas as pd
import os
import cv2
import yaml
import shutil

train_imag = r'D:\..'
test_imag  = r'D:\..'
csv_file   = r'D:\...csv'
output     = r'C:\..'

# الملفات الي راح نعملهن
folders = [r'images\train', r'images\val', r'labels\train', r'labels\val']

for f in folders:  # انشاء الملفات
    os.makedirs(os.path.join(output,f), exist_ok=True)

df = pd.read_csv(csv_file)

if 'class' not in df.columns:  # اذا ما موجود عمود التصنيفات 
    df['class']= 0

def convert(xmin,ymin, xmax,ymax, img_w,img_h):
    x_centor = ((xmin+xmax)/2)/ img_w
    y_centor = ((ymin+ymax)/2)/ img_h
    w = (xmax - xmin)/ img_w
    h = (ymax - ymin)/ img_h

    return x_centor, y_centor, w,h

for idx, row in df.iterrows():
    image_name = row['image']
    image_path = os.path.join(train_imag, image_name)
    
    if not os.path.exists(image_path):
        continue

    img = cv2.imread(image_path)
    if img is None:
        continue

    img_h, img_w = img.shape[:2]
    
    x_centor, y_centor, w,h = convert(row['xmin'], row['ymin'], row['xmax'], row['ymax'], img_w, img_h)

    label_path = os.path.join(output, r'labels\train', image_name.replace('.jpg','.txt'))

    with open (label_path,'a') as f:
        f.write(f'{0} {x_centor} {y_centor} {w} {h}\n')

    # 
    shutil.copy(image_path, os.path.join(output, r'images\train', image_name))

print('✔Training set was done')

for img_name in os.listdir(test_imag):
    image_path = os.path.join(test_imag, img_name)

    if image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        shutil.copy(image_path, os.path.join(output, r'images\val', image_name))
print('done')

daly_yaml = {
    'train': os.path.abspath(os.path.join(output, r'images\train')),
    'val': os.path.abspath(os.path.join(output, r'images\val')),
    'nc': df['class'].nunique(),
    'names': 'car' # or only [str(c) for c in df['class'].unique()]
}

with open(os.path.join(output, 'data.yaml'), 'w') as f:
    yaml.dump(daly_yaml, f, default_flow_style=False)

print('done: data yaml')

















