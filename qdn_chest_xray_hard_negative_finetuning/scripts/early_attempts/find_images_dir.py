# find_images_dir.py
# Early debugging attempt, copied from the first chat (LEGENDARY_DL.pdf). Kept for the record.
# Lines that the PDF cut off at the page edge are marked [clipped].
import os
import pandas as pd
def locate_images():
    df = pd.read_csv("./splits/train.csv")
    img_col = next((col for col in df.columns if 'path' in col.lower() or 'image' in col.lower() or 'file' in col.lower()), df.columns[0])  # [clipped] ending restored
    sample_img = str(df.iloc[0][img_col])
    
    print(f"Searching for sample image filename: {sample_img}")
    
    search_roots = ["/", "./", "../", "/mnt/data"]
    for root_dir in ["/mnt/data", "/home/debarshi", "."]:
        for root, dirs, files in os.walk(root_dir):
            if sample_img in files:
                print(f"FOUND MATCHING DIRECTORY: {root}")
                return
locate_images()
