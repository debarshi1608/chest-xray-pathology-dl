# update_img_dir.py (second version)
# Early debugging attempt, copied from the first chat (LEGENDARY_DL.pdf). Kept for the record.
# Lines that the PDF cut off at the page edge are marked [clipped].
import os
import pandas as pd
def find_correct_path():
    df = pd.read_csv("./splits/train.csv")
    img_col = next((col for col in df.columns if 'path' in col.lower() or 'image' in col.lower() or 'file' in col.lower()), df.columns[0])  # [clipped] ending restored
    sample_img = str(df.iloc[0][img_col])
    
    # Search parent and sibling directories
    for root, dirs, files in os.walk("../"):
        if sample_img in files:
            print(f"FOUND_DIR:{os.path.abspath(root)}")
            return root
            
    print("NOT_FOUND")
    return "./images"
if __name__ == "__main__":
    find_correct_path()
