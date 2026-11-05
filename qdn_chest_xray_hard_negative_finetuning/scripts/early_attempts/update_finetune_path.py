# update_finetune_path.py
# Early debugging attempt, copied from the first chat (LEGENDARY_DL.pdf). Kept for the record.
# Lines that the PDF cut off at the page edge are marked [clipped].
with open("finetune_qdn.py", "r") as f:
    code = f.read()
# Replace img_dir = "./images" with img_dir = "./cache_224"
code = code.replace('img_dir = "./images"', 'img_dir = "./cache_224"')
with open("finetune_qdn.py", "w") as f:
    f.write(code)
print("Updated finetune_qdn.py to use ./cache_224 as the image directory.")
