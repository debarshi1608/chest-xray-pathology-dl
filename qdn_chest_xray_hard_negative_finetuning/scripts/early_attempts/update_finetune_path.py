
with open("finetune_qdn.py", "r") as f:
    code = f.read()

code = code.replace('img_dir = "./images"', 'img_dir = "./cache_224"')
with open("finetune_qdn.py", "w") as f:
    f.write(code)
print("Updated finetune_qdn.py to use ./cache_224 as the image directory.")
