# fix_nan_finetune.py
# Early debugging attempt, copied from the first chat (LEGENDARY_DL.pdf). Kept for the record.
# Lines that the PDF cut off at the page edge are marked [clipped].
import re
with open("finetune_qdn.py", "r") as f:
    code = f.read()
# Add gradient clipping and mixed precision (GradScaler) to prevent NaN loss
old_loop = """    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()"""
new_loop = """    scaler = torch.cuda.amp.GradScaler()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            if torch.isnan(loss):
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()"""
if old_loop in code:
    code = code.replace(old_loop, new_loop)
    with open("finetune_qdn.py", "w") as f:
        f.write(code)
    print("Successfully added GradScaler, NaN check, and gradient clipping to finetune_qdn.py")
else:
    print("Could not match exact training loop block. Check script structure.")
EOF
