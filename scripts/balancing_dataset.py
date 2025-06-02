import os
import random
import shutil

src_nothing = 'data/temp/nothing'
src_jump = 'data/temp/jump'

dst_nothing = 'data/temp_balanced/0'
dst_jump = 'data/temp_balanced/1'

os.makedirs(dst_nothing, exist_ok=True)
os.makedirs(dst_jump, exist_ok=True)

# Copy all nothing images to balanced folder
for img in os.listdir(src_nothing):
    shutil.copy(os.path.join(src_nothing, img), os.path.join(dst_nothing, img))

# Copy all jump images to balanced folder first
for img in os.listdir(src_jump):
    shutil.copy(os.path.join(src_jump, img), os.path.join(dst_jump, img))

num_nothing = len(os.listdir(src_nothing))
num_jump = len(os.listdir(src_jump))

print(f"Nothing images: {num_nothing}")
print(f"Jump images: {num_jump}")

num_to_add = num_nothing - num_jump
jump_images = os.listdir(src_jump)

for i in range(num_to_add):
    img_name = random.choice(jump_images)
    src_path = os.path.join(src_jump, img_name)
    new_img_name = f"dup_{i}_{img_name}"
    dst_path = os.path.join(dst_jump, new_img_name)
    shutil.copy(src_path, dst_path)

print(f"Oversampled jump images from {num_jump} to {len(os.listdir(dst_jump))}")
print(f"Balanced dataset saved to 'data/temp_balanced'")
