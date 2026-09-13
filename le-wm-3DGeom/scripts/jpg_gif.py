import os
from PIL import Image

input_folder = "/home/student/users/Yangjie_workspace/data_visualization/images/extracted_frames_episode_4"
output_gif = "/home/student/users/Yangjie_workspace/data_visualization/images/extracted_frames_episode_4/episode_4.gif"

images = []

files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

for file in files:
    path = os.path.join(input_folder, file)
    img = Image.open(path)
    images.append(img)

images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=250,   
    loop=0          
)

print("GIF created!")