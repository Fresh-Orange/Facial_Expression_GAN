import os

dir = r"C:\Users\FreshOrange\Desktop\result921\level2"

for f in os.listdir(dir):
    id = f.split("m")[0]
    os.rename(os.path.join(dir, f), os.path.join(dir, id+".mp4"))