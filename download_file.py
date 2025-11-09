import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("wenewone/cub2002011")
os.system(f"cp -r {path}/CUB_200_2011 ./CUB_200_2011")