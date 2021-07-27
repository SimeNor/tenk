import os, warnings, json

warnings.filterwarnings('ignore')
os.chdir("tenk")
os.system('pip install -r utils/requirements.txt')
os.system('unzip cropped.zip && rm uncropped.zip')

temp = {'workers': 0 if os.name == 'nt' else 2}

with open('_temp_.json', 'w') as f:
    json.dump(temp, f)