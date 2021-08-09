import os, warnings, json, random, numpy, torch

def setup(seed:int=42):
    warnings.filterwarnings('ignore')
    os.chdir("tenk")
    os.system('pip install -r utils/requirements.txt')
    os.system('unzip komp_kjendiser.zip')

    random.seed(seed)
    torch.manual_seed(seed+1)
    numpy.random.seed(seed+2)

    temp = {'workers': 0 if os.name == 'nt' else 2, "kjendis_chache":"kjendis_cache.json"}

    with open('_temp_.json', 'w') as f:
        json.dump(temp, f)

    os.makedirs("dine_opplastninger", exist_ok=True)