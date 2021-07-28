import os, warnings, json

def setup():
    warnings.filterwarnings('ignore')
    os.chdir("tenk")
    os.system('pip install -r utils/requirements.txt')
    os.system('unzip komp_kjendiser.zip')

    temp = {'workers': 0 if os.name == 'nt' else 2}

    with open('_temp_.json', 'w') as f:
        json.dump(temp, f)

    os.makedirs("dine_opplastninger", exist_ok=True)