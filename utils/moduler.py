import os
import warnings
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import json
import pandas as pd
from PIL import Image
from IPython.display import Image as IMG
from IPython.display import display

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def last_inn_bilder(lokasjon_bilder:str):
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    datasett = datasets.ImageFolder(lokasjon_bilder, transform=trans)

    with open('_temp_.json', 'r') as f:
        temp = json.load(f)
    
    temp["num_classes"] = len(datasett.class_to_idx)
    try:
        _ = temp["kjendisbilder_lokasjon"]
    except KeyError:
        temp["kjendisbilder_lokasjon"] = lokasjon_bilder
    datasett.idx_to_class = {i:c for c, i in datasett.class_to_idx.items()}
    
    with open('_temp_.json', 'w') as f:
        json.dump(temp, f)

    print(f'Datasett klart. {len(datasett)} bilder funnet.')

    return datasett


def del_opp_datasett(datasett, andel_test:float, størrelse_treningsgrupper:int=64, ignore_print:bool=False):
    with open('_temp_.json', 'r') as f:
        temp = json.load(f)

    img_inds = np.arange(len(datasett))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int((1-andel_test) * len(img_inds))]
    val_inds = img_inds[int((1-andel_test) * len(img_inds)):]
    
    train_loader = DataLoader(
        datasett,
        num_workers=temp['workers'],
        batch_size=størrelse_treningsgrupper,
        sampler=SubsetRandomSampler(train_inds)
    )
    
    val_loader = DataLoader(
        datasett,
        num_workers=temp['workers'],
        batch_size=størrelse_treningsgrupper,
        sampler=SubsetRandomSampler(val_inds)
    )

    if not ignore_print:
        print(f'Datasett delt opp.\n{len(train_inds)} treningsbilder.\n{len(val_inds)} testbilder.')

    return train_loader, val_loader

def last_ned_modell():
    global device

    print("Laster ned modell..")
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
    ).to(device)
    print("Modell klar.")

    return resnet

def tren_modell(modell, data, læringsrate:int, treningsiterasjoner:int, test_data=None):
    global device

    optimizer = optim.Adam(modell.parameters(), lr=læringsrate)
    scheduler = MultiStepLR(optimizer, [5, 10])

    loss_fn, metrics = loss_metrics()

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 1

    modell.eval()

    for epoch in range(treningsiterasjoner):
        print('\nEpoch {}/{}'.format(epoch + 1, treningsiterasjoner))
        print('-' * 10)

        modell.train()
        training.pass_epoch(
            modell, loss_fn, data, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        if test_data is not None:
            modell.eval()
            training.pass_epoch(
                modell, loss_fn, test_data,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )

    writer.close()

    return modell


def loss_metrics():
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'Treffsikkerhet': training.accuracy
    }
    return loss_fn, metrics


def test_modell(modell, data):
    global device

    loss_fn, metrics = loss_metrics()

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 1

    modell.eval()
    training.pass_epoch(
        modell, loss_fn, data,
        batch_metrics=metrics, show_running=False, device=device,
        writer=writer
    )

    writer.close()


def generer_modellrepresentasjon(modell, datasett):
    global device

    with open('_temp_.json', 'r') as f:
        temp = json.load(f)

    if len(datasett) > 10:
        img_inds = np.arange(len(datasett))
        np.random.shuffle(img_inds)
        inds = img_inds[:10]

        data_loader = DataLoader(
            datasett,
            num_workers=temp['workers'],
            batch_size=64,
            sampler=SubsetRandomSampler(inds)
        )
    else:
        data_loader = DataLoader(
            datasett,
            num_workers=temp['workers'],
            batch_size=64)

    modell.eval()
    embeddings = {}

    idx = 0
    for x, _ in data_loader:
        y_pred = modell(x.to(device))

        for emb in y_pred.cpu().detach().numpy():
            embeddings[datasett.imgs[idx][0]] = emb
            idx += 1

    return embeddings


def beregn_likhet(modell, kjendis_datasett, dine_bilder, antall_mest_like:int=1):
    with open('_temp_.json', 'r') as f:
        temp = json.load(f)
    modell_representasjon = generer_modellrepresentasjon(modell, kjendis_datasett)
    dine_representasjoner = generer_modellrepresentasjon(modell, dine_bilder)

    kjendiser = pd.read_csv("kjendiser.csv")
    kjendiser.set_index("fil_lokasjon", inplace=True)

    distances = {}
    done = []

    for idx1, e1 in dine_representasjoner.items():
        distances[idx1] = pd.DataFrame(columns=['ditt_bilde', 'kjendis_bilde', 'ulikhet', 'link'])
        
        for idx2, e2 in modell_representasjon.items():
            key = create_key([idx1, idx2])
            
            if key not in done:
                kjendis_link = idx2.split(temp["kjendisbilder_lokasjon"])[-1][1:]
                distances[idx1] = distances[idx1].append({'ditt_bilde': idx1, 'kjendis_bilde': idx2, 'ulikhet':np.linalg.norm(e1 - e2), 'link': kjendis_link}, ignore_index=True)
                done.append(key)

        distances[idx1] = distances[idx1].sort_values(by="ulikhet").head(antall_mest_like)
        distances[idx1] = distances[idx1].join(kjendiser, on="link")
        distances[idx1].drop("link", axis=1, inplace=True)
        
    return distances

def create_key(idxs):
    idxs.sort()
    return str(idxs[0]) + '_' + str(idxs[1])


def extract_face(file_name: str, save_path:str) -> np.array:
    global device

    # Load image
    img = Image.open(file_name)
    
    # Instantiate detector
    face_detector = MTCNN(
        image_size=160, margin=5, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
        )
    
    face = face_detector(img, save_path=save_path)
    if face is None:
        rot_img = img.rotate(-90)
        face = face_detector(rot_img, save_path=save_path)

    if face is None:
        rot_img = img.rotate(-90)
        face = face_detector(rot_img, save_path=save_path)
    if face is None:
        rot_img = img.rotate(-90)
        face = face_detector(rot_img, save_path=save_path)

    # Detect face
    return face


def finn_ansikter(lokasjon_bilder:str, path_cropped_images:str="dine_ansikter"):
    # Confirm path to store images to
    os.makedirs(path_cropped_images, exist_ok=True)

    for folder, _, files in os.walk(lokasjon_bilder):
        for file in files:
            file_path = os.path.join(folder, file)
            output_folder = os.path.join(path_cropped_images, folder)
            new_path = os.path.join(output_folder, file)

            _ = extract_face(file_path, save_path=new_path)
            print(f'Fant ansikt i bilde "{file}"')

    return path_cropped_images


def vis_bilder(lokasjon_bilder, antall_bilder_totalt:int=None, antall_bilder_per_kjendis:int=None):
    numb_images = 0
    
    if lokasjon_bilder[-4:] in [".jpg", ".JPG"]:
        display(IMG(filename=lokasjon_bilder))
    elif type(lokasjon_bilder) is list:
        for file_path in lokasjon_bilder[:antall_bilder_totalt]:
            display(IMG(filename=file_path))
    else:
        for folder, subs, files in os.walk(lokasjon_bilder):
            for file in files:
                file_path = os.path.join(folder, file)
                display(IMG(filename=file_path))
                numb_images += 1
                if (antall_bilder_per_kjendis is not None) and (numb_images >= antall_bilder_per_kjendis):
                    break
            if (antall_bilder_totalt is not None) and (numb_images >= antall_bilder_totalt):
                    break

def vis_resultater(resultater):
    for ditt_bilde, resultat in resultater.items():
        print("Ditt bilde:")
        vis_bilder([ditt_bilde])
        print("Dine mest like kjendiser:\n\n")

        for i in range(len(resultat)):
            row = resultat.iloc[i]
            print(f'{row["navn"]} - Ulikhet: {row["ulikhet"]}')
            vis_bilder([row["kjendis_bilde"]])
            print('\n\n')