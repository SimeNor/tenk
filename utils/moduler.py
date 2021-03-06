import os
import warnings
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import json
import pandas as pd
from PIL import Image, ImageOps
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
    datasett.idx_to_class = {i:c for c, i in datasett.class_to_idx.items()}
    temp["idx_to_class"] = datasett.idx_to_class
    
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

def last_ned_modell(vis_fremgang:bool=True):
    global device

    with open('_temp_.json', 'r') as f:
        temp = json.load(f)
    
    if vis_fremgang: print("Laster ned modell..")
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=temp["num_classes"]
    ).to(device)
    if vis_fremgang: print("Modell klar.")

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


def finn_kjendis(modell, datasett):
    global device

    with open('_temp_.json', 'r') as f:
        temp = json.load(f)

    data_loader = DataLoader(
        datasett,
        num_workers=temp['workers'],
        batch_size=16)

    kjendiser = pd.read_csv("kjendiser.csv")
    kjendiser.set_index("kjendis_id", inplace=True)
    kjendiser = kjendiser[['navn','kjønn']].drop_duplicates()
    
    preds = []

    for i_batch, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y_raw = modell(x)
        preds = torch.argmax(F.softmax(y_raw, dim=0), dim=1).detach().cpu().numpy()

    results = pd.DataFrame(preds, columns=['kjendis_id'])
    results['kjendis_id'] = results['kjendis_id'].apply(lambda x: int(temp["idx_to_class"][str(x)]))
    results = results.join(kjendiser, on="kjendis_id")

    for idx, (img_loc, cls) in enumerate(datasett.imgs):
        print("Ditt bilde:")
        vis_bilder([img_loc])
        print(f'Din kjendis: {results.iloc[idx]["navn"]}\n')
    
    return results


activation = {}

def get_activation(name):
        def hook(model, input, output):
            global activation

            activation[name] = output.detach()
        return hook

def generer_modellrepresentasjon(modell, datasett):
    global device

    with open('_temp_.json', 'r') as f:
        temp = json.load(f)

    data_loader = DataLoader(
        datasett,
        num_workers=temp['workers'],
        batch_size=64)

    modell.eval()
    embeddings =  []
    indices =  []

    modell.last_linear.register_forward_hook(get_activation('emb'))

    idx = 0
    embeddings = []
    for index, (x, _) in enumerate(data_loader):
        y_pred = modell(x.to(device))
        
        indices += [x[0] for x in datasett.imgs[idx:idx+len(y_pred)]]
        embeddings.append(activation['emb'].cpu().detach())
        
        idx += len(y_pred)

    return indices, torch.vstack(embeddings)


def beregn_likhet(kjendis_datasett, dine_bilder, antall_mest_like:int=1, women:bool=True):
    print(f'Finner kjendis for {len(dine_bilder)} bilder.')
    modell = last_ned_modell(vis_fremgang=False)

    with open('_temp_.json', 'r') as f:
        temp = json.load(f)

    kjendis_lokasjoner, kjendiser_representasjon = generer_modellrepresentasjon(modell, kjendis_datasett)
    dine_bilder_lokasjoner, dine_representasjoner = generer_modellrepresentasjon(modell, dine_bilder)

    kjendiser = pd.read_csv("kjendiser.csv")
    kjendiser.set_index("fil_lokasjon", inplace=True)

    distances = {}
    resultater = pd.DataFrame()

    for i_1, e1 in enumerate(dine_representasjoner):
        idx1 = dine_bilder_lokasjoner[i_1]

        distances[idx1] = pd.DataFrame(columns=['ditt_bilde', 'kjendis_bilde', 'ulikhet', 'link'])

        e1_tensor = e1.repeat(len(kjendiser_representasjon), 1)

        distances[idx1]['ulikhet'] = np.array(torch.norm(e1_tensor.to(device) - kjendiser_representasjon.to(device), 2, dim=1).cpu())
        distances[idx1]['kjendis_bilde'] = kjendis_lokasjoner
        distances[idx1]['link'] = distances[idx1]['kjendis_bilde'].apply(lambda x: x.split(temp["kjendisbilder_lokasjon"])[-1][1:])
        distances[idx1]['ditt_bilde'] = idx1
        distances[idx1] = distances[idx1].join(kjendiser, on="link")
        if women:
            distances[idx1] = distances[idx1][distances[idx1]["kjønn"] == "dame"].sort_values(by="ulikhet").head(antall_mest_like)
        else:
            distances[idx1] = distances[idx1].sort_values(by="ulikhet").head(antall_mest_like)
        distances[idx1].drop(["link", "Unnamed: 0"], axis=1, inplace=True)

        resultater = resultater.append(distances[idx1])
        
        print(f'Funnet likhet for {i_1 + 1}/{len(dine_representasjoner)}.')

    del distances

    return resultater


def create_key(idxs):
    idxs.sort()
    return str(idxs[0]) + '_' + str(idxs[1])


def extract_face(file_name: str, save_path:str) -> np.array:
    global device
    
    # Load image
    img = Image.open(file_name).convert("RGB")
    img = ImageOps.exif_transpose(img)

    # Instantiate detector
    face_detector = MTCNN(
        image_size=160, margin=5, min_face_size=20,
        thresholds=[0.1, 0.1, 0.1], factor=0.709, post_process=True,
        selection_method="probability", device=device
        )
    
    # Detect face

    cropped_img = face_detector(img, save_path=save_path)

    if cropped_img is not None:
        return cropped_img
    else:
        reimg = img.resize((160, 160))
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
            img.save(save_path)
        return np.array(reimg, np.int8)


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
    for i in range(len(resultater)):
        row = resultater.iloc[i]
        print("Ditt bilde:")
        vis_bilder([row["ditt_bilde"]])
        print("Dine mest like kjendis:")
        print(f'{row["navn"]}({row["bilde_år"]}) - Ulikhet: {row["ulikhet"]}')
        vis_bilder([row["kjendis_bilde"]])
        print('\n\n')


def generer_prediksjoner(model, x):
    y_pred = model(x)
    return y_pred


def sammenlign_med_fakta(y, y_pred, loss_fn):
    return loss_fn(y_pred, y)


def oppdater_modell(loss_batch, optimizer):
    loss_batch.backward()
    optimizer.step()
    optimizer.zero_grad()


def loss_metrics():
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'Treffsikkerhet': training.accuracy
    }
    return loss_fn, metrics