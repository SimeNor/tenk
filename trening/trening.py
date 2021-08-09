import torch
from torch import optim
import numpy as np
import time
from facenet_pytorch import training

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generer_prediksjoner(model, x):
    y_pred = model(x)
    return y_pred


def sammenlign_med_fakta(y, y_pred, loss_fn):
    return loss_fn(y_pred, y)


def oppdater_modell(loss_batch, optimizer):
    loss_batch.backward()
    optimizer.step()
    optimizer.zero_grad()


def kjør_iterasjon(data, modell, avviksfunksjon, optimizer, batch_metrics):
    global device
    mode = 'Train' if modell.training else 'Valid'
    logger = training.Logger(mode, length=len(data), calculate_mean=True)
    avvik = 0
    metrics = {}

    for i, (eksempler, fakta) in enumerate(data):
        eksempler, fakta = eksempler.to(device), fakta.to(device)

        #######################################################################################
        # På radene under er det markert hvor vi må fylle ut med en "#"
        #######################################################################################

        """
        Først må vi bruke modellen til å gjøre prediksjoner med de eksemplene som vises
        """
        predisksjoner = generer_prediksjoner(modell, eksempler)
        #predisksjoner = # Bruk funksjon for å gjøre prediksjoner med dataen

        """
        Prediksjonene sammenlignes med faktaen fra datasettet for å se hvor stort avviket er
        """
        avvik_gruppe = sammenlign_med_fakta(fakta, predisksjoner, avviksfunksjon)
        #avvik_gruppe = # Bruk funksjonen for å sammenligne fakta mot prediksjoner og spesifiser hvordan dette skal beregnes

        if modell.training:
            """
            Når modellen er satt i treningsmodus skal vi oppdatere den basert på avvikene vi finner
            
            HINT: Det er en egen funksjon for å oppdatere modellen
            """
            oppdater_modell(avvik_gruppe, optimizer)
            # Bruk avvikene du fant for gruppen og en optimizer til å gjøre oppdateringer

        #######################################################################################
        # Slutt på oppgaven, resten av koden kan bli stående som det er
        #######################################################################################


        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(predisksjoner, fakta).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

        avvik_gruppe = avvik_gruppe.detach().cpu()
        avvik += avvik_gruppe

        logger(avvik, metrics, i)

    avvik = avvik / (i + 1)
    metrics = {k: v / (i + 1) for k, v in metrics.items()}

    return avvik, metrics


def tren_modell(modell, trening_data, læringsrate:float, treningsiterasjoner:int, test_data=None):
    global device

    optimizer = optim.Adam(modell.parameters(), lr=læringsrate)
    avviksfunksjon, metrics = loss_metrics()

    for epoch in range(treningsiterasjoner):
        print('\nEpoch {}/{}'.format(epoch + 1, treningsiterasjoner))
        print('-' * 10)

        #######################################################################################
        # På radene under er det markert hvor vi må fylle ut med en "#"
        #######################################################################################

        """
        Første steg i hver iterasjon er å gjøre treningen av modellen. Modellen settes til treningsmodus

        Sett inn riktig datasett for å trene opp ML-modellen
        """
        modell.train()
        avvik, metrics = kjør_iterasjon(trening_data, modell, avviksfunksjon, optimizer, metrics)
        #avvik, metrics = kjør_iterasjon(# Sett inn riktig datasett her!,
        #                                modell, avviksfunksjon, optimizer, metrics)

        """
        Deretter skal vi teste modellen på data den ikke har fått lære av. Modellen settes til evalueringsmodus
        
        Sett inn riktig datasett for å teste ML-modellen
        """
        modell.eval()
        avvik, metrics = kjør_iterasjon(test_data, modell, avviksfunksjon, optimizer, metrics)
        #avvik, metrics = kjør_iterasjon(# Sett inn riktig datasett her!
        #                                , modell, avviksfunksjon, optimizer, metrics)

        #######################################################################################
        # Slutt på oppgaven, resten kan bli stående som det er
        #######################################################################################

    return modell


def loss_metrics():
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'Treffsikkerhet': training.accuracy
    }
    return loss_fn, metrics
