# Rilevamento Segnali Stradali con YOLOv10
Questo progetto, sviluppato per il corso di Machine Learning dell'Università di Catania (Anno Accademico 2024-2025), si concentra sul rilevamento di segnali stradali utilizzando YOLOv10. L'obiettivo è valutare e confrontare le prestazioni di tre diverse varianti del modello: YOLOv10n, YOLOv10s e YOLOv10m.

### Descrizione del Progetto

Il progetto utilizza YOLOv10, un modello all'avanguardia per il rilevamento di oggetti, per identificare segnali stradali all'interno di immagini. YOLOv10 è noto per la sua accuratezza ed efficienza. Questa applicazione affronta specificamente il rilevamento dei segnali stradali, cruciale per i sistemi di guida autonoma e la gestione del traffico.

### Caratteristiche del Progetto

- **Confronto tra Modelli:** Il progetto mette a confronto le prestazioni di tre versioni di YOLOv10:
  - **YOLOv10n:** Versione più compatta e veloce, ottimizzata per applicazioni in tempo reale.
  - **YOLOv10s:** Versione bilanciata che offre un buon compromesso tra velocità e accuratezza.
  - **YOLOv10m:** Modello più grande con un'accuratezza superiore, adatto per rilevamenti dettagliati.
- **Rilevamento Oggetti:** Rileva e classifica segnali stradali nelle immagini.
- **Confronte con Ground Truth:** Visualizza i segnali rilevati affiancati alle annotazioni di ground truth per valutare le prestazioni del modello tramite un'applicazione Python.

## Requisiti

Per eseguire questo progetto, installa i seguenti pacchetti Python. Puoi installarli usando `pip`:

```bash
pip install -r requirements.txt
```

## Utility

La repository contiene alcuni script utili nella cartella `utility` per preprocessare le immagini del dataset:

1.  `convertGTSDB.py`: converte il dataset GTSDB in formato YOLO.
2.  `createbackgoundimages.py`: estrae delle immagini dal dataser per creare il backgorund.

## Demo di Addestramento (Notebook .ipynb)

Il file `Training_Traffic_Sign_Detection.ipynb` fornisce una dimostrazione pratica dell'addestramento e della validazione dei modelli YOLOv10 utilizzati nel progetto. Il notebook guida attraverso le seguenti fasi:

1.  **Addestramento del Modello**:
    - Utilizza il dataset ibrido per addestrare i modelli YOLOv10n, YOLOv10s e YOLOv10m.
    - Mostra come inizializzare e addestrare ciascun modello utilizzando il dataset specificato.
    - Per addestrare i modelli su un dataset diverso, posiziona il tuo dataset nella cartella `datasets`, modifica il file `data.yaml` in modo che punti al nuovo dataset e assicurati di mantenere la struttura descritta nella relazione del progetto.

2.  **Validazione del Modello**:
    - Esegue la validazione del modello addestrato utilizzando un dataset personalizzato di segnali stradali.
    - Dimostra come calcolare e visualizzare le metriche di prestazione come mAP, precisione, recall e F1 Score.

3.  **Confronto delle Metriche**:
    - Confronta graficamente le metriche mAP, precisione, recall e F1 Score tra i diversi modelli YOLOv10 per determinarne le prestazioni relative.

4.  **Test Visivo**:
    - Esegue test visivi su immagini di segnali stradali raccolte per il progetto per valutare l'accuratezza delle previsioni dei modelli su immagini reali.

## **Applicazione di Rilevamento Segnali Stradali**

L'applicazione di Rilevamento Segnali Stradali è un'interfaccia grafica (GUI) costruita con Tkinter, progettata per il rilevamento di segnali stradali utilizzando YOLOv10. Questa applicazione permette di:

-   **Caricare e Visualizzare Immagini**: Puoi caricare immagini di segnali stradali, che verranno mostrate insieme alle loro annotazioni di ground truth. L'immagine d'esempio e il relativo ground truth, utlizzato nel video dimostrativo è disponibile nella cartella `images`.
-   **Prevedere Segnali Stradali**: L'applicazione utilizza i modelli YOLOv10 per effettuare previsioni sulle immagini caricate e visualizzare i risultati.
-   **Visualizzare le Previsioni**: Le previsioni vengono mostrate affiancate all'immagine originale e alle annotazioni di ground truth. I risultati includono bounding box ed etichette delle classi.

## Esempio

https://github.com/Matteovullo/Progetto-ML/assets/video_dimostrativo_demo.mov

### **Esecuzione dell'Applicazione**

Per utilizzare l'Applicazione di Rilevamento Segnali Stradali, esegui il seguente comando:

```bash
python trafficSignsDetectionApp.py
```

## Contributi

I contributi sono benvenuti! Apri pure una issue o invia una pull request per qualsiasi miglioramento o correzione di bug.
