# Person Re-ID
Progetto svolto durante il periodo di tirocinio per la tesi di Laurea Triennale in Informatica.

Il progetto consiste in un framework di re-identificazione basato su features estratte da uno skeleton 3D generato tramite Unity.
Il riconoscimento dei soggetti avviene attraverso reti neurali, in particolare sono stati testati 6 diversi modelli di rete neurale.
Tutti i dettagli sulle features e sull'architettura implementati sono disponibili nella tesi, disponibile al seguente [link](https://mega.nz/file/SMgUwbCJ#N4F3ho9f072BR4Yi3jA6_D2MoyZ4QjfcdkORLirLmVw).

## Requisiti
Per poter eseguire gli esperimenti è necessario seguire i seguenti step:
1. Clonare la repository `git clone https://github.com/valerio-pescatori/person_re-id`
    - Alternativamente è possibile scaricare l'archivio zip
3. Scaricare il dataset, disponibile al seguente [link](https://mega.nz/file/nVJ2BY5I#s_RMEE3Wtt5zzKHnZYb6ljvNs4F3qNqqAeb0LK-6awM)
4. Estrarre il contenuto dell'archivio nella cartella `Data/`
  - Alternativamente è possibile generare il dataset manualmente, è sufficiente: 
    - Aprire il progetto in Unity
    - Caricare la scena tramite File --> Load Scene --> `Scenes\SampleScene.unity` 
    - Premere il tasto Play, la collezione del dataset impiegherà circa 3 ore.

## Quickstart
Una volta collezionato il dataset è possibile eseguire i test sui vari modelli di neural network implementati.
È sufficiente eseguire il file python `python Python/neuralnets.py`, è possibile specificare 3 flag tramite cli:

- `--ablation` / `-a`: Da inserire se si voglio eseguire i test sia sul modello completo che con le ablazioni
- `--save_states` / `-ss`: Da inserire se si vogliono salvare i dizionari dello stato dei modelli e dei relativi optimizer nella cartella Python/model_states/
- `--load_states` / `-ls`: Da inserire se si vogliono caricare i dizionari dello stato dei modelli e dei relativi optimizer dalla cartella Python/model_states/

