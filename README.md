# Person Re-ID
Progetto svolto durante il periodo di tirocinio per la tesi di Laurea Triennale in Informatica.
La tesi finale è disponibile al seguente [link](https://mega.nz/file/SMgUwbCJ#N4F3ho9f072BR4Yi3jA6_D2MoyZ4QjfcdkORLirLmVw)

Per eseguire gli esperimenti è necessario scaricare il dataset, disponibile al seguente [link](https://mega.nz/file/nVJ2BY5I#s_RMEE3Wtt5zzKHnZYb6ljvNs4F3qNqqAeb0LK-6awM).
I file contenuti nell'archivio vanno estratti nella cartella Data/.
Alternativamente è possibile generare il dataset manualmente, è sufficiente aprire il progetto in Unity e premere il tasto Play.
In automatico le animazioni vengono messe in riproduzione e al termine dell'esecuzione i dati verranno splittati e salvati in 3 files JSON contenuti nella cartella Data/.

Una volta collezionato il dataset è possibile eseguire i test sui vari modelli di nn implementati.
Basta eseguire il file python "Python/neuralnets.py", è possibile specificare 3 flag tramite cli:

1. __--ablation/-a__: Da inserire se si voglio eseguire i test sia sul modello completo che con le ablazioni
2. __--save_states/-ss__: Da inserire se si vogliono salvare i dizionari dello stato dei modelli e dei relativi optimizer nella cartella Python/model_states/
3. __--load_states/-ls__: Da inserire se si vogliono caricare i dizionari dello stato dei modelli e dei relativi optimizer dalla cartella Python/model_states/

