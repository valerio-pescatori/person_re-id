using System;

namespace ObjectsForJson
{
    [Serializable]
    public class JFrame
    {
        public float hunchback;
        public float outToeingR;
        public float outToeingL;
        public float bodyOpennessU;
        public float bodyOpennessL;
        public float bctU;
        public float bctL;
        public float bctF;
        public float feetDist { get; private set; }
        // non salvo la lunghezza del passo poichè la feature non è una media delle distanze tra i piedi di ogni frame.
        // Partendo da un vettore contenente la distanza tra i due piedi(joint['feet']), 
        // ricavo solo i frame in cui si suppone ci sia la massima estensione del passo (quindi i "picchi" dei valori nel vettore)


        // opzione: potrei salvare le distanze tra i piedi in ogni frame e creare un metodo in animations che calcola la media finale.
        // forse questa è la più pulita. 
        // \\
        // devo (?) per forza poichè non so alla fine quanti frame ho, quindi non so quando chiamare la funzione finale di media in steplength
        // potre

        public JFrame(float outToeingR, float outToeingL, float hunchback, float feetDist,
                        float bodyOpennessU, float bodyOpennessL, float bctU, float bctL, float bctF)
        {
            this.outToeingL = outToeingL;
            this.outToeingR = outToeingR;
            this.hunchback = hunchback;
            this.feetDist = feetDist;
            this.bodyOpennessL = bodyOpennessL;
            this.bodyOpennessU = bodyOpennessU;
            this.bctF = bctF;
            this.bctL = bctL;
            this.bctU = bctU;
        }
    }
}