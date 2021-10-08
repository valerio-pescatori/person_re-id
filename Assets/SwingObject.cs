using System;
using System.Collections.Generic;
using UnityEngine;

namespace SwingClass
{
    // Uno SwingObject tiene traccia dell'oscillazione della distanza tra due punti
    public class SwingObject
    {
        private float lastValue;
        private List<float> distances; // solo x il plot
        private List<float> swingPeaks; // tiene traccia dei picchi positivi (massima estensione del passo)
        private bool trend;
        private float lambda;
        private float max;
        private float min;

        public SwingObject(float lambda)
        {
            lastValue = 0f;
            distances = new List<float>();
            swingPeaks = new List<float>();
            this.lambda = lambda;
            this.max = float.NegativeInfinity;
            this.min = float.PositiveInfinity;
        }

        // AvgDistance prende la distanza tra i due punti dati in input, la salva
        // e poi decide, in base ai valori dei campi dell'oggetto, se si tratta di un picco positivo.
        public void AvgDistance(Vector3 p1, Vector3 p2)
        {
            float newValue = Vector3.Distance(p1, p2);
            distances.Add(newValue);

            // idea per nuovo modo di calcolare i passi
            // dopo aver letto tutti i valori prendo il max e il min
            // prendo poi tutti i valori che rientrano nel max +- lambda e min +- lambda
            // li accoppio in ordine e prendo le distanze

            if (lastValue != 0)
            {
                // se il valore è in crescita non si tratta di un picco
                // lambda serve per ignorare i cambiamenti irrilevanti
                if (newValue > lastValue && Math.Abs(newValue - lastValue) > lambda)
                    trend = true;
                // se il valore è in calo potrei aver passato il picco
                else if (newValue < lastValue && Math.Abs(newValue - lastValue) > lambda)
                {
                    // se prima dell'ultimo aggiornamento il valore era in crescita allora si 
                    // tratta di un picco
                    if (trend)
                        swingPeaks.Add(lastValue);
                    trend = false;
                }
            }
            lastValue = newValue;
        }

        // idea per nuovo modo di calcolare i passi
        // dopo aver letto tutti i valori prendo il max e il min
        // prendo poi tutti i valori che rientrano nel max +- lambda e min +- lambda
        // li accoppio in ordine e prendo le distanze

        // problema: prendendo i punti come descritto sopra avrei un mucchietto di punti per ogni picco
        // risolvo prendendo i punti che si distano di almeno x pixels (e.g 60) 

        public void AvgDistance2(Vector3 p1, Vector3 p2)
        {
            float newValue = Vector3.Distance(p1, p2);
            distances.Add(newValue);
            if (newValue > max)
                max = newValue;
            else if (newValue < min)
                min = newValue;
        }

        public void calculateSteps()
        {
            var lastIndex = 0;
            for (int i = 0; i < distances.Count; i++)
            {
                // Debug.Log((max - lambda < distances[i] || distances[i] < max + lambda)
                //     + "\n" +
                //     (min - lambda > distances[i] || distances[i] > min + lambda) + "\n" + i);
                // peak positivo e negativo
                if ((max - lambda < distances[i] || distances[i] < max + lambda)
                    ||
                    (min - lambda > distances[i] || distances[i] > min + lambda))
                {
                    // Debug.Log(i + " " + lastIndex);
                    if (swingPeaks.Count == 0)
                    {
                        swingPeaks.Add(distances[i]);
                        Debug.Log("ADDING PEAK: " + distances[i] + "\nAT INDEX: " + i + "\nLAST INDEX WAS: " + lastIndex);
                        lastIndex = i;
                    }
                    else if (i - lastIndex > 40)
                    {
                        swingPeaks.Add(distances[i]);
                        Debug.Log("ADDING PEAK: " + distances[i] + "\nAT INDEX: " + i + "\nLAST INDEX WAS: " + lastIndex);
                        lastIndex = i;
                    }
                }
            }

        }

        // getters and setters
        public float LastValue { get => lastValue; set => lastValue = value; }
        public List<float> SwingPeaks { get => swingPeaks; set => swingPeaks = value; }
        public bool Trend { get => trend; set => trend = value; }
        public float Lambda { get => lambda; set => lambda = value; }
        public List<float> Distances { get => distances; set => distances = value; }
    }


}
