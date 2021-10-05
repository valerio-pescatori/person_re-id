using System;
using System.Collections.Generic;
using UnityEngine;

namespace SwingClass
{
    public class SwingObject
    {
        private float lastValue;
        private List<float> distances; // solo x il plot
        private List<float> swingPeaks;
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
                // controllo se cresce
                if (newValue > lastValue && Math.Abs(newValue - lastValue) > lambda)
                    trend = true;
                // controllo se decresce
                else if (newValue < lastValue && Math.Abs(newValue - lastValue) > lambda)
                {
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
