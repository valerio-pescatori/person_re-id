using System;
using System.Collections.Generic;
using UnityEngine;

namespace SwingClass
{
    public class SwingObject{
    private float lastValue;
    private List<float> distances; // solo x il plot
    private List<float> swingPeaks;
    private bool trend;
    private float lambda;

    public SwingObject(float lambda)
    {
        lastValue = 0f;
        distances = new List<float>();
        swingPeaks = new List<float>();
        this.lambda = lambda;
    }

    // getters and setters
    public float LastValue { get => lastValue; set => lastValue = value; }
    public List<float> SwingPeaks { get => swingPeaks; set => swingPeaks = value; }
    public bool Trend { get => trend; set => trend = value; }
    public float Lambda { get => lambda; set => lambda = value; }
    public List<float> Distances { get => distances; set => distances = value; }

        public void AvgDistance(Vector2 p1, Vector2 p2)
    {
        float newValue = Vector2.Distance(p1, p2);
        distances.Add(newValue);

        if(lastValue != 0)
        {
            // controllo se cresce
            if(newValue > lastValue && Math.Abs(newValue - lastValue) > lambda)
                trend = true;
            // controllo se decresce
            else if(newValue < lastValue && Math.Abs(newValue - lastValue) > lambda)
            {
                if(trend)
                    swingPeaks.Add(lastValue);
                trend = false;
            }
        }
        lastValue = newValue;
    }
}
}
