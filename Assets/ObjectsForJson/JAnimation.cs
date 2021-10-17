using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace ObjectsForJson
{
    [Serializable]
    public class JAnimation
    {
        public float mediaLungPass;
        // indice dell'animazione per identificarla
        public int index;
        public List<JFrame> frames;
        private float lambda;
        public const string pyScriptPath = @"Python\plot.py";
        public const string pyExePath = @"C:\Users\Valerio\AppData\Local\Programs\Python\Python39\pythonw.exe";

        public JAnimation(float lambda, int index)
        {
            this.lambda = lambda;
            this.index = index;
            this.frames = new List<JFrame>(750);
        }

        public void AddFrame(JFrame frame)
        {
            if (frames.Count < 750)
                frames.Add(frame);
        }

        public float CalculateSteps(bool plot)
        {
            // calcola picchi e media passi

            // scorro i frames
            List<float> peaks = new List<float>(20);

            // taglio i valori in eccesso 
            frames.RemoveRange(750, frames.Count - 750);



            for (int i = 2; i < frames.Count; i++)
            {
                // indica se nel frame attuale il trend è in crescita
                bool curTrend = frames[i].feetDist > frames[i - 1].feetDist
                        && (Math.Abs(frames[i].feetDist - frames[i - 1].feetDist) > lambda);
                // indica se nel frame precedente il trend è in crescita
                bool lastTrend = frames[i - 1].feetDist > frames[i - 2].feetDist
                        && (Math.Abs(frames[i - 1].feetDist - frames[i - 2].feetDist) > lambda);

                // se il trend tra i-2 e i-1 cresce
                // e il trend tra i-1 ed i decresce
                // allora ho un picco 
                if (!curTrend && lastTrend)
                    peaks.Add(frames[i].feetDist);
            }
            var sum = 0f;
            // ricavati i picchi, calcolo la media
            foreach (float peak in peaks)
                sum += peak;
            mediaLungPass = sum / peaks.Count;
            if (plot)
                this.Plot(index);
            return mediaLungPass;
        }

        private void Plot(int index)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < frames.Count - 1; i++)
                sb.Append(frames[i].feetDist.ToString("F4").Replace(",", ".") + ",");
            sb.Append(frames[frames.Count - 1].feetDist.ToString("F4").Replace(",", "."));
            string arg1 = sb.ToString();
            string arg2 = frames.Count.ToString();

            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = pyExePath;
            start.Arguments = $" -i \"{pyScriptPath}\" \"{arg1}\" \"{arg2}\" \"{index}\"";
            start.UseShellExecute = true;
            start.CreateNoWindow = true;
            using (Process process = Process.Start(start))
            {
            }
        }
    }
}