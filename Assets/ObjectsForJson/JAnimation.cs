using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;
using Microsoft.Win32;

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
        public const string pyScriptPath = @"Python\walkplot.py";

        public JAnimation(float lambda, int index)
        {
            this.lambda = lambda;
            this.index = index;
            this.frames = new List<JFrame>(PlayerController.NUMBER_OF_FRAMES);
        }

        public JAnimation() { }

        public void AddFrame(JFrame frame)
        {
            if (frames.Count < PlayerController.NUMBER_OF_FRAMES)
                frames.Add(frame);
        }

        public float CalculateSteps(bool plot)
        {
            // calcola picchi e media passi
            var nFrames = PlayerController.NUMBER_OF_FRAMES;
            // scorro i frames
            List<float> peaks = new List<float>(20);

            // taglio i valori in eccesso 
            if (frames.Count > nFrames)
                frames.RemoveRange(nFrames, frames.Count - nFrames);

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
            mediaLungPass = peaks.Count == 0 ? 0 : sum / peaks.Count;
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
            start.FileName = FindPythonPath();
            start.Arguments = $" -i \"{pyScriptPath}\" \"{arg1}\" \"{arg2}\" \"{index}\"";
            start.UseShellExecute = true;
            start.CreateNoWindow = true;
            using (Process process = Process.Start(start))
            {
            }
        }

        private static string FindPythonPath(){
            object pyPath = Registry.GetValue("HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\3.9\\InstallPath", "ExecutablePath", "err");
            if(pyPath != null){
                string s = pyPath.ToString();
                if (s == "err")
                    throw new PythonExeNotFoundException();
                return s;
            }
            else
               throw new PythonExeNotFoundException("Registry key doesn't exist.");
        }
    }
}