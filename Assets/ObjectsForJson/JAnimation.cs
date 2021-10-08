using System;
using System.Collections.Generic;
namespace ObjectsForJson
{
    [Serializable]
    public class JAnimation
    {
        public float mediaLungPass;
        public List<JFrame> frames;
        private float lambda;

        public JAnimation(float lambda) : this()
        {
            this.lambda = lambda;
        }

        public JAnimation() { }

        public void addFrame(JFrame frame)
        {
            if (frames == null)
                frames = new List<JFrame>();
            frames.Add(frame);
        }

        public float calculateSteps()
        {
            // calcola picchi e media passi

            // scorro il vettore
            // lastvalue diventa i-1 newvalue è i
            // devo salvare lambda come campo di janimation (?)
            // trend non mi serve poichè è (newval > lastval) --> se true è in crescita
            List<float> peaks = new List<float>(20);
            for (int i = 2; i < frames.Count; i++)
            {

                bool curTrend = frames[i].feetDist > frames[i - 1].feetDist
                        && (Math.Abs(frames[i].feetDist - frames[i - 1].feetDist) > lambda);
                bool lastTrend = frames[i - 1].feetDist > frames[i - 2].feetDist
                        && (Math.Abs(frames[i].feetDist - frames[i - 1].feetDist) > lambda);

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
            return mediaLungPass;
        }
    }
}