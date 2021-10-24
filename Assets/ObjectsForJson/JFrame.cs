using System;
using UnityEngine;

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
        public float[] positions;
        public float[] velocities;
        public float[] accelerations;
        public float feetDist { get; private set; }

        public JFrame(float outToeingR, float outToeingL, float hunchback, float feetDist,
                        float bodyOpennessU, float bodyOpennessL, float bctU, float bctL, float bctF, Vector3[] accel, Vector3[] pos, Vector3[] vel)
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

            this.velocities = new float[pos.Length * 3];
            this.accelerations = new float[pos.Length * 3];
            this.positions = new float[pos.Length * 3];
            // flattening degli array di vector3
            for (int i = 0, j = 0; i < pos.Length; i++, j += 3)
            {
                for (int x = 0; x < 3; x++)
                {
                    positions[j + x] = pos[i][x];
                }
            }

            // accel e vel potrebbero essere null (1° frame)
            if (accel != null)
                for (int i = 0, j = 0; i < accel.Length; i++, j += 3)
                    for (int x = 0; x < 3; x++)
                    {
                        velocities[j + x] = vel[i][x];
                        accelerations[j + x] = accel[i][x];
                    }
            else
                for (int i = 0; i < accelerations.Length; i++)
                {
                    velocities[i] = 0f;
                    accelerations[i] = 0f;
                }
        }
    }
}
