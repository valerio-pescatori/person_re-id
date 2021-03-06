using UnityEngine;
using UnityEngine.UI;
using System;
using ObjectsForJson;
using Debug = UnityEngine.Debug;
using System.IO;

// 0  => "mixamorig9:Neck",
// 1  => "mixamorig9:Spine2",
// 2  => "mixamorig9:RightArm", (shoulder)
// 3  => "mixamorig9:RightForeArm", (elbow)
// 4  => "mixamorig9:RightHand", (wrist)
// 5  => "mixamorig9:LeftArm", (shoulder)
// 6  => "mixamorig9:LeftForeArm", (elbow)
// 7  => "mixamorig9:LeftHand", (wrist)
// 8  => "mixamorig9:Hips", (hip middle)
// 9  => "mixamorig9:RightUpLeg", (hip right)
// 10 => "mixamorig9:RightLeg", (knee)
// 11 => "mixamorig9:RightFoot", (ankle)
// 12 => "mixamorig9:LeftUpLeg", (hip left)
// 13 => "mixamorig9:LeftLeg", (knee)
// 14 => "mixamorig9:LeftFoot", (ankle)
// 15 => "mixamorig9:Head",
// 16 => "mixamorig9:LeftToeBase",
// 17 => "mixamorig9:RightToeBase"
// 18 => "mixamorig9:Spine1"
// 19 => "mixamorig9:Spine"

public class ReId : MonoBehaviour
{
    public GameObject[] joints = new GameObject[20];
    public Text textObject;
    public Animation anim;
    // indice dell'animazione in riproduzione
    private int currAnim;
    public JAnimation[] jAnims;
    private float characterHeight;
    private Vector3[] lastVel;
    private Vector3[] lastPos;
    // if WALK_PLOT is true, plot of distance between ankles will be plotted for each animation
    public const bool WALK_PLOT = false;
    public const int nSamples = 7;
    public const int numClasses = 56;
    public const string JSON_PATH = @".\Data\data.json";

    // Start is called before the first frame update
    void Start()
    {
        currAnim = 0;
        jAnims = new JAnimation[numClasses * nSamples];
        characterHeight = joints[15].transform.position.y - joints[16].transform.position.y;
        // inizializzo la prima JAnimation
        jAnims[currAnim] = new JAnimation(characterHeight / 110, currAnim);
    }


    // Update is called once per frame
    void Update()
    {
        // se l'Animation component non sta riproducendo alcuna animazione
        if (!anim.isPlaying)
        {
            // serializzo e salvo in json
            SplitAndSave();
            //stop play mode
            UnityEditor.EditorApplication.ExitPlaymode();
        }
        var animId = currAnim % numClasses;
        // se in questo frame sto eseguendo un'animazione diversa da quella precedente
        if (!anim.IsPlaying("mixamo.com" + (animId == 0 ? "" : " " + animId)))
        {
            // chiamo il metodo calculateSteps() dell'animazione appena terminata
            jAnims[currAnim].CalculateSteps(WALK_PLOT);
            Debug.Log(jAnims[currAnim].frames.Count);
            
            // e creo la nuova animazione
            if (currAnim < (numClasses * nSamples) - 1)
                jAnims[++currAnim] = new JAnimation(characterHeight / 110, (animId+1)%numClasses);
        }

        // feature extraction
        var hunchAngles = HunchbackFeature();
        var otAngles = OutToeingFeature();
        var bct = BodyConvexTriangulation();
        var bo = BodyOpenness();
        var positions = Positions();
        var velocities = Velocities(positions);
        var accelerations = Accelerations(velocities);
        var leftFoot = joints[14].transform.position;
        var rightFoot = joints[11].transform.position;
        JFrame f = new JFrame(otAngles[1], otAngles[0], hunchAngles[1],
                    Vector3.Distance(leftFoot, rightFoot), bo[0], bo[1], bct[0], bct[1], bct[2],
                                     accelerations, positions, velocities);
        // salvo il frame corrente
        jAnims[currAnim].AddFrame(f);
        // aggiorno positions e velocities
        lastPos = positions;
        lastVel = velocities;
    }

    private void SplitAndSave()
    {
        // splitto l'array jAnims e lo salvo in pi?? json
        int splitLen = (int)Math.Floor((double)jAnims.Length / 3);
        var arr1 = new JAnimation[splitLen];
        var arr2 = new JAnimation[splitLen];
        var arr3 = new JAnimation[jAnims.Length - (splitLen * 2)];

        Array.Copy(jAnims, 0, arr1, 0, splitLen);
        Array.Copy(jAnims, splitLen, arr2, 0, splitLen);
        Array.Copy(jAnims, splitLen * 2, arr3, 0, jAnims.Length - splitLen * 2);

        int fileCouter = 0;
        File.WriteAllText(JSON_PATH.Insert(JSON_PATH.IndexOf(".j"), fileCouter.ToString()), JsonHelper.ToJson<JAnimation>(arr1));
        fileCouter++;
        File.WriteAllText(JSON_PATH.Insert(JSON_PATH.IndexOf(".j"), fileCouter.ToString()), JsonHelper.ToJson<JAnimation>(arr2));
        fileCouter++;
        File.WriteAllText(JSON_PATH.Insert(JSON_PATH.IndexOf(".j"), fileCouter.ToString()), JsonHelper.ToJson<JAnimation>(arr3));
    }
    private Vector3[] Positions()
    {
        var pos = new Vector3[joints.Length];
        for (var i = 0; i < joints.Length; i++)
            pos[i] = joints[i].transform.position;
        return pos;
    }
    private Vector3[] Velocities(Vector3[] positions)
    {
        Vector3[] velocities = new Vector3[positions.Length];
        for (int i = 0; i < positions.Length; i++)
            if (lastPos != null)
                velocities[i] = (positions[i] - lastPos[i]) / Time.deltaTime;
            else
                velocities[i] = positions[i] / Time.deltaTime;
        return velocities;
    }
    private Vector3[] Accelerations(Vector3[] velocities)
    {
        Vector3[] accelerations = new Vector3[velocities.Length];
        for (int i = 0; i < velocities.Length; i++)
            if (lastVel! != null)
                accelerations[i] = (velocities[i] - lastVel[i]) / Time.deltaTime;
            else
                accelerations[i] = velocities[i] / Time.deltaTime;

        return accelerations;
    }
    private float[] HunchbackFeature()
    {
        // creo dei triangoli usando i punti del bacino, schiena e collo
        // misuro poi i vari angoli e cerco di capire se la persona ?? curva

        // punti dall'alto al basso:
        // 0 => neck
        // 18 => spine 1
        // 8 => hips

        //triangolo: hips, spine1, neck
        float[] angles = CalculateAngles(joints[8].transform.position, joints[18].transform.position, joints[0].transform.position);

        // controllo non ci siano NaN (compare 1 NaN nel primo frame di una sola animazione)
        for (int i = 0; i < angles.Length; i++)
            if (angles[i] == float.NaN)
                angles[i] = 0;

        return angles;
    }
    private float[] OutToeingFeature()
    {
        // creo triangoli usando caviglie e punta del piede
        // misuro poi l'angolo della caviglia (+- 90?? dovrebbe essere la norma)

        // punti:
        // 11 => RightFoot (ankle)
        // 17 => RightToeBase
        // 14 => LeftFoot
        // 16 => LeftToeBase

        // 3?? punto (si da per scontato che il persoanggio cammina in direzione di z crescente, quindi si prende punto per rendere il terzo lato perpendicolare 
        // alla direzione in cui cammina)
        var point3r = new Vector3(joints[11].transform.position.x - 2, joints[11].transform.position.y, joints[11].transform.position.z);
        var point3l = new Vector3(joints[14].transform.position.x + 2, joints[14].transform.position.y, joints[14].transform.position.z);

        var leftFootAngles = CalculateAngles(joints[16].transform.position, joints[14].transform.position, point3l);
        var rightFootAngles = CalculateAngles(joints[17].transform.position, joints[11].transform.position, point3r);

        return new float[] { leftFootAngles[1], rightFootAngles[1] };
    }
    private float[] BodyOpenness()
    {
        Vector3 hipMiddle = joints[8].transform.position;
        Vector3 kneeLeft = joints[13].transform.position;
        Vector3 kneeRight = joints[10].transform.position;
        Vector3 neck = joints[1].transform.position; //(in realt?? ?? Spine2 invece che Neck)
        Vector3 elbowLeft = joints[6].transform.position;
        Vector3 elbowRight = joints[3].transform.position;
        Vector3 ankleLeft = joints[11].transform.position;
        Vector3 ankleRight = joints[14].transform.position;
        Vector3 anklesAvg = Vector3.zero;
        for (int i = 0; i < 3; i++)
        {
            anklesAvg[i] = (ankleLeft[i] + ankleRight[i]) / 2;
        }

        float[] bodyOpenness = new float[2];
        bodyOpenness[0] = Vector3.Distance(neck, hipMiddle) / Vector3.Distance(elbowLeft, elbowRight); //upper
        bodyOpenness[1] = Vector3.Distance(hipMiddle, anklesAvg) / Vector3.Distance(kneeLeft, kneeRight); //lower
        return bodyOpenness;
    }
    private float[] BodyConvexTriangulation()
    {
        float[] res = new float[3];
        // upper body bct
        res[0] = BodyConvexTriangulation(joints[4].transform.position,
                                         joints[7].transform.position,
                                         joints[1].transform.position);
        // lower body bct
        res[1] = BodyConvexTriangulation(joints[8].transform.position,
                                         joints[11].transform.position,
                                         joints[14].transform.position);
        // full body bct
        res[2] = BodyConvexTriangulation(joints[14].transform.position,
                                         joints[11].transform.position,
                                         joints[0].transform.position);
        return res;
    }
    private float[] CalculateAngles(Vector3 point1, Vector3 point2, Vector3 point3)
    {
        // lunghezze lati ab, ac, bc
        float lungA = Vector3.Distance(point1, point2);
        float lungB = Vector3.Distance(point1, point3);
        float lungC = Vector3.Distance(point2, point3);

        // lungh al quadrato
        float aSqr = lungA * lungA;
        float bSqr = lungB * lungB;
        float cSqr = lungC * lungC;

        // alfa e beta
        float thetaA = (float)Math.Acos((double)((bSqr + cSqr - aSqr) / (2 * lungB * lungC)));
        float thetaB = (float)Math.Acos((double)((aSqr + cSqr - bSqr) / (2 * lungA * lungC)));

        // converto in gradi
        thetaA *= 180.0f / (float)Math.PI;
        thetaB *= 180.0f / (float)Math.PI;


        // ricavo gamma
        float thetaG = 180.0f - thetaA - thetaB;
        return new float[] { thetaA, thetaB, thetaG };
    }
    private float BodyConvexTriangulation(Vector3 point1, Vector3 point2, Vector3 point3)
    {
        float[] angles = CalculateAngles(point1, point2, point3);
        return (angles[1] / angles[0]) - (angles[2] / angles[0]);
    }
    private void OnDrawGizmos()
    {
        float radius = 0.06f;
        // first sphere
        Gizmos.color = Color.black;
        Gizmos.DrawSphere(joints[15].transform.position, radius);
        // spine2 to spine1
        DrawLineAndSphere(joints[1].transform.position, joints[18].transform.position, Color.black);
        // spine1 to spine
        DrawLineAndSphere(joints[18].transform.position, joints[19].transform.position, Color.black);
        // spine to hip
        DrawLineAndSphere(joints[19].transform.position, joints[8].transform.position, Color.black);
        // neck to head
        DrawLineAndSphere(joints[15].transform.position, joints[0].transform.position, Color.black);
        // hip middle to l hip 
        DrawLineAndSphere(joints[8].transform.position, joints[12].transform.position, Color.black);
        // l hip to l knee
        DrawLineAndSphere(joints[12].transform.position, joints[13].transform.position, Color.black);
        // l knee to l ankle
        DrawLineAndSphere(joints[13].transform.position, joints[14].transform.position, Color.black);
        // l ankle to l foot
        DrawLineAndSphere(joints[14].transform.position, joints[16].transform.position, Color.black);
        // hip middle to r hip
        DrawLineAndSphere(joints[8].transform.position, joints[9].transform.position, Color.black);
        // r hip to r knee
        DrawLineAndSphere(joints[9].transform.position, joints[10].transform.position, Color.black);
        // r knee to r ankle
        DrawLineAndSphere(joints[10].transform.position, joints[11].transform.position, Color.black);
        // r ankle to r foot
        DrawLineAndSphere(joints[11].transform.position, joints[17].transform.position, Color.black);
        // neck to chest
        DrawLineAndSphere(joints[0].transform.position, joints[1].transform.position, Color.black);
        // chest to  shoulder
        DrawLineAndSphere(joints[1].transform.position, joints[5].transform.position, Color.black);
        // l shoulder to l elbow
        DrawLineAndSphere(joints[5].transform.position, joints[6].transform.position, Color.black);
        // l elbow to l wrist
        DrawLineAndSphere(joints[6].transform.position, joints[7].transform.position, Color.black);
        // chest to r shoulder
        DrawLineAndSphere(joints[1].transform.position, joints[2].transform.position, Color.black);
        // r shoulder to r elbow
        DrawLineAndSphere(joints[2].transform.position, joints[3].transform.position, Color.black);
        // r elbow to r wrist
        DrawLineAndSphere(joints[3].transform.position, joints[4].transform.position, Color.black);

        // step avg
        Gizmos.color = Color.red;
        Gizmos.DrawSphere(joints[11].transform.position, radius);
        Gizmos.DrawSphere(joints[14].transform.position,radius);
        Gizmos.DrawLine(joints[11].transform.position, joints[14].transform.position);
        
        // // ot
        // var point3r = new Vector3(joints[11].transform.position.x - 2, joints[11].transform.position.y, joints[11].transform.position.z);
        // DrawLineAndSphere(joints[11].transform.position, joints[17].transform.position, Color.red);
        // DrawLineAndSphere(joints[17].transform.position, point3r, Color.red);
        // DrawLineAndSphere(point3r, joints[11].transform.position, Color.red);

        // // hb
        // DrawLineAndSphere(joints[8].transform.position, joints[18].transform.position, Color.red);
        // DrawLineAndSphere(joints[18].transform.position, joints[0].transform.position, Color.red);
        // DrawLineAndSphere(joints[0].transform.position, joints[8].transform.position, Color.red);

        // //bo 1, 8, 6, 3
        // DrawLineAndSphere(joints[1].transform.position, joints[8].transform.position, Color.blue);
        // DrawLineAndSphere(joints[8].transform.position, joints[1].transform.position, Color.blue);
        // DrawLineAndSphere(joints[6].transform.position, joints[3].transform.position, Color.blue);
        // DrawLineAndSphere(joints[3].transform.position, joints[6].transform.position, Color.blue);
        // // 13 10
        // Vector3 ankleLeft = joints[11].transform.position;
        // Vector3 ankleRight = joints[14].transform.position;
        // Vector3 anklesAvg = Vector3.zero;
        // for (int i = 0; i < 3; i++)
        // {
        //     anklesAvg[i] = (ankleLeft[i] + ankleRight[i]) / 2;
        // }
        // DrawLineAndSphere(joints[8].transform.position, anklesAvg, Color.red);
        // DrawLineAndSphere(anklesAvg, joints[8].transform.position, Color.red);
        // DrawLineAndSphere(joints[13].transform.position, joints[10].transform.position, Color.red);
        // DrawLineAndSphere(joints[10].transform.position, joints[13].transform.position, Color.red);

        // //bct
        // // 4 7 1
        // // 8 11 14
        // // 14 11 0
        // DrawLineAndSphere(joints[4].transform.position, joints[7].transform.position, Color.blue);
        // DrawLineAndSphere(joints[7].transform.position, joints[1].transform.position, Color.blue);
        // DrawLineAndSphere(joints[1].transform.position, joints[4].transform.position, Color.blue);

        // DrawLineAndSphere(joints[8].transform.position, joints[11].transform.position, Color.yellow);
        // DrawLineAndSphere(joints[11].transform.position, joints[14].transform.position, Color.yellow);
        // DrawLineAndSphere(joints[14].transform.position, joints[8].transform.position, Color.yellow);

        // DrawLineAndSphere(joints[14].transform.position, joints[11].transform.position, Color.red);
        // DrawLineAndSphere(joints[11].transform.position, joints[0].transform.position, Color.red);
        // DrawLineAndSphere(joints[0].transform.position, joints[14].transform.position, Color.red);
    }

    // private void OnDrawGizmos()
    // {
    //     float radius = 0.06f;
    //     // first sphere
    //     Gizmos.color = Color.magenta;
    //     Gizmos.DrawSphere(joints[15].transform.position, radius);
    //     // spine2 to spine1
    //     DrawLineAndSphere(joints[1].transform.position, joints[18].transform.position, new Color(1, 0.41f, 0.2f, 1));
    //     // spine1 to spine
    //     DrawLineAndSphere(joints[18].transform.position, joints[19].transform.position, new Color(1, 0.56f, 0.4f, 1));
    //     // spine to hip
    //     DrawLineAndSphere(joints[19].transform.position, joints[8].transform.position, new Color(1, 0.61f, 0.2f, 1));
    //     // neck to head
    //     DrawLineAndSphere(joints[15].transform.position, joints[0].transform.position, Color.magenta);
    //     // hip middle to l hip 
    //     DrawLineAndSphere(joints[8].transform.position, joints[12].transform.position, Color.blue);
    //     // l hip to l knee
    //     DrawLineAndSphere(joints[12].transform.position, joints[13].transform.position, Color.blue);
    //     // l knee to l ankle
    //     DrawLineAndSphere(joints[13].transform.position, joints[14].transform.position, Color.blue);
    //     // l ankle to l foot
    //     DrawLineAndSphere(joints[14].transform.position, joints[16].transform.position, Color.blue);
    //     // hip middle to r hip
    //     DrawLineAndSphere(joints[8].transform.position, joints[9].transform.position, Color.cyan);
    //     // r hip to r knee
    //     DrawLineAndSphere(joints[9].transform.position, joints[10].transform.position, Color.cyan);
    //     // r knee to r ankle
    //     DrawLineAndSphere(joints[10].transform.position, joints[11].transform.position, Color.cyan);
    //     // r ankle to r foot
    //     DrawLineAndSphere(joints[11].transform.position, joints[17].transform.position, Color.cyan);
    //     // neck to chest
    //     DrawLineAndSphere(joints[0].transform.position, joints[1].transform.position, Color.red);
    //     // chest to  shoulder
    //     DrawLineAndSphere(joints[1].transform.position, joints[5].transform.position, Color.green);
    //     // l shoulder to l elbow
    //     DrawLineAndSphere(joints[5].transform.position, joints[6].transform.position, Color.green);
    //     // l elbow to l wrist
    //     DrawLineAndSphere(joints[6].transform.position, joints[7].transform.position, Color.green);
    //     // chest to r shoulder
    //     DrawLineAndSphere(joints[1].transform.position, joints[2].transform.position, Color.yellow);
    //     // r shoulder to r elbow
    //     DrawLineAndSphere(joints[2].transform.position, joints[3].transform.position, Color.yellow);
    //     // r elbow to r wrist
    //     DrawLineAndSphere(joints[3].transform.position, joints[4].transform.position, Color.yellow);

    //     Gizmos.color = Color.black;
    //     // step avg
    //     // Gizmos.DrawSphere(joints[11].transform.position, radius);
    //     // Gizmos.DrawSphere(joints[14].transform.position,radius);
    //     // Gizmos.DrawLine(joints[11].transform.position, joints[14].transform.position);
    //     // ot
    //     // var point3r = new Vector3(joints[11].transform.position.x - 2, joints[11].transform.position.y, joints[11].transform.position.z);
    //     // var point3l = new Vector3(joints[14].transform.position.x + 2, joints[14].transform.position.y, joints[14].transform.position.z);
    //     // DrawLineAndSphere(joints[11].transform.position, joints[17].transform.position, Color.black);
    //     // DrawLineAndSphere(joints[17].transform.position, point3r, Color.black);
    //     // DrawLineAndSphere(point3r, joints[11].transform.position, Color.black);
    //     // hb
    //     // DrawLineAndSphere(joints[8].transform.position, joints[18].transform.position, Color.black);
    //     // DrawLineAndSphere(joints[18].transform.position, joints[0].transform.position, Color.black);
    //     // DrawLineAndSphere(joints[0].transform.position, joints[8].transform.position, Color.black);

    //     // //bo 1, 8, 6, 3
    //     // DrawLineAndSphere(joints[1].transform.position, joints[8].transform.position, Color.black);
    //     // DrawLineAndSphere(joints[8].transform.position, joints[1].transform.position, Color.black);
    //     // DrawLineAndSphere(joints[6].transform.position, joints[3].transform.position, Color.black);
    //     // DrawLineAndSphere(joints[3].transform.position, joints[6].transform.position, Color.black);
    //     // // 13 10
    //     // Vector3 ankleLeft = joints[11].transform.position;
    //     // Vector3 ankleRight = joints[14].transform.position;
    //     // Vector3 anklesAvg = Vector3.zero;
    //     // for (int i = 0; i < 3; i++)
    //     // {
    //     //     anklesAvg[i] = (ankleLeft[i] + ankleRight[i]) / 2;
    //     // }
    //     // DrawLineAndSphere(joints[8].transform.position, anklesAvg, Color.red);
    //     // DrawLineAndSphere(anklesAvg, joints[8].transform.position, Color.red);
    //     // DrawLineAndSphere(joints[13].transform.position, joints[10].transform.position, Color.red);
    //     // DrawLineAndSphere(joints[10].transform.position, joints[13].transform.position, Color.red);

    //     //bct
    //     // 4 7 1
    //     // 8 11 14
    //     // 14 11 0
    //     DrawLineAndSphere(joints[4].transform.position, joints[7].transform.position, Color.blue);
    //     DrawLineAndSphere(joints[7].transform.position, joints[1].transform.position, Color.blue);
    //     DrawLineAndSphere(joints[1].transform.position, joints[4].transform.position, Color.blue);

    //     DrawLineAndSphere(joints[8].transform.position, joints[11].transform.position, Color.yellow);
    //     DrawLineAndSphere(joints[11].transform.position, joints[14].transform.position, Color.yellow);
    //     DrawLineAndSphere(joints[14].transform.position, joints[8].transform.position, Color.yellow);

    //     DrawLineAndSphere(joints[14].transform.position, joints[11].transform.position, Color.red);
    //     DrawLineAndSphere(joints[11].transform.position, joints[0].transform.position, Color.red);
    //     DrawLineAndSphere(joints[0].transform.position, joints[14].transform.position, Color.red);
    // }
    private void DrawLineAndSphere(Vector3 from, Vector3 to, Color color, float radius = 0.06f)
    {
        Gizmos.color = color;
        Gizmos.DrawLine(from, to);
        Gizmos.DrawSphere(to, radius);
    }
}