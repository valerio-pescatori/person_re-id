using System.Text;
using UnityEngine;
using UnityEngine.UI;
using System;
using System.Diagnostics;
using ObjectsForJson;
using Debug = UnityEngine.Debug;
using System.IO;
using System.Collections.Generic;

// 0  => "mixamorig9:Neck",
// 1  => "mixamorig9:Spine2",
// 2  => "mixamorig9:RightArm", (shoulder)
// 3  => "mixamorig9:RightForeArm", (elbow)
// 4  => "mixamorig9:RightHand", 
// 5  => "mixamorig9:LeftArm", (shoulder)
// 6  => "mixamorig9:LeftForeArm", (elbow)
// 7  => "mixamorig9:LeftHand",
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
    // array con indici shuffle per la creazione di testing dataset
    private int[] randIndexes;
    public JAnimation[] jAnims;
    private float characterHeight;
    public const bool PLOT = false;
    // la variabile training indica se sto registrando dati per la fase di training oppure per la fase di testing
    // nel primo caso le animazioni vengono eseguite e salvate in ordine crescende da "mixamo.com" a "mixamo.com 55"
    // nel secondo caso vengono salvate in ordine casuale
    public static bool TRAINING = false;
    public const int numClasses = 56;
    public const string JSON_TESTING_PATH = @".\testing.json";
    public const string JSON_TRAINING_PATH = @".\training.json";



    // Start is called before the first frame update
    void Start()
    {
        currAnim = 0;
        jAnims = new JAnimation[numClasses];
        characterHeight = joints[15].transform.position.y - joints[16].transform.position.y;

        // randomizzo gli indici per il testing
        randIndexes = new int[numClasses];
        for (int i = 0; i < numClasses; i++)
            randIndexes[i] = i;
        System.Random rng = new System.Random();
        ShuffleArray(randIndexes, rng);
        // inizializzo la prima JAnimation
        jAnims[GetIndex()] = new JAnimation(characterHeight / 110, currAnim);
    }


    // Update is called once per frame
    void Update()
    {
        // feature exctraction
        var hunchAngles = HunchbackFeature();
        var otAngles = OutToeingFeature();
        var bct = BodyConvexTriangulation();
        var bo = BodyOpenness();

        int index = GetIndex();

        // se in questo frame eseguo un'animazione diversa da quella precedente
        if (!anim.IsPlaying("mixamo.com" + (currAnim == 0 ? "" : " " + currAnim.ToString())))
        {
            // nuova animazione
            // chiamo il metodo calculateSteps() dell'animation
            jAnims[index].CalculateSteps(PLOT);
            Debug.Log(jAnims[index].frames.Count);

            // creo la nuova animazione
            if (currAnim < 55)
            {
                currAnim++;
                //aggiorno l'indice
                index = GetIndex();
                jAnims[index] = new JAnimation(characterHeight / 110, currAnim);

            }
        }

        // se l'Animation component non sta riproducendo alcuna animazione
        if (!anim.isPlaying)
        {
            // serializzo e salvo in json
            string json = JsonHelper.ToJson<JAnimation>(jAnims, true);
            if (TRAINING)
                File.WriteAllText(JSON_TRAINING_PATH, json);
            else
                File.WriteAllText(JSON_TESTING_PATH, json);

            //stop play mode
            UnityEditor.EditorApplication.ExitPlaymode();
        }

        // calcolo e salvo i valori per il frame attuale
        var leftFoot = joints[14].transform.position;
        var rightFoot = joints[11].transform.position;
        JFrame f = new JFrame(otAngles[1], otAngles[0], hunchAngles[1],
                    Vector3.Distance(leftFoot, rightFoot), bo[0], bo[1], bct[0], bct[1], bct[2]);
        jAnims[index].AddFrame(f);


        // printo gli output sulla canvas
        StringBuilder sb = new StringBuilder("OUTPUTS\n");
        // Hunchback outs
        sb.AppendLine("\nHUNCHBACK\nSpine1 angle: " + hunchAngles[1].ToString());
        // OutToeing outs
        sb.AppendLine("\nOUTTOEING\nLeft foot angle: " + otAngles[0].ToString());
        sb.AppendLine("Right foot angle: " + otAngles[1].ToString());
        // print
        textObject.text = sb.ToString();

    }

    private int GetIndex()
    {
        if (TRAINING)
            return currAnim;
        else
            return randIndexes[currAnim];
    }

    private float[] HunchbackFeature()
    {
        // creo dei triangoli usando i punti del bacino, schiena e collo
        // misuro poi i vari angoli e cerco di capire se la persona è curva

        // punti dall'alto al basso:
        // 0 => neck
        // 1 => spine 2
        // 18 => spine 1
        // 19 => spine
        // 8 => hips

        //primo triangolo: hips, spine1, neck
        float[] angles = CalculateAngles(joints[8].transform.position, joints[18].transform.position, joints[0].transform.position);

        //textObject.text = "angoli: \n" + angles[0].ToString() + "\n" + angles[1].ToString() + "\n" + angles[2].ToString()+ "\n\n" + angles[1]/180;
        return angles;
    }

    private float[] OutToeingFeature()
    {
        // creo triangoli usando caviglie e punta del piede
        // misuro poi l'angolo della caviglia (+- 90° dovrebbe essere la norma)

        // punti:
        // 11 => RightFoot (ankle)
        // 17 => RightToeBase
        // 14 => LeftFoot
        // 16 => LeftToeBase

        // 3° punto (si da per scontato che il persoanggio cammina in direzione di z crescente, quindi si prende punto per rendere il terzo lato perpendicolare 
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
        Vector3 neck = joints[1].transform.position; //(in realtà è Spine2 invece che Neck)
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
        // first sphere
        Gizmos.color = Color.magenta;
        Gizmos.DrawSphere(joints[15].transform.position, 0.06f);
        // neck to head
        DrawLineAndSphere(joints[15].transform.position, joints[0].transform.position, Color.magenta);
        // neck to chest
        DrawLineAndSphere(joints[0].transform.position, joints[1].transform.position, Color.red);
        // chest to  shoulder
        DrawLineAndSphere(joints[1].transform.position, joints[5].transform.position, Color.green);
        // l shoulder to l elbow
        DrawLineAndSphere(joints[5].transform.position, joints[6].transform.position, Color.green);
        // l elbow to l wrist
        DrawLineAndSphere(joints[6].transform.position, joints[7].transform.position, Color.green);
        // chest to r shoulder
        DrawLineAndSphere(joints[1].transform.position, joints[2].transform.position, Color.yellow);
        // r shoulder to r elbow
        DrawLineAndSphere(joints[2].transform.position, joints[3].transform.position, Color.yellow);
        // r elbow to r wrist
        DrawLineAndSphere(joints[3].transform.position, joints[4].transform.position, Color.yellow);
        // spine2 to spine1
        DrawLineAndSphere(joints[1].transform.position, joints[18].transform.position, new Color(1, 0.41f, 0.2f, 1));
        // spine1 to spine
        DrawLineAndSphere(joints[18].transform.position, joints[19].transform.position, new Color(1, 0.56f, 0.4f, 1));
        // spine to hip
        DrawLineAndSphere(joints[19].transform.position, joints[8].transform.position, new Color(1, 0.61f, 0.2f, 1));
        // hip middle to l hip 
        DrawLineAndSphere(joints[8].transform.position, joints[12].transform.position, Color.blue);
        // l hip to l knee
        DrawLineAndSphere(joints[12].transform.position, joints[13].transform.position, Color.blue);
        // l knee to l ankle
        DrawLineAndSphere(joints[13].transform.position, joints[14].transform.position, Color.blue);
        // l ankle to l foot
        DrawLineAndSphere(joints[14].transform.position, joints[16].transform.position, Color.blue);
        // hip middle to r hip
        DrawLineAndSphere(joints[8].transform.position, joints[9].transform.position, Color.cyan);
        // r hip to r knee
        DrawLineAndSphere(joints[9].transform.position, joints[10].transform.position, Color.cyan);
        // r knee to r ankle
        DrawLineAndSphere(joints[10].transform.position, joints[11].transform.position, Color.cyan);
        // r ankle to r foot
        DrawLineAndSphere(joints[11].transform.position, joints[17].transform.position, Color.cyan);
    }

    private void DrawLineAndSphere(Vector3 from, Vector3 to, Color color, float radius = 0.06f)
    {
        Gizmos.color = color;
        Gizmos.DrawLine(from, to);
        Gizmos.DrawSphere(to, radius);
    }

    private static void ShuffleArray(int[] array, System.Random rng)
    {
        int n = array.Length;
        while (n > 1)
        {
            int k = rng.Next(n--);
            int temp = array[n];
            array[n] = array[k];
            array[k] = temp;
        }
    }

}