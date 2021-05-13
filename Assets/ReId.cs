using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using System;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using SwingClass;

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

public class ReId : MonoBehaviour
{
    public GameObject[] joints = new GameObject[18];
    public Text textObject;

    // campi per misura1
    private float characterHeight;
    // campi per steplength
    private SwingObject feetSwing;
    private SwingObject armsSwing;
    public const string pyScriptPath = @"C:\Users\pesca\Person Re-Id\Assets\plot.py";

    // Start is called before the first frame update
    void Start()
    {
        characterHeight = joints[15].transform.position.y - joints[16].transform.position.y;
        feetSwing = new SwingObject(characterHeight/110);
        armsSwing = new SwingObject(characterHeight/110);
    }

    // Update is called once per frame
    void Update()
    {
        // StepLength();
        ArmSwingLength();
    }


    // StepLength
    private void StepLength()
    {   
        // calcolo ampiezza del passo
        // la calcolo come la distanza media (prendendo solo le coordinate x e z) dei piedi
        Vector2 leftFoot = new Vector2(joints[14].transform.position.x, joints[14].transform.position.z);
        Vector2 rightFoot = new Vector2(joints[11].transform.position.x, joints[11].transform.position.z);
        feetSwing.AvgDistance(leftFoot, rightFoot);


        if(feetSwing.Distances.Count == 300)
        {
            float sum = 0f;
            foreach(float f in feetSwing.SwingPeaks)
                sum += f;
            float avg = sum/feetSwing.SwingPeaks.Count;
            Debug.Log("AVG: " + avg );
            Debug.Log("H: " + characterHeight);
            Debug.Log("RAPPORTO: " + characterHeight/avg); // >1 --> passi più corti dell'altezza
            PlotValues(feetSwing);
        }
    }

    private void ArmSwingLength()
    {
        Vector2 leftArm = new Vector2(joints[7].transform.position.x, joints[7].transform.position.z);
        Vector2 rightArm = new Vector2(joints[4].transform.position.x, joints[4].transform.position.z);
        armsSwing.AvgDistance(leftArm, rightArm);

        if(armsSwing.Distances.Count == 300)
        {
            float sum = 0f;
            foreach(float f in armsSwing.SwingPeaks)
                sum += f;
            float avg = sum/armsSwing.SwingPeaks.Count;
            Debug.Log("AVG: " + avg );
            Debug.Log("H: " + characterHeight);
            Debug.Log("RAPPORTO: " + characterHeight/avg); // >1 --> bracciate più corte dell'altezza
            PlotValues(armsSwing);
        }
    }

  
    public void PlotValues(SwingObject obj)
    {
        StringBuilder sb = new StringBuilder();
            for (int i = 0; i < obj.Distances.Count -1 ; i++)
                sb.Append(obj.Distances[i].ToString("F4").Replace(",", ".") + ",");
            sb.Append(obj.Distances[obj.Distances.Count-1].ToString("F4").Replace(",", "."));
        string arg1 = sb.ToString();
        string arg2 = obj.Distances.Count.ToString();

        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = @"C:\Users\pesca\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\python3.exe";
        start.Arguments = $" -i \"{pyScriptPath}\" \"{arg1}\" \"{arg2}\"";
        start.UseShellExecute = true;
        start.CreateNoWindow = true;
        using (Process process = Process.Start(start))
        {
            
        }
    }

    private void PrintJointsCoord()
    {
        StringBuilder sb = new StringBuilder("Joint positions\n");
        foreach( GameObject joint in joints)
        {
            sb.AppendLine(joint.name +  " : " + joint.transform.position);
        }
        textObject.text = sb.ToString();
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
        for (int i = 0; i<3; i++)
        {
            anklesAvg[i] = (ankleLeft[i] + ankleRight[i]) / 2;
        }

        float[] bodyOpenness = new float[2];
        bodyOpenness[0] = Vector3.Distance(hipMiddle, anklesAvg) / Vector3.Distance(kneeLeft, kneeRight); //lower
        bodyOpenness[1] = Vector3.Distance(neck, hipMiddle) / Vector3.Distance(elbowLeft, elbowRight); //upper
        return bodyOpenness;
    }

    private float[] BodyConvexTriangulation()
    {
        float[] res = new float[3];
        // lower body bct
        res[0] = BodyConvexTriangulation(joints[8].transform.position,
                                         joints[11].transform.position,
                                         joints[14].transform.position);
        // upper body bct
        res[1] = BodyConvexTriangulation(joints[4].transform.position,
                                         joints[7].transform.position,
                                         joints[1].transform.position);
        // full body bct
        res[2] = BodyConvexTriangulation(joints[14].transform.position,
                                         joints[11].transform.position,
                                         joints[0].transform.position);
        return res;
    }

    private float BodyConvexTriangulation(Vector3 point1, Vector3 point2, Vector3 point3)
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
        float thetaA = (float)Math.Acos( (double) ((bSqr + cSqr - aSqr) / (2* lungB * lungC)) );
        float thetaB = (float)Math.Acos( (double) ((aSqr + cSqr - bSqr) / (2* lungA * lungC)) );

        // converto in gradi
        thetaA *= 180.0f/(float)Math.PI; 
        thetaB *= 180.0f/(float)Math.PI;

        
        // ricavo gamma
        float thetaG = 180.0f - thetaA - thetaB;
        return (thetaB / thetaA) - (thetaG / thetaA);
    }
}