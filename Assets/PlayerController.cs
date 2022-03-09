using UnityEngine;
using System;
using System.Collections.Generic;
public class PlayerController : MonoBehaviour
{
    public const int NUMBER_OF_FRAMES = 5000;
    Animation animationComponent;
    public const int targetFrameRate = 60;  // setto il target 3 fps in più per evitare che scenda sotto 

    // fps limitator    
    void Awake()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = targetFrameRate;
    }

    // Use this for initialization
    void Start()
    {
        animationComponent = GetComponent<Animation>();
        
        // tests
        // //4.63 1.09 -3,54
        // string anim = "mixamo.com 16";
        // animationComponent.Play(anim);
        // for(int x = 1; x < 57; x++)
        //     // animationComponent.PlayQueued(anim + " " + x);
        //     animationComponent.PlayQueued(anim);


        // la prima animazione va messa in play manualmente
        animationComponent.Play("mixamo.com");


        // costruisco la coda di riproduzione
        var playQueue = GeneratePlayQueue();

        // riproduco ogni animazione nella coda per 750 frames
        foreach (var anim in playQueue)
            for (int i = 0; i < (int)Math.Ceiling(NUMBER_OF_FRAMES / (animationComponent[anim].length * (targetFrameRate - 3))); i++)
                animationComponent.PlayQueued(anim);
    }

    private Queue<String> GeneratePlayQueue()
    {
        var queue = new Queue<String>();
        for (int j = 0; j < ReId.nSamples; j++)
            for (int i = 0; i < 56; i++)
                queue.Enqueue(i == 0 ? "mixamo.com" : "mixamo.com " + i);
        return queue;
    }

    // Update is called once per frame
    void Update()
    {
        if (Application.targetFrameRate != targetFrameRate)
            Application.targetFrameRate = targetFrameRate;
    }
}


