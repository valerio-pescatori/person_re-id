using UnityEngine;
using System;
using System.Collections.Generic;
public class PlayerController : MonoBehaviour
{
    // public Animator animator;
    public const int NUMBER_OF_FRAMES = 750;
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

        // ###################### TEST ###########################
        int c = 0;
        animationComponent.Play("mixamo.com 4" + (c == 0 ? "" : " " + c));
        for (int i = 0; i < 30; i++)
            animationComponent.PlayQueued("mixamo.com 4" + (c == 0 ? "" : " " + c));

        // ########################################################

        // // la prima animazione va messa in play manualmente
        // animationComponent.Play("mixamo.com");

        // // costruisco la coda di riproduzione
        // var playQueue = GeneratePlayQueue();

        // // riproduco ogni animazione nella coda per 750 frames
        // foreach (var anim in playQueue)
        //     for (int i = 0; i < (int)Math.Ceiling(NUMBER_OF_FRAMES / (animationComponent[anim].length * (targetFrameRate - 3))); i++)
        //         animationComponent.PlayQueued(anim);
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

        // tutto il blocco sottostante serve per il controllo del character tramite l'Animator Component
        // (usato per testare il corretto funzionamento delle animazioni)
        // dato che l'esecuzione delle animazioni parte in automatico non ne ho bisogno

        // animator.SetFloat("vertical", Input.GetAxis("Vertical"));
        // animator.SetFloat("horizontal", Input.GetAxis("Horizontal"));

        // if (Input.GetKeyDown(KeyCode.Space))
        // {
        //     animator.SetTrigger("trigger1");
        // }

        // // Lean left
        // if (Input.GetKeyDown(KeyCode.Q))
        //     if (animator.GetFloat("lean") < -0.1)
        //         animator.SetFloat("lean", 0);
        //     else
        //         animator.SetFloat("lean", -1);

        // // Lean right
        // if (Input.GetKeyDown(KeyCode.E))
        //     if (animator.GetFloat("lean") > 0.1)
        //         animator.SetFloat("lean", 0);
        //     else
        //         animator.SetFloat("lean", 1);

        // // Crouch
        // if (Input.GetKeyDown(KeyCode.C))
        //     if (animator.GetFloat("crouch") == 1)
        //         animator.SetFloat("crouch", 0);
        //     else
        //         animator.SetFloat("crouch", 1);

        // // Short walk
        // if (Input.GetKey(KeyCode.T))
        //     animator.SetFloat("shortwalk", 1);
        // else
        //     animator.SetFloat("shortwalk", 0);

        // // Long walk
        // if (Input.GetKey(KeyCode.Y))
        //     animator.SetFloat("longwalk", 1);
        // else
        //     animator.SetFloat("longwalk", 0);

        // // Hunchback Walk
        // if (Input.GetKey(KeyCode.G))
        //     animator.SetFloat("hunchwalk", 1);
        // else
        //     animator.SetFloat("hunchwalk", 0);


        // // Hunchback Idle
        // if (Input.GetKeyDown(KeyCode.H))
        //     if (animator.GetFloat("hunchback") == 1)
        //         animator.SetFloat("hunchback", 0);
        //     else
        //         animator.SetFloat("hunchback", 1);

        // // Back Arc
        // if (Input.GetKeyDown(KeyCode.B))
        //     if (animator.GetFloat("backarc") == 1)
        //         animator.SetFloat("backarc", 0);
        //     else
        //         animator.SetFloat("backarc", 1);

        // // OutToeing1
        // if (Input.GetKeyDown(KeyCode.M))
        //     if (animator.GetFloat("ot1") == 1)
        //         animator.SetFloat("ot1", 0);
        //     else
        //         animator.SetFloat("ot1", 1);
    }
}


