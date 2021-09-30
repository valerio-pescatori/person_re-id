using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

public class PlayerController : MonoBehaviour
{
    // public Animator anim;
    public AnimationClip[] animations;
    public new Animation animation;
    PlayableGraph playableGraph;


     // Use this for initialization
    void Start() 
    {
        // anim = GetComponent<Animator>();    

        // test playables

        // number of loops for each clip
        int loops = 5;
        foreach (var clip in animations)
        {
            // Wrap the clip in a playable
            // Connect the Playable to an output    
            clip.wrapMode = WrapMode.Loop;
            AnimationPlayableUtilities.PlayClip(GetComponent<Animator>(), clip, out playableGraph);
            
            // trova il modo di far loopare una animazione 5 volte e poi cambiare 


        }        
    }

    // Update is called once per frame
    void Update()
    {   


    //     anim.SetFloat("vertical", Input.GetAxis("Vertical"));
    //     anim.SetFloat("horizontal", Input.GetAxis("Horizontal"));

    //     if(Input.GetKeyDown(KeyCode.Space))
    //     {
    //         anim.SetTrigger("trigger1");
    //     }

    //     // Lean left
    //     if(Input.GetKeyDown(KeyCode.Q))
    //         if(anim.GetFloat("lean") < -0.1)
    //             anim.SetFloat("lean", 0);
    //         else
    //             anim.SetFloat("lean", -1);

    //     // Lean right
    //     if(Input.GetKeyDown(KeyCode.E))
    //         if(anim.GetFloat("lean") > 0.1)
    //             anim.SetFloat("lean", 0);
    //         else 
    //             anim.SetFloat("lean", 1);

    //     // Crouch
    //     if (Input.GetKeyDown(KeyCode.C))
    //         if(anim.GetFloat("crouch") == 1)
    //             anim.SetFloat("crouch", 0);
    //         else
    //             anim.SetFloat("crouch", 1);

    //     // Short walk
    //     if (Input.GetKey(KeyCode.T))
    //         anim.SetFloat("shortwalk", 1);
    //     else
    //         anim.SetFloat("shortwalk", 0);

    //     // Long walk
    //     if (Input.GetKey(KeyCode.Y))
    //         anim.SetFloat("longwalk", 1);
    //     else
    //         anim.SetFloat("longwalk", 0);
        
    //     // Hunchback Walk
    //     if (Input.GetKey(KeyCode.G))
    //         anim.SetFloat("hunchwalk", 1);
    //     else
    //         anim.SetFloat("hunchwalk", 0);


    //     // Hunchback Idle
    //     if (Input.GetKeyDown(KeyCode.H))
    //         if(anim.GetFloat("hunchback") == 1)
    //             anim.SetFloat("hunchback", 0);
    //         else
    //             anim.SetFloat("hunchback", 1);
        
    //     // Back Arc
    //     if (Input.GetKeyDown(KeyCode.B))
    //         if(anim.GetFloat("backarc") == 1)
    //             anim.SetFloat("backarc", 0);
    //         else
    //             anim.SetFloat("backarc", 1);
        
    //     // OutToeing1
    //     if (Input.GetKeyDown(KeyCode.M))
    //         if(anim.GetFloat("ot1") == 1)
    //             anim.SetFloat("ot1", 0);
    //         else
    //             anim.SetFloat("ot1", 1);
    }

    void OnDisable()
    {
        // Destroys all Playables and PlayableOutputs created by the graph.
        playableGraph.Destroy();
    }
}
