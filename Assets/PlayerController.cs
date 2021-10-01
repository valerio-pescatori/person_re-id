using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

public class PlayerController : MonoBehaviour
{
    public Animator anim;
    public AnimationClip[] animations;
    public new Animation animation;
    PlayableGraph playableGraph;

    private class AnimatorReference{
        public Animator anim;
        public AnimationClip clip;
        public PlayableGraph playableGraph;

        public AnimatorReference(Animator anim, AnimationClip clip, PlayableGraph graph){
            this.anim = anim;
            this.clip = clip;
            this.playableGraph = graph;
        } 

        
    }
    private static void CountLoops(object obj)
    {   
        Debug.Log("test");
        AnimatorReference reference = (AnimatorReference) obj;
        AnimationPlayableUtilities.PlayClip(reference.anim,
                                            reference.clip,
                                            out reference.playableGraph);

    }
     // Use this for initialization
    void Start() 
    {  
        // test playables
        Timer timer = null;
        // number of loops for each clip

        AnimationPlayableUtilities.PlayClip(GetComponent<Animator>(),
                                            animations[0],
                                            out playableGraph);

        foreach (var clip in animations)
        {
            for( var loops = 0; loops < 5; loops++)
            {
                timer = new Timer(CountLoops, new AnimatorReference(GetComponent<Animator>(), clip, playableGraph), 0, (int)clip.length * 1000);
            }
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

