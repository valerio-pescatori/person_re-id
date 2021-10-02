using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

public class PlayerController : MonoBehaviour
{
    public Animator anim;
    public AnimationClip[] animations;
    PlayableGraph playableGraph;

    public Animation test;

    // Use this for initialization
    void Start()
    {
        playableGraph = PlayableGraph.Create();
        playableGraph.SetTimeUpdateMode(DirectorUpdateMode.GameTime);
        var playableOutput = AnimationPlayableOutput.Create(playableGraph, "Animation", anim);


        // Wrap the clip in a playable
        var clip1 = AnimationClipPlayable.Create(playableGraph, animations[0]);
        var clip2 = AnimationClipPlayable.Create(playableGraph, animations[1]);

        // Use the AnimationMixerPlayable as the source for the AnimationPlayableOutput.
        playableOutput.SetSourcePlayable(clip1);

        // playableGraph.Play();
        test.Play(PlayMode.StopAll);


        // foreach (var clip in animations)
        // {
        //     // Wrap the clip in a playable
        //     var clipPlayable = AnimationClipPlayable.Create(playableGraph, clip);
        //     // Connect the Playable to an output
        //     playableOutput.SetSourcePlayable(clipPlayable);

        //     var loops = 0;

        //     while (loops < 5)
        //     {
        //         playableGraph.Play();
        //         while (playableGraph.IsPlaying())
        //         {
        //             //wait
        //         }
        //         //fine riproduzione, incremento
        //         loops++;
        //     }
        // }
    }

    // Update is called once per frame
    void Update()
    {

        // anim.SetFloat("vertical", Input.GetAxis("Vertical"));
        // anim.SetFloat("horizontal", Input.GetAxis("Horizontal"));

        // if (Input.GetKeyDown(KeyCode.Space))
        // {
        //     anim.SetTrigger("trigger1");
        // }

        // // Lean left
        // if (Input.GetKeyDown(KeyCode.Q))
        //     if (anim.GetFloat("lean") < -0.1)
        //         anim.SetFloat("lean", 0);
        //     else
        //         anim.SetFloat("lean", -1);

        // // Lean right
        // if (Input.GetKeyDown(KeyCode.E))
        //     if (anim.GetFloat("lean") > 0.1)
        //         anim.SetFloat("lean", 0);
        //     else
        //         anim.SetFloat("lean", 1);

        // // Crouch
        // if (Input.GetKeyDown(KeyCode.C))
        //     if (anim.GetFloat("crouch") == 1)
        //         anim.SetFloat("crouch", 0);
        //     else
        //         anim.SetFloat("crouch", 1);

        // // Short walk
        // if (Input.GetKey(KeyCode.T))
        //     anim.SetFloat("shortwalk", 1);
        // else
        //     anim.SetFloat("shortwalk", 0);

        // // Long walk
        // if (Input.GetKey(KeyCode.Y))
        //     anim.SetFloat("longwalk", 1);
        // else
        //     anim.SetFloat("longwalk", 0);

        // // Hunchback Walk
        // if (Input.GetKey(KeyCode.G))
        //     anim.SetFloat("hunchwalk", 1);
        // else
        //     anim.SetFloat("hunchwalk", 0);


        // // Hunchback Idle
        // if (Input.GetKeyDown(KeyCode.H))
        //     if (anim.GetFloat("hunchback") == 1)
        //         anim.SetFloat("hunchback", 0);
        //     else
        //         anim.SetFloat("hunchback", 1);

        // // Back Arc
        // if (Input.GetKeyDown(KeyCode.B))
        //     if (anim.GetFloat("backarc") == 1)
        //         anim.SetFloat("backarc", 0);
        //     else
        //         anim.SetFloat("backarc", 1);

        // // OutToeing1
        // if (Input.GetKeyDown(KeyCode.M))
        //     if (anim.GetFloat("ot1") == 1)
        //         anim.SetFloat("ot1", 0);
        //     else
        //         anim.SetFloat("ot1", 1);
    }

    void OnDisable()
    {
        playableGraph.Destroy();
    }
}


