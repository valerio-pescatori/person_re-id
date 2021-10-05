using System.Threading;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

public class PlayerController : MonoBehaviour
{
    public Animator animator;
    Animation anim;


    // Use this for initialization
    void Start()
    {

        anim = GetComponent<Animation>();

        // TODO: PRIMA CAMBIARE IL METODO DI RILEVAMENTO DEI PASSI
        // SCRIVI IL JSON E VEDI QUANTI DANNO NaN
        anim.Play("mixamo.com 5");
        for (int i = 0; i < 30; i++)
            anim.PlayQueued("mixamo.com 5");

        // for (var i = 1; i < 20; i++)
        //     // le animazioni ripartono dal punto 0,0 
        //     for (int x = 0; x < 10; x++)
        //         anim.PlayQueued("mixamo.com " + i);


    }

    // Update is called once per frame
    void Update()
    {
        animator.SetFloat("vertical", Input.GetAxis("Vertical"));
        animator.SetFloat("horizontal", Input.GetAxis("Horizontal"));

        if (Input.GetKeyDown(KeyCode.Space))
        {
            animator.SetTrigger("trigger1");
        }

        // Lean left
        if (Input.GetKeyDown(KeyCode.Q))
            if (animator.GetFloat("lean") < -0.1)
                animator.SetFloat("lean", 0);
            else
                animator.SetFloat("lean", -1);

        // Lean right
        if (Input.GetKeyDown(KeyCode.E))
            if (animator.GetFloat("lean") > 0.1)
                animator.SetFloat("lean", 0);
            else
                animator.SetFloat("lean", 1);

        // Crouch
        if (Input.GetKeyDown(KeyCode.C))
            if (animator.GetFloat("crouch") == 1)
                animator.SetFloat("crouch", 0);
            else
                animator.SetFloat("crouch", 1);

        // Short walk
        if (Input.GetKey(KeyCode.T))
            animator.SetFloat("shortwalk", 1);
        else
            animator.SetFloat("shortwalk", 0);

        // Long walk
        if (Input.GetKey(KeyCode.Y))
            animator.SetFloat("longwalk", 1);
        else
            animator.SetFloat("longwalk", 0);

        // Hunchback Walk
        if (Input.GetKey(KeyCode.G))
            animator.SetFloat("hunchwalk", 1);
        else
            animator.SetFloat("hunchwalk", 0);


        // Hunchback Idle
        if (Input.GetKeyDown(KeyCode.H))
            if (animator.GetFloat("hunchback") == 1)
                animator.SetFloat("hunchback", 0);
            else
                animator.SetFloat("hunchback", 1);

        // Back Arc
        if (Input.GetKeyDown(KeyCode.B))
            if (animator.GetFloat("backarc") == 1)
                animator.SetFloat("backarc", 0);
            else
                animator.SetFloat("backarc", 1);

        // OutToeing1
        if (Input.GetKeyDown(KeyCode.M))
            if (animator.GetFloat("ot1") == 1)
                animator.SetFloat("ot1", 0);
            else
                animator.SetFloat("ot1", 1);
    }
}


