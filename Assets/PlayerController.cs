using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public Animator anim;


     // Use this for initialization
    void Start() 
    {
        anim = GetComponent<Animator>();      
    }

    // Update is called once per frame
    void Update()
    {   
        anim.SetFloat("vertical", Input.GetAxis("Vertical"));
        anim.SetFloat("horizontal", Input.GetAxis("Horizontal"));

        // Lean left
        if(Input.GetKeyDown(KeyCode.Q))
            if(anim.GetFloat("lean") < -0.1)
                anim.SetFloat("lean", 0);
            else
                anim.SetFloat("lean", -1);

        // Lean right
        if(Input.GetKeyDown(KeyCode.E))
            if(anim.GetFloat("lean") > 0.1)
                anim.SetFloat("lean", 0);
            else 
                anim.SetFloat("lean", 1);

        // Crouch
        if (Input.GetKeyDown(KeyCode.C))
            if(anim.GetFloat("crouch") == 1)
                anim.SetFloat("crouch", 0);
            else
                anim.SetFloat("crouch", 1);

        // Short walk
        if (Input.GetKey(KeyCode.T))
            anim.SetFloat("shortwalk", 1);
        else
            anim.SetFloat("shortwalk", 0);

        // Long walk
        if (Input.GetKey(KeyCode.Y))
            anim.SetFloat("longwalk", 1);
        else
            anim.SetFloat("longwalk", 0);
    }
}
