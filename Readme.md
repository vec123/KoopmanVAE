Experimentations of Koopman Operator learning.
Initial deterministic version.
Possibility of a tensor-train parameterized matrix A. 

Inspired by the papers:

DeSKO - Stability-Assured Robust Control with a Deep Stochastic Koopman Operator 

Chaos as an intermittently forced linear system

Learn a linear system z_{k+1} = A z_k
where x = d(z) and z = e(z).

This is an encoder-decoder architecture with linear dynamics in the latent space.

DeSKO uses a variational approach, similar to the variational auto-encoder (VAE)
to consider uncertainty in measurements and dynamics.
Furthermore, they learn a control system  z_{k+1} = A z_k + Bu(t) 
by applying random u(t) trajectories during training
and then use this to apply optimal control techniques (e.g. model predictive control)
to non-linear systems. 
Very interesting

HAVOC introduces a forcing term for chaotic systems:
 z_{k+1} = A z_k + Bv(t)
 This enables the learning of chaotic dynamics which can not be represented by a finite-dimensional Koopman operator due to continous spectrum properties.
 

 The code applies a deterministic auto-encoder approach to learn the Koopman-matrix. 
 Tested for several non-linear systems. 

 Interesting extensions/application:

    - learn the HAVOC term and apply to the Lorenz system
      to the best of my knowledge novel as a deep-learning extension to
      "Chaos as an intermittently forced linear system"
    
    - deviate from the koopman operator theory and examine symbolic regression approaches, 
        i.e. linear -> polynomial latent space evolution + library of nonlinear functions. 
        Compare to: 
        "State estimation of a physical system with unknown governing equations"
    
    - examine the influence of stochastic dynamics 
        implemented by \mu_{k+1} = A \mu_k
        and \z__{k+1} = N(  A \mu_k, A \Sigma_{k} A^T)


    - Maybe usable to speed-up simulations by enabling bigger time-steps.




