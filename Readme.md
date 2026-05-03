Implementing a libary (and possibly a SDK) for Deep Learning approximations of the Koopman Operator.
<br>
<br>
The Koopman operator is a relatively fascinating mathematical object that arose in the times of von-Neuman, the time when quantum mechanics was birthed. The landmark paper was published in 1931 titled "Hamiltonian Systems and Transformation in Hilbert Space."
<br>
<br>
This remarkable work connects the mathematical machinery used for quantum physics (Hilbert Spaces) to classical systems. The former is well known to be expressed in the language of linear operators on an infinite dimensional space. The latter often expresses non-linear systems which can exhibit chaotic behaviour that is difficult to tame.
<br>
<br>
Koopman Operator theory transfors the non-linear dynamics on a finite-dimensional state-space into linear dynamics governed by the Koopman operator on infinite dimensional Hilbert space of observables. There are also fruitful connection between this point of view and its dual, the Perron-Frobenius operator, one well known example being the generator of the Fokker-Plank equation.
<br>
<br>
For many years, Koopman Operator Theory was considered a purely theoretical toy with to practical utility. However, with the 21st century data-driven finite-dimensional approximations of the linear infinite dimensional operator started to become popular. A famous example is dynamic mode decomposition, which arose as a linear approxmation method to study high-dimensional fluid simulations.<br>
<br>
In the recent years, with the rise of deep learning, neural networks can be applied to discovered non-linear lifting functions into a finite-dimensional subspace on which one can appoximate the Koopman operator. In contrast to the 20th century, the theory has found numerous applications, ranging from medical imaging, molecular systems, robotics, finance, quantum mechanics, causality, quantum computing. All these applications share a common theme which will be explained and implemented in the following. 