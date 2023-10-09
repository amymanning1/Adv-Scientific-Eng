# SVD-Spring-Mass-Solver
This repository contains a single value decomposition method from scratch, a black box program to check the output of the algorithm, and a spring-mass displacement SVD solver. 
In order to run the SVD functions without the application of the spring-mass problem, ensure that array A is not commented out in main() and the arrays in the test program and mysvd.py match. 
To run spring_mass_svd.py comment out the first two lines of main() in mysvd.py and run from the spring_mass_svd.py tab. This program will output K (stiffness matrix), u (displacement matrix), e (elongation),
w (internal forces), eigenvalues, condition number, and singular values. 


In the spring-mass program, the user has the option to have two fixed ends or a fixed end and a free end. There is not a free-free option because this is physically impossible. The elongation equation is unsolvable by itself due to there being more displacements to calculate than elongations. It is also known that matrix rank informs the chances of solving a system of equations, and a matrix must have full rank to be solvable.
Another way to prove that a free-free system is not compatible with the spring-mass program is due to its null space not being empty, hence the matrix does not contain a full rank and cannot be solved. 