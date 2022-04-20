#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>


#include <fstream>
#include <iostream>

using namespace dealii;

/********************************************************
Example solving a simple system of equations: x+y=10, x-y=2
*********************************************************/
int main(){

    FullMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    system_matrix.reinit(2,2);
    solution.reinit(2);
    system_rhs.reinit(2);

    system_matrix.set(0,0,1);
    system_matrix.set(0,1,1);
    system_matrix.set(1,0,1);
    system_matrix.set(1,1,-1);
    
    system_rhs[0] = 10;
    system_rhs[1] = 2;


    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

    std::cout << solution;
    system_matrix.print(std::cout);
    return 0;
}
