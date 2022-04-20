#include <iostream>
#include <fstream>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

/************************************************************
Quadrature formula: how to approximate/interpolate, which is independent of finite element
*************************************************************/
int main(){

    const int dim {1};
    const unsigned int n_quadrature_points = 2;    // for quadrature (interpolation)
    const int polynomial_degree {1};    // for finite element

    Triangulation<dim>  triangulation;
    DoFHandler<dim>     dof_handler(triangulation);
    FE_Q<dim>           fe(polynomial_degree);
    
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(1);
    dof_handler.distribute_dofs(fe);

    QGauss<dim>     quadrature_formula(n_quadrature_points);
    FEValues<dim>   fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
    
    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;
    Vector<double>          solution;
    Vector<double>          system_rhs;
    FullMatrix<double>      full_system_matrix(dof_handler.n_dofs());
    
    // Setup sysem
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);    
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Assemble
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i,j) += fe_values.shape_grad(i,q_index) *
                                        fe_values.shape_grad(j,q_index) *
                                        fe_values.JxW(q_index);
                }
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                cell_rhs(i) +=  fe_values.shape_value(i,q_index) *
                                100 * 
                                fe_values.JxW(q_index);
            }
        }
        // copy from local to global
        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                system_matrix.add(  local_dof_indices[i], 
                                    local_dof_indices[j],
                                    cell_matrix(i,j));
            }
        }
        for (const unsigned int i : fe_values.dof_indices()) {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    // Show how it is look like before applying B.C
    for (unsigned int i = 0; i<dof_handler.n_dofs(); i++) {
        for (unsigned int j = 0; j < dof_handler.n_dofs(); j++) {
            full_system_matrix(i,j) = system_matrix.el(i,j);
        }
    }    
    std::cout << "Before applying B.C: " << std::endl;
    full_system_matrix.print(std::cout);
    std::cout << "system_rhs: " << system_rhs << std::endl;
    std::cout << "solution: " << solution << std::endl;


    // check boundary ID
    std::cout << "Check if cell is at boundary: " << std::endl;
    for (auto &cell : dof_handler.active_cell_iterators()) {
        std::cout << cell->at_boundary() << ", ";
    }
    std::cout << std::endl;

    // Apply B.C
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ConstantFunction<dim>(10), boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
    // print the map
    std::cout << "boundary_values: ";
    for (const auto& [key, value] : boundary_values) {
        std::cout << '[' << key << "] = " << value << "; ";
    }
    std::cout << std::endl;

    // Show how it is look like after applying B.C
    for (unsigned int i = 0; i<dof_handler.n_dofs(); i++) {
        for (unsigned int j = 0; j < dof_handler.n_dofs(); j++) {
            full_system_matrix(i,j) = system_matrix.el(i,j);
        }
    }    
    std::cout << "After applying B.C: " << std::endl;
    full_system_matrix.print(std::cout);
    // system_matrix.print(std::cout);
    std::cout << "system_rhs: " << system_rhs << std::endl;
    std::cout << "solution: " << solution << std::endl;
    

    // Show how it is look like after solving
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    std::cout << "Solved solution: " << solution << std::endl;

    return 0;
}
