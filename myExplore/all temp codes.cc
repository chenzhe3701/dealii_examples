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
    const unsigned int n_quadrature_points = 3;    // for quadrature (interpolation)
    const int polynomial_degree {1};    // for finite element

    QGauss<dim> quadrature_formula(n_quadrature_points);
    FE_Q<dim> fe(polynomial_degree);
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
    
    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(1);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    fe_values.reinit(dof_handler.begin_active());
    
    SparseMatrix<double> system_matrix;
    SparsityPattern sparsity_pattern;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);    
    system_matrix.reinit(sparsity_pattern);
    
    FullMatrix<double> full_system_matrix(dof_handler.n_dofs());
    Vector<double> solution(dof_handler.n_dofs());
    Vector<double> system_rhs(dof_handler.n_dofs());
    

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(),boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

    
    full_system_matrix.print(std::cout);

    
    system_matrix.print(std::cout);
    solution.print(std::cout);
    return 0;
}
