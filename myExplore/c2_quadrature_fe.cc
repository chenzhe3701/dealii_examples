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
Quadrature formula: How to approximate/interpolate, which is independent of finite element
Finte element: We can have many dofs per cell, but only need to do interpolation at quadrature points
FEValues::shape_grad, etc: is dependent on mesh size (mapping considered?)
  For example:
    Unit cell, gradient = 1
    If hyper cube (0,10), then gradient is 0.1.  
    But if also refined once, then gradient is 0.2.
*************************************************************/
int main(){

    const int dim {1};  // space dimension
    const unsigned int n_quadrature_points = 2;    // for quadrature (interpolation)
    const int polynomial_degree {1};    // for finite element

    QGauss<dim> quadrature_formula(n_quadrature_points);

    // This print the quadrature points (or weights possibly)
    // Quadrature points are where we want to do the interpolation
    std::cout << "QGauss<dim=1>(n_quadrature_points=2): " << std::endl;
    for (const auto &element : quadrature_formula.get_points())
        std::cout << element << std::endl;
    std::cout << std::endl;


    // for a 1-dimensional, polynomial of degree(order)
    // print the 3 shape function values at some points (can be arbitrarily selected) 
    // We can have many dofs per cell, but only need to do interpolation at quadrature points
    FE_Q<dim> fe(polynomial_degree);
    std::cout << "FE_Q<dim=1> fe(poly_degree) shape_values:" << std::endl;
    for (unsigned int i = 0; i<fe.n_dofs_per_cell(); i++){
        for (double f = 0; f <= 1; f+=0.125){
            std::cout << fe.shape_value(i, Point<1>(f)) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


    // FEValues, can get shape_values at qth quadrature point
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
    
    std::cout << "fe.n_dofs_per_cell() = " << fe.n_dofs_per_cell() << std::endl;;
    std::cout << "fe_values.dofs_per_cell = " << fe_values.dofs_per_cell << std::endl;
    std::cout << "fe_values.n_quadrature_points() = " << fe_values.n_quadrature_points << std::endl;
    std::cout << std::endl;

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler(triangulation);
    GridGenerator::hyper_cube(triangulation,0,10);
    triangulation.refine_global(1);
    dof_handler.distribute_dofs(fe);

    std::cout << "fe_values at quadrature points (defined by quadrature formula):" << std::endl; 
    fe_values.reinit(dof_handler.begin_active());
    for (const auto q_index : fe_values.quadrature_point_indices()){
        std::cout << "at quadrature point # " << q_index << ", each shape function:" << std::endl;
        for (const unsigned int i : fe_values.dof_indices()){
            std::cout << fe_values.shape_value(i, q_index) << "  ";
        }
        std::cout << std::endl;

        std::cout << "at quadrature point # " << q_index << ", each shape grad:" << std::endl;
        for (const unsigned int i : fe_values.dof_indices()){
            std::cout << fe_values.shape_grad(i, q_index) << "  ";
        }
        std::cout << std::endl;

        std::cout << "at quadrature point # " << q_index << ", each JxW:" << std::endl;
        for (const unsigned int i : fe_values.dof_indices()){
            std::cout << fe_values.JxW(q_index) << "  ";
        }
        std::cout << std::endl;
    }
    

    return 0;
}
