#include <iostream>
#include <fstream>
#include <cmath>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/mapping_q1.h>

using namespace dealii;

void distribute_dofs(DoFHandler<2> &dof_handler){
  const FE_Q<2> finite_element(1);
  dof_handler.distribute_dofs(finite_element);
  
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  std::ofstream out("sparsity_pattern.svg");
  sparsity_pattern.print_svg(out);
}

int main()
{
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation, 10.0, 20.0);
  triangulation.refine_global(1);

  DoFHandler<2> dof_handler(triangulation);

  distribute_dofs(dof_handler);

  // plot triangulation mesh
  std::ofstream out("gnuplot.gpl");

  out << "plot '-' using 1:2 with linespoints linewidth 2 pointsize 2 pointtype 7,"
      << " '-' with labels point pt 2 offset 0,0 textcolor 'red' font ',24'"
      << std::endl;

  GridOut grid_out;
  grid_out.write_gnuplot(triangulation, out);

  out << "e" << std::endl;
  const int dim = 2;
  if (dim==2)
  {    
    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);
    DoFTools::write_gnuplot_dof_support_point_info(out, support_points);
    out << "e" << std::endl;
  }

  std::cout << "Grid written to grid plot" << std::endl;

}