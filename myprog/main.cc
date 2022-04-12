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

template <int dim>
void gnuplot_mesh(Triangulation<dim> &triangulation, DoFHandler<dim> &dof_handler){
  std::ofstream out("gnuplot.gpl");

  out << "plot '-' using 1:2 with linespoints linewidth 2 pointsize 2 pointtype 7,"
    << " '-' with labels point pt 2 offset 0,0 textcolor 'red' font ',24'"
    << std::endl;
  GridOut grid_out;
  grid_out.write_gnuplot(triangulation, out);

  std::map<types::global_dof_index, Point<dim>> support_points;
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);

  out << "e" << std::endl;  // additional entry
  if (dim==2) {
    DoFTools::write_gnuplot_dof_support_point_info(out, support_points);
  } 
  else if (dim==1) {
    for (const auto &[index, point] : support_points) {
      out << point << " " << 1 << " " << '"' << index << '"' << std::endl;
    }
  }
  out << "e" << std::endl;  // additional entry

  std::cout << "Grid written to grid plot" << std::endl;
}

int main()
{
  const int dim {2};

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 10.0, 20.0);
  triangulation.refine_global(1);

  DoFHandler<dim> dof_handler(triangulation);
  
  const FE_Q<dim> finite_element(1);
  dof_handler.distribute_dofs(finite_element);
  
  DoFRenumbering::Cuthill_McKee(dof_handler); // Renumber and plot sparsity pattern

  // plot sparsity pattern
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  SparsityPattern sparsity_pattern;  
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  std::ofstream out_sp("sparsity_pattern_1.svg");
  sparsity_pattern.print_svg(out_sp);  
  

  // plot triangulation mesh
  gnuplot_mesh(triangulation, dof_handler);

}