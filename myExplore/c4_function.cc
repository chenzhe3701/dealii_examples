#include <deal.II/base/function.h>

#include <iostream>

using namespace dealii;

// Define a class
template <int dim>
class MyFunctionClass : public Function<dim> {
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};

// Define the function 'value' in this class
template <int dim>
double MyFunctionClass<dim>::value(const Point<dim> & p,
                                  const unsigned int /*component*/) const {
    return p.square();
}

int main(){
    const int dim = 2;
    Point<dim> p(1,2);
    MyFunctionClass<dim> myfunction;
    
    std::cout <<  myfunction.value(p) << std::endl;   
}