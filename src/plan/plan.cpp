#include "varblockspmm/vbsr.hpp"
#include <stdexcept>
namespace vbsr { Plan::Plan(const Matrix&m,PlanOptions o):a_(m.device_view()),options_(o){if(o.rhs_width!=8&&o.rhs_width!=16&&o.rhs_width!=32&&o.rhs_width!=64)throw std::invalid_argument("rhs_width must be 8,16,32,64");if(o.kernel==Kernel::SplitRow)throw std::invalid_argument("split-row was not justified by the 1.0 regime map; use RowOwned or GroupedGemmPlan");} void Plan::execute(const float*b,float*c,cudaStream_t s)const{launch_row_owned(a_,b,c,options_.rhs_width,s);} }
