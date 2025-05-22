
#include "nbody_basic.h"

int main(){
  int n = 1000; 
  int n_steps = 1000; 
  double delta_t = 0.1; 
  char g_i = 'g';
  int output_freq = 100; 

  main_do_basic(n,
    n_steps,
    delta_t,
    output_freq,
    g_i);

  return 1;
}