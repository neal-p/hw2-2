#include "common.h"
#include <mpi.h>

#include <cmath>
#include <algorithm>
#include <vector>


#define DEBUG

#ifdef DEBUG
#include <iostream>
#include <iomanip>
#endif

// Put any static global variables here that you will use throughout the simulation.
int MAX_DEPTH;
int NUM_PROCS;


// Each thread starts with full broadcasted
// list of particles
// this makes a balanced split of the data
void recursive_bisect(int rank, std::vector<particle_t>& particles, int depth, double x_min, double x_max, double y_min, double y_max) {

	// base case: return
	if (depth >= MAX_DEPTH) {

               #ifdef DEBUG
		for (int i=0; i < NUM_PROCS; ++i) {
		    MPI_Barrier(MPI_COMM_WORLD);
		    if (i == rank) {

			    std::cout << "Rank " << rank 
				      << " will handle:\n\tx_min: " << x_min 
				      << " -> x_max: " << x_max 
				      << "\n\ty_min: " << y_min 
				      << " -> y_max: " << y_max;

			    std::cout << "\n\tn_particles: " << particles.size() << "\n";
			    for (int p_idx=0; p_idx < particles.size(); ++p_idx) {
				    particle_t p = particles[p_idx];
				    std::cout << "\tpid: " << p.id 
					    << "\n\t\tx=" << std::fixed << std::setprecision(4) << p.x 
					        << ", y=" << std::fixed << std::setprecision(4) << p.y << "\n";
			    }
		    }
		}

            #endif
            return;
	}

	// non-base case:
	//     if it is an even iteration, split on X axis
	//     otherwise split on Y axis
	int midpoint = particles.size() / 2;
	
	if (depth % 2 == 0) {
		// can get away with partial sort since
		// basically always splitting in half
		std::partial_sort(particles.begin(), particles.begin() + midpoint, particles.end(), [](const particle_t &a, const particle_t &b) {return a.x < b.x;});

	} else {
		std::partial_sort(particles.begin(), particles.begin() + midpoint, particles.end(), [](const particle_t &a, const particle_t &b) {return a.y < b.y;});
	}


	// distribute each half to 
	// appropriate workers
	// trick from se
	

	if (rank < (1 << depth)) {
		// keep left half
		particles.resize(midpoint);

		// adjust boundary for next iter
		if (depth % 2 == 0) {
			x_max = (x_min + x_max) / 2;
		} else {
			y_max = (y_min + y_max) / 2;
		}

	} else {
		// keep right half
		particles.erase(particles.begin(), particles.begin() + midpoint);
		
		// adjust boundary for next iter
		if (depth % 2 == 0) {
			x_min = (x_min + x_max) / 2;
		} else {
			y_min = (y_min + y_max) / 2;
		}
	}

	// CAL RECURSIVELY
	recursive_bisect(rank, particles, depth+1, x_min, x_max, y_min, y_max);
}



////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
	
        NUM_PROCS = num_procs;	
	MAX_DEPTH = std::log2(num_procs);

#ifdef DEBUG
	if (rank == 0) {
	     std::cout << "init_simulation:\n\tnum_procs: " << num_procs << "\n\tnum_parts: " << num_parts
	               << "\n\tsize: " << size << "\n" << "\tmax_depth: " << MAX_DEPTH << "\n";
	}
#endif
	
	std::vector<particle_t> particles(num_parts);
	particles.assign(parts, parts + num_parts);

	recursive_bisect(rank, particles, 0, 
			0,    // x_min
			size, // x_max
			0,    // y_min
			size  // y_max
			);

	MPI_Barrier(MPI_COMM_WORLD);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}
