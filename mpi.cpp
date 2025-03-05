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
std::vector<particle_t> PARTS;
std::vector<particle_t> GHOSTS;
float CUTOFF = cutoff;

bool is_overlapping(float* a_bounds, float* b_bounds) {

	float a_x_min = a_bounds[0];
	float a_x_max = a_bounds[1];
	float a_y_min = a_bounds[2];
	float a_y_max = a_bounds[3];

	float b_x_min = b_bounds[0];
	float b_x_max = b_bounds[1];
	float b_y_min = b_bounds[2];
	float b_y_max = b_bounds[3];

	return (a_x_min < b_x_max && a_x_max > b_x_min &&
		a_y_min < b_y_max && a_y_max > b_y_min);
}


// Each thread starts with full broadcasted
// list of particles
// this makes a balanced split of the data
void recursive_bisect(int rank, int depth) {

	// base case: return
	if (depth >= MAX_DEPTH) {
            return;
	}

	// non-base case:
	//     if it is an even iteration, split on X axis
	//     otherwise split on Y axis
	int midpoint = PARTS.size() / 2;
	
	if (depth % 2 == 0) {
		// can get away with partial sort since
		// basically always splitting in half
		std::partial_sort(PARTS.begin(), 
				 PARTS.begin() + midpoint, 
				 PARTS.end(), 
			 	 [](const particle_t &a, const particle_t &b) {return a.x < b.x;});
 
	} else {
		std::partial_sort(PARTS.begin(), 
				  PARTS.begin() + midpoint, 
				  PARTS.end(), 
				  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});
	}


	// distribute each half to 
	// appropriate workers
	// trick from se
	

	//if (rank < (1 << depth)) {
	if (((rank >> (MAX_DEPTH - depth - 1)) & 1) == 0) {
		// keep left half
		PARTS.resize(midpoint);

	} else {
		// keep right half
		PARTS.erase(PARTS.begin(), PARTS.begin() + midpoint);
		
	}

	// CAL RECURSIVELY
	recursive_bisect(rank, depth+1);
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

        // Assigns areas of the simulation
	// to each rank with recursive bisection	
	PARTS.assign(parts, parts + num_parts);
	recursive_bisect(rank, 0);

	float x_max = std::max_element(PARTS.begin(),
		                       PARTS.end(),
                                       [](const particle_t &a, const particle_t &b) {return a.x < b.x;})->x;

	float x_min = std::min_element(PARTS.begin(),
		                       PARTS.end(),
                                       [](const particle_t &a, const particle_t &b) {return a.x < b.x;})->x;

	float y_max = std::max_element(PARTS.begin(),
		                       PARTS.end(),
                                       [](const particle_t &a, const particle_t &b) {return a.y < b.y;})->y;

	float y_min = std::min_element(PARTS.begin(),
		                       PARTS.end(),
                                       [](const particle_t &a, const particle_t &b) {return a.y < b.y;})-> y;

	float x_max_halo = x_max + CUTOFF;
	float x_min_halo = x_min - CUTOFF;
	float y_max_halo = y_max + CUTOFF;
	float y_min_halo = y_min - CUTOFF;


       #ifdef DEBUG
	for (int i=0; i < NUM_PROCS; ++i) {
	    MPI_Barrier(MPI_COMM_WORLD);
	    if (i == rank) {

		    std::cout << "Rank " << rank 
			      << " will handle:\n\tx_min: " << x_min 
			      << " -> x_max: " << x_max 
			      << "\n\ty_min: " << y_min 
			      << " -> y_max: " << y_max

		              << "\n\tHalo x_min: " << x_min_halo 
			      << "-> Halo x_max: " << x_max_halo
			      << "\n\tHalo y_min: " << y_min_halo
			      << "-> Halo y_max: " << y_max_halo;

		    std::cout << "\n\tn_particles: " << PARTS.size() << "\n";
		    for (int p_idx=0; p_idx < PARTS.size(); ++p_idx) {
			    particle_t p = PARTS[p_idx];
			    std::cout << "\tpid: " << p.id 
				    << "\n\t\tx=" << std::fixed << std::setprecision(4) << p.x 
					<< ", y=" << std::fixed << std::setprecision(4) << p.y << "\n";
		    }
	    }
	}

    #endif

    // Communicate the bounds 
    float my_bounds[4] = {x_min_halo, x_max_halo, y_min_halo, y_max_halo};
    float other_bounds[4] = {x_min, x_max, y_min, y_max};
    MPI_Status status;
    std::vector<int> relevant_ranks;

    int send_to = (rank + 1) % NUM_PROCS;
    int recv_from = (rank - 1 + NUM_PROCS) % NUM_PROCS;
    int data_source = recv_from;


     #ifdef DEBUG

            for (int i=0; i < NUM_PROCS; ++i) {
	        MPI_Barrier(MPI_COMM_WORLD);
	        if (i == rank) {
			std::cout << "Rank " << rank << " will send to " << send_to << " and recieve from " << recv_from << "\n";
		}
	    }

    #endif         



    for (int ring_count=0; ring_count < NUM_PROCS-1; ring_count++){


            #ifdef DEBUG
            for (int i=0; i < NUM_PROCS; ++i) {
	        MPI_Barrier(MPI_COMM_WORLD);
	        if (i == rank) {
			std::cout << "Rank " << rank << " getting data from rank " << data_source << "\n";
		}
	    }

            #endif         

	    MPI_Sendrecv_replace(&other_bounds,
			         4, MPI_FLOAT,
				 send_to, 0,
				 recv_from, 0,
				 MPI_COMM_WORLD, &status);


	    // Check if bounds of other rank
	    // overlap with halo region of this rank
	    if (is_overlapping(my_bounds, other_bounds)) {
		    relevant_ranks.push_back(data_source);
            }
    }
    
     #ifdef DEBUG
     for (int i=0; i < NUM_PROCS; ++i) {
	  MPI_Barrier(MPI_COMM_WORLD);
	  if (i == rank) {
		  std::cout << "Rank " << rank << " needs particles from: ";
		  for (int relevant_rank : relevant_ranks ) {
			  std::cout << relevant_rank << " ";
		  }
	      
		  std::cout << "\n";
	    }
     }

     #endif         


     


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
