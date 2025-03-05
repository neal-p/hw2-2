#include "common.h"
#include <mpi.h>

#include <cmath>
#include <algorithm>
#include <vector>


#define DEBUG 1

#ifdef DEBUG
#include <iostream>
#include <iomanip>
#endif

///////////////////////////////////////////////////////////////

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact

    #if (DEBUG >=3)
    std::cout << "Pid " << particle.id << " is " << r2 << " away from Pid " << neighbor.id << "\n";
    #endif


    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method

    #if (DEBUG >= 3)
    float start_x = p.x;
    float start_y = p.y;
    #endif

    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    #if (DEBUG >=3)
    std::cout << "Pid " << p.id << " went form x: " << start_x << " -> " << p.x << " and y: " << start_y << " -> " << p.y << "\n";
    #endif

}

///////////////////////////////////////////////////////////////


// Put any static global variables here that you will use throughout the simulation.
int MAX_DEPTH;
int NUM_PROCS;
std::vector<particle_t> PARTS;
std::vector<particle_t> GHOSTS;
std::vector<int> RELEVANT_RANKS;
std::vector<float> RELEVANT_RANKS_X_MIN;
std::vector<float> RELEVANT_RANKS_X_MAX;
std::vector<float> RELEVANT_RANKS_Y_MAX;
std::vector<float> RELEVANT_RANKS_Y_MIN;
float CUTOFF = cutoff;
int STEP;
bool NEED_TO_REDISTRIBUTE = false;
bool GLOBAL_NEED_TO_REDISTRIBUTE;
int MIN_N_RELEVANT_RANKS;
int MIN_N_GHOSTS;

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
	STEP = 0;

        #if (DEBUG >= 1)
	if (rank == 0) {
	     std::cout << "init_simulation:\n\tnum_procs: " << num_procs << "\n\tnum_parts: " << num_parts
	               << "\n\tsize: " << size << "\n" << "\tmax_depth: " << MAX_DEPTH << "\n";
	}
        #endif

        // Assigns areas of the simulation
	// to each rank with recursive bisection	
	PARTS.assign(parts, parts + num_parts);

        #if (DEBUG >= 1)
	if (rank == 0) {
		std::cout << "Initial Particle Locations:\n";
		for (particle_t p : PARTS) {
			std::cout << "\tPid: " << p.id << " x: " << p.x << " y: " << p.y << "\n";
		}
	}
        #endif

	recursive_bisect(rank, 0);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	if (STEP % 5 == 0) {
	    MPI_Allreduce(&NEED_TO_REDISTRIBUTE, &GLOBAL_NEED_TO_REDISTRIBUTE, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

		if (GLOBAL_NEED_TO_REDISTRIBUTE) {
		    STEP = 0; /// reset internal step counter
		    gather_for_save(parts, num_parts, size, rank, num_procs);
		    MPI_Bcast(parts, num_parts, PARTICLE, 0, MPI_COMM_WORLD);
		    PARTS.assign(parts, parts + num_parts);

		    #if (DEBUG >= 1)
			if (rank == 0) {
			    std::cout << "NEED TO REDISTRIBUTE! Current Locations:\n";
			    for (particle_t p : PARTS) {
				std::cout << "\tPid: " << p.id << " x: " << p.x << " y: " << p.y << "\n";
			    }
			}
		    #endif

		    recursive_bisect(rank, 0);
		}
	}

   
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

       #if (DEBUG >= 2)
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
    RELEVANT_RANKS.clear();
    MPI_Status status;

    int send_to = (rank + 1) % NUM_PROCS;
    int recv_from = (rank - 1 + NUM_PROCS) % NUM_PROCS;
    int data_source = recv_from;


     #if (DEBUG >= 2)

            for (int i=0; i < NUM_PROCS; ++i) {
	        MPI_Barrier(MPI_COMM_WORLD);
	        if (i == rank) {
			std::cout << "Rank " << rank << " will send to " << send_to << " and recieve from " << recv_from << "\n";
		}
	    }

    #endif         


    for (int ring_count=0; ring_count < NUM_PROCS-1; ring_count++){


            #if (DEBUG >= 2)
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
		    RELEVANT_RANKS.push_back(data_source);
		    RELEVANT_RANKS_X_MIN.push_back(other_bounds[0]);
		    RELEVANT_RANKS_X_MAX.push_back(other_bounds[1]);
		    RELEVANT_RANKS_Y_MIN.push_back(other_bounds[2]);
		    RELEVANT_RANKS_Y_MAX.push_back(other_bounds[3]);
            }

	    data_source = (data_source -1 + NUM_PROCS) % NUM_PROCS;
    }
    
     #if (DEBUG >= 1)
     for (int i=0; i < NUM_PROCS; ++i) {
	  MPI_Barrier(MPI_COMM_WORLD);
	  if (i == rank) {
		  std::cout << "Rank " << rank << " needs particles from: ";
		  for (int relevant_rank : RELEVANT_RANKS) {
			  std::cout << relevant_rank << " ";
		  }
	      
		  std::cout << "\n";
	    }
     }

     #endif         

     // Get ghosts in the halo region
     GHOSTS.clear();


     //for (int relevant_rank : RELEVANT_RANKS) {
     for (int i=0; i < RELEVANT_RANKS.size(); i++) {
	    int relevant_rank = RELEVANT_RANKS[i];
	    float other_x_min = RELEVANT_RANKS_X_MIN[i];
	    float other_x_max = RELEVANT_RANKS_X_MAX[i];
	    float other_y_min = RELEVANT_RANKS_Y_MIN[i];
	    float other_y_max = RELEVANT_RANKS_Y_MAX[i];

	    std::vector<particle_t> to_send;

	    for (particle_t p : PARTS) {
		    if (other_x_min < p.x && other_x_max > p.x && other_y_min < p.y && other_y_max > p.y) {
			    to_send.push_back(p);
		    }
	    }
	    int my_n = to_send.size();
	    int other_n;
	    
	    MPI_Sendrecv(&my_n, 1, MPI_INT,
			    relevant_rank, 0,
			    &other_n, 1, MPI_INT,
			    relevant_rank, 0, MPI_COMM_WORLD, &status);


            #if (DEBUG >= 1)
	        std::cout << "Rank " << rank << " will need to recieve " << other_n << " ghosts from " << relevant_rank << "\n";
            #endif


	    // Make enough room	
	    int current_n_ghosts = GHOSTS.size();
	    GHOSTS.resize(current_n_ghosts + other_n);


	    MPI_Sendrecv(to_send.data(), my_n, PARTICLE,
			    relevant_rank, 0,
			    &GHOSTS[current_n_ghosts], other_n, PARTICLE,
			    relevant_rank, 0, MPI_COMM_WORLD, &status);

            #if (DEBUG >= 2)
	    for (particle_t p : GHOSTS) {
		    std::cout << "Rank " << rank << " got ghost Pid " << p.id << "\n";
	    }
            #endif
     }

     // See if we have strayed into unbalanced work territory
     // and need to rebalance next iteration
     if (STEP != 0) {
	     if (RELEVANT_RANKS.size() > MIN_N_RELEVANT_RANKS * 2) {
		     NEED_TO_REDISTRIBUTE = true;
	     } else if(GHOSTS.size() > MIN_N_GHOSTS * 2) {
		     NEED_TO_REDISTRIBUTE = true;
	     }
     } else {
	     MIN_N_RELEVANT_RANKS = RELEVANT_RANKS.size();
	     MIN_N_GHOSTS = GHOSTS.size();
     }


     // NOW CAN FINALLY DO THE CALCULATIONS!

     // Calculate forces
     for (particle_t p_outer : PARTS) {
	     for (particle_t p_inner : PARTS) {
		     if (p_outer.id != p_inner.id) {
			     apply_force(p_outer, p_inner);
		     }
	     }

             for (particle_t p_inner : GHOSTS) {
                 apply_force(p_outer, p_inner);
	     }
     }

     // Move based on forces
     for (particle_t p : PARTS) {
	     move(p, size);
     }


     STEP++;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

	// Place to recieve all parts
	// only actually allocate space on rank0
	std::vector<particle_t> recv;
	if (rank == 0) {
		recv.resize(num_parts);
	}

	MPI_Gather(PARTS.data(), PARTS.size(), PARTICLE,
		   recv.data(), num_parts, PARTICLE, 
		   0, MPI_COMM_WORLD
			);

	for (particle_t p : recv) { 
		parts[p.id - 1] = p;
	}
}
