#include "common.h"
#include <mpi.h>

#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>


#define DEBUG 2

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
int NUM_PARTS;
int NUM_PROCS;
int STEP;
float CUTOFF;
float SIZE;
float DOMAIN_SIZE;
float MY_START;
float MY_END;

std::vector<particle_t> PARTS;
std::vector<particle_t> MY_LEFT_GHOSTS;
std::vector<particle_t> MY_RIGHT_GHOSTS;
std::vector<particle_t> OTHER_LEFT_GHOSTS;
std::vector<particle_t> OTHER_RIGHT_GHOSTS;

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
	
	NUM_PARTS = num_parts;
	NUM_PROCS = num_procs;
	STEP = 0;
	CUTOFF = cutoff;
	SIZE = size;


#if DEBUG > 1
	if (rank == 0) {
		std::cout << "Initial Particle Locations:\n";
		for (int i=0; i < NUM_PARTS; i++) {
			particle_t& p = parts[i];
			std::cout << "\tPid: " << p.id << " x: " << p.x << " y: " << p.y << "\n"
				  << "\t\t vx: " << p.vx << " vy: " << p.vy << "\n";
		}
	}
#endif


        // Split along X dim, one slice per rank
	DOMAIN_SIZE = std::max(CUTOFF, SIZE / (float)NUM_PROCS);   // if domains get really small, thats going to be an issue
	MY_START = DOMAIN_SIZE * (float)rank;
	MY_END = MY_START + DOMAIN_SIZE;
	
	// Gather particles in this rank's domain
	for (int i=0; i < NUM_PARTS; i++) {

		if (parts[i].x >= MY_START && parts[i].x < MY_END) {
			PARTS.push_back(parts[i]);
		}
	}

#if DEBUG > 0
	for (int i=0; i < NUM_PROCS; i++) {
		if (i == rank) { 
			std::cout << "Rank " << rank << " has " << PARTS.size() << " particles:\n";
			std::cout << "\tstart: " << MY_START << " -> end: " << MY_END << "\n";

			for (particle_t& p : PARTS) {
				std::cout << "\t\tpid: " << p.id << "\n";
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
#endif

}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	// Check what is near my left border
	if (rank != 0) {
		MY_LEFT_GHOSTS.clear();
		float other_halo = MY_START + CUTOFF;

		for (particle_t& p : PARTS) {
			if (p.x <= other_halo) {
				MY_LEFT_GHOSTS.push_back(p);
			}
		}
	}

	// Check what is near my right border
	if (rank != NUM_PROCS-1) {
		MY_RIGHT_GHOSTS.clear();
		float other_halo =  MY_END - CUTOFF;

		for (particle_t& p : PARTS) {
			if (p.x >= other_halo) {
				MY_RIGHT_GHOSTS.push_back(p);
			}
		}
	}


	MPI_Status status;
	MPI_Request reqs[4];

	int req_count = 0;

	int other_left_count = 0;
	int my_right_count = MY_RIGHT_GHOSTS.size();

	int other_right_count = 0;
	int my_left_count = MY_LEFT_GHOSTS.size();

	if (rank > 0) {

		MPI_Irecv(&other_left_count, 1, MPI_INT, 
				rank-1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
		MPI_Isend(&my_right_count, 1, MPI_INT,
				rank-1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
	}

	if (rank < NUM_PROCS-1) {

		MPI_Irecv(&other_right_count, 1, MPI_INT, 
				rank+1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
		MPI_Isend(&my_left_count, 1, MPI_INT,
				rank+1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
	}


	MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);


	// Allocate space for each other rank
	OTHER_LEFT_GHOSTS.resize(other_left_count);
	OTHER_RIGHT_GHOSTS.resize(other_right_count);


	if (rank > 0) {

		MPI_Irecv(OTHER_LEFT_GHOSTS.data(), other_left_count, PARTICLE, 
				rank-1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
		MPI_Isend(MY_RIGHT_GHOSTS.data(), my_right_count, PARTICLE,
				rank-1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
	}

	if (rank < NUM_PROCS-1) {

		MPI_Irecv(OTHER_RIGHT_GHOSTS.data(), other_right_count, PARTICLE, 
				rank+1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
		MPI_Isend(MY_LEFT_GHOSTS.data(), my_left_count, PARTICLE,
				rank+1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
	}

	MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);


	// If any of my particles have gone outside my domain
	// they will have been passed as ghosts to the proper place
	// so I should delete them from my domain
	// and they will be taken up from by the other rank
	auto not_in_my_domain = [&](const particle_t p){ return p.x < MY_START || p.x >= MY_END;};

#if DEBUG > 1
	int before = PARTS.size();
#endif

	PARTS.erase(
			std::remove_if(
				PARTS.begin(),
				PARTS.end(),
				not_in_my_domain),
			PARTS.end());
#if DEBUG > 1
	int after = PARTS.size();
	if (before != after) {
		std::cout << "Rank " << rank << " removed a particle!\n";
	}
#endif

	// Move from Ghosts to PARTS if it is inside domain
	auto left_it = std::stable_partition(OTHER_LEFT_GHOSTS.begin(), OTHER_LEFT_GHOSTS.end(), not_in_my_domain);
	auto right_it = std::stable_partition(OTHER_RIGHT_GHOSTS.begin(), OTHER_RIGHT_GHOSTS.end(), not_in_my_domain);

#if DEBUG > 1
	before = PARTS.size();
#endif

	PARTS.insert(PARTS.end(), std::make_move_iterator(left_it), std::make_move_iterator(OTHER_LEFT_GHOSTS.end()));

#if DEBUG > 1
	after = PARTS.size();
	if (before != after) {
		std::cout << "Rank " << rank << " got a particle from the left!\n";
	}

	before = PARTS.size();
#endif

	PARTS.insert(PARTS.end(), std::make_move_iterator(right_it), std::make_move_iterator(OTHER_RIGHT_GHOSTS.end()));

#if DEBUG > 1
	after = PARTS.size();
	if (before != after) {
		std::cout << "Rank " << rank << " got a particle from the right!\n";
	}

	before = PARTS.size();
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	OTHER_LEFT_GHOSTS.erase(left_it, OTHER_LEFT_GHOSTS.end());
	OTHER_RIGHT_GHOSTS.erase(right_it, OTHER_RIGHT_GHOSTS.end());


	// Now should have everything we need communication wise


	// Sort by Y dim so we can leverage the cutoff better
        std::sort(PARTS.begin(), 
		  PARTS.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});

        std::sort(OTHER_LEFT_GHOSTS.begin(), 
		  OTHER_LEFT_GHOSTS.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});
 
	std::sort(OTHER_RIGHT_GHOSTS.begin(), 
		  OTHER_RIGHT_GHOSTS.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});

	// Start computing interactions
	for (particle_t& p_outer : PARTS) {

	        // Owned parts interaction	
		for (particle_t& p_inner : PARTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}


	        // LEFT ghosts
		for (particle_t& p_inner : OTHER_LEFT_GHOSTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}


	        // RIGHT ghosts
		for (particle_t& p_inner : OTHER_LEFT_GHOSTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}
	}

	// Perform moves now that all PARTS
	// have been updated
	for (particle_t& p_outer : PARTS) {
		move(p_outer, SIZE);
	}

    STEP++;
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    int my_count = PARTS.size();
    std::vector<int> counts;
    std::vector<int> disps;
    std::vector<particle_t> recv;

    if (rank == 0) {
	    counts.resize(NUM_PROCS);
	    disps.resize(NUM_PROCS);
            recv.resize(NUM_PARTS);
    }

    MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_parts = 0;
        for (int i = 0; i < NUM_PROCS; i++) {
            disps[i] = total_parts;
            total_parts += counts[i];

	}

	if (total_parts != NUM_PARTS) {
		std::cout << "ISSUEEEEEE~~~!!!!!\n";
	}
   }


    MPI_Gatherv(PARTS.data(), my_count, PARTICLE,
                recv.data(), counts.data(), disps.data(), PARTICLE, 0, MPI_COMM_WORLD);


    std::sort(recv.begin(), 
	      recv.end(),
	      [](const particle_t &a, const particle_t &b) {return a.id < b.id;});

    std::move(recv.begin(), recv.end(), parts);
}

