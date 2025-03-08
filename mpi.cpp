#include "common.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define DEBUG 0

// Put any static global variables here that you will use throughout the simulation.
float CUTOFF = cutoff;

float MY_START;
float MY_END;
float MY_LEFT_HALO;
float MY_RIGHT_HALO;

std::vector<particle_t> MY_PARTS;




void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;


    r2 = fmax(r2, min_r * min_r);

//    std::cout << particle.id << " v " << neighbor.id << ": " << r2 << " p(" << particle.x << ", " << particle.y << ", " << particle.vx << ", " << particle.vy << ", " << particle.ax << ", " << particle.ay << ")" << " v n(" << neighbor.x << ", " << neighbor.y << ", " << neighbor.vx << ", " << neighbor.vy << ", " << neighbor.ax << ", " << neighbor.ay << ")\n";

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
}






// Helper functions for determining
// if a partcle belongs to this rank
// or this rank's halo region

bool in_my_domain(particle_t& p) {
	return p.x <= MY_END && p.x > MY_START;
}

bool in_my_right_halo(particle_t& p) {
	return p.x >= MY_RIGHT_HALO;
}

bool in_my_left_halo(particle_t& p) {
	return p.x <= MY_LEFT_HALO;
}




void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	// Each rank owns a slice of the X dimension
	// whose width is at least 3x the cutoff
	
	float min_width = 3.0 * CUTOFF;
	float width = size / num_procs;

	if (width < min_width) {
		if (rank == 0) {
		    std::cout << "HEY! an even width would be " << width << ", but that is too small. Using " << min_width << " and some ranks will not get work\n";
		}
		width = min_width;
	}

#if DEBUG > 0
	if (rank == 0) {
		std::cout << "X bins of width: " << width << "\n";
	}
#endif


	// Assign each rank's domain
	MY_START = rank * width;
	MY_END = MY_START + width;

#if DEBUG > 0
	std::cout << "Rank " << rank << " MY_START: " << MY_START << " -> MY_END: " << MY_END << "\n";
#endif

	// Calculate halo region 
	//   this is inside my current region
	//   and what i need to share with adjoining 
	//   ranks
	MY_LEFT_HALO = MY_START + CUTOFF;
	MY_RIGHT_HALO = MY_END - CUTOFF;

#if DEBUG > 0
	std::cout << "Rank " << rank << " MY_LEFT_HALO: " << MY_LEFT_HALO << ", MY_RIGHT_HALO: " << MY_RIGHT_HALO << "\n";
#endif


#if DEBUG > 3
        if (rank == 0) {
                std::cout << "Initial Particle Locations:\n";
                for (int i=0; i < num_parts; i++) {
                        particle_t& p = parts[i];
                        std::cout << "\tPid: " << p.id << " x: " << p.x << " y: " << p.y << "\n"
                                  << "\t\t vx: " << p.vx << " vy: " << p.vy << "\n";
                }
        }

#endif

        // Get particles in my region
	for (int i=0; i < num_parts; i++) {
		particle_t p = parts[i];

		if (in_my_domain(p)) {

#if DEBUG > 2
			std::cout << "Rank " << rank << " has taken ownership of pid " << p.id << "\n";
#endif

			MY_PARTS.emplace_back(p);
		}
	}
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	// ALWAYS REFER TO LEFT AND RIGHT WRT THE CURRENT RANK

	MPI_Request count_reqs[4];
	int count_req_count=0;
	MPI_Request particle_reqs[4];
	int particle_req_count=0;

	// See what is near my left border -> this will be sent to rank-1
	std::vector<particle_t> my_left_parts;

	for (particle_t& p : MY_PARTS) {
		if (in_my_left_halo(p)) {
			my_left_parts.push_back(p);
		}
	}

	int count_my_left_parts = my_left_parts.size();
	if (rank > 0) { // only send if I am not rank 0
        	MPI_Isend(&count_my_left_parts, 1, MPI_INT,
			  rank-1, // to my left
			  0, MPI_COMM_WORLD, &count_reqs[count_req_count++]);
	}


	// The rank to my right will have also 
	// done the above, so I can recieve them
	int count_other_right_parts=0;
	if (rank < num_procs-1) {
		MPI_Irecv(&count_other_right_parts, 1, MPI_INT,
			rank+1, // from my right
			0, MPI_COMM_WORLD, &count_reqs[count_req_count++]);
	}


	// See what is near my right border -> this will be sent to rank+1
	std::vector<particle_t> my_right_parts;

	for (particle_t& p : MY_PARTS) {
		if (in_my_right_halo(p)) {
			my_right_parts.push_back(p);
		}
	}

	int count_my_right_parts = my_right_parts.size();
	if (rank < num_procs-1) { // only send if I am not the last rank
		MPI_Isend(&count_my_right_parts, 1, MPI_INT,
			rank+1, // to my right
		 	0, MPI_COMM_WORLD, &count_reqs[count_req_count++]);
	}

	// The rank to my left will have also
	// done the above, so I can recieve them
	int count_other_left_parts=0;
	if (rank > 0) {
		MPI_Irecv(&count_other_left_parts, 1, MPI_INT,
				rank-1, // from my left
				0, MPI_COMM_WORLD, &count_reqs[count_req_count++]);
	}


	MPI_Waitall(count_req_count, count_reqs, MPI_STATUSES_IGNORE);

	// Now I have the info about how many particles I will
	// recieve from my left and right
	std::vector<particle_t> other_right_parts(count_other_right_parts);
	std::vector<particle_t> other_left_parts(count_other_left_parts);

	// Same pattern, get particles now

	if (rank > 0) { // only send if I am not rank 0

#if DEBUG > 2
		std::cout << "Rank " << rank << " sending " << count_my_left_parts << " my_left_parts to rank " << rank-1 << ": ";
		for (particle_t& p : my_left_parts) {
			std::cout << p.id << ", ";
		}
		std::cout << "\n";
#endif
        	MPI_Isend(my_left_parts.data(), count_my_left_parts, PARTICLE,
			  rank-1, // to my left
		          0, MPI_COMM_WORLD, &particle_reqs[particle_req_count++]);
	}

	if (rank < num_procs-1) {
#if DEBUG > 2
		std::cout << "Rank " << rank << " recieving " << count_other_right_parts << " other_right_parts to rank " << rank+1 << "\n";
#endif

		MPI_Irecv(other_right_parts.data(), count_other_right_parts, PARTICLE,
			rank+1, // from my right
			0, MPI_COMM_WORLD, &particle_reqs[particle_req_count++]);
	}


	if (rank < num_procs-1) { // only send if I am not the last rank
#if DEBUG > 2
		std::cout << "Rank " << rank << " sending " << count_my_right_parts << " my_right_parts to rank " << rank+1 << ": ";
		for (particle_t& p : my_right_parts) {
			std::cout << p.id << ", ";
		}
		std::cout << "\n";

#endif

		MPI_Isend(my_right_parts.data(), count_my_right_parts, PARTICLE,
			rank+1, // to my right
		 	0, MPI_COMM_WORLD, &particle_reqs[particle_req_count++]);
	}

	if (rank > 0) {
#if DEBUG > 2
		std::cout << "Rank " << rank << " recieving " << count_other_left_parts << " other_left_parts to rank " << rank-1 << "\n";
#endif

		MPI_Irecv(other_left_parts.data(), count_other_left_parts, PARTICLE,
			rank-1, // from my left
			0, MPI_COMM_WORLD, &particle_reqs[particle_req_count++]);
	}

	MPI_Waitall(particle_req_count, particle_reqs, MPI_STATUSES_IGNORE);

	MPI_Barrier(MPI_COMM_WORLD);


	// At this point I should have everything I need
	// to calculate forces and do movements
	
	// But, need to be careful about particle ownership
	// check if any of the ghosts I was passed have actually
	// left their rank's domain and come into mine

        // do this the stupid way first, then the C++ way	
	std::vector<int> other_left_parts_to_delete;

        int left_ghost_idx = 0;	
	for (particle_t& p : other_left_parts) {

#if DEBUG > 3
		std::cout << "Rank " << rank << " got pid: " << p.id << " (" << p.x << ") from rank " << rank-1 << "\n";
#endif

		if (in_my_domain(p)) {
#if DEBUG > 1
			std::cout << "Rank " << rank << " got pid: " << p.id << " from rank " << rank-1 << " and it has entered rank " << rank << "'s domain\n";
#endif
			other_left_parts_to_delete.push_back(left_ghost_idx);
		}

		left_ghost_idx++;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	

	std::vector<int> other_right_parts_to_delete;

        int right_ghost_idx = 0;	
	for (particle_t& p : other_right_parts) {
#if DEBUG > 3
		std::cout << "Rank " << rank << " got pid: " << p.id << " (" << p.x << ") from rank " << rank+1 << "\n";
#endif

		if (in_my_domain(p)) {
#if DEBUG > 1
			std::cout << "Rank " << rank << " got pid: " << p.id << " from rank " << rank+1 << " and it has entered rank " << rank << "'s domain\n";
#endif
			other_right_parts_to_delete.push_back(right_ghost_idx);
		}

		right_ghost_idx++;
	}


	MPI_Barrier(MPI_COMM_WORLD);


	std::sort(other_right_parts_to_delete.begin(), other_right_parts_to_delete.end(), std::greater<int>());
	std::sort(other_left_parts_to_delete.begin(), other_left_parts_to_delete.end(), std::greater<int>());


	// Add to MY_PARTS, remove from ghosts
	for (int idx : other_left_parts_to_delete) {
#if DEBUG > 1
		std::cout << "Rank " << rank << " adding pid: " << other_left_parts[idx].id << " to MY_PARTS from left\n";
#endif

		MY_PARTS.push_back(other_left_parts[idx]);
		other_left_parts.erase(other_left_parts.begin() + idx);
	}
	
	for (int idx : other_right_parts_to_delete) {

#if DEBUG > 1
		std::cout << "Rank " << rank << " adding pid: " << other_right_parts[idx].id << " to MY_PARTS from right\n";
#endif


		MY_PARTS.push_back(other_right_parts[idx]);
		other_right_parts.erase(other_right_parts.begin() + idx);
	}



	MPI_Barrier(MPI_COMM_WORLD);


	// Now the same for MY_PARTS and ghosts!
	
	std::vector<int> my_parts_to_delete;
	int part_idx=0; 

	for (particle_t& p : MY_PARTS) {
		if (!in_my_domain(p)) {
#if DEBUG > 1
			std::cout << "Rank " << rank << " has pid: " << p.id << " but it is outside its domain (" << p.x << "), releasing ownership\n";
#endif
			my_parts_to_delete.push_back(part_idx);
		}
		part_idx ++;
	}



	std::sort(my_parts_to_delete.begin(), my_parts_to_delete.end(), std::greater<int>());

	for (int idx : my_parts_to_delete) {
#if DEBUG > 1
		std::cout << "Rank " << rank << " deleting pid: " << MY_PARTS[idx].id << " from MY_PARTS\n";
#endif
		// arbitrarily add to right ghost list
		// we need it to still interact with other
		// particles, we just are no longer responsible
		// for moving in this step
		other_right_parts.push_back(MY_PARTS[idx]);
		MY_PARTS.erase(MY_PARTS.begin() + idx);
	}

	MPI_Barrier(MPI_COMM_WORLD);


	// FINALLY ready for calculations now that ownership is sorted
	

	// Sort by Y so that I can threshold what needs to be computed
	
        std::sort(MY_PARTS.begin(), 
		  MY_PARTS.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});

        std::sort(other_left_parts.begin(), 
		  other_left_parts.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});
 
	std::sort(other_right_parts.begin(), 
		  other_right_parts.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});

	for (particle_t& p_outer : MY_PARTS) {
		p_outer.ax = p_outer.ay = 0;

	        // Owned parts interaction	
		for (particle_t& p_inner : MY_PARTS) {

			if (p_outer.id == p_inner.id) {
				continue;
			} else if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}


	        // LEFT ghosts
		for (particle_t& p_inner : other_left_parts) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}


	        // RIGHT ghosts
		for (particle_t& p_inner : other_right_parts) {
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
	for (particle_t& p_outer : MY_PARTS) {
		move(p_outer, size);
	}

//#if DEBUG > 0
	if (rank == 0) {
	    std::cout << "END OF STEP\n";
	}
	MPI_Barrier(MPI_COMM_WORLD);
//#endif

}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.


    int my_count = MY_PARTS.size();
    std::vector<int> counts;
    std::vector<int> disps;

    if (rank == 0) {
	    counts.resize(num_procs);
	    disps.resize(num_procs);
    }

    MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_parts = 0;
        for (int i = 0; i < num_procs; i++) {
            disps[i] = total_parts;
            total_parts += counts[i];

	}

	if (total_parts != num_parts) {
		std::cout << "ISSUEEEEEE~~~!!!!!\n";
	}
   }


    MPI_Gatherv(MY_PARTS.data(), my_count, PARTICLE,
                parts, counts.data(), disps.data(), PARTICLE, 0, MPI_COMM_WORLD);


    std::sort(parts,
	      parts + num_parts,
	      [](const particle_t &a, const particle_t &b) {return a.id < b.id;});
}
