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
std::vector<particle_t> MY_NEW_PARTS;
std::vector<particle_t> OTHER_LEFT_PARTS;
std::vector<particle_t> OTHER_RIGHT_PARTS;




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

inline
bool in_my_domain(const particle_t& p) {
	return p.x <= MY_END && p.x > MY_START;
}

inline
bool in_my_right_halo(const particle_t& p) {
	return p.x >= MY_RIGHT_HALO;
}

inline
bool in_my_left_halo(const particle_t& p) {
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



        // MOVE THE SHUFFLING OF MY_PARTS to now
	// so that while that communication is goin
	// we can move any non-in domain parts
	// to the ghost list where they belong	

	// Now the same for MY_PARTS and ghosts!
	auto my_partition = std::stable_partition(
			         MY_PARTS.begin(),
				 MY_PARTS.end(),
				 [](const particle_t& p) {return in_my_domain(p); });

	// arbitrarily insert into right ghosts
	OTHER_RIGHT_PARTS.clear();
	OTHER_RIGHT_PARTS.insert(OTHER_RIGHT_PARTS.end(),
		      	std::make_move_iterator(my_partition),
			std::make_move_iterator(MY_PARTS.end()));


        // remove	
	MY_PARTS.erase(my_partition, MY_PARTS.end());

	// Now wait
	MPI_Waitall(count_req_count, count_reqs, MPI_STATUSES_IGNORE);


	// Now I have the info about how many particles I will
	// recieve from my left and right
	int current_other_right_parts_size = OTHER_RIGHT_PARTS.size();
	OTHER_RIGHT_PARTS.resize(current_other_right_parts_size + count_other_right_parts);
	OTHER_LEFT_PARTS.resize(count_other_left_parts);

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

		MPI_Irecv(OTHER_RIGHT_PARTS.data() + current_other_right_parts_size, count_other_right_parts, PARTICLE,
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

		MPI_Irecv(OTHER_LEFT_PARTS.data(), count_other_left_parts, PARTICLE,
			rank-1, // from my left
			0, MPI_COMM_WORLD, &particle_reqs[particle_req_count++]);
	}


	// OVERLAP COMPUTATION WHILE WE WAIT FOR 
	// THE LARGER COMMUNICATIONS
	// can start working on the particles
	// that are already in our domain 
	// right away
	

	std::sort(MY_PARTS.begin(), 
		  MY_PARTS.end(),
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
	}



	MPI_Waitall(particle_req_count, particle_reqs, MPI_STATUSES_IGNORE);

	
        // Need to manage particle ownership
	
	// partiton based on if any ghosts are
	// in this ranks domain
	auto left_partition = std::stable_partition(
			         OTHER_LEFT_PARTS.begin(),
				 OTHER_LEFT_PARTS.end(),
				 [](const particle_t& p) {return !in_my_domain(p); });


	auto right_partition = std::stable_partition(
			         OTHER_RIGHT_PARTS.begin(),
				 OTHER_RIGHT_PARTS.end(),
				 [](const particle_t& p) {return !in_my_domain(p); });


	// If they are, insert to MY_NEW_PARTS
	MY_NEW_PARTS.clear();
	MY_NEW_PARTS.insert(MY_NEW_PARTS.end(),
			std::make_move_iterator(left_partition),
			std::make_move_iterator(OTHER_LEFT_PARTS.end()));

	MY_NEW_PARTS.insert(MY_NEW_PARTS.end(),
			std::make_move_iterator(right_partition),
			std::make_move_iterator(OTHER_RIGHT_PARTS.end()));

	// Remove from ghosts
	OTHER_LEFT_PARTS.erase(left_partition, OTHER_LEFT_PARTS.end());
	OTHER_RIGHT_PARTS.erase(right_partition, OTHER_RIGHT_PARTS.end());


	// FINALLY ready for calculations now that ownership is sorted
	// Sort by Y so that I can threshold what needs to be computed
	std::sort(MY_NEW_PARTS.begin(), 
		  MY_NEW_PARTS.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});

        std::sort(OTHER_LEFT_PARTS.begin(), 
		  OTHER_LEFT_PARTS.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});
 
	std::sort(OTHER_RIGHT_PARTS.begin(), 
		  OTHER_RIGHT_PARTS.end(),
		  [](const particle_t &a, const particle_t &b) {return a.y < b.y;});


	for (particle_t& p_outer : MY_PARTS) {

		// MY_NEW_PARTS
		for (particle_t& p_inner : MY_NEW_PARTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}

	        // LEFT ghosts
		for (particle_t& p_inner : OTHER_LEFT_PARTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}


	        // RIGHT ghosts
		for (particle_t& p_inner : OTHER_RIGHT_PARTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}
	}


	// MY_NEW_PARTS
	//    hasn't had any forces applied yet
	for (particle_t& p_outer : MY_NEW_PARTS) {
		p_outer.ax = p_outer.ay = 0; 

		// MY_PARTS
		for (particle_t& p_inner : MY_PARTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}

		// MY_NEW_PARTS
		for (particle_t& p_inner : MY_NEW_PARTS) {
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
		for (particle_t& p_inner : OTHER_LEFT_PARTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}


	        // RIGHT ghosts
		for (particle_t& p_inner : OTHER_RIGHT_PARTS) {
			if (p_inner.y > (p_outer.y + CUTOFF)) {
				break;
			} else if (p_inner.y < (p_outer.y - CUTOFF)) {
				continue;
			}

			apply_force(p_outer, p_inner);
		}
	}


	// Move MY_NEW_PARTS in with MY_PARTS
	MY_PARTS.insert(MY_PARTS.end(),
		      	std::make_move_iterator(MY_NEW_PARTS.begin()),
			std::make_move_iterator(MY_NEW_PARTS.end()));

	// Perform moves now that all PARTS
	// have been updated
	for (particle_t& p_outer : MY_PARTS) {
		move(p_outer, size);
	}

#if DEBUG > 0
	if (rank == 0) {
	    std::cout << "END OF STEP\n";
	}
#endif

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
