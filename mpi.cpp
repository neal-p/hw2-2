#include "common.h"
#include <mpi.h>

#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>


#define DEBUG 0

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

    if (particle.ax != particle.ax) {
	    std::cout << "ax is nan! r2=" << r2 << " r=" << r << " coef=" << coef << " a_x: " << particle.x << " a_y: " << particle.y << " b_x: " << neighbor.x << " b_y: " << neighbor.y << "\n";
    }

    if (particle.ay != particle.ay) {
	    std::cout << "ay is nan! r2=" << r2 << " r=" << r << " coef=" << coef << " a_x: " << particle.x << " a_y: " << particle.y << " b_x: " << neighbor.x << " b_y: " << neighbor.y << "\n";
    }

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
int N_CELLS;
int MY_CELL_START;
int MY_CELL_END;

std::unordered_map<int, std::vector<particle_t>> CELL_TO_PARTS;




void row_col_to_flat(const int row, const int col, int& flat) {
	flat = (row * N_CELLS) + col;
}

void flat_to_row_col(const int flat,int& row, int& col) {
	row = flat / N_CELLS;
	col = flat % N_CELLS;
}



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

	// Even grid with cells of size CUTOFFxCUTOFF
	N_CELLS = SIZE / CUTOFF;
	if ((float)N_CELLS * CUTOFF < SIZE) {
		N_CELLS++;
	}

	// Divide cells among the ranks
	// would be better to pack in square blocks - revisit later
	MY_CELL_START = 0;
	MY_CELL_END = 0;
	int even_split = (N_CELLS * N_CELLS) / NUM_PROCS;
	int remainder = (N_CELLS * N_CELLS) % NUM_PROCS;

	for (int i=0; i < NUM_PROCS; i++) {
		MY_CELL_START = MY_CELL_END;
		MY_CELL_END = MY_CELL_START + even_split;

		if (i < remainder) {
			MY_CELL_END ++;
		}

		if (rank == i) {
			break;
		}
	}
#if DEBUG > 0
	if (rank == 0) {
                std::cout << "Simulation params:\n";
		std::cout << "SIZE: " << size << ", CUTOFF: " << CUTOFF << ", N_CELLS: " << N_CELLS << "\n";
	}
#endif

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

#if DEBUG > 0
	for (int i=0; i < NUM_PROCS; i++) {
		if (rank == i) {
			std::cout << "Rank " << rank << " will handle cells: " << MY_CELL_START << " -> " << MY_CELL_END << "\n";
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}
#endif



}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	// Assign particles to cells
	CELL_TO_PARTS.clear();
	//PARTS_TO_CELL.clear();

	for (int i=0; i < NUM_PARTS; i++) {
		particle_t& p = parts[i];

		int flat;
		int col = p.x / CUTOFF;
		int row = p.y / CUTOFF;

		row_col_to_flat(row, col, flat);
		CELL_TO_PARTS[flat].push_back(p);
		//PARTS_TO_CELL[p.id] = flat;
	}

	// Loop over all cells that ths rank is responsible for
	// calculate forces
	for (int current_flat=MY_CELL_START; current_flat < MY_CELL_END; current_flat++) {
		int current_row;
		int current_col;


		flat_to_row_col(current_flat, current_row, current_col);
#if DEBUG > 1
		std::cout << "Rank " << rank << " working on cell " << current_flat << " (row: " << current_row << ", col: " << current_col << ")\n";
#endif

		// Neighboring Cells
		for (int other_row=current_row-1; other_row <= current_row+1; other_row++) {
		    for (int other_col=current_col-1; other_col <= current_col+1; other_col++) {

			    if (other_row < 0 || other_col < 0 || other_row >= N_CELLS || other_col >=N_CELLS) {
				    continue;
			    }

			    int other_flat;
			    row_col_to_flat(other_row, other_col, other_flat);

			    for (particle_t& p_outer : CELL_TO_PARTS[current_flat]) {
			        for (particle_t& p_inner : CELL_TO_PARTS[other_flat]) {
					if (p_outer.id != p_inner.id) {
				            apply_force(p_outer, p_inner);
					}
				}
			    }
		    }
		}
	}

	// Loop over all cells that ths rank is responsible for
	// perform moves
	// gather all for sending
	
	std::vector<particle_t> send;

	for (int current_flat=MY_CELL_START; current_flat < MY_CELL_END; current_flat++) {
		int current_row;
		int current_col;

		flat_to_row_col(current_flat, current_row, current_col);
		std::vector<particle_t>& cell_parts = CELL_TO_PARTS[current_flat];

		for (particle_t& p_outer : cell_parts) {
			move(p_outer, SIZE);
		}

		std::move(cell_parts.begin(), cell_parts.end(), std::back_inserter(send));
	}

	// All ranks communicate their updated cell's particles
	// two parts, first sending the counts
	int send_count = send.size();
	std::vector<int> recv_counts(NUM_PROCS);
	std::vector<int> recv_disps(NUM_PROCS);

	MPI_Allgather(&send_count, 1, MPI_INT, 
			recv_counts.data(), 1, MPI_INT,
			MPI_COMM_WORLD);

	int total = 0;
	for (int i=0; i < NUM_PROCS; i++) {
		if (i == 0) {
			recv_disps[0] = 0;
		} else {
			recv_disps[i] = recv_disps[i-1] + recv_counts[i-1];
		}

		total += recv_counts[i];
	}

	if (total != NUM_PARTS) {
		std::cout << "HHJJ:SDJF:LKSJDF:   " << NUM_PARTS << " vs " << total << "\n";
	}

	MPI_Allgatherv(send.data(), send_count, PARTICLE,
		      parts, recv_counts.data(), recv_disps.data(), 
		      PARTICLE, MPI_COMM_WORLD);
}




void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    if (rank == 0) {
        std::sort(parts,
	          parts + NUM_PARTS,
	          [](const particle_t &a, const particle_t &b) {return a.id < b.id;});
    }
}


