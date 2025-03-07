#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

/* 
* Problem 
* A simple 2D particle interaction simulation parallelized using OpenMPI.
* 
* Solution
* - Create a grid representing the number of ranks to be performing computations. There will either be:
*    - One Node with 64 ranks, or 
*    - Two nodes with 128 ranks.
*
* - Subdivide this grid into boxes approximately but no less than the length of the cutoff distance.
* 
* - Each rank will perform calculations for particles contained within it's grid. Each rank will communicate with
*   neighboring ranks to obtain information about nearby particles. 
* 
* - For ranks on the node-node boundary, ranks will directly communicate across nodes to obtains the necessary
*  information.
*/

//==============================
// Containers
//==============================

// Create a container for particles based on the particle interation cutoff distance.
struct particle_container {
    std::vector<particle_t> particles;
};

// Create a container for the particles stored in each rank
struct rank_container {
    int xi, yj;                                                 // The rank xy-location                                                     // The rank y-location
    double x1, x2, y1, y2;                                      // The edges of the rank
    std::vector<std::vector<particle_container>> sub_grid;      // The sub-grid of particle containers
    std::vector<std::vector<particle_container>> ghost_grid;    // The grid of particles +1 cutoff distance
    std::vector<particle_t> particle_list;                      // The list of particles in the rank
};

//========================================
// Global Variables
//========================================

// MPI grid properties
int num_mpi_grid_x;
int num_mpi_grid_y;
double mpi_grid_dx;
double mpi_grid_dy;

// Sub-grid properties
double num_sg_box_x;
double num_sg_box_y;
double sg_box_dx;
double sg_box_dy;

// Initialize a rank container
rank_container mpi_rank;

/*
// Create some booleans to represent rank locations
bool mpi_left_edge = (mpi_rank.x1 == 0);                                                    // 1=Left
bool mpi_right_edge = (mpi_rank.x2 >= size - cutoff);                                       // 2=Right
bool mpi_top_edge = (mpi_rank.y2 >= size - cutoff);                                         // 3=Up
bool mpi_bottom_edge = (mpi_rank.y1 == 0);                                                   // 4=Down
bool mpi_grid_corner_1 = (mpi_rank.x1 == 0) && (mpi_rank.y2 >= size - cutoff);              // 5=Up/Left
bool mpi_grid_corner_2 = (mpi_rank.x2 >= size - cutoff) && (mpi_rank.y2 >= size - cutoff);  // 6=Up/Right
bool mpi_grid_corner_3 = (mpi_rank.x1 == 0) && (mpi_rank.y1 == 0);                          // 7=Down/Left
bool mpi_grid_corner_4 = (mpi_rank.x2 >= size - cutoff) && (mpi_rank.y1 == 0);              // 8=Down/Right
*/

//===========================================
// Define the MPI grid and sub-grid.
//===========================================
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
// Note: Each process calls init_simulation() before the simulation starts.

    // Determine if one or two nodes is being used.
    bool two_nodes = (num_procs > 64);

    // Define global MPI grid properties
    num_mpi_grid_x = 3;                                         // MPI grid boxes in x-direction
    num_mpi_grid_y = two_nodes ? 16 : 3;                        // MPI grid boxes in y-direction
    mpi_grid_dx = size / num_mpi_grid_x;
    mpi_grid_dy = size / num_mpi_grid_y;

    // Define the local MPI rank
    mpi_rank.xi = rank % num_mpi_grid_x;
    mpi_rank.yj = rank / num_mpi_grid_x;

    // Define the local MPI rank domain
    mpi_rank.x1 = mpi_rank.xi * mpi_grid_dx;                    // Left boundary
    mpi_rank.x2 = mpi_rank.x1 + mpi_grid_dx;                    // Right boundary
    mpi_rank.y1 = mpi_rank.yj * mpi_grid_dy;                    // Bottom boundary
    mpi_rank.y2 = mpi_rank.y1 + mpi_grid_dy;                    // Top Boundary

    // Define global sub grid properties
    num_sg_box_x = std::floor(mpi_grid_dx / cutoff);         // Sub-grid boxes in x-direction
    num_sg_box_y = std::floor(mpi_grid_dy / cutoff);         // Sub-grid boxes in y-direction
    sg_box_dx = mpi_grid_dx / num_sg_box_x;
    sg_box_dy = mpi_grid_dy / num_sg_box_y;

    // Resize the sub grid based on the global properties
    mpi_rank.sub_grid.resize(num_sg_box_x, std::vector<particle_container>(num_sg_box_y));

    // Define the MPI communication variables
    std::vector<int> send_counts(num_procs, 0);      // The number of particles to send to each rank
    std::vector<int> displ(num_procs, 0);           // The start index of the particle
    std::vector<particle_t> send_buf;

    if (rank == 0) {
        std::cout << "Before rank 0 distributes particles" << std::endl;
    }
    

    // Rank 0 divides particles between ranks and communicates them.
    if (rank == 0) {
        // Create a vector to store the particles for each rank.
        std::vector<std::vector<particle_t>> local_particles(num_procs);

        // Loop through all of the particles and assign them to a rank based on location
        for (int i = 0; i < num_parts; i++) {
            int particle_xi = std::floor((parts[i].x / size) * num_mpi_grid_x);
            int particle_yj = std::floor((parts[i].y / size) * num_mpi_grid_y);

            // Compute the rank to send the particle to and add it to the appropriate rank.
            int particle_rank_idx = particle_yj * num_mpi_grid_x + particle_xi;
            local_particles[particle_rank_idx].push_back(parts[i]);
        }

        // Fill the send buffer with all of the particles ordered by rank
        int offset = 0;
        for (int i = 0; i < num_procs; i++) {

            // Assign values to communication variables
            send_counts[i] = local_particles[i].size();
            displ[i] = offset;
            offset += send_counts[i];

            send_buf.insert(send_buf.end(),
                            local_particles[i].begin(),
                            local_particles[i].end()
            );
        }
    }
    
    // Scatter the send counts to the ranks so they know how many particles they are receiving
    int num_particles_on_rank = 0;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &num_particles_on_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    // Allocate the storage for the particles on the rank.
    mpi_rank.particle_list.resize(num_particles_on_rank);
    
    // Scatter the particles to the ranks
    MPI_Scatterv(send_buf.data(), send_counts.data(), displ.data(), PARTICLE,
                 mpi_rank.particle_list.data(), num_particles_on_rank, PARTICLE, 0, MPI_COMM_WORLD);
    
    /*
    // Shape the ghost particle grids based on the position of the rank in the MPI grid
    if (mpi_grid_corner_1 || mpi_grid_corner_2 || mpi_grid_corner_3 || mpi_grid_corner_4) {
        // Shape the ghost grid for corner ranks (+1 col, +1 row)
        mpi_rank.ghost_grid.resize(num_sg_box_x + 1, std::vector<particle_container>(num_sg_box_y + 1));

    } else if ((mpi_rank.x1 == 0) || (mpi_rank.x2 >= size - cutoff)) {
        // Shape the ghost grid for left/right boundary ranks (+1 col, +2 row)
        mpi_rank.ghost_grid.resize(num_sg_box_x + 1, std::vector<particle_container>(num_sg_box_y + 2));

    } else if ((mpi_rank.y1 == 0) || (mpi_rank.y2 >= size - cutoff)) {
        // Shape the ghost grid for top/bottom boundary ranks (+2 col, +1 row)
        mpi_rank.ghost_grid.resize(num_sg_box_x + 2, std::vector<particle_container>(num_sg_box_y + 1));

    }else {
        // Shape the ghost grid for internal ranks (+2 col, +2 row)
        mpi_rank.ghost_grid.resize(num_sg_box_x + 2, std::vector<particle_container>(num_sg_box_y + 2));
    }
    */
}

//================================================
// Functions For Moving Particles Between Ranks
//================================================

int get_destination_rank_id(int direction, int rank) {

    // Start from the current rank
    int dest_x = mpi_rank.xi;
    int dest_y = mpi_rank.yj;

    // Move in x and y based on the direction argument
    switch (direction) {
        case 1: dest_x -= 1; break;                 // 1=Left
        case 2: dest_x += 1; break;                 // 2=Right
        case 3: dest_y += 1; break;                 // 3=Up
        case 4: dest_y -= 1; break;                 // 4=Down
        case 5: dest_y += 1; dest_x -= 1; break;    // 5=Up/Left
        case 6: dest_y += 1; dest_x += 1; break;    // 6=Up/Right
        case 7: dest_y -= 1; dest_x -= 1; break;    // 7=Down/Left
        case 8: dest_y -= 1; dest_x += 1; break;    // 8=Down/Right
    }

    // Verify move against boundaries of simulation domain
    if (dest_x < 0 || dest_x >= num_mpi_grid_x || dest_y < 0 || dest_y >= num_mpi_grid_y) {
        return MPI_PROC_NULL;
    }

    return dest_x + dest_y * num_mpi_grid_x;
}

int get_inverse_tag(int direction) {
    switch (direction) {
        case 1: return 2;  // Left  <-> Right
        case 2: return 1;  // Right <-> Left
        case 3: return 4;  // Up    <-> Down
        case 4: return 3;  // Down  <-> Up
        case 5: return 8;  // Up/Left  <-> Down/Right
        case 6: return 7;  // Up/Right <-> Down/Left
        case 7: return 6;  // Down/Left <-> Up/Right
        case 8: return 5;  // Down/Right <-> Up/Left
        default: return -1;
    }
}
 
void send_recv_particles(std::vector<std::vector<particle_t>>& send_buf,
                         std::vector<std::vector<particle_t>>& recv_buf, int rank) {
    MPI_Status status;

    // Loop through surrounding ranks
    for (int i = 1; i <= 8; i++) {

        // Obtain the destination rank
        int dest = get_destination_rank_id(i, rank);
        int inv_tag = get_inverse_tag(i);

        

        // If the destination rank is 
        if (dest == MPI_PROC_NULL) continue;

        if (rank == 0) {
            std::cout << "Destination rank id " << dest << " iteration " << i  << std::endl;
        }

        // Initialize the communication variables
        int send_size = send_buf[i - 1].size();
        int recv_size = 0;
        
        /*
        for (int r = 0; r < 64; r++) {
            if (rank == r) {
                std::cout << "Rank " << rank << " Send to " 
            << dest << " send tag " << i <<  " send size " << send_buf[i - 1].size()
            << " recv tag " << inv_tag << std::endl;
                std::cout.flush();  // Ensure output is flushed before continuing
            }
            MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before next rank prints
        }
        */

        if (rank == 0) {
            std::cout << "Rank " << rank << " Send to " 
            << dest << " send tag " << i <<  " send size " << send_buf[i - 1].size()
            << " recv tag " << inv_tag << std::endl;
        }
        
        
        // Sendrecv the size of the buffer with the destination rank.
        MPI_Sendrecv(&send_size, 1, MPI_INT, dest, i,
                     &recv_size, 1, MPI_INT, dest, inv_tag,
                     MPI_COMM_WORLD, &status);

        if (rank == 0) {
            std::cout << "Working before allocating memory to receive buffer, rank " << rank << std::endl;
        }
                 
        // Allocate memory for the receive buffer based on the receive counts
        recv_buf[i - 1].resize(recv_size);
                 
        if (rank == 0){
            std::cout << "Working before sending data" << std::endl;
        }
                 
        // Sendrecv the buffer with the destination rank
        MPI_Sendrecv(send_buf[i - 1].data(), send_size, PARTICLE, dest, i,
                     recv_buf[i - 1].data(), recv_size, PARTICLE, dest, inv_tag,
                     MPI_COMM_WORLD, &status);
        
        
        

        if ((rank == 0) && (i == 1)) {
            std::cout << "Completed sendrecv" << std::endl;
        } 
        
        
    }
}

void update_mpi_particle_list(int rank) {
    
    if (rank ==0) {
        std::cout << "Woring before allocating memory to buffers" << std::endl;
    }
    
    // Initalize send and receive buffers inside an iterable container
    std::vector<std::vector<particle_t>> send_buf(8), recv_buf(8);
    // 1=Left, 2=Right, 3=Up, 4=Down, 5=Up/Left, 6=Up/Right, 7=Down/Left, 8 = Down/Right

    for (int i = 0; i < 8; i++){
        send_buf[i].clear();
        recv_buf[i].clear();
    }

    if (rank == 0) {
        std::cout << "working before looping through particles on the rank" << std::endl;
    }
    
    // Iterate through the particles in the rank and determine if they need to move ranks
    for (auto it = mpi_rank.particle_list.begin(); it != mpi_rank.particle_list.end();) {

        // Define the particle 
        particle_t &p = *it;

        // Create a flag to determine if the particle moved
        bool it_moved = false;

        if (p.y > mpi_rank.y2 && p.x < mpi_rank.x1) {           // Up-Left
            send_buf[5].push_back(p);
            it_moved = true;
        } else if (p.y > mpi_rank.y2 && p.x > mpi_rank.x2) {    // Up-Right
            send_buf[6].push_back(p);
            it_moved = true;
        } else if (p.y < mpi_rank.y1 && p.x < mpi_rank.x1) {    // Down-Left
            send_buf[7].push_back(p);
            it_moved = true;
        } else if (p.y < mpi_rank.y1 && p.x > mpi_rank.x2) {    // Down-Right
            send_buf[8].push_back(p);
            it_moved = true;
        } else if (p.y > mpi_rank.y2) {                         // Up
            send_buf[3].push_back(p);
            it_moved = true;
        } else if (p.y < mpi_rank.y1) {                         // Down
            send_buf[4].push_back(p);
            it_moved = true;
        } else if (p.x < mpi_rank.x1) {                         // Left
            send_buf[1].push_back(p);
            it_moved = true;
        } else if (p.x > mpi_rank.x2) {                         // Right
            send_buf[2].push_back(p);
            it_moved = true;
        }

        // Remove the particle from the rank if it moved
        if (it_moved) {
            it = mpi_rank.particle_list.erase(it);
        } else {
            // Continue iterating if it didn't move
            ++it;
        }
    }

    if (rank == 0) {
        std::cout << "working before sendrecv particles" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Perform MPI communications to send and receive particles
    send_recv_particles(send_buf, recv_buf, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Before allocating memory for particle list based on receive buffer" << std::endl;
    }

    // Allocate memory for the particle list based on the size of the receive buffers
    size_t total_recv_size = 0;
    for (const auto& buf : recv_buf) {
        total_recv_size += buf.size();
    }
    mpi_rank.particle_list.reserve(mpi_rank.particle_list.size() + total_recv_size);

    if (rank == 0) {
        std::cout << "Before moving buffers into the particle list" << std::endl;
    }

    // Move the buffers into the particle list to avoid unecessary copies
    for (auto& buf : recv_buf){
        mpi_rank.particle_list.insert(mpi_rank.particle_list.end(),
                                      std::make_move_iterator(buf.begin()),
                                      std::make_move_iterator(buf.end()));
    }
}

//=============================================
// Function For Updating the MPI rank sub grid
//=============================================

void update_sub_grid() {
    
    // Clear the sub grid 
    for (int i = 0; i < num_sg_box_x; i++) {
        for (int j = 0; j < num_sg_box_y; j++) {
            mpi_rank.sub_grid[i][j].particles.clear();
        }
    }

    // Loop through all of the particles in the list and assign them to a sub grid box
    for (const auto& particle : mpi_rank.particle_list) {

        // Determine the sub grid xy-location
        int particle_xi = std::floor((particle.x - mpi_rank.x1) * sg_box_dx);
        int particle_yj = std::floor((particle.y - mpi_rank.y1) * sg_box_dy);

        // Assign particle to sub grid box
        mpi_rank.sub_grid[particle_xi][particle_yj].particles.push_back(particle);
    }
}

//============================================
// Functions for Updating the Ghost Grid
//============================================
/*
void pack_LR_buffer(std::vector<std::vector<particle_t>>& ghost_send_buf) {

    // Pack the left and right columns
    for (int j = 0; j < num_sg_box_y; j++) {
        //Left column                                  
        ghost_send_buf[1].insert(ghost_send_buf[1].end(),
                                 mpi_rank.sub_grid[0][j].particles.begin(),
                                 mpi_rank.sub_grid[0][j].particles.end());      
        // Right column
        ghost_send_buf[2].insert(ghost_send_buf[2].end(),
                                 mpi_rank.sub_grid[num_sg_box_x - 1][j].particles.begin(),
                                 mpi_rank.sub_grid[num_sg_box_x - 1][j].particles.end());
    }
}

void sendrecv_LR_buffer(std::vector<std::vector<particle_container>>& ghost_send_buf,
                        std::vector<std::vector<particle_container>>& ghost_recv_buf) {
    // The local rank will:
    // SEND BUFFER      RECEIVE BUFFER      DIRECTION
    // 1 Left           1 Left              1 Left
    // 2 Right          2 Right             2 Right
    // 3 Up             3 Up                3 Up
    // 4 Down           4 Down              4 Down
    
    // Define working variables
    MPI_Status status;
    MPI_Request requests[4];
    int req_count = 0;

    // Obtain the ranks to the left and right
    int left_rank = get_destination_rank_id(1);
    int right_rank = get_destination_rank_id(2);

    // Send and receive the left column
    if (left_rank != MPI_PROC_NULL) {
        // Send the left column of the local rank to the rank on the left
        // This is the right side of the ghost grid for the rank on the left
        MPI_Isend(ghost_send_buf[1], ghost_send_buf[1].size(), PARTICLE, left_rank, 1,
                  MPI_COMM_WORLD, &requests[req_count++]);

        // Receive the right column from the rank on the left.
        // This represents the left side of the ghost grid for the local rank.
        MPI_Irecv(ghost_recv_buf[1], ghost_recv_buf[1].size(), PARTICLE, left_rank, 2,
                  MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Send and receive the right column
    if (right_rank != MPI_PROC_NULL) {

        // Send the right column of the local rank to the rank on the right
        // This represents the left side of ghost grid of the rank on the right
        MPI_Isend(ghost_send_buf[2], ghost_send_buf[2].size(), PARTICLE, right_rank, 2,
                  MPI_COMM_WORLD, &requests[req_count++]);

        // Receive the left column from the rank on the right.
        // This represents the right side of the ghost grid for the local rank.
        MPI_Irecv(ghost_recv_buf[2], ghost_recv_buf[2].size(), PARTICLE, right_rank, 1,
                  MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait to complete communicating
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

}

void unpack_LR_buffer(std::vector<std::vector<particle_container>>& ghost_recv_buf) {
    
    if (mpi_grid_corner_1) {                //Top-Left
        // Ghost grid shape (+1 col to right, +1 row down)
        
        // Buffer 1 Null
        
        // Buffer 2 (Right column)
        for (const auto& particle : ghost_recv_buf[2]) {
            int ghost_xi = num_sg_box_x;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        } 
    } else if (mpi_grid_corner_2) {         //Top-Right
        // Ghost grid shape (+1 col to left, +1 row down)

        // Buffer 1 (Left Column)
        for (const auto& particle : ghost_recv_buf[1]) {
            int ghost_xi = 0;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 2 Null
        
    } else if (mpi_grid_corner_3) {         //Bottom-Left
        // Ghost grid shape (+1 col to right, +1 row up)

        // Buffer 1 Null
        
        // Buffer 2 (Right column)
        for (const auto& particle : ghost_recv_buf[2]) {
            int ghost_xi = num_sg_box_x;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy);

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
        
    } else if (mpi_grid_corner_4) {         //Bottom-Right
        // Ghost grid shape (+1 col to left, +1 row up)
        
        // Buffer 1 (Left column)
        for (const auto& particle : ghost_recv_buf[1]) {
            int ghost_xi = 0;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy);

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 2 Null

    } else if (mpi_left_edge) {             //Left
        // Ghost grid shape (+1 col to right, +2 row)
        
        // Buffer 1 Null
        
        // Buffer 2 (Right column)
        for (const auto& particle : ghost_recv_buf[2]) {
            int ghost_xi = num_sg_box_x;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
    } else if (mpi_right_edge) {            //Right
        // Ghost grid shape (+1 col to left, +2 row)
        
        // Buffer 1 (Left column)
        for (const auto& particle : ghost_recv_buf[1]) {
            int ghost_xi = 0;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 2 Null

    } else if (mpi_top_edge) {              //Top
        // Ghost grid shape (+ 2 col, + 1 row down)

        // Buffer 1 (Left column)
        for (const auto& particle : ghost_recv_buf[1]) {
            int ghost_xi = 0;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 2 (Right column)
        for (const auto& particle : ghost_recv_buf[2]) {
            int ghost_xi = num_sg_box_x;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
        
    } else if (mpi_bottom_edge) {           //Bottom
        // Ghost grid shape (+2 col, + 1 row up)
        
        // Buffer 1 (Left column)
        for (const auto& particle : ghost_recv_buf[1]) {
            int ghost_xi = 0;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy);

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 2 (Right column)
        for (const auto& particle : ghost_recv_buf[2]) {
            int ghost_xi = num_sg_box_x;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy);

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
    } else {                                //Internal
        // Ghost grid shape (+2 col, +2 row)
        
        // Buffer 1 (Left column)
        for (const auto& particle : ghost_recv_buf[1]) {
            int ghost_xi = 0;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 2 (Right column)
        for (const auto& particle : ghost_recv_buf[2]) {
            int ghost_xi = num_sg_box_x;
            int ghost_yj = std::floor((particle.y - mpi_rank.y1) / sg_box_dy) + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
    }
}                       

void pack_TB_buffer(std::vector<std::vector<particle_container>>& ghost_send_buf) {
    
    // Pack the top and bottom rows and the forwarded corners
    if (mpi_grid_corner_1) {                //Top-Left
        // Ghost grid shape (+1 col to right, +1 row down)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 Null

            // Buffer 4 (Bottom row + 1 forwarded corner)
            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                     mpi_rank.sub_grid[i][0].particles.begin(),
                                     mpi_rank.sub_grid[i][0].particles.end());
        }
            
        ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.ghost_grid[num_sg_box_x][1].particles.begin(),
                                 mpi_rank.ghost_grid[num_sg_box_x][1].particles.end());
            
    } else if (mpi_grid_corner_2) {         //Top-Right
        // Ghost grid shape (+1 col to left, +1 row down)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 Null

            // Buffer 4 (Bottom row + 1 forwarded corner)
            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.sub_grid[i][0].particles.begin(),
                                 mpi_rank.sub_grid[i][0].particles.end());
        }
            
        ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.ghost_grid[0][1].particles.begin(),
                                 mpi_rank.ghost_grid[0][1].particles.end());
            
    } else if (mpi_grid_corner_3) {         //Bottom-Left
        // Ghost grid shape (+1 col to right, +1 row up)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 (Top row + 1 forwarded corner)
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.begin(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.end());
        }

        ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.ghost_grid[num_sg_box_x][num_sg_box_y].begin(),
                                 mpi_rank.ghost_grid[num_sg_box_x][num_sg_box_y].end());

        // Buffer 4 Null
            
    } else if (mpi_grid_corner_4) {         //Bottom-Right
        // Ghost grid shape (+1 col to left, +1 row up)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 (Top row + 1 forwarded corner)
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.begin(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.end());
        }

        ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.ghost_grid[0][num_sg_box_y].begin(),
                                 mpi_rank.ghost_grid[0][num_sg_box_y].end());

        // Buffer 4 Null
    } else if (mpi_left_edge) {             //Left
        // Ghost grid shape (+1 col to right, +2 row)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 (Top row + 1 forwarded corner)
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.begin(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.end());

            // Buffer 4 (Bottom row + 1 forwarded corner)
            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.sub_grid[i][0].particles.begin(),
                                 mpi_rank.sub_grid[i][0].particles.end());

        }   
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                     mpi_rank.ghost_grid[num_sg_box_x][num_sg_box_y + 1].begin(),
                                     mpi_rank.ghost_grid[num_sg_box_x][num_sg_box_y + 1].end());

            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                     mpi_rank.ghost_grid[num_sg_box_x][1].particles.begin(),
                                     mpi_rank.ghost_grid[num_sg_box_x][1].particles.end());
            
    } else if (mpi_right_edge) {            //Right
        // Ghost grid shape (+1 col to left, +2 row)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 (Top row + 1 forwarded corner)
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.begin(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.end());

            // Buffer 4 (Bottom row + 1 forwarded corner)
            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.sub_grid[i][0].particles.begin(),
                                 mpi_rank.sub_grid[i][0].particles.end());
                                 
            }

            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                     mpi_rank.ghost_grid[0][num_sg_box_y + 1].begin(),
                                     mpi_rank.ghost_grid[0][num_sg_box_y + 1].end());

            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                     mpi_rank.ghost_grid[0][1].particles.begin(),
                                     mpi_rank.ghost_grid[0][1].particles.end());
            
    
    } else if (mpi_top_edge) {              //Top
        // Ghost grid shape (+ 2 col, + 1 row down)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 Null

            // Buffer 4 (Bottom row + 2 forwarded corners)
            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.sub_grid[i][0].particles.begin(),
                                 mpi_rank.sub_grid[i][0].particles.end());
        }
            
        ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.ghost_grid[0][1].particles.begin(),
                                 mpi_rank.ghost_grid[0][1].particles.end());

        ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.ghost_grid[num_sg_box_x + 1][1].particles.begin(),
                                 mpi_rank.ghost_grid[num_sg_box_x + 1][1].particles.end());
            
    } else if (mpi_bottom_edge) {           //Bottom
        // Ghost grid shape (+2 col, + 1 row up)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 (Top row + 2 forwarded corners)
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.begin(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.end());
        }
            
        ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.ghost_grid[0][num_sg_box_y].particles.begin(),
                                 mpi_rank.ghost_grid[0][num_sg_box_y].particles.end());

        ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.ghost_grid[num_sg_box_x + 1][num_sg_box_y].particles.begin(),
                                 mpi_rank.ghost_grid[num_sg_box_x + 1][num_sg_box_y].particles.end());

        // Buffer 4 Null

    } else {                                //Internal
        // Ghost grid shape (+2 col, +2 row)
        for (int i = 0; i < num_sg_box_x; i++) {
            // Buffer 3 (Top row + 2 forwarded corner)
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.begin(),
                                 mpi_rank.sub_grid[i][num_sg_box_y - 1].particles.end());

            // Buffer 4 (Bottom row + 2 forwarded corner)
            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                 mpi_rank.sub_grid[i][0].particles.begin(),
                                 mpi_rank.sub_grid[i][0].particles.end());
        }
            
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                     mpi_rank.ghost_grid[0][num_sg_box_y + 1].begin(),
                                     mpi_rank.ghost_grid[0][num_sg_box_y + 1].end());
            
            ghost_send_buf[3].insert(ghost_send_buf[3].end(),
                                     mpi_rank.ghost_grid[num_sg_box_x + 1][num_sg_box_y + 1].begin(),
                                     mpi_rank.ghost_grid[num_sg_box_x + 1][num_sg_box_y + 1].end());

            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                     mpi_rank.ghost_grid[0][1].particles.begin(),
                                     mpi_rank.ghost_grid[0][1].particles.end());

            ghost_send_buf[4].insert(ghost_send_buf[4].end(),
                                     mpi_rank.ghost_grid[num_sg_box_x + 1][1].particles.begin(),
                                     mpi_rank.ghost_grid[num_sg_box_x + 1][1].particles.end());
            
    }  
}

void sendrecv_TB_buffer(std::vector<std::vector<particle_container>>& ghost_send_buf,
                        std::vector<std::vector<particle_container>>& ghost_recv_buf) {

    // The local rank will:
    // SEND BUFFER      RECEIVE BUFFER      DIRECTION
    // 1 Left           1 Left              1 Left
    // 2 Right          2 Right             2 Right
    // 3 Up             3 Up                3 Up
    // 4 Down           4 Down              4 Down
    
    // Define working variables
    MPI_Status status;
    MPI_Request requests[4];
    int req_count = 0;

    // Obtain the ranks to the left and right
    int top_rank = get_destination_rank_id(3);
    int bottom_rank = get_destination_rank_id(4);

    // Send and receive the left column
    if (top_rank != MPI_PROC_NULL) {
        // Send the top row of the local rank to the rank above
        // This is the bottom of the ghost grid for the rank above
        MPI_Isend(ghost_send_buf[3], ghost_send_buf[3].size(), PARTICLE, top_rank, 3,
                  MPI_COMM_WORLD, &requests[req_count++]);

        // Receive the the bottom column from the rank above
        // This represents the top of the ghost grid for the local rank.
        MPI_Irecv(ghost_recv_buf[3], ghost_recv_buf[3].size(), PARTICLE, top_rank, 4,
                  MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Send and receive the right column
    if (bottom_rank != MPI_PROC_NULL) {

        // Send the bottom row of the local rank to the rank below
        // This represents the top of the ghost grid for the rank below.
        MPI_Isend(ghost_send_buf[4], ghost_send_buf[4].size(), PARTICLE, bottom_rank, 4,
                  MPI_COMM_WORLD, &requests[req_count++]);

        // Receive the top row from the rank above
        // This represents the bottom of the ghost grid for the local rank.
        MPI_Irecv(ghost_recv_buf[4], ghost_recv_buf[4].size(), PARTICLE, bottom_rank, 3,
                  MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait to complete communicating
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

void unpack_TB_buffer(std::vector<std::vector<particle_container>>& ghost_recv_buf) {

    if (mpi_grid_corner_1) {                //Top-Left
        // Ghost grid shape (+1 col to right, +1 row down)
        
        // Buffer 3 Null
        
        // Buffer 4 (Bottom row)
        for (const auto& particle : ghost_recv_buf[4]) {
            bool FC1 = particle.x < mpi_rank.x2;
            int ghost_xi = FC1 ? std::floor((particle.x - mpi_rank.x1) / sg_box_dx) : num_sg_box_x;
            int ghost_yj = 0;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        } 
    } else if (mpi_grid_corner_2) {         //Top-Right
        // Ghost grid shape (+1 col to left, +1 row down)

        // Buffer 3 Null
        
        // Buffer 4 (Bottom row)
        for (const auto& particle : ghost_recv_buf[4]) {
            bool FC1 = particle.x < mpi_rank.x1;
            int ghost_xi = FC1 ? 0 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;
            int ghost_yj = 0;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
        
    } else if (mpi_grid_corner_3) {         //Bottom-Left
        // Ghost grid shape (+1 col to right, +1 row up)

        // Buffer 3 (Top row)
        for (const auto& particle : ghost_recv_buf[3]) {
            bool FC1 = particle.x < mpi_rank.x2;
            int ghost_xi = FC1 ? std::floor((particle.x - mpi_rank.x1) / sg_box_dx) : num_sg_box_x;
            int ghost_yj = num_sg_box_y;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 4 Null
        
    } else if (mpi_grid_corner_4) {         //Bottom-Right
        // Ghost grid shape (+1 col to left, +1 row up)
        
        // Buffer 3 (Top row)
        for (const auto& particle : ghost_recv_buf[3]) {
            bool FC1 = particle.x < mpi_rank.x1;
            int ghost_xi = FC1 ? 0 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;;
            int ghost_yj = num_sg_box_y;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 4 Null

    } else if (mpi_left_edge) {             //Left
        // Ghost grid shape (+1 col to right, +2 row)
        
        // Buffer 3 (Top row)
        for (const auto& particle : ghost_recv_buf[3]) {
            bool FC1 = particle.x < mpi_rank.x2;
            int ghost_xi = FC1 ? std::floor((particle.x - mpi_rank.x1) / sg_box_dx) : num_sg_box_x;
            int ghost_yj = num_sg_box_y + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
        
        // Buffer 4 (Bottom row)
        for (const auto& particle : ghost_recv_buf[4]) {
            bool FC1 = particle.x < mpi_rank.x2;
            int ghost_xi = FC1 ? std::floor((particle.x - mpi_rank.x1) / sg_box_dx) : num_sg_box_x;
            int ghost_yj = 0;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

    } else if (mpi_right_edge) {            //Right
        // Ghost grid shape (+1 col to left, +2 row)
        
        // Buffer 3 (Top row)
        for (const auto& particle : ghost_recv_buf[3]) {
            bool FC1 = particle.x < mpi_rank.x1;
            int ghost_xi = FC1 ? 0 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;
            int ghost_yj = num_sg_box_y + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
        
        // Buffer 4 (Bottom row)
        for (const auto& particle : ghost_recv_buf[4]) {
            bool FC1 = particle.x < mpi_rank.x1;
            int ghost_xi = FC1 ? 0 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;
            int ghost_yj = 0;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

    } else if (mpi_top_edge) {              //Top
        // Ghost grid shape (+ 2 col, + 1 row down)

        // Buffer 3 Null
        
        // Buffer 4 (Bottom row)
        for (const auto& particle : ghost_recv_buf[4]) {
            bool FC1 = particle.x < mpi_rank.x1;
            bool FC2 = particle.x > mpi_rank.x2;
            int ghost_xi = FC1 ? 0 : FC2 ? num_sg_box_x + 1 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;
            int ghost_yj = 0;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
        
    } else if (mpi_bottom_edge) {           //Bottom
        // Ghost grid shape (+2 col, + 1 row up)
        
        // Buffer 3 (Top row)
        for (const auto& particle : ghost_recv_buf[3]) {
            bool FC1 = particle.x < mpi_rank.x1;
            bool FC2 = particle.x > mpi_rank.x2;
            int ghost_xi = FC1 ? 0 : FC2 ? num_sg_box_x + 1 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;
            int ghost_yj = num_sg_box_y;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }

        // Buffer 4 Null

    } else {                                //Internal
        // Ghost grid shape (+2 col, +2 row)
        
        // Buffer 3 (Top row)
        for (const auto& particle : ghost_recv_buf[3]) {
            bool FC1 = particle.x < mpi_rank.x1;
            bool FC2 = particle.x > mpi_rank.x2;
            int ghost_xi = FC1 ? 0 : FC2 ? num_sg_box_x + 1 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;
            int ghost_yj = num_sg_box_y + 1;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
        
        // Buffer 4 (Bottom row)
        for (const auto& particle : ghost_recv_buf[4]) {
            bool FC1 = particle.x < mpi_rank.x1;
            bool FC2 = particle.x > mpi_rank.x2;
            int ghost_xi = FC1 ? 0 : FC2 ? num_sg_box_x + 1 : std::floor((particle.x - mpi_rank.x1) / sg_box_dx) + 1;
            int ghost_yj = 0;

            mpi_rank.ghost_grid[ghost_xi][ghost_yj].particles.push_back(particle);
        }
    }
}

// Send and recieve ghost buffers to and from appropriate ranks
void update_ghost_grid(double size) {

    // Clear the ghost grid
    for (int i = 0; i < num_sg_box_x; i++) {
        for (int j = 0; j < num_sg_box_y; j++) {
            mpi_rank.ghost_grid[i][j].particles.clear();
        }
    }

    // Initialize send and receive ghost buffers.
    std::vector<std::vector<particle_t>> ghost_send_buf(4), ghost_recv_buf(4);
    // The local rank will:
    // SEND BUFFER      RECEIVE BUFFER      DIRECTION
    // 1 Left           2 Right             1 Left
    // 2 Right          1 Left              2 Right
    // 3 Up             4 Down              3 Up
    // 4 Down           3 Up                4 Down

    // Pack the left and right columns send buffer for this rank
    pack_LR_buffer(ghost_send_buf);

    // Send and receive left and right ghost buffers from appropriate ranks
    sendrecv_LR_buffer(ghost_send_buf, ghost_recv_buf);

    // Unpack the received left and right ghost buffers
    unpack_LR_buffer(ghost_recv_buf);

    // Pack the top and bottom rows send buffer for this rank
    pack_TB_buffer(ghost_send_buf);

    // Send and receive the top and bottom ghost buffers from appropriate ranks
    sendrecv_TB_buffer(ghost_send_buf, ghost_recv_buf);

    // Unpack the received top and bottom ghost buffers
    unpack_TB_buffer(ghost_recv_buf);
}

*/
//====================================
// Functions for particle computations
//====================================

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Function to determine which particles to evaluate against.
void check_nearby_containers(particle_t &particle) {

    // Loop through the 3x3 grid of containers centered on the current particle.
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            // Define the current container to be evaluated
            int cx = particle.x + dx;
            int cy = particle.y + dy;

            std::vector<particle_t>& neighbor_particles =mpi_rank.sub_grid[cx][cy].particles;

            // Loop through the particles in the container
            for (auto &neighbor : neighbor_particles) {
                if (&particle != &neighbor) {
                    apply_force(particle, neighbor);
                } 
            }
        }
    }
    /*
    // Get the xy-position of the particle being evaluated
    int particle_xi = std::floor((particle.x - mpi_rank.x1) * sg_box_dx);
    int particle_yj = std::floor((particle.y - mpi_rank.y1) * sg_box_dy);

    // Loop through the 3x3 grid of containers centered on the current particle.
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            // Define the current container to be evaluated
            int cx = particle_xi + dx;
            int cy = particle_yj + dy;

            // Create a bool if cx or cy is outside the domain of the sub grid
            bool check_ghost_grid = ((cx < 0) || (cy < 0) || (cx >= num_sg_box_x) || (cy >= num_sg_box_y));
            
            if (check_ghost_grid) {
                if (mpi_grid_corner_1) {                //Top-Left
                    // Ghost grid shape (+1 col to right, +1 row down)
                    if ((cx < 0) || (cy >= num_sg_box_y)) {
                        continue;
                    } 
                    int gy = (cy < 0) ? 0 : cy + 1;
                    std::vector<particle_t>& neighbor_particles = mpi_rank.ghost_grid[cx][gy];
    
                } else if (mpi_grid_corner_2) {         //Top-Right
                    // Ghost grid shape (+1 col to left, +1 row down)
                    if ((cx >= num_sg_box_x) || (cy >= num_sg_box_y)) {
                        continue;
                    }
                    int gx = (cx < 0) ? 0 : cx + 1;
                    int gy = (cy < 0) ? 0 : cy + 1;
                    std::vector<particle_t>& neighbor_particles = mpi_rank.ghost_grid[gx][gy];
                    
                } else if (mpi_grid_corner_3) {         //Bottom-Left
                    // Ghost grid shape (+1 col to right, +1 row up)
                    if ((cx < 0) || (cy < 0)) {
                        continue;
                    }
                    std::vector<particle_t>& neighbor_particles =mpi_rank.ghost_grid[cx][cy];
                    
                } else if (mpi_grid_corner_4) {         //Bottom-Right
                    // Ghost grid shape (+1 col to left, +1 row up)
                    if ((cx >= num_sg_box_x) || (cy < 0)) {
                        continue;
                    }
                    int gx = (cx < 0) ? 0 : cx + 1;
                    std::vector<particle_t>& neighbor_particles = mpi_rank.ghost_grid[gx][cy];
            
                } else if (mpi_left_edge) {             //Left
                    // Ghost grid shape (+1 col to right, +2 row)
                    if (cx < 0) {
                        continue;
                    }
                    int gy = (cy < 0) ? 0 : cy + 1;
                    std::vector<particle_t>& neighbor_particles = mpi_rank.ghost_grid[cx][gy];
            
                } else if (mpi_right_edge) {            //Right
                    // Ghost grid shape (+1 col to left, +2 row)
                    if (cx > 0) {
                        continue;
                    }
                    int gx = (cx < 0) ? 0 : cx + 1;
                    int gy = (cy < 0) ? 0 : cy + 1;
                    std::vector<particle_t>& neighbor_particles = mpi_rank.ghost_grid[gx][gy];
            
                } else if (mpi_top_edge) {              //Top
                    // Ghost grid shape (+ 2 col, + 1 row down)
                    if (cy > 0) {
                        continue;
                    }
                    int gx = (cx < 0) ? 0 : cx + 1;
                    int gy = (cy < 0) ? 0 : cy + 1;
                    std::vector<particle_t>& neighbor_particles = mpi_rank.ghost_grid[gx][gy];
                    
                } else if (mpi_bottom_edge) {           //Bottom
                    // Ghost grid shape (+2 col, + 1 row up)
                    if (cy < 0) {
                        continue;
                    }
                    int gx = (cx < 0) ? 0 : cx + 1;
                    std::vector<particle_t>& neighbor_particles = mpi_rank.ghost_grid[gx][cy];
                } 
                
            } else {
                std::vector<particle_t>& neighbor_particles =mpi_rank.ghost_grid[cx][cy];
            }
            
            // Loop through the particles in the container
            for (auto &neighbor : neighbor_particles) {
                if (&particle != &neighbor) {
                    apply_force(particle, neighbor);
                } 
            }
        }
    }

    */
}

// Function to move particles based on their acceleration
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

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // Update the sub grid
    update_sub_grid();

    // Update ghost grid
    //update_ghost_grid();
    if (rank == 0) {
        std::cout << "Working before computing forces" << std::endl;
    }
    
    // Loop through all of the particles in the rank and compute the forces
    for (particle_t& particle : mpi_rank.particle_list) {

        // Reset accelerations to avoid accumulation error
        particle.ax = particle.ay = 0;

        // Check particles within one cutoff distance
        check_nearby_containers(particle);
    }
    if (rank == 0) {
        std::cout << "Working before moving particles" << std::endl;
    }
    
    // Move Particles after the forces have been computed for all particles
    for (int i = 0; i < mpi_rank.particle_list.size(); ++i) {
        move(mpi_rank.particle_list[i], size);
    }

    if (rank == 0) {
        std::cout << "Working before updating particles moving between ranks" << std::endl;
    }

    // Update the particles in the MPI grid box particle list
    update_mpi_particle_list(rank);

    if (rank == 0) {
        std::cout << "Completed simulation step" << std::endl;
    }
    
    

    MPI_Barrier(MPI_COMM_WORLD);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    
    std::cout << "---------------------------------------------------------------------------------------------------------" << std::endl;
    

    // Step 1: Get the number of particles on each rank
    int num_particles_on_rank = mpi_rank.particle_list.size();

    if (rank == 0) {
        std::cout << "Gather for save error" << std::endl;
    }

    // Step 2: Gather the counts of particles on each rank at rank 0
    std::vector<int> recv_counts(num_procs, 0); // Stores number of particles each rank sends
    MPI_Gather(&num_particles_on_rank, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 3: Compute displacements at rank 0
    std::vector<int> displacements(num_procs, 0); // Starting index for each rank’s data in the final array
    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }

    // Step 4: Gather all particles into `parts` on rank 0
    MPI_Gatherv(mpi_rank.particle_list.data(), num_particles_on_rank, PARTICLE,
                parts, recv_counts.data(), displacements.data(), PARTICLE, 
                0, MPI_COMM_WORLD);

    // Step 5: Ensure particles are sorted by ID (only rank 0 does this)
    if (rank == 0) {
        std::sort(parts, parts + num_parts, [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });
    }
}