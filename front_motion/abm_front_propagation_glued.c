#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

/* ==========================================================
   CONSTANTS AND MACROS
   ========================================================== */

#define MAX_PARTICLES 200000
#define FRONT_NTHRESH 3

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/* ==========================================================
   RANDOM NUMBER GENERATOR (xoshiro256++)
   ========================================================== */

static uint64_t s[4];

static inline uint64_t rotl64(const uint64_t x, int k){
    return (x << k) | (x >> (64-k));
}

static inline uint64_t xoshiro256pp(void){
    uint64_t result;
    uint64_t t;

    result = rotl64(s[0] + s[3], 23) + s[0];
    t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl64(s[3], 45);

    return result;
}

static inline void seed_xoshiro256pp(uint64_t seed){
    uint64_t z;

    /* SplitMix64 initialization */
    for(int i = 0; i < 4; i++){
        seed += 0x9e3779b97f4a7c15ULL;
        z = seed;
        z = (z^(z>>30))*0xbf58476d1ce4e5b9ULL;
        z = (z^(z>>27))*0x94d049bb133111ebULL;
        s[i] = z^(z>>31);
    }
}

static inline double rand_uniform(void){
    return (xoshiro256pp()>>11)*(1.0/9007199254740992.0);
}

/* Box-Muller normal RNG polar method */
static inline double rand_normal(void){
    static bool hasSpare = false;
    static double spare;
    double u, v, r;

    if(hasSpare){
        hasSpare = false;
        return spare;
    }

    do{
        u = rand_uniform()*2.0 - 1.0;
        v = rand_uniform()*2.0 - 1.0;
        r = u*u + v*v;
    } while(r >= 1.0 || r == 0.0);

    r = sqrt(-2.0*log(r)/r);
    spare = v*r;
    hasSpare = true;

    return u*r;
}

/* ==========================================================
   PARTICLE STRUCTURE
   ========================================================== */

typedef struct {
    double x;
    double y;
    double theta;
} Particle;

/* ==========================================================
   PARAMETER STRUCTURE
   ========================================================== */

typedef struct {
    int run_id;
    uint64_t seed;

    double p0;
    double q0;
    int Ns;
    double R_inter;
    double rho0;

    int save_per_step;
    int front_per_step;
    int rho_profile_every_front;

    double isolation_buffer_factor;

    double v0;
    double Dr;
    double Dtheta;
    double dt;
    double T;

    double Lx;
    double Ly;
    double x_init_min;
    double x_init_max;
    double warmup_T;

    double rho_sat;
    int nbins_x;
    double threshold_frac[FRONT_NTHRESH];
} Params;

/* ==========================================================
   FRONT MEASUREMENT STRUCTURES
   ========================================================== */

typedef struct {
    double Lx;
    double Ly;
    double x_center;
    double rho_sat;
    int nbins_x;
    double dx;
    int nthresh;
    double threshold_frac[FRONT_NTHRESH];
} FrontParams;

typedef struct {
    int *count_x;
    double *rho_x;
} FrontWork;

/* ==========================================================
   GLUED CENTRAL BULK STRUCTURE
   ========================================================== */

typedef struct {
    bool has_region;
    double buffer_factor;
    double buffer;
    double min_delete_width;
    double density_threshold;
    int n_detect_bins;
    double x_delete_L;
    double x_delete_R;
} GluedBulkState;

/* Compactified coordinate helpers for the glued central-bulk method. */
double glued_bulk_deleted_width(GluedBulkState *iso);
double glued_bulk_compact_length(GluedBulkState *iso, double Lx);
double glued_bulk_x_to_s(double x, GluedBulkState *iso);
double glued_bulk_s_to_x(double s, GluedBulkState *iso);

typedef struct {
    int max_total_events;
    int births_at_max;
    int deaths_at_max;
    int step_at_max;
    int N_before_at_max;
    int N_after_at_max;
    double time_at_max;
} EventStats;

/* ==========================================================
   PERIODIC BOUNDARY CONDITIONS
   ========================================================== */

double wrap(double x, double L){
    while(x < 0.0){
        x += L;
    }
    while(x >= L){
        x -= L;
    }
    return x;
}

double wrap_interval(double x, double x_min, double L){
    return x_min + wrap(x - x_min, L);
}

double pbc_displacement(double dx, double L){
    while(dx > 0.5 * L){
        dx -= L;
    }
    while(dx < -0.5 * L){
        dx += L;
    }
    return dx;
}

bool validate_pbc_displacement_dimensions(double R_inter,
                                          double x_init_min,
                                          double x_init_max,
                                          double Ly){
    double x_init_width;

    x_init_width = x_init_max - x_init_min;

    if(R_inter <= 0.0 || x_init_width <= 0.0 || Ly <= 0.0){
        printf("Error: periodic boundary conditions cannot be computed due to dimensions. ");
        printf("Need R_inter > 0, x_init_max > x_init_min, and Ly > 0. ");
        printf("Got R_inter = %.10g, x_init_width = %.10g, Ly = %.10g.\n",
               R_inter, x_init_width, Ly);
        return false;
    }

    if(R_inter >= 0.5 * x_init_width){
        printf("Error: periodic boundary conditions cannot be computed due to dimensions. ");
        printf("The warmup x-periodic box is too small for pbc_displacement: ");
        printf("R_inter = %.10g must be smaller than (x_init_max - x_init_min)/2 = %.10g.\n",
               R_inter, 0.5 * x_init_width);
        return false;
    }

    if(R_inter >= 0.5 * Ly){
        printf("Error: periodic boundary conditions cannot be computed due to dimensions. ");
        printf("The y-periodic box is too small for pbc_displacement: ");
        printf("R_inter = %.10g must be smaller than Ly/2 = %.10g.\n",
               R_inter, 0.5 * Ly);
        return false;
    }

    return true;
}

/* ==========================================================
   PARAMETER FILE READING
   ========================================================== */

void trim_in_place(char *s){
    int start;
    int end;
    int len;
    int i;

    len = (int)strlen(s);
    start = 0;

    while(start < len &&
          (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')){
        start++;
    }

    end = len - 1;
    while(end >= start &&
          (s[end] == ' ' || s[end] == '\t' || s[end] == '\n' || s[end] == '\r')){
        end--;
    }

    if(start > end){
        s[0] = '\0';
        return;
    }

    if(start > 0){
        for(i = start; i <= end; i++){
            s[i - start] = s[i];
        }
        s[end - start + 1] = '\0';
    }
    else{
        s[end + 1] = '\0';
    }
}

void remove_inline_comment(char *line){
    char *comment;

    comment = strchr(line, '#');
    if(comment != NULL){
        *comment = '\0';
    }
}

void init_params_missing(Params *par){
    par->run_id = -1;
    par->seed = UINT64_MAX;

    par->p0 = NAN;
    par->q0 = NAN;
    par->Ns = -1;
    par->R_inter = NAN;
    par->rho0 = NAN;

    par->save_per_step = -1;
    par->front_per_step = -1;
    par->rho_profile_every_front = 0;

    par->isolation_buffer_factor = NAN;

    par->v0 = NAN;
    par->Dr = NAN;
    par->Dtheta = NAN;
    par->dt = NAN;
    par->T = NAN;

    par->Lx = NAN;
    par->Ly = NAN;
    par->x_init_min = NAN;
    par->x_init_max = NAN;
    par->warmup_T = NAN;

    par->rho_sat = NAN;
    par->nbins_x = -1;

    for(int k = 0; k < FRONT_NTHRESH; k++){
        par->threshold_frac[k] = NAN;
    }
}

int assign_param_value(Params *par, const char *key, const char *value){
    unsigned long long seed_tmp;

    if(strcmp(key, "run_id") == 0){
        par->run_id = atoi(value);
    }
    else if(strcmp(key, "seed") == 0){
        seed_tmp = strtoull(value, NULL, 10);
        par->seed = (uint64_t)seed_tmp;
    }
    else if(strcmp(key, "p0") == 0){
        par->p0 = atof(value);
    }
    else if(strcmp(key, "q0") == 0){
        par->q0 = atof(value);
    }
    else if(strcmp(key, "Ns") == 0){
        par->Ns = atoi(value);
    }
    else if(strcmp(key, "R_inter") == 0){
        par->R_inter = atof(value);
    }
    else if(strcmp(key, "rho0") == 0){
        par->rho0 = atof(value);
    }
    else if(strcmp(key, "save_per_step") == 0){
        par->save_per_step = atoi(value);
    }
    else if(strcmp(key, "front_per_step") == 0){
        par->front_per_step = atoi(value);
    }
    else if(strcmp(key, "rho_profile_every_front") == 0){
        par->rho_profile_every_front = atoi(value);
    }
    else if(strcmp(key, "isolation_buffer_factor") == 0){
        par->isolation_buffer_factor = atof(value);
    }
    else if(strcmp(key, "v0") == 0){
        par->v0 = atof(value);
    }
    else if(strcmp(key, "Dr") == 0){
        par->Dr = atof(value);
    }
    else if(strcmp(key, "Dtheta") == 0){
        par->Dtheta = atof(value);
    }
    else if(strcmp(key, "dt") == 0){
        par->dt = atof(value);
    }
    else if(strcmp(key, "T") == 0){
        par->T = atof(value);
    }
    else if(strcmp(key, "Lx") == 0){
        par->Lx = atof(value);
    }
    else if(strcmp(key, "Ly") == 0){
        par->Ly = atof(value);
    }
    else if(strcmp(key, "x_init_min") == 0){
        par->x_init_min = atof(value);
    }
    else if(strcmp(key, "x_init_max") == 0){
        par->x_init_max = atof(value);
    }
    else if(strcmp(key, "warmup_T") == 0){
        par->warmup_T = atof(value);
    }
    else if(strcmp(key, "rho_sat") == 0){
        par->rho_sat = atof(value);
    }
    else if(strcmp(key, "nbins_x") == 0){
        par->nbins_x = atoi(value);
    }
    else if(strcmp(key, "threshold_frac1") == 0){
        par->threshold_frac[0] = atof(value);
    }
    else if(strcmp(key, "threshold_frac2") == 0){
        par->threshold_frac[1] = atof(value);
    }
    else if(strcmp(key, "threshold_frac3") == 0){
        par->threshold_frac[2] = atof(value);
    }
    else{
        return 1;
    }

    return 0;
}

int check_required_params(Params *par, char *missing, int max_len){
    missing[0] = '\0';

    if(par->run_id < 0){ strcat(missing, " run_id"); }
    if(par->seed == UINT64_MAX){ strcat(missing, " seed"); }

    if(isnan(par->p0)){ strcat(missing, " p0"); }
    if(isnan(par->q0)){ strcat(missing, " q0"); }
    if(par->Ns < 0){ strcat(missing, " Ns"); }
    if(isnan(par->R_inter)){ strcat(missing, " R_inter"); }
    if(isnan(par->rho0)){ strcat(missing, " rho0"); }

    if(par->save_per_step < 0){ strcat(missing, " save_per_step"); }
    if(par->front_per_step < 0){ strcat(missing, " front_per_step"); }

    if(isnan(par->v0)){ strcat(missing, " v0"); }
    if(isnan(par->Dr)){ strcat(missing, " Dr"); }
    if(isnan(par->Dtheta)){ strcat(missing, " Dtheta"); }
    if(isnan(par->dt)){ strcat(missing, " dt"); }
    if(isnan(par->T)){ strcat(missing, " T"); }

    if(isnan(par->Lx)){ strcat(missing, " Lx"); }
    if(isnan(par->Ly)){ strcat(missing, " Ly"); }
    if(isnan(par->x_init_min)){ strcat(missing, " x_init_min"); }
    if(isnan(par->x_init_max)){ strcat(missing, " x_init_max"); }
    if(isnan(par->warmup_T)){ strcat(missing, " warmup_T"); }

    if(isnan(par->rho_sat)){ strcat(missing, " rho_sat"); }
    if(par->nbins_x < 0){ strcat(missing, " nbins_x"); }

    if(isnan(par->threshold_frac[0])){ strcat(missing, " threshold_frac1"); }
    if(isnan(par->threshold_frac[1])){ strcat(missing, " threshold_frac2"); }
    if(isnan(par->threshold_frac[2])){ strcat(missing, " threshold_frac3"); }

    if((int)strlen(missing) >= max_len){
        missing[max_len - 1] = '\0';
    }

    if(missing[0] != '\0'){
        return 1;
    }

    return 0;
}

int read_params_named_blocks(FILE *f, int target_run_id, Params *par){
    char line[2048];
    char clean_line[2048];
    char key[256];
    char value[256];
    char missing[2048];
    char *equal;
    Params candidate;
    int current_block;
    int block_unknown_count;
    bool in_block;

    rewind(f);
    init_params_missing(par);
    init_params_missing(&candidate);

    current_block = -1;
    block_unknown_count = 0;
    in_block = false;

    while(fgets(line, sizeof(line), f) != NULL){
        strcpy(clean_line, line);
        remove_inline_comment(clean_line);
        trim_in_place(clean_line);

        if(clean_line[0] == '\0'){
            continue;
        }

        if(strcmp(clean_line, "[run]") == 0){
            if(in_block && candidate.run_id == target_run_id){
                *par = candidate;

                if(check_required_params(par, missing, sizeof(missing)) != 0){
                    printf("Error: missing required parameters in [run] block with run_id %d:%s\n", target_run_id, missing);
                    return 2;
                }

                if(block_unknown_count > 0){
                    printf("Warning: ignored %d unknown parameter key(s) in selected run_id %d.\n",
                           block_unknown_count, target_run_id);
                }

                return 0;
            }

            current_block++;
            init_params_missing(&candidate);
            block_unknown_count = 0;
            in_block = true;
            continue;
        }

        if(!in_block){
            printf("Error: parameter file must use [run] blocks. Found text before first [run]: %s\n",
                   clean_line);
            return 2;
        }

        equal = strchr(clean_line, '=');
        if(equal == NULL){
            printf("Error: expected 'key = value' inside [run] block %d, got: %s\n",
                   current_block, clean_line);
            return 2;
        }

        *equal = '\0';

        strncpy(key, clean_line, sizeof(key) - 1);
        key[sizeof(key) - 1] = '\0';
        strncpy(value, equal + 1, sizeof(value) - 1);
        value[sizeof(value) - 1] = '\0';

        trim_in_place(key);
        trim_in_place(value);

        if(assign_param_value(&candidate, key, value) != 0){
            block_unknown_count++;
        }
    }

    if(in_block && candidate.run_id == target_run_id){
        *par = candidate;

        if(check_required_params(par, missing, sizeof(missing)) != 0){
            printf("Error: missing required parameters in [run] block with run_id %d:%s\n", target_run_id, missing);
            return 2;
        }

        if(block_unknown_count > 0){
            printf("Warning: ignored %d unknown parameter key(s) in selected run_id %d.\n",
                   block_unknown_count, target_run_id);
        }

        return 0;
    }

    return 3;
}

int read_params(const char *filename, int target_run_id, Params *par){
    FILE *f;
    int status;

    f = fopen(filename, "r");
    if(f == NULL){
        return 1;
    }

    status = read_params_named_blocks(f, target_run_id, par);

    fclose(f);
    return status;
}

/* ==========================================================
   OUTPUT FILE NAME
   ========================================================== */

void get_base_name_without_extension(const char *input_name, char *base_name, int max_len){
    int len;
    int start;
    int end;
    int i,j;
    bool found;

    len = (int)strlen(input_name);

    end = len;
    /* Find the start of the base name after last '/' or '\\' */
    i = len - 1;
    start = 0;
    found = false;
    while(i >= 0 && !found){
        if(input_name[i] == '/' || input_name[i] == '\\'){
            start = i + 1;
            found = 1;
        }
        i--;
    }

    /* Find the end of the base name before last '.' */
    i = len - 1;
    found = 0;
    while(i >= start && !found){
        if(input_name[i] == '.'){
            end = i;
            found = 1;
        }
        i--;
    }

    j = 0;
    for(i = start; i < end && j < max_len - 1; i++){
        base_name[j++] = input_name[i];
    }
    base_name[j] = '\0';
}

/* ==========================================================
   INITIALIZATION
   ========================================================== */

void initialize_front_seed(Particle *p, int N0, double x_min, double x_max, double Ly){
    double width;

    width = x_max - x_min;

    for(int i = 0; i < N0; i++){
        p[i].x = x_min + width * rand_uniform();
        p[i].y = Ly * rand_uniform();
        p[i].theta = 2.0 * M_PI * rand_uniform();
    }
}



/* ==========================================================
   NEIGHBOR COUNT IN A TUBE
   ========================================================== */

bool cell_was_visited(int *visited, int nvisited, int cell){
    for(int k = 0; k < nvisited; k++){
        if(visited[k] == cell){
            return true;
        }
    }

    return false;
}

int count_neighbors_rect(Particle *p, int i, int *head, int *next,
                         int Nx_cells, int Ny_cells,
                         double cell_size,
                         double x_min, double Lx_domain, double Ly,
                         double R_inter, bool periodic_x,
                         GluedBulkState *iso){
    int Ni;
    int cx, cy;
    int nx, ny;
    int cell;
    int j;
    int visited[9];
    int nvisited;
    double dist_x, dist_y;
    double R2;
    double xi;
    double xj;

    Ni = 0;
    R2 = R_inter * R_inter;

    xi = glued_bulk_x_to_s(p[i].x, iso);

    cx = (int)((xi - x_min) / cell_size);
    cy = (int)(p[i].y / cell_size);

    if(cx < 0){
        cx = 0;
    }
    if(cx >= Nx_cells){
        cx = Nx_cells - 1;
    }
    if(cy < 0){
        cy = 0;
    }
    if(cy >= Ny_cells){
        cy = Ny_cells - 1;
    }

    nvisited = 0;

    for(int dx = -1; dx <= 1; dx++){
        for(int dy = -1; dy <= 1; dy++){

            nx = cx + dx;
            ny = cy + dy;

            if(periodic_x){
                nx = (nx + Nx_cells) % Nx_cells;
            }
            else{
                if(nx < 0 || nx >= Nx_cells){
                    continue;
                }
            }

            ny = (ny + Ny_cells) % Ny_cells;

            cell = nx + Nx_cells * ny;

            if(cell_was_visited(visited, nvisited, cell)){
                continue;
            }

            visited[nvisited] = cell;
            nvisited++;

            j = head[cell];
            while(j != -1){
                if(j != i){
                    xj = glued_bulk_x_to_s(p[j].x, iso);
                    dist_x = xj - xi;
                    if(periodic_x){
                        dist_x = pbc_displacement(dist_x, Lx_domain);
                    }

                    dist_y = pbc_displacement(p[j].y - p[i].y, Ly);

                    if((dist_x*dist_x + dist_y*dist_y) <= R2){
                        Ni++;
                    }
                }
                j = next[j];
            }
        }
    }

    return Ni;
}





/* ==========================================================
   SAVE TRAJECTORY
   ========================================================== */

void save_particles(FILE *f, Particle *p, int N, int step, double t){

    fprintf(f, "FRAME %d %.8f %d\n", step, t, N);

    for(int i = 0; i < N; i++){
        fprintf(f, "%d %.8f %.8f %.8f\n", i, p[i].x, p[i].y, p[i].theta);
    }

    fprintf(f, "\n");
}

/* ==========================================================
   FRONT OBSERVABLES
   ========================================================== */

double quantile_from_histogram(int *count_x, int nbins_x, double dx, int N, double q){
    int target;
    int cumulative;

    if(N <= 0){
        return NAN;
    }

    target = (int)ceil(q * (double)N);
    if(target < 1){
        target = 1;
    }

    cumulative = 0;

    for(int j = 0; j < nbins_x; j++){
        cumulative += count_x[j];
        if(cumulative >= target){
            return (j + 0.5) * dx;
        }
    }

    return (nbins_x - 0.5) * dx;
}

double interpolate_crossing(double x0, double x1, double rho0, double rho1, double rho_star){
    double denom;

    denom = rho1 - rho0;

    if(fabs(denom) < 1e-14){
        return 0.5 * (x0 + x1);
    }

    return x0 + (rho_star - rho0) * (x1 - x0) / denom;
}

double find_left_threshold_front(double *rho_x, int nbins_x, double dx, double rho_star){
    double x0, x1;

    if(rho_x[0] >= rho_star){
        return 0.0;
    }

    for(int j = 0; j < nbins_x - 1; j++){
        if(rho_x[j] < rho_star && rho_x[j+1] >= rho_star){
            x0 = (j + 0.5) * dx;
            x1 = (j + 1.5) * dx;
            return interpolate_crossing(x0, x1, rho_x[j], rho_x[j+1], rho_star);
        }
    }

    return NAN;
}

double find_right_threshold_front(double *rho_x, int nbins_x, double dx, double Lx, double rho_star){
    double x_left;
    double x_right;

    if(rho_x[nbins_x - 1] >= rho_star){
        return Lx;
    }

    for(int j = nbins_x - 1; j > 0; j--){
        if(rho_x[j] < rho_star && rho_x[j - 1] >= rho_star){
            x_left = (j - 0.5) * dx;
            x_right = (j + 0.5) * dx;
            return interpolate_crossing(x_left, x_right,
                                        rho_x[j - 1], rho_x[j],
                                        rho_star);
        }
    }

    return NAN;
}

void save_density_profile(FILE *frho, int step, double t, int N,
                          FrontParams *fp, FrontWork *fw){
    if(frho == NULL){
        return;
    }

    fprintf(frho, "%d %.8f %d", step, t, N);

    for(int j = 0; j < fp->nbins_x; j++){
        fprintf(frho, " %.10g", fw->rho_x[j]);
    }

    fprintf(frho, "\n");
}

bool should_save_density_profile(int rho_profile_every_front,
                                 int front_measurement_index,
                                 bool force){
    if(rho_profile_every_front <= 0){
        return false;
    }

    if(force){
        return true;
    }

    return (front_measurement_index % rho_profile_every_front) == 0;
}

void measure_front(FILE *ffront, FILE *frho, Particle *p, int N, int step, double t,
                   FrontParams *fp, FrontWork *fw, bool hit_boundary,
                   bool write_density_profile){
    int bin;
    double x_left_tip;
    double x_right_tip;
    double x_q01;
    double x_q99;
    double rho_star;
    double x_left_th[FRONT_NTHRESH];
    double x_right_th[FRONT_NTHRESH];

    for(int j = 0; j < fp->nbins_x; j++){
        fw->count_x[j] = 0;
        fw->rho_x[j] = 0.0;
    }

    x_left_tip = NAN;
    x_right_tip = NAN;

    if(N > 0){
        x_left_tip = p[0].x;
        x_right_tip = p[0].x;
    }

    for(int i = 0; i < N; i++){
        if(p[i].x < x_left_tip){
            x_left_tip = p[i].x;
        }
        if(p[i].x > x_right_tip){
            x_right_tip = p[i].x;
        }

        bin = (int)(p[i].x / fp->dx);

        if(bin < 0){
            bin = 0;
        }
        if(bin >= fp->nbins_x){
            bin = fp->nbins_x - 1;
        }

        fw->count_x[bin]++;
    }

    for(int j = 0; j < fp->nbins_x; j++){
        fw->rho_x[j] = (double)fw->count_x[j] / (fp->dx * fp->Ly);
    }

    x_q01 = quantile_from_histogram(fw->count_x, fp->nbins_x, fp->dx, N, 0.01);
    x_q99 = quantile_from_histogram(fw->count_x, fp->nbins_x, fp->dx, N, 0.99);

    for(int k = 0; k < fp->nthresh; k++){
        rho_star = fp->threshold_frac[k] * fp->rho_sat;
        x_left_th[k] = find_left_threshold_front(fw->rho_x, fp->nbins_x, fp->dx, rho_star);
        x_right_th[k] = find_right_threshold_front(fw->rho_x, fp->nbins_x, fp->dx, fp->Lx, rho_star);
    }


    fprintf(ffront,
            "%d %.8f %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %d\n",
            step, t, N,
            x_left_tip, x_right_tip,
            x_q01, x_q99,
            x_left_th[0], x_right_th[0],
            x_left_th[1], x_right_th[1],
            x_left_th[2], x_right_th[2],
            hit_boundary ? 1 : 0);

    if(write_density_profile){
        save_density_profile(frho, step, t, N, fp, fw);
    }
}


/* ==========================================================
   GLUED CENTRAL BULK LOGIC
   ========================================================== */

void initialize_glued_bulk_state(GluedBulkState *iso,
                                double isolation_buffer_factor,
                                double R_inter,
                                double rho_sat,
                                double dx,
                                int nbins_x){
    iso->has_region = false;
    iso->buffer_factor = isolation_buffer_factor;
    iso->buffer = isolation_buffer_factor * R_inter;

    iso->min_delete_width = 5.0 * R_inter;
    iso->density_threshold = 0.8 * rho_sat;

    /* Bulk detection is averaged over a wider region to avoid reacting to
       one noisy bin. */
    iso->n_detect_bins = (int)ceil((5.0 * R_inter) / dx);

    iso->x_delete_L = NAN;
    iso->x_delete_R = NAN;

    if(iso->n_detect_bins < 1){
        iso->n_detect_bins = 1;
    }
    if(iso->n_detect_bins > nbins_x){
        iso->n_detect_bins = nbins_x;
    }
}

bool glued_bulk_particle_is_deleted(GluedBulkState *iso, double x){
    if(!iso->has_region){
        return false;
    }

    if(x > iso->x_delete_L && x < iso->x_delete_R){
        return true;
    }

    return false;
}

double glued_bulk_deleted_width(GluedBulkState *iso){
    if(!iso->has_region){
        return 0.0;
    }

    return iso->x_delete_R - iso->x_delete_L;
}


double glued_bulk_compact_length(GluedBulkState *iso, double Lx){
    double W;

    if(!iso->has_region){
        return Lx;
    }

    W = glued_bulk_deleted_width(iso);

    if(W <= 0.0){
        return Lx;
    }

    return Lx - W;
}


double glued_bulk_x_to_s(double x, GluedBulkState *iso){
    double W;

    if(!iso->has_region){
        return x;
    }

    W = glued_bulk_deleted_width(iso);

    if(x <= iso->x_delete_L){
        return x;
    }

    if(x >= iso->x_delete_R){
        return x - W;
    }

    return iso->x_delete_L;
}


double glued_bulk_s_to_x(double s, GluedBulkState *iso){
    double W;

    if(!iso->has_region){
        return s;
    }

    W = glued_bulk_deleted_width(iso);

    if(s <= iso->x_delete_L){
        return s;
    }

    return s + W;
}


int remove_deleted_bulk_particles(Particle *p, int *N,
                                  GluedBulkState *iso){
    int old_N;
    int new_N;
    int removed;

    old_N = *N;
    new_N = 0;

    if(!iso->has_region){
        return 0;
    }

    /* Remove the central deleted interval. */
    for(int i = 0; i < old_N; i++){
        if(!glued_bulk_particle_is_deleted(iso, p[i].x)){
            p[new_N] = p[i];
            new_N++;
        }
    }

    removed = old_N - new_N;
    *N = new_N;
    return removed;
}

double find_left_bulk_anchor(double *rho_x, int nbins_x, double dx,
                             double rho_threshold, int n_detect_bins){
    int j;
    int k;
    bool found;
    double sum_rho;
    double avg_rho;
    double x_anchor;

    found = false;
    x_anchor = NAN;
    j = 0;

    while(j <= nbins_x - n_detect_bins && !found){
        sum_rho = 0.0;
        for(k = 0; k < n_detect_bins; k++){
            sum_rho += rho_x[j + k];
        }

        avg_rho = sum_rho / (double)n_detect_bins;

        if(avg_rho >= rho_threshold){
            x_anchor = (j + 0.5) * dx;
            found = true;
        }

        j++;
    }

    return x_anchor;
}

double find_right_bulk_anchor(double *rho_x, int nbins_x, double dx, double Lx,
                              double rho_threshold, int n_detect_bins){
    int j;
    int k;
    bool found;
    double sum_rho;
    double avg_rho;
    double x_anchor;

    found = false;
    x_anchor = NAN;
    j = nbins_x - n_detect_bins;

    while(j >= 0 && !found){
        sum_rho = 0.0;
        for(k = 0; k < n_detect_bins; k++){
            sum_rho += rho_x[j + k];
        }

        avg_rho = sum_rho / (double)n_detect_bins;

        if(avg_rho >= rho_threshold){
            x_anchor = (j + n_detect_bins - 0.5) * dx;
            if(x_anchor > Lx){
                x_anchor = Lx;
            }
            found = true;
        }

        j--;
    }

    return x_anchor;
}

bool update_glued_bulk_boundaries(GluedBulkState *iso, FrontParams *fp, FrontWork *fw){
    double x_bulk_L;
    double x_bulk_R;
    double x_delete_L_candidate;
    double x_delete_R_candidate;
    bool valid_candidate;
    bool changed;

    changed = false;

    x_bulk_L = find_left_bulk_anchor(fw->rho_x, fp->nbins_x, fp->dx,
                                     iso->density_threshold,
                                     iso->n_detect_bins);
    x_bulk_R = find_right_bulk_anchor(fw->rho_x, fp->nbins_x, fp->dx, fp->Lx,
                                      iso->density_threshold,
                                      iso->n_detect_bins);

    valid_candidate = false;

    if(!isnan(x_bulk_L) && !isnan(x_bulk_R)){
        x_delete_L_candidate = x_bulk_L + iso->buffer;
        x_delete_R_candidate = x_bulk_R - iso->buffer;

        if(x_delete_L_candidate < 0.0){
            x_delete_L_candidate = 0.0;
        }
        if(x_delete_L_candidate > fp->Lx){
            x_delete_L_candidate = fp->Lx;
        }
        if(x_delete_R_candidate < 0.0){
            x_delete_R_candidate = 0.0;
        }
        if(x_delete_R_candidate > fp->Lx){
            x_delete_R_candidate = fp->Lx;
        }

        if(x_delete_R_candidate - x_delete_L_candidate >= iso->min_delete_width){
            valid_candidate = true;
        }
    }

    if(valid_candidate){
        if(!iso->has_region){
            iso->x_delete_L = x_delete_L_candidate;
            iso->x_delete_R = x_delete_R_candidate;
            iso->has_region = true;
            changed = true;
        }
        else{
            if(x_delete_L_candidate < iso->x_delete_L){
                iso->x_delete_L = x_delete_L_candidate;
                changed = true;
            }
            if(x_delete_R_candidate > iso->x_delete_R){
                iso->x_delete_R = x_delete_R_candidate;
                changed = true;
            }
        }
    }
    return changed;
}

void save_glued_bulk_state(FILE *fbulk, int N, int step, double t,
                           GluedBulkState *iso, double Lx,
                           int N_removed_previous_update, long long N_removed_cumulative){
    int N_active;
    double deleted_width;
    double compact_length;

    if(fbulk == NULL){
        return;
    }

    N_active = N;

    deleted_width = glued_bulk_deleted_width(iso);
    compact_length = glued_bulk_compact_length(iso, Lx);

    fprintf(fbulk, "%d %.8f %d %d %.10g %.10g %.10g %.10g %d %d %lld\n",
            step, t, N,
            iso->has_region ? 1 : 0,
            iso->x_delete_L, iso->x_delete_R,
            deleted_width, compact_length,
            N_active, N_removed_previous_update, N_removed_cumulative);
}



/* ==========================================================
   RECTANGULAR CELL LIST BUILDER
   ========================================================== */

void build_cells_rect_glued(Particle *p, int N, int *head, int *next,
                               int Nx_cells, int Ny_cells,
                               double cell_size, double x_min,
                               GluedBulkState *iso){
    int total;
    int cx, cy, c;
    double sx;

    total = Nx_cells * Ny_cells;

    for(int i = 0; i < total; i++){
        head[i] = -1;
    }

    for(int i = 0; i < N; i++){
        next[i] = -1;

        sx = glued_bulk_x_to_s(p[i].x, iso);
        cx = (int)((sx - x_min) / cell_size);
        cy = (int)(p[i].y / cell_size);

        if(cx < 0){
            cx = 0;
        }
        if(cx >= Nx_cells){
            cx = Nx_cells - 1;
        }
        if(cy < 0){
            cy = 0;
        }
        if(cy >= Ny_cells){
            cy = Ny_cells - 1;
        }

        c = cx + Nx_cells * cy;
        next[i] = head[c];
        head[c] = i;
    }
}

void compute_all_neighbors_rect_glued(Particle *p, int N, int *head, int *next,
                                         int *neighbors,
                                         int Nx_cells, int Ny_cells,
                                         double cell_size,
                                         double x_min, double Lx_domain, double Ly,
                                         double R_inter, bool periodic_x,
                                         GluedBulkState *iso){

    for(int i = 0; i < N; i++){
        neighbors[i] = count_neighbors_rect(p, i, head, next,
                                            Nx_cells, Ny_cells,
                                            cell_size,
                                            x_min, Lx_domain, Ly,
                                            R_inter, periodic_x, iso);
    }
}


/* ==========================================================
   HEUN INTEGRATION (ACTIVE BROWNIAN MOTION)
   ========================================================== */
/* In this code:
   Dr     = translational diffusion
   Dtheta = angular diffusion
*/

void move_particles_tube_glued(Particle *p, int N,
                                  double v0, double Dr, double Dtheta, double dt,
                                  double x_min, double x_max, double Ly,
                                  bool periodic_x, bool *hit_boundary,
                                  GluedBulkState *iso){
    double theta;
    double nx, ny, nt;
    double sqrt_2Dr_dt, sqrt_2Dtheta_dt;
    double theta_pred;
    double vx0, vy0, vx1, vy1;
    double Lx_domain;
    double L_eff;
    double dx_motion;
    double s_coord;

    sqrt_2Dr_dt = sqrt(2.0 * Dr * dt);
    sqrt_2Dtheta_dt = sqrt(2.0 * Dtheta * dt);
    Lx_domain = x_max - x_min;

    for(int i = 0; i < N; i++){
        theta = p[i].theta;

        nx = rand_normal();
        ny = rand_normal();
        nt = rand_normal();

        theta_pred = theta + sqrt_2Dtheta_dt * nt;

        vx0 = cos(theta);
        vy0 = sin(theta);

        vx1 = cos(theta_pred);
        vy1 = sin(theta_pred);

        dx_motion = 0.5 * v0 * (vx0 + vx1) * dt + sqrt_2Dr_dt * nx;

        if(!periodic_x && iso->has_region){
            L_eff = glued_bulk_compact_length(iso, Lx_domain);
            s_coord = glued_bulk_x_to_s(p[i].x, iso);
            s_coord = s_coord + dx_motion;

            if(s_coord <= x_min){
                s_coord = x_min;
                *hit_boundary = true;
            }
            if(s_coord >= L_eff){
                s_coord = L_eff;
                *hit_boundary = true;
            }

            p[i].x = glued_bulk_s_to_x(s_coord, iso);
        }
        else{
            p[i].x = p[i].x + dx_motion;

            if(periodic_x){
                p[i].x = wrap_interval(p[i].x, x_min, Lx_domain);
            }
            else{
                if(p[i].x <= x_min){
                    p[i].x = x_min;
                    *hit_boundary = true;
                }
                if(p[i].x >= x_max){
                    p[i].x = x_max;
                    *hit_boundary = true;
                }
            }
        }

        p[i].y = p[i].y + 0.5 * v0 * (vy0 + vy1) * dt + sqrt_2Dr_dt * ny;
        p[i].theta = p[i].theta + sqrt_2Dtheta_dt * nt;

        p[i].y = wrap(p[i].y, Ly);

        while(p[i].theta >= 2.0 * M_PI){
            p[i].theta -= 2.0 * M_PI;
        }
        while(p[i].theta < 0.0){
            p[i].theta += 2.0 * M_PI;
        }

        /* In compact coordinates, particles cross the artificial boundary instead
           of entering the deleted interval. */
    }
}


/* ==========================================================
   BIRTH / DEATH LOGIC
   ========================================================== */

int birth_death_glued(Particle *p, Particle *p_new, int *N, int *neighbors,
                         double dt, double p0, double q0, int Ns,
                         int *births_this_step, int *deaths_this_step){
    int old_N;
    int new_N;
    double birth;
    double u;

    old_N = *N;
    new_N = 0;
    *births_this_step = 0;
    *deaths_this_step = 0;

    for(int i = 0; i < old_N; i++){
        birth = p0 * (1.0 - (double)neighbors[i] / Ns);
        if(birth < 0.0){
            birth = 0.0;
        }

        u = rand_uniform();

        if(u < birth * dt){
            if(new_N + 2 > MAX_PARTICLES){
                return 1;
            }

            p_new[new_N] = p[i];
            new_N++;

            p_new[new_N] = p[i];
            new_N++;

            (*births_this_step)++;
        }
        else if(u < (birth + q0) * dt){
            (*deaths_this_step)++;
        }
        else{
            if(new_N + 1 > MAX_PARTICLES){
                return 1;
            }
            p_new[new_N] = p[i];
            new_N++;
        }
    }

    for(int i = 0; i < new_N; i++){
        p[i] = p_new[i];
    }

    *N = new_N;

    return 0;
}

/* ==========================================================
   EVENT COUNTER LOGIC
   ========================================================== */

void initialize_event_stats(EventStats *events){
    events->max_total_events = -1;
    events->births_at_max = 0;
    events->deaths_at_max = 0;
    events->step_at_max = -1;
    events->N_before_at_max = -1;
    events->N_after_at_max = -1;
    events->time_at_max = NAN;
}

void update_event_stats(EventStats *events, int step, double t,
                        int N_before, int N_after,
                        int births_this_step, int deaths_this_step){
    int total_events;

    total_events = births_this_step + deaths_this_step;

    if(total_events > events->max_total_events){
        events->max_total_events = total_events;
        events->births_at_max = births_this_step;
        events->deaths_at_max = deaths_this_step;
        events->step_at_max = step;
        events->N_before_at_max = N_before;
        events->N_after_at_max = N_after;
        events->time_at_max = t;
    }
}

void save_event_summary(FILE *fevents, Params *par, EventStats *events,
                        double R_inter, double Ly, double Dr, double v0,
                        double Dtheta, double dt, double T,
                        GluedBulkState *iso){
    fprintf(fevents, "# Columns are one summary row per run.\n");
    fprintf(fevents, "# Deleted central-bulk particles are not counted as death events.\n");
    fprintf(fevents, "# run_id seed max_event_step max_event_time N_before_at_max N_after_at_max max_total_events births_at_max deaths_at_max isolation_buffer_factor isolation_buffer isolation_min_delete_width R_inter Ly Dr v0 Dtheta dt T\n");
    fprintf(fevents, "%d %llu %d %.10g %d %d %d %d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
            par->run_id, (unsigned long long)par->seed,
            events->step_at_max, events->time_at_max,
            events->N_before_at_max, events->N_after_at_max,
            events->max_total_events,
            events->births_at_max, events->deaths_at_max,
            iso->buffer_factor, iso->buffer, iso->min_delete_width,
            R_inter, Ly, Dr, v0, Dtheta, dt, T);
}

/* ==========================================================
   MAIN
   ========================================================== */

int main(int argc, char *argv[]){
    Particle *p;
    Particle *p_new;
    int *head;
    int *next;
    int *neighbors;

    FrontParams fp;
    FrontWork fw;
    GluedBulkState iso;
    EventStats event_stats;

    Params par;

    int N;
    int N0;
    int Nx_cells;
    int Ny_cells;
    int Nx_warm_cells;
    int head_size;
    int steps, step;
    int warmup_steps;
    int status;
    int Ns;
    int save_per_step;
    int front_per_step;
    int rho_profile_every_front;
    int front_measurement_index;
    int N_active;
    int N_removed_previous_update;
    long long N_removed_cumulative;
    int N_before_step;
    int births_this_step;
    int deaths_this_step;
    int progress_stride;
    int selected_run_id;
    bool hit_boundary;
    bool write_density_profile;
    bool snapshot_saved_this_step;

    double p0;
    double q0;
    double R_inter;
    double rho0;
    double v0;
    double Dr;
    double Dtheta;
    double dt;
    double T;
    double Lx;
    double Ly;
    double x_init_min;
    double x_init_max;
    double x_init_width;
    double warmup_T;
    double rho_sat;
    double cell_size;
    double t;
    double elapsed_seconds;
    double init_area;
    double rho_sat_input;
    double rho_warmup_estimate;
    double isolation_buffer_factor;
    char rho_sat_source[64];

    uint64_t seed;

    char *param_filename;
    char base_name[256];
    char output_name[512];
    char front_output_name[512];
    char rho_output_name[512];
    char glued_bulk_output_name[512];
    char event_output_name[512];
    char warmup_output_name[512];

    clock_t start_clock;
    clock_t current_clock;

    FILE *f;
    FILE *ffront;
    FILE *frho;
    FILE *fbulk;
    FILE *fevents;
    FILE *fwarmup;

    f = NULL;
    ffront = NULL;
    frho = NULL;
    fbulk = NULL;
    fevents = NULL;
    fwarmup = NULL;

    if(argc != 3){
        printf("Usage: %s params.txt run_id\n", argv[0]);
        return 1;
    }

    param_filename = argv[1];
    selected_run_id = atoi(argv[2]);

    if(selected_run_id < 0){
        printf("Error: run_id must be >= 0\n");
        return 1;
    }

    status = read_params(param_filename, selected_run_id, &par);
    if(status == 1){
        printf("[ERROR] [run_id=%d] could not open parameter file %s\n", selected_run_id, param_filename);
        fflush(stdout);
        return 1;
    }
    if(status == 2){
        printf("Error: invalid format in selected run of parameter file %s\n", param_filename);
        printf("Use only the [run] key = value format.\n");
        return 1;
    }
    if(status == 3){
        printf("Error: run_id %d not found in parameter file %s\n", selected_run_id, param_filename);
        return 1;
    }

    p0 = par.p0;
    q0 = par.q0;
    Ns = par.Ns;
    R_inter = par.R_inter;
    rho0 = par.rho0;
    save_per_step = par.save_per_step;
    front_per_step = par.front_per_step;
    rho_profile_every_front = par.rho_profile_every_front;
    isolation_buffer_factor = par.isolation_buffer_factor;
    if(isnan(isolation_buffer_factor)){
        isolation_buffer_factor = 20.0;
    }
    v0 = par.v0;
    Dr = par.Dr;
    Dtheta = par.Dtheta;
    dt = par.dt;
    T = par.T;
    Lx = par.Lx;
    Ly = par.Ly;
    x_init_min = par.x_init_min;
    x_init_max = par.x_init_max;
    warmup_T = par.warmup_T;
    rho_sat = par.rho_sat;
    rho_sat_input = rho_sat;
    rho_warmup_estimate = NAN;
    strcpy(rho_sat_source, "input");
    seed = par.seed;

    x_init_width = x_init_max - x_init_min;
    init_area = x_init_width * Ly;
    N0 = (int)(rho0 * init_area + 0.5);

    if(Lx <= 0.0 || Ly <= 0.0){
        printf("Error: Lx and Ly must be positive.\n");
        return 1;
    }
    if(x_init_min < 0.0 || x_init_max > Lx || x_init_width <= 0.0){
        printf("Error: invalid initial interval. Need 0 <= x_init_min < x_init_max <= Lx.\n");
        return 1;
    }

    if(!validate_pbc_displacement_dimensions(R_inter, x_init_min, x_init_max, Ly)){
        return 1;
    }
    if(par.nbins_x < 2){
        printf("Error: nbins_x must be >= 2.\n");
        return 1;
    }
    if(front_per_step < 1 || save_per_step < 1){
        printf("Error: front_per_step and save_per_step must be >= 1.\n");
        return 1;
    }
    if(rho_profile_every_front < 0){
        printf("Error: rho_profile_every_front must be >= 0. Use 0 to disable density-profile output.\n");
        return 1;
    }
    if(isolation_buffer_factor <= 0.0){
        printf("Error: isolation_buffer_factor must be positive.\n");
        return 1;
    }
    if(N0 < 1){
        printf("Error: initial number of particles is zero. Increase rho0 or seed area.\n");
        return 1;
    }
    if(N0 > MAX_PARTICLES){
        printf("Error: initial number of particles (%d) is larger than MAX_PARTICLES (%d)\n", N0, MAX_PARTICLES);
        return 1;
    }

    if(v0 * dt > 0.1 * R_inter){
        printf("Error: dt is too large: v0 * dt is not much smaller than R_inter.\n");
        printf("v0 * dt = %.8f, 0.1 * R_inter = %.8f\n", v0 * dt, 0.1 * R_inter);
        return 1;
    }

    if(sqrt(2.0 * Dr * dt) > 0.1 * R_inter){
        printf("Error: dt is too large: translational diffusion step is not much smaller than R_inter.\n");
        printf("sqrt(2 * Dr * dt) = %.8f, 0.1 * R_inter = %.8f\n",
               sqrt(2.0 * Dr * dt), 0.1 * R_inter);
        return 1;
    }

    if(sqrt(2.0 * Dtheta * dt) > 0.1){
        printf("Error: dt is too large: angular diffusion step is not much smaller than 1 radian.\n");
        printf("sqrt(2 * Dtheta * dt) = %.8f, %.8f\n",
               sqrt(2.0 * Dtheta * dt), 0.1);
        return 1;
    }

    if(p0 * dt > 0.01 || q0 * dt > 0.01){
        printf("Error: dt is too large for a good fixed-time-step birth/death approximation.\n");
        printf("p0 * dt = %.8f, q0 * dt = %.8f, maximum allowed = 0.01\n",
               p0 * dt, q0 * dt);
        return 1;
    }

    cell_size = R_inter;
    Nx_cells = (int)ceil(Lx / cell_size);
    Nx_warm_cells = (int)ceil(x_init_width / cell_size);
    Ny_cells = (int)ceil(Ly / cell_size);

    if(Nx_cells < 1){
        Nx_cells = 1;
    }
    if(Nx_warm_cells < 1){
        Nx_warm_cells = 1;
    }
    if(Ny_cells < 1){
        Ny_cells = 1;
    }

    head_size = Nx_cells * Ny_cells;
    if(Nx_warm_cells * Ny_cells > head_size){
        head_size = Nx_warm_cells * Ny_cells;
    }

    p = malloc(MAX_PARTICLES * sizeof(Particle));
    p_new = malloc(MAX_PARTICLES * sizeof(Particle));
    head = malloc(head_size * sizeof(int));
    next = malloc(MAX_PARTICLES * sizeof(int));
    neighbors = malloc(MAX_PARTICLES * sizeof(int));
    fw.count_x = malloc(par.nbins_x * sizeof(int));
    fw.rho_x = malloc(par.nbins_x * sizeof(double));

    if(p == NULL || p_new == NULL || head == NULL || next == NULL || neighbors == NULL ||
       fw.count_x == NULL || fw.rho_x == NULL){
        printf("Error allocating memory.\n");
        free(p);
        free(p_new);
        free(head);
        free(next);
        free(neighbors);
        free(fw.count_x);
        free(fw.rho_x);
        return 1;
    }

    fp.Lx = Lx;
    fp.Ly = Ly;
    fp.x_center = 0.5 * (x_init_min + x_init_max);
    fp.rho_sat = rho_sat;
    fp.nbins_x = par.nbins_x;
    fp.dx = Lx / (double)par.nbins_x;
    fp.nthresh = FRONT_NTHRESH;
    for(int k = 0; k < FRONT_NTHRESH; k++){
        fp.threshold_frac[k] = par.threshold_frac[k];
    }

    seed_xoshiro256pp(seed);
    initialize_front_seed(p, N0, x_init_min, x_init_max, Ly);

    N = N0;

    get_base_name_without_extension(param_filename, base_name, sizeof(base_name));
    sprintf(output_name, "snapshot_%s_run_%03d.dat", base_name, par.run_id);
    sprintf(front_output_name, "front_%s_run_%03d.dat", base_name, par.run_id);
    sprintf(rho_output_name, "rho_%s_run_%03d.dat", base_name, par.run_id);
    sprintf(glued_bulk_output_name, "glued_bulk_%s_run_%03d.dat", base_name, par.run_id);
    sprintf(event_output_name, "events_%s_run_%03d.dat", base_name, par.run_id);
    sprintf(warmup_output_name, "warmup_%s_run_%03d.dat", base_name, par.run_id);

    steps = (int)(T / dt);
    warmup_steps = (int)(warmup_T / dt);

    progress_stride = steps / 10;
    if(progress_stride < 1){
        progress_stride = 1;
    }

    start_clock = clock();

    printf("[INFO] [run_id=%d] Parameter file: %s\n", par.run_id, param_filename);
    printf("[INFO] [run_id=%d] Trajectory output: %s\n", par.run_id, output_name);
    printf("[INFO] [run_id=%d] Front output: %s\n", par.run_id, front_output_name);
    if(rho_profile_every_front > 0){
        printf("[INFO] [run_id=%d] Density-profile output: %s, every %d front measurements\n",
               par.run_id, rho_output_name, rho_profile_every_front);
    }
    else{
        printf("[INFO] [run_id=%d] Density-profile output: disabled\n", par.run_id);
    }
    printf("[INFO] [run_id=%d] Glued-bulk output: %s\n", par.run_id, glued_bulk_output_name);
    printf("[INFO] [run_id=%d] Event summary output: %s\n", par.run_id, event_output_name);
    printf("[INFO] [run_id=%d] Isolation buffer factor: %.10g, glued-bulk seam, density threshold fixed to 0.8 * rho_sat, detection window fixed to 5 * R_inter, minimum deleted width fixed to 5 * R_inter\n",
           par.run_id, isolation_buffer_factor);
    printf("[INFO] [run_id=%d] Initial particles before warmup: %d\n", par.run_id, N);
    fflush(stdout);

    /*
       During warmup there is no glued/deleted region.
    */
    iso.has_region = false;

    if(warmup_steps > 0){
        printf("[INFO] [run_id=%d] Warmup steps: %d\n", par.run_id, warmup_steps);
        printf("[INFO] [run_id=%d] Warmup population output: %s\n",
               par.run_id, warmup_output_name);
        fflush(stdout);

        fwarmup = fopen(warmup_output_name, "w");
        if(fwarmup == NULL){
            printf("Error opening warmup output file.\n");
            free(p);
            free(p_new);
            free(head);
            free(next);
            free(neighbors);
            free(fw.count_x);
            free(fw.rho_x);
            return 1;
        }

        fprintf(fwarmup, "# source_param_file %s\n", param_filename);
        fprintf(fwarmup, "# selected_run_id %d\n", selected_run_id);
        fprintf(fwarmup, "# run_id %d seed %llu\n", par.run_id, (unsigned long long)par.seed);
        fprintf(fwarmup, "# warmup_T %.8f dt %.8f x_init_min %.8f x_init_max %.8f Ly %.8f seed_area %.10g\n",
                warmup_T, dt, x_init_min, x_init_max, Ly, init_area);
        fprintf(fwarmup, "# p0 %.8f q0 %.8f Ns %d R %.8f rho0 %.8f v0 %.8f Dr %.8f Dtheta %.8f\n",
                p0, q0, Ns, R_inter, rho0, v0, Dr, Dtheta);
        fprintf(fwarmup, "# step time N rho_seed\n");
        fprintf(
            fwarmup,
            "%d %.8f %d %.10g\n",
            0,
            0.0,
            N,
            (double)N / init_area
        );
    }

    
    int n_warmup_outputs = (int)(warmup_T + 0.5);

    if(n_warmup_outputs < 1){
        n_warmup_outputs = 1;
    }

    if(n_warmup_outputs > warmup_steps){
        n_warmup_outputs = warmup_steps;
    }

    int warmup_outputs_written = 0;
    int rho_warmup_count = 0;

    rho_warmup_estimate = 0.0;
    for(int wstep = 0; wstep < warmup_steps && N > 0; wstep++){

        hit_boundary = false;

        move_particles_tube_glued(p, N, v0, Dr, Dtheta, dt,
                                     x_init_min, x_init_max, Ly,
                                     true, &hit_boundary, &iso);

        build_cells_rect_glued(p, N, head, next,
                                  Nx_warm_cells, Ny_cells,
                                  cell_size, x_init_min, &iso);

        compute_all_neighbors_rect_glued(p, N, head, next, neighbors,
                                            Nx_warm_cells, Ny_cells,
                                            cell_size,
                                            x_init_min, x_init_width, Ly,
                                            R_inter, true, &iso);

        status = birth_death_glued(p, p_new, &N, neighbors, dt, p0, q0, Ns,
                                      &births_this_step, &deaths_this_step);
        if(status != 0){
            printf("[ERROR] [run_id=%d] number of particles exceeded MAX_PARTICLES during warmup = %d\n",
                   par.run_id, MAX_PARTICLES);
            if(f != NULL){
                fclose(f);
            }
            if(ffront != NULL){
                fclose(ffront);
            }
            if(fwarmup != NULL){
                fclose(fwarmup);
            }
            free(p);
            free(p_new);
            free(head);
            free(next);
            free(neighbors);
            free(fw.count_x);
            free(fw.rho_x);
            return 1;
        }

        /* We will check the last 50 time steps to obtain the mean rho_sat*/

        if((double)(wstep+1)*dt >= warmup_T - 50.0){
            rho_warmup_estimate += (double)N;
            rho_warmup_count++;
        }

        if(fwarmup != NULL){
            int step_done = wstep + 1;

            int target_outputs_written = (step_done * n_warmup_outputs) / warmup_steps;

            while(warmup_outputs_written < target_outputs_written){
                warmup_outputs_written++;

                fprintf(
                    fwarmup,
                    "%d %.8f %d %.10g\n",
                    step_done,
                    step_done * dt,
                    N,
                    (double)N / init_area
                );
            }
        }        
    }

    if(fwarmup != NULL){
        fclose(fwarmup);
        fwarmup = NULL;
    }

    printf("[INFO] [run_id=%d] Particles after warmup: %d\n", par.run_id, N);
    fflush(stdout);

    if(rho_warmup_count > 0){
        rho_warmup_estimate = rho_warmup_estimate / ((double)rho_warmup_count * init_area);
    }
    else{
        rho_warmup_estimate = (double)N / init_area;
    }
    
    if(rho_sat <= 0.0){
        if(N <= 0){
            printf("Error: cannot estimate rho_sat from warmup because N after warmup is zero.\n");
            free(p);
            free(p_new);
            free(head);
            free(next);
            free(neighbors);
            free(fw.count_x);
            free(fw.rho_x);
            return 1;
        }

        rho_sat = rho_warmup_estimate;
        strcpy(rho_sat_source, "warmup");
    }

    if(rho_sat <= 0.0){
        printf("Error: rho_sat_used must be positive. Use rho_sat > 0 or rho_sat = -1 for warmup estimation.\n");
        free(p);
        free(p_new);
        free(head);
        free(next);
        free(neighbors);
        free(fw.count_x);
        free(fw.rho_x);
        return 1;
    }

    fp.rho_sat = rho_sat;

    initialize_glued_bulk_state(&iso, isolation_buffer_factor, R_inter, rho_sat,
                               fp.dx, fp.nbins_x);
    initialize_event_stats(&event_stats);

    printf("[INFO] [run_id=%d] rho_warmup_estimate = %.10g\n",
           par.run_id, rho_warmup_estimate);
    printf("[INFO] [run_id=%d] rho_sat_used = %.10g, source = %s\n",
           par.run_id, rho_sat, rho_sat_source);
    fflush(stdout);

    f = fopen(output_name, "w");
    ffront = fopen(front_output_name, "w");
    fbulk = fopen(glued_bulk_output_name, "w");
    fevents = fopen(event_output_name, "w");
    if(rho_profile_every_front > 0){
        frho = fopen(rho_output_name, "w");
    }

    if(f == NULL || ffront == NULL || fbulk == NULL || fevents == NULL || (rho_profile_every_front > 0 && frho == NULL)){
        printf("Error opening output files.\n");
        if(f != NULL){
            fclose(f);
        }
        if(ffront != NULL){
            fclose(ffront);
        }
        if(fbulk != NULL){
            fclose(fbulk);
        }
        if(fevents != NULL){
            fclose(fevents);
        }
        if(frho != NULL){
            fclose(frho);
        }
        free(p);
        free(p_new);
        free(head);
        free(next);
        free(neighbors);
        free(fw.count_x);
        free(fw.rho_x);
        return 1;
    }

    fprintf(f, "# source_param_file %s\n", param_filename);
    fprintf(f, "# selected_run_id %d\n", selected_run_id);
    fprintf(f, "# run_id %d seed %llu\n", par.run_id, (unsigned long long)par.seed);
    fprintf(f, "# Lx %.8f Ly %.8f dt %.8f v0 %.8f Dr %.8f Dtheta %.8f R %.8f p0 %.8f q0 %.8f Ns %d rho0 %.8f save_per_step %d front_per_step %d rho_profile_every_front %d\n",
            Lx, Ly, dt, v0, Dr, Dtheta, R_inter, p0, q0, Ns, rho0, save_per_step, front_per_step, rho_profile_every_front);
    fprintf(f, "# x_init_min %.8f x_init_max %.8f warmup_T %.8f seed_area %.8f N_after_warmup %d nbins_x %d threshold_frac1 %.8f threshold_frac2 %.8f threshold_frac3 %.8f\n",
            x_init_min, x_init_max, warmup_T, init_area, N, par.nbins_x, par.threshold_frac[0], par.threshold_frac[1], par.threshold_frac[2]);
    fprintf(f, "# rho_sat_input %.10g rho_warmup_estimate %.10g rho_sat_used %.10g rho_sat_source %s\n",
            rho_sat_input, rho_warmup_estimate, rho_sat, rho_sat_source);
    fprintf(f, "# glued_central_bulk 1 isolation_buffer_factor %.10g isolation_buffer %.10g isolation_min_delete_width %.10g isolation_density_threshold %.10g isolation_n_detect_bins %d\n",
            iso.buffer_factor, iso.buffer, iso.min_delete_width, iso.density_threshold, iso.n_detect_bins);

    fprintf(ffront, "# source_param_file %s\n", param_filename);
    fprintf(ffront, "# selected_run_id %d\n", selected_run_id);
    fprintf(ffront, "# run_id %d seed %llu\n", par.run_id, (unsigned long long)par.seed);
    fprintf(ffront, "# Lx %.8f Ly %.8f nbins_x %d threshold_frac1 %.8f threshold_frac2 %.8f threshold_frac3 %.8f x_center %.8f front_per_step %d rho_profile_every_front %d\n",
            Lx, Ly, par.nbins_x, par.threshold_frac[0], par.threshold_frac[1], par.threshold_frac[2], fp.x_center, front_per_step, rho_profile_every_front);
    fprintf(ffront, "# rho_sat_input %.10g rho_warmup_estimate %.10g rho_sat_used %.10g rho_sat_source %s seed_area %.10g N_after_warmup %d\n",
            rho_sat_input, rho_warmup_estimate, rho_sat, rho_sat_source, init_area, N);
    fprintf(ffront, "# glued_central_bulk 1 isolation_buffer_factor %.10g isolation_buffer %.10g isolation_min_delete_width %.10g isolation_density_threshold %.10g isolation_n_detect_bins %d\n",
            iso.buffer_factor, iso.buffer, iso.min_delete_width, iso.density_threshold, iso.n_detect_bins);
    fprintf(ffront, "# step time N x_left_tip x_right_tip x_q01 x_q99 x_left_th_1 x_right_th_1 x_left_th_2 x_right_th_2 x_left_th_3 x_right_th_3 hit_boundary\n");

    fprintf(fbulk, "# source_param_file %s\n", param_filename);
    fprintf(fbulk, "# selected_run_id %d\n", selected_run_id);
    fprintf(fbulk, "# run_id %d seed %llu\n", par.run_id, (unsigned long long)par.seed);
    fprintf(fbulk, "# glued_central_bulk 1 isolation_buffer_factor %.10g isolation_buffer %.10g isolation_min_delete_width %.10g isolation_density_threshold %.10g isolation_n_detect_bins %d\n",
            iso.buffer_factor, iso.buffer, iso.min_delete_width, iso.density_threshold, iso.n_detect_bins);
    fprintf(fbulk, "# step time N has_region x_delete_L x_delete_R deleted_width compact_length N_active N_removed_previous_update N_removed_cumulative\n");

    if(frho != NULL){
        fprintf(frho, "# source_param_file %s\n", param_filename);
        fprintf(frho, "# selected_run_id %d\n", selected_run_id);
        fprintf(frho, "# run_id %d seed %llu\n", par.run_id, (unsigned long long)par.seed);
        fprintf(frho, "# Lx %.8f Ly %.8f nbins_x %d dx %.10g front_per_step %d rho_profile_every_front %d\n",
                Lx, Ly, par.nbins_x, fp.dx, front_per_step, rho_profile_every_front);
        fprintf(frho, "# threshold_frac1 %.8f threshold_frac2 %.8f threshold_frac3 %.8f x_center %.8f\n",
                par.threshold_frac[0], par.threshold_frac[1], par.threshold_frac[2], fp.x_center);
        fprintf(frho, "# rho_sat_input %.10g rho_warmup_estimate %.10g rho_sat_used %.10g rho_sat_source %s seed_area %.10g N_after_warmup %d\n",
                rho_sat_input, rho_warmup_estimate, rho_sat, rho_sat_source, init_area, N);
        fprintf(frho, "# bin_centers");
        for(int j = 0; j < fp.nbins_x; j++){
            fprintf(frho, " %.10g", (j + 0.5) * fp.dx);
        }
        fprintf(frho, "\n");
        fprintf(frho, "# step time N");
        for(int j = 0; j < fp.nbins_x; j++){
            fprintf(frho, " rho_%d", j);
        }
        fprintf(frho, "\n");
    }

    step = 0;
    t = 0.0;
    hit_boundary = false;

    front_measurement_index = 0;
    N_removed_previous_update = 0;
    N_removed_cumulative = 0;

    save_particles(f, p, N, step, t);
    write_density_profile = should_save_density_profile(rho_profile_every_front,
                                                        front_measurement_index,
                                                        false);
    measure_front(ffront, frho, p, N, step, t, &fp, &fw, hit_boundary,
                  write_density_profile);
    save_glued_bulk_state(fbulk, N, step, t, &iso, Lx,
                          N_removed_previous_update, N_removed_cumulative);

    /* Update the compactified deleted interval only after saving the current state. */
    if(update_glued_bulk_boundaries(&iso, &fp, &fw)){
        N_removed_previous_update = remove_deleted_bulk_particles(p, &N, &iso);
        N_removed_cumulative += N_removed_previous_update;
    }
    else{
        N_removed_previous_update = 0;
    }
    front_measurement_index++;

    while(step < steps && N > 0 && !hit_boundary){

        /* 1. Move only active front particles in the tube: periodic y, open x */
        move_particles_tube_glued(p, N, v0, Dr, Dtheta, dt,
                                     0.0, Lx, Ly,
                                     false, &hit_boundary, &iso);

        step++;
        t = step * dt;

        if(hit_boundary){
            write_density_profile = should_save_density_profile(rho_profile_every_front,
                                                                front_measurement_index,
                                                                true);
            measure_front(ffront, frho, p, N, step, t, &fp, &fw, hit_boundary,
                          write_density_profile);
            save_glued_bulk_state(fbulk, N, step, t, &iso, Lx,
                                  N_removed_previous_update, N_removed_cumulative);
            save_particles(f, p, N, step, t);

            /* Update the compactified deleted interval only after saving the current state. */
            if(update_glued_bulk_boundaries(&iso, &fp, &fw)){
                N_removed_previous_update = remove_deleted_bulk_particles(p, &N, &iso);
                N_removed_cumulative += N_removed_previous_update;
            }
            else{
                N_removed_previous_update = 0;
            }
            front_measurement_index++;
            printf("[STOP] [run_id=%d] A particle touched x = 0 or x = Lx at step %d, t = %.8f\n",
                   par.run_id, step, t);
            fflush(stdout);
        }
        else{

        /* 2. Build cell list */
        build_cells_rect_glued(p, N, head, next,
                                  Nx_cells, Ny_cells,
                                  cell_size, 0.0, &iso);

        /* 3. Compute all neighbor counts */
        compute_all_neighbors_rect_glued(p, N, head, next, neighbors,
                                            Nx_cells, Ny_cells,
                                            cell_size,
                                            0.0, Lx, Ly,
                                            R_inter, false, &iso);

        /* 4. Apply birth/death to the compactified active system */
        N_before_step = N;
        status = birth_death_glued(p, p_new, &N, neighbors, dt, p0, q0, Ns,
                                      &births_this_step, &deaths_this_step);
        update_event_stats(&event_stats, step, t, N_before_step, N,
                           births_this_step, deaths_this_step);
        if(status != 0){
            printf("[ERROR] [run_id=%d] number of particles exceeded MAX_PARTICLES = %d\n",
                   par.run_id, MAX_PARTICLES);
            fflush(stdout);
            fclose(f);
            fclose(ffront);
            fclose(fbulk);
            fclose(fevents);
            if(frho != NULL){
                fclose(frho);
            }
            free(p);
            free(p_new);
            free(head);
            free(next);
            free(neighbors);
            free(fw.count_x);
            free(fw.rho_x);
            return 1;
        }

        snapshot_saved_this_step = false;

        /* 5. Measure fronts, then update the compactified deleted interval */
        if(step % front_per_step == 0 || N == 0){
            write_density_profile = should_save_density_profile(rho_profile_every_front,
                                                                front_measurement_index,
                                                                N == 0);

            measure_front(ffront, frho, p, N, step, t, &fp, &fw, hit_boundary,
                        write_density_profile);
            save_glued_bulk_state(fbulk, N, step, t, &iso, Lx,
                                  N_removed_previous_update, N_removed_cumulative);

            if(step % save_per_step == 0 || N == 0){
                save_particles(f, p, N, step, t);
                snapshot_saved_this_step = true;
            }

            /* Update the compactified deleted interval only after saving the current state. */
            if(update_glued_bulk_boundaries(&iso, &fp, &fw)){
                N_removed_previous_update = remove_deleted_bulk_particles(p, &N, &iso);
                N_removed_cumulative += N_removed_previous_update;
            }
            else{
                N_removed_previous_update = 0;
            }

            front_measurement_index++;
        }

        /* 6. Save occasional full particle snapshots, if not already saved above */
        if((step % save_per_step == 0 || N == 0) && !snapshot_saved_this_step){
            save_particles(f, p, N, step, t);
        }

        /* Progress */
        if(step % progress_stride == 0 || N == 0){
            current_clock = clock();
            elapsed_seconds = (double)(current_clock - start_clock) / CLOCKS_PER_SEC;
            N_active = N;
            printf("[PROGRESS] [run_id=%d] %3d%%, N=%d, active=%d, removed_previous=%d, removed_cumulative=%lld, elapsed=%.2f s\n",
                   par.run_id, (step * 100) / steps, N, N_active,
                   N_removed_previous_update, N_removed_cumulative, elapsed_seconds);
            fflush(stdout);
        }
        }
    }

    save_event_summary(fevents, &par, &event_stats, R_inter, Ly, Dr, v0,
                       Dtheta, dt, T, &iso);

    fclose(f);
    fclose(ffront);
    fclose(fbulk);
    fclose(fevents);
    if(frho != NULL){
        fclose(frho);
    }

    free(p);
    free(p_new);
    free(head);
    free(next);
    free(neighbors);
    free(fw.count_x);
    free(fw.rho_x);

    current_clock = clock();
    elapsed_seconds = (double)(current_clock - start_clock) / CLOCKS_PER_SEC;

    printf("[EVENTS] [run_id=%d] max_total_events_per_step=%d births=%d deaths=%d step=%d time=%.8f N_before=%d N_after=%d\n",
           par.run_id, event_stats.max_total_events,
           event_stats.births_at_max, event_stats.deaths_at_max,
           event_stats.step_at_max, event_stats.time_at_max,
           event_stats.N_before_at_max, event_stats.N_after_at_max);
    printf("[DONE] [run_id=%d] Final particle number: %d, total elapsed time: %.2f s\n",
           par.run_id, N, elapsed_seconds);
    fflush(stdout);

    return 0;
}
