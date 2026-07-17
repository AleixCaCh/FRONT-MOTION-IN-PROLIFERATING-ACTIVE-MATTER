#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ==========================================================
   CONSTANTS AND MACROS
   ========================================================== */

#define MAX_PARTICLES 50000
#define BIRTH_ORIENTATION_LABEL "random"

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
    // SplitMix64 initialization
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
    static int hasSpare = 0;
    static double spare;
    double u, v, r;

    if(hasSpare){
        hasSpare = 0;
        return spare;
    }

    do{
        u = rand_uniform()*2.0 - 1.0;
        v = rand_uniform()*2.0 - 1.0;
        r = u*u + v*v;
    } while(r >= 1.0 || r == 0.0);

    r = sqrt(-2.0*log(r)/r);
    spare = v*r;
    hasSpare = 1;

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
    double v0;
    double Dr;
    double Dtheta;
    double dt;
    double T;
    double L;
} Params;

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

double pbc_displacement(double dx, double L){
    while(dx > 0.5 * L){
        dx -= L;
    }
    while(dx < -0.5 * L){
        dx += L;
    }
    return dx;
}

/* ==========================================================
   PARAMETER FILE READING
   ========================================================== */

void trim_in_place(char *text){
    int start = 0;
    int end = (int)strlen(text) - 1;

    while(text[start] == ' ' || text[start] == '\t' || text[start] == '\n' || text[start] == '\r'){
        start++;
    }
    while(end >= start &&
          (text[end] == ' ' || text[end] == '\t' || text[end] == '\n' || text[end] == '\r')){
        end--;
    }

    if(end < start){
        text[0] = '\0';
        return;
    }

    if(start > 0){
        memmove(text, text + start, (size_t)(end - start + 1));
    }
    text[end - start + 1] = '\0';
}

void remove_inline_comment(char *line){
    char *comment = strchr(line, '#');
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
    par->v0 = NAN;
    par->Dr = NAN;
    par->Dtheta = NAN;
    par->dt = NAN;
    par->T = NAN;
    par->L = NAN;
}

int assign_param_value(Params *par, const char *key, const char *value){
    if(strcmp(key, "run_id") == 0){
        par->run_id = atoi(value);
    }
    else if(strcmp(key, "seed") == 0){
        par->seed = (uint64_t)strtoull(value, NULL, 10);
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
    else if(strcmp(key, "L") == 0){
        par->L = atof(value);
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
    if(isnan(par->v0)){ strcat(missing, " v0"); }
    if(isnan(par->Dr)){ strcat(missing, " Dr"); }
    if(isnan(par->Dtheta)){ strcat(missing, " Dtheta"); }
    if(isnan(par->dt)){ strcat(missing, " dt"); }
    if(isnan(par->T)){ strcat(missing, " T"); }
    if(isnan(par->L)){ strcat(missing, " L"); }

    if((int)strlen(missing) >= max_len){
        missing[max_len - 1] = '\0';
    }

    return missing[0] == '\0' ? 0 : 1;
}

int read_params(const char *filename, int target_run_id, Params *par){
    FILE *f;
    char line[2048];
    char clean_line[2048];
    char key[256];
    char value[256];
    char missing[1024];
    char *equal;
    Params candidate;
    int in_block = 0;
    int unknown_count = 0;

    f = fopen(filename, "r");
    if(f == NULL){
        return 1;
    }

    init_params_missing(&candidate);

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
                fclose(f);
                if(check_required_params(par, missing, sizeof(missing)) != 0){
                    printf("Error: missing required parameters for run_id %d:%s\n", target_run_id, missing);
                    return 2;
                }
                if(unknown_count > 0){
                    printf("Warning: ignored %d unknown parameter key(s) for run_id %d.\n",
                           unknown_count, target_run_id);
                }
                return 0;
            }

            init_params_missing(&candidate);
            unknown_count = 0;
            in_block = 1;
            continue;
        }

        if(!in_block){
            fclose(f);
            printf("Error: parameter file must use [run] blocks.\n");
            return 2;
        }

        equal = strchr(clean_line, '=');
        if(equal == NULL){
            fclose(f);
            printf("Error: expected 'key = value', got: %s\n", clean_line);
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
            unknown_count++;
        }
    }

    fclose(f);

    if(in_block && candidate.run_id == target_run_id){
        *par = candidate;
        if(check_required_params(par, missing, sizeof(missing)) != 0){
            printf("Error: missing required parameters for run_id %d:%s\n", target_run_id, missing);
            return 2;
        }
        if(unknown_count > 0){
            printf("Warning: ignored %d unknown parameter key(s) for run_id %d.\n",
                   unknown_count, target_run_id);
        }
        return 0;
    }

    return 3;
}

/* ==========================================================
   OUTPUT FILE NAME
   ========================================================== */

void get_base_name_without_extension(const char *input_name, char *base_name, int max_len){
    int len;
    int start;
    int end;
    int i,j;
    int found;

    len = (int)strlen(input_name);

    end = len;
    // Find the start of the base name (after last '/' or '\')
    i = len - 1;
    start = 0;
    found = 0;
    while (i >= 0 && !found) {
        if (input_name[i] == '/' || input_name[i] == '\\') {
            start = i + 1;
            found = 1;
        }
        i--;
    }

    // Find the end of the base name (before last '.')
    i = len - 1;
    found = 0;
    while (i >= start && !found) {
        if (input_name[i] == '.') {
            end = i;
            found = 1;
        }
        i--;
    }

    // Copy characters to base_name
    j = 0;
    for (i = start; i < end && j < max_len - 1; i++) {
        base_name[j++] = input_name[i];
    }
    base_name[j] = '\0';
}

/* ==========================================================
   INITIALIZATION
   ========================================================== */

void initialize(Particle *p, int N0, double L){

    for(int i = 0; i < N0; i++){
        p[i].x = L * rand_uniform();
        p[i].y = L * rand_uniform();
        p[i].theta = 2.0 * M_PI * rand_uniform();
    }
}

/* ==========================================================
   CELL LIST BUILDER
   ========================================================== */

void build_cells(Particle *p, int N, int *head, int *next, int Nc, double cell_size){
    int total;
    int cx, cy, c;

    total = Nc * Nc;

    for(int i = 0; i < total; i++){
        head[i] = -1;
    }

    for(int i = 0; i < N; i++){
        cx = (int)(p[i].x / cell_size);
        cy = (int)(p[i].y / cell_size);


        /* In case of floating point numbers, these should not happen */
        if(cx >= Nc){
            cx = Nc - 1;
        }
        if(cy >= Nc){
            cy = Nc - 1;
        }

        /* Add the particle to the corresponding head of the cell */
        c = cx + Nc * cy;
        next[i] = head[c];
        head[c] = i;
    }
}

/* ==========================================================
   NEIGHBOR COUNT
   ========================================================== */

int count_neighbors(Particle *p, int i, int *head, int *next, int Nc, double cell_size, double L, double R_inter){
    int Ni;
    int cx, cy;
    int nx, ny;
    int cell;
    int j;
    double dist_x, dist_y;
    double R2;

    Ni = 0;
    R2 = R_inter * R_inter;

    cx = (int)(p[i].x / cell_size);
    cy = (int)(p[i].y / cell_size);

    /* Look at the 3 x 3 squares around the selected particle */
    for(int dx = -1; dx <= 1; dx++){
        for(int dy = -1; dy <= 1; dy++){
            nx = (cx + dx + Nc) % Nc;
            ny = (cy + dy + Nc) % Nc;
            cell = nx + Nc * ny;

            j = head[cell];
            while(j != -1){
                if(j != i){
                    dist_x = pbc_displacement(p[j].x - p[i].x, L);
                    dist_y = pbc_displacement(p[j].y - p[i].y, L);

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

void compute_all_neighbors(Particle *p, int N, int *head, int *next, int *neighbors, int Nc, double cell_size, double L, double R_inter){

    for(int i = 0; i < N; i++){
        neighbors[i] = count_neighbors(p, i, head, next, Nc, cell_size, L, R_inter);
    }
}

/* ==========================================================
   HEUN INTEGRATION (ACTIVE BROWNIAN MOTION)
   ========================================================== */
/* Following the notation of the paper DOI: https://doi.org/10.1103/v3gm-vhry:
   Dr     = translational diffusion
   Dtheta = angular diffusion
*/

void move_particles(Particle *p, int N, double v0, double Dr, double Dtheta, double dt, double L){
    double theta;
    double nx, ny, nt;
    double sqrt_2Dr_dt, sqrt_2Dtheta_dt;
    double theta_pred;
    double vx0, vy0, vx1, vy1;

    sqrt_2Dr_dt = sqrt(2.0 * Dr * dt);
    sqrt_2Dtheta_dt = sqrt(2.0 * Dtheta * dt);


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

        p[i].x = p[i].x + 0.5 * v0 * (vx0 + vx1) * dt + sqrt_2Dr_dt * nx;
        p[i].y = p[i].y + 0.5 * v0 * (vy0 + vy1) * dt + sqrt_2Dr_dt * ny;
        p[i].theta = p[i].theta + sqrt_2Dtheta_dt * nt;

        p[i].x = wrap(p[i].x, L);
        p[i].y = wrap(p[i].y, L);

        /* To mantain angles between 0 and 2*pi */
        while(p[i].theta >= 2.0 * M_PI){
            p[i].theta -= 2.0 * M_PI;
        }
        while(p[i].theta < 0.0){
            p[i].theta += 2.0 * M_PI;
        }
    }
}

/* ==========================================================
   BIRTH / DEATH LOGIC
   ========================================================== */

int birth_death(Particle *p, Particle *p_new, int *N, int *neighbors, double dt, double p0, double q0, int Ns){
    int old_N;
    int new_N;
    double birth;
    double u;

    old_N = *N;
    new_N = 0;

    for(int i = 0; i < old_N; i++){
        birth = p0 * (1.0 - (double)neighbors[i] / Ns);
        if(birth < 0.0){
            birth = 0.0;
        }

        u = rand_uniform();

        if(u < birth * dt){ // Birth
            if(new_N + 2 > MAX_PARTICLES){
                return 1;
            }
            // Parent
            p_new[new_N] = p[i];
            new_N++;

            // Child
            p_new[new_N].x = p[i].x;
            p_new[new_N].y = p[i].y;
            p_new[new_N].theta = 2.0 * M_PI * rand_uniform();  // random orientation
            new_N++;
        }
        else if(u < (birth + q0) * dt){ // Death
        }
        else{// Nothing
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
   SAVE TRAJECTORY
   ========================================================== */

void save_particles(FILE *f, Particle *p, int N, int step, double t){

    fprintf(f, "FRAME %d %.8f %d\n", step, t, N);

    for(int i = 0; i < N; i++){
        fprintf(f, "%d %.8f %.8f %.8f\n", i, p[i].x, p[i].y, p[i].theta);
    }

    fprintf(f, "\n");
}

/* Observables computations */

double compute_S_order(Particle *p, int N){
    double sx = 0.0;
    double sy = 0.0;

    if(N <= 0){
        return 0.0;
    }

    for(int i = 0; i < N; i++){
        sx += cos(p[i].theta);
        sy += sin(p[i].theta);
    }

    return sqrt(sx*sx + sy*sy) / (double)N;
}

void compute_gr(Particle *p, int N, double L, double dr, int Nr, double *g){

    int k;
    double dx, dy, r;
    double C_pair = 0.0;

    for(int k = 0; k < Nr; k++){
        g[k] = 0.0;
    }

    for(int i = 0; i < N; i++){
        for(int j = i + 1; j < N; j++){
            dx = pbc_displacement(p[j].x - p[i].x, L);
            dy = pbc_displacement(p[j].y - p[i].y, L);
            r = sqrt(dx*dx + dy*dy);

            if(r < 0.5 * L){
                k = (int)(r / dr);
                if(k >= 0 && k < Nr){
                    g[k] += 1.0;
                }
                C_pair += 1.0;
            }
        }
    }

    if(C_pair > 0.0){
        for(int k = 0; k < Nr; k++){
            g[k] = g[k]*0.25 * L * L / (C_pair * (2.0*k + 1.0) * dr * dr);
        }
    }
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
    double *g_inst;


    Params par;

    int N;
    int N0;
    int Nc;
    int steps, step;
    int status;
    int Ns;
    int Nr;
    int save_per_step;
    int S_stride;
    int progress_stride;
    int selected_run_id;

    double p0;
    double q0;
    double R_inter;
    double rho0;
    double v0;
    double Dr;
    double Dtheta;
    double dt;
    double T;
    double L;
    double cell_size;
    double t;
    double elapsed_seconds;
    double dr;
    double r_mid;

    uint64_t seed;

    char *param_filename;
    char base_name[256];
    char output_name[512];
    char g_output_name[1024];
    char S_output_name[1024];

    clock_t start_clock;
    clock_t current_clock;

    FILE *f;
    FILE *fg;
    FILE *fS;

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
        printf("Error: invalid [run] block in parameter file %s\n", param_filename);
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
    v0 = par.v0;
    Dr = par.Dr;
    Dtheta = par.Dtheta;
    dt = par.dt;
    T = par.T;
    L = par.L;
    seed = par.seed;
    N0 = (int)(rho0 * L * L);

    S_stride = 100;
    dr = 0.01;
    Nr = (int)(0.5 * L / dr);

    if(Nr < 1){
        printf("Error: invalid Nr for g(r). Check L and dr.\n");
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
    Nc = (int)(L / cell_size); /* Number of cells */

    if(Nc < 1){
        printf("Error: invalid number of cells. Check L and R_inter.\n");
        return 1;
    }

    /* Vector definitions */
    p = malloc(MAX_PARTICLES * sizeof(Particle));
    p_new = malloc(MAX_PARTICLES * sizeof(Particle));
    head = malloc(Nc * Nc * sizeof(int));
    next = malloc(MAX_PARTICLES * sizeof(int));
    neighbors = malloc(MAX_PARTICLES * sizeof(int));
    g_inst = malloc(Nr * sizeof(double));

    if(p == NULL || p_new == NULL || head == NULL || next == NULL || neighbors == NULL || g_inst == NULL){
        printf("Error allocating memory.\n");
        free(p);
        free(p_new);
        free(head);
        free(next);
        free(neighbors);
        free(g_inst);
        return 1;
    }

    seed_xoshiro256pp(seed);
    initialize(p, N0, L);

    N = N0;

    /* Output files */
    get_base_name_without_extension(param_filename, base_name, sizeof(base_name));
    sprintf(output_name, "%s_run_%03d.dat", base_name, par.run_id);

    f = fopen(output_name, "w");
    if(f == NULL){
        printf("Error opening output file %s\n", output_name);
        free(p);
        free(p_new);
        free(head);
        free(next);
        free(neighbors);
        free(g_inst);
        return 1;
    }

    fprintf(f, "# source_param_file %s\n", param_filename);
    fprintf(f, "# selected_run_id %d\n", selected_run_id);
    fprintf(f, "# run_id %d seed %llu\n", par.run_id, (unsigned long long)par.seed);
    fprintf(f, "# birth_orientation %s\n", BIRTH_ORIENTATION_LABEL);
    fprintf(f, "# L %.8f dt %.8f v0 %.8f Dr %.8f Dtheta %.8f R %.8f p0 %.8f q0 %.8f Ns %d rho0 %.8f save_per_step %d\n",
            L, dt, v0, Dr, Dtheta, R_inter, p0, q0, Ns, rho0, save_per_step);



    /* Output g(r) and S(t)*/
    sprintf(g_output_name, "g_%s", output_name);
    sprintf(S_output_name, "S_%s", output_name);

    fg = fopen(g_output_name, "w");
    fS = fopen(S_output_name, "w");

    if(fg == NULL || fS == NULL){
        printf("Error opening g or S output files\n");
        fclose(f);
        if(fg != NULL){
            fclose(fg);
        }
        if(fS != NULL){
            fclose(fS);
        }
        free(p);
        free(p_new);
        free(head);
        free(next);
        free(neighbors);
        free(g_inst);
        return 1;
    }

    fprintf(fg, "# g(r) file\n");
    fprintf(fg, "# birth_orientation %s\n", BIRTH_ORIENTATION_LABEL);
    fprintf(fg, "# dr %.8f Nr %d L %.8f\n", dr, Nr, L);

    fprintf(fS, "# S(t) file\n");
    fprintf(fS, "# birth_orientation %s\n", BIRTH_ORIENTATION_LABEL);
    fprintf(fS, "# columns: step time N S\n");



    /* Start simulation */
    steps = (int)(T / dt);

    progress_stride = steps / 10;
    if(progress_stride < 1){
        progress_stride = 1;
    }

    
    start_clock = clock();

    printf("[INFO] [run_id=%d] Parameter file: %s\n", par.run_id, param_filename);
    printf("[INFO] [run_id=%d] Output file: %s\n", par.run_id, output_name);
    printf("[INFO] [run_id=%d] Initial particles: %d\n", par.run_id, N);
    fflush(stdout);

    /* Save initial condition at t = 0 */
    fprintf(fS, "%d %.8f %d %.8f\n", 0, 0.0, N, compute_S_order(p, N));

    save_particles(f, p, N, 0, 0.0);

    compute_gr(p, N, L, dr, Nr, g_inst);

    fprintf(fg, "FRAME %d %.8f %d\n", 0, 0.0, N);
    fprintf(fg, "# r g\n");

    for(int k = 0; k < Nr; k++){
        double r_mid = (k + 0.5) * dr;
        fprintf(fg, "%.8f %.8f\n", r_mid, g_inst[k]);
    }

    fprintf(fg, "\n");


    step = 0;
    t = 0.0;

    while(step < steps && N > 0){

        /* 1. Move particles */
        move_particles(p, N, v0, Dr, Dtheta, dt, L);

        /* 2. Build cell list */
        build_cells(p, N, head, next, Nc, cell_size);

        /* 3. Compute all neighbor counts */
        compute_all_neighbors(p, N, head, next, neighbors, Nc, cell_size, L, R_inter);

        /* 4. Apply birth/death */
        status = birth_death(p, p_new, &N, neighbors, dt, p0, q0, Ns);
        if(status != 0){
            printf("[ERROR] [run_id=%d] number of particles exceeded MAX_PARTICLES = %d\n",
                   par.run_id, MAX_PARTICLES);
            fflush(stdout);
            fclose(f);
            fclose(fg);
            fclose(fS);
            free(p);
            free(p_new);
            free(head);
            free(next);
            free(neighbors);
            free(g_inst);
            return 1;
        }

        /* Advance step/time AFTER one full update */
        step++;
        t = step * dt;

        /* 5. Save data */
        /* If the absorbing state is reached, report it and stop */
        if(step % S_stride == 0 || N == 0){
            fprintf(fS, "%d %.8f %d %.8f\n", step, t, N, compute_S_order(p, N));
        }

        if(step % save_per_step == 0 || N == 0){
            save_particles(f, p, N, step, t);

            compute_gr(p, N, L, dr, Nr, g_inst);

            fprintf(fg, "FRAME %d %.8f %d\n", step, t, N);
            fprintf(fg, "# r g\n");

            for(int k = 0; k < Nr; k++){
                r_mid = (k + 0.5) * dr;
                fprintf(fg, "%.8f %.8f\n", r_mid, g_inst[k]);
            }

            fprintf(fg, "\n");
        }

        /* Progress */
        if(step % progress_stride == 0 || N == 0){
            current_clock = clock();
            elapsed_seconds = (double)(current_clock - start_clock) / CLOCKS_PER_SEC;
            printf("[PROGRESS] [run_id=%d] %3d%%, N=%d, elapsed=%.2f s\n",
            par.run_id, (step * 100) / steps, N, elapsed_seconds);
            fflush(stdout);
        }      
    }


    fclose(f);
    fclose(fg);
    fclose(fS);

    free(p);
    free(p_new);
    free(head);
    free(next);
    free(neighbors);
    free(g_inst);

    current_clock = clock();
    elapsed_seconds = (double)(current_clock - start_clock) / CLOCKS_PER_SEC;

    printf("[DONE] [run_id=%d] Final particle number: %d, total elapsed time: %.2f s\n",
       par.run_id, N, elapsed_seconds);
    fflush(stdout);

    return 0;
}