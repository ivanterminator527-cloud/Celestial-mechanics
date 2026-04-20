#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <functional>
#include <cmath>
#include <tuple>
#include <numeric>
#include <algorithm>

double G = 6.674e-11;
double Msun = 1.989e30, kpc = 206265.0*149598000000.0*1e3;
double c=0.2672*kpc, Mb = 1.03e10 *Msun, Md=6.51e10*Msun, a=4.4*kpc, b=0.3084*kpc, Mh=2.90e11*Msun, d=7.7*kpc;

float dist(const std::vector<float>& point_to, const std::vector<float>& point_from)
{
    return std::sqrt(std::pow(point_to[0]-point_from[0], 2)+std::pow(point_to[1]-point_from[1], 2)+std::pow(point_to[2]-point_from[2], 2));
}

std::vector<int>  ind_for_matrix(int idx)
{
    int i = static_cast<int>((sqrt(1+8*idx)-1)/2);
    int j = idx - static_cast<int> (i*(i+1))/2;
    std::vector<int> vec = {i, j};
    return vec;
}

float Z_model(const std::vector<float>& coord, std::vector<float>& z)
{
    std::vector<int> ind (2, 0);
    float Z=0.0;
    for (size_t i=0; i<z.size(); i++)
    {
        ind = ind_for_matrix(i);
        Z += z[i]*std::pow(coord[0], ind[0]-ind[1])*std::pow(coord[1], ind[1]);
    }
    return Z;
}

std::vector<float> find_per2(const std::vector<float>& star, std::vector<float>& z, float d, float& r, const std::vector<float>&ref)
{
    std::vector<float> distances(4, 0); int i; float b = 1/2;
    distances[0] = dist({ref[0]+d, ref[1], Z_model({ref[0]+d, ref[1]},z)}, star);
    distances[1] = dist({ref[0], ref[1]+d, Z_model({ref[0], ref[1]+d},z)}, star);
    distances[2] = dist({ref[0]-d, ref[1], Z_model({ref[0]-d, ref[1]},z)}, star);
    distances[3] = dist({ref[0], ref[1]-d, Z_model({ref[0], ref[1]-d},z)}, star);
    i = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    if (d<r*2.0f and *std::max_element(distances.begin(), distances.end())- distances[i]<r){
        std::vector<float> out = {distances[i], ref[0], ref[1]};
        return out;
    }
    if (i==0){
        return find_per2(star, z, d*b, r, {ref[0]+d*b, ref[1], Z_model({ref[0]+d*b, ref[1]},z)});
    } else if (i==1){
        return find_per2(star, z, d*b, r, {ref[0], ref[1]+d*b, Z_model({ref[0], ref[1]+d*b},z)});
    } else if (i==2){
        return find_per2(star, z, d*b, r, {ref[0]-d*b, ref[1], Z_model({ref[0]-d*b, ref[1]},z)});
    } else {
        return find_per2(star, z, d*b, r, {ref[0], ref[1]-d*b, Z_model({ref[0], ref[1]-d*b},z)});
    }

}

static float U_balge(float& R, float& Z){
    return -G*Mb/std::sqrt(R*R + Z*Z + c*c);
}

static float U_disk(const std::vector<float>& coord, std::vector<float>& z, float d, float r){
    std::vector<float> solve = find_per2(coord, z, d, r, coord);
    float Z = coord[2] - Z_model({solve[1], solve[2]}, z);
    double R = std::sqrt(coord[0]*coord[0]+coord[1]*coord[1]);
    return -G*Md*std::sqrt(R*R + std::pow(a + std::sqrt(Z*Z + b*b), 2.0));
}

static float U_halo(float& R, float& Z){
    return -G*Mh/std::sqrt(R*R + Z*Z) * (1.0 + std::sqrt(R*R + Z*Z)/d);
}

std::vector<float> energy (std::vector<std::vector<float>>& stars, 
                            std::vector<float>& z, float& d, float& r,
                            std::vector<std::vector<float>>& velocity){
    std::vector<float> ene (stars.size(), 0.0f);
    float R;

    for (size_t i=0; i<stars.size(); i++){
        R = std::sqrt(stars[i][0]*stars[i][0] + stars[i][1]*stars[i][1]);
        ene[i] = U_balge(R, stars[i][2])+U_disk(stars[i], z, d, r)+U_halo(R, stars[i][2])+
        (velocity[i][0]*velocity[i][0] + velocity[i][1]*velocity[i][1] + velocity[i][2]*velocity[i][2])/2;
    }
    return ene;
}

float LKH_potencial(std::vector<float>& solve, std::vector<std::vector<float>>& stars, 
                        float& d, float& r,
                        std::vector<std::vector<float>>& velocity){
    std::vector<float> z (solve.size()-1, 0.0f);
    float sigma = solve[solve.size()-1];
    for (size_t i=0; i<z.size(); i++){
        z[i] = solve[i];
    }

    std::vector<float> ene = energy(stars, z, d, r, velocity);
    float summ=0;
    for (float el: ene){
        summ+= (el-sigma)*(el-sigma);
    }

    return summ/2/sigma/sigma
    +stars.size()*std::log(sigma)
    +stars.size()/2*std::log(2*std::acos(0.0));
    
}

PYBIND11_MODULE(potencials, m) {
    m.def("find_per2", &find_per2, "finding the nearest nbor");
    m.def("find_dist", &dist, "Return the distancion between point to and point ");
    m.def("zeta", &Z_model);
    m.def("LKH_potencial", &LKH_potencial, "Return Likelyhood-func for distrib of potencials");
}