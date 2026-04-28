#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <functional>
#include <cmath>
#include <tuple>
#include <numeric>
#include <algorithm>

double G = 6.674e-11;
double Msun = 1.989e30, kpc = 3.08567758128e19;
double c=0.2672, Mb = 1.03e10 *Msun, Md=6.51e10*Msun, a=4.4, b=0.3084, Mh=2.90e11*Msun, d=7.7;

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
    float Z=0.0f;
    for (size_t i=0; i<z.size(); i++)
    {
        ind = ind_for_matrix(i);
        Z += z[i]*std::pow(coord[0], ind[0]-ind[1])*std::pow(coord[1], ind[1]);
    }
    return Z;
}

std::vector<float> find_per2(const std::vector<float>& star, std::vector<float>& z, const float& step, float& r, const std::vector<float>&ref)
{
    std::vector<float> distances(4, 0); int i; float b = 0.5f;
    distances[0] = dist({ref[0]+step, ref[1], Z_model({ref[0]+step, ref[1]},z)}, star);
    distances[1] = dist({ref[0], ref[1]+step, Z_model({ref[0], ref[1]+step},z)}, star);
    distances[2] = dist({ref[0]-step, ref[1], Z_model({ref[0]-step, ref[1]},z)}, star);
    distances[3] = dist({ref[0], ref[1]-step, Z_model({ref[0], ref[1]-step},z)}, star);
    i = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    if (step<r*2.0f and *std::max_element(distances.begin(), distances.end())- distances[i]<r){
        std::vector<float> out = {distances[i], ref[0], ref[1]};
        return out;
    }
    if (i==0){
        return find_per2(star, z, step*b, r, {ref[0]+step*b, ref[1], Z_model({ref[0]+step*b, ref[1]},z)});
    } else if (i==1){
        return find_per2(star, z, step*b, r, {ref[0], ref[1]+step*b, Z_model({ref[0], ref[1]+step*b},z)});
    } else if (i==2){
        return find_per2(star, z, step*b, r, {ref[0]-step*b, ref[1], Z_model({ref[0]-step*b, ref[1]},z)});
    } else {
        return find_per2(star, z, step*b, r, {ref[0], ref[1]-step*b, Z_model({ref[0], ref[1]-step*b},z)});
    }

}

static  float U_balge(float& R, float& Z){
    return -G*Mb/std::sqrt(R*R + Z*Z + c*c)/kpc;
}

static float U_disk(const std::vector<float>& coord, std::vector<float>& z){
    float Z = coord[2] - Z_model(coord, z);
    double R = std::sqrt(coord[0]*coord[0]+coord[1]*coord[1]);
    return -G*Md/std::sqrt(R*R + std::pow(a + std::sqrt(Z*Z + b*b), 2.0))/kpc;
}

static float U_halo(float& R, float& Z){
    return -G*Mh/std::sqrt(R*R + Z*Z) * (1.0 + std::sqrt(R*R + Z*Z)/d)/kpc;
}

std::vector<float> energy (std::vector<std::vector<float>>& stars, 
                            std::vector<float>& z,
                            std::vector<std::vector<float>>& velocity){
    std::vector<float> ene (stars.size(), 0.0f);
    float R;

    for (size_t i=0; i<stars.size(); i++){
        R = std::sqrt(stars[i][0]*stars[i][0] + stars[i][1]*stars[i][1]);
        ene[i] = U_balge(R, stars[i][2])+U_disk(stars[i], z)/2+U_halo(R, stars[i][2])+ //U_disk/2 because disk that structure of stars
        (velocity[i][0]*velocity[i][0] + velocity[i][1]*velocity[i][1] + velocity[i][2]*velocity[i][2])/2;
    }
    return ene;
}
/*
float LKH_potencial(std::vector<float>& z, std::vector<std::vector<float>>& stars, 
                        float& d, float& r, float& sigma,
                        std::vector<std::vector<float>>& velocity){

    std::vector<float> ene = energy(stars, z, d, r, velocity);
    float summ=0.0f;
    for (float el: ene){
        summ += (el-sigma)*(el-sigma);
    }

    return summ;
    
}
*/
PYBIND11_MODULE(potencials, m) {
    m.def("energy", &energy, "Show full energy for that particle");
    m.def("find_per2", &find_per2, "finding the nearest nbor");
    m.def("find_dist", &dist, "Return the distancion between point to and point");
    m.def("zeta", &Z_model);
    //m.def("LKH_potencial", &LKH_potencial, "Return Likelyhood-func for distrib of potencials");
}