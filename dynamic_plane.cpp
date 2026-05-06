#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <functional>
#include <cmath>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <fstream>

double G = 6.674e-11;
double Msun = 1.989e30, kpc = 206265.0*149598000000.0*1e3;
double c=0.2672*kpc, Mb = 1.03e10 *Msun, Md=6.51e10*Msun, a=4.4*kpc, b=0.3084*kpc, Mh=2.90e11*Msun, d=7.7*kpc;



float dist(const std::vector<float>& point_to, const std::vector<float>& point_from)
{
    return std::sqrt(std::pow(point_to[0]-point_from[0], 2)+std::pow(point_to[1]-point_from[1], 2)+std::pow(point_to[2]-point_from[2], 2));
}


static float abs_vector(std::vector<float> r){
    float summ=0;
    for (float el:r){
        summ+=el*el;
    }
    return summ;
};

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
    std::vector<float> distances(4, 0); int i; float b = 0.6f;
    distances[0] = dist({ref[0]+d, ref[1], Z_model({ref[0]+d, ref[1]},z)}, star);
    distances[1] = dist({ref[0], ref[1]+d, Z_model({ref[0], ref[1]+d},z)}, star);
    distances[2] = dist({ref[0]-d, ref[1], Z_model({ref[0]-d, ref[1]},z)}, star);
    distances[3] = dist({ref[0], ref[1]-d, Z_model({ref[0], ref[1]-d},z)}, star);
    i = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    if (d<r and *std::max_element(distances.begin(), distances.end())- distances[i]<r){
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

static double force_buldge_R(float& R, float& z){
    return -R/std::pow(R*R + z*z + c*c, 3.0/2)*G*Mb;
};
static double force_disk_R (const std::vector<float>& coord, std::vector<float>& z){
    float Z = coord[2] - Z_model({coord[0], coord[1]}, z);
    double R = std::sqrt(coord[0]*coord[0]+coord[1]*coord[1]);
    return -G*Md*R/std::pow(R*R+std::pow(a+std::sqrt(Z*Z+b*b), 2), 3.0/2);
};
static double force_halo_R (float& R, float& z){
    double r = std::sqrt(R*R + z*z);
    return  G*Mh/r*(1/d/(1+r/d) - std::log(1+r/d)/r)*R/r;
};
static double force_buldge_z(float& R, float& z){
    return -z/std::pow(R*R + z*z + c*c, 3.0/2)*G*Mb;
};
static double force_disk_z (const std::vector<float>& coord, std::vector<float>& z){
    std::vector<float> solve = find_per2(coord, z, d, r, coord);
    float Z = coord[2] - Z_model({coord[0], coord[1]}, z);
    double R = std::sqrt(coord[0]*coord[0]+coord[1]*coord[1]);
    return -G*Md*Z/std::pow(R*R+std::pow(a+std::sqrt(Z*Z+b*b), 2), 3.0/2)/std::sqrt(Z*Z+b*b)*(a+std::sqrt(Z*Z+b*b));
};
static double force_halo_z (float& R, float& z){
    double r = std::sqrt(R*R + z*z);
    return  G*Mh/r*(1/d/(1+r/d) - std::log(1+r/d)/r)*z/r;
};


//Tmax, z, X, Y, vz, vt, vr, dt, frequency, z_ij, d, r
std::vector<std::vector<double>> RK4_cpp(double Tmax, std::vector<double> Z, 
    std::vector<double> X, std::vector<double> Y, std::vector<double> vz, 
    std::vector<double> vx, std::vector<double> vy, double dt,
     std::vector<float> z_ij)
{

    auto force_R = [](const std::vector<float>& coord, std::vector<float>& z){
        float Z = coord[2];
        double R = std::sqrt(coord[0]*coord[0]+coord[1]*coord[1]);
        return force_buldge_R(R, Z)+force_disk_R(coord, z, d, r)+force_halo_R(R, Z); //
    };
    auto force_z = [](const std::vector<float>& coord, std::vector<float>& z){
        float Z = coord[2];
        double R = std::sqrt(coord[0]*coord[0]+coord[1]*coord[1]);
        return force_buldge_z(R, Z)+force_disk_z(coord, z, d, r)+force_halo_z(R, Z);
    };
    auto func_abs = [](std::vector<float> vec){ 
        float summ = 0;
        for (float i: vec){
            summ += i*i;
        }
        return std::sqrt(summ);
    };
    int n = Tmax/dt; int ind;
    std::vector<double> r_temp(3, 0);
    std::vector<double> v_temp(3, 0);
    std::vector<std::vector<double>> r(3, std::vector<double>(X.size(), 0));
    std::vector<std::vector<double>> v(3, std::vector<double>(X.size(), 0));
    double kx1, kx2, kx3, kx4, ky1, ky2, ky3, ky4, kz1, kz2, kz3, kz4, Rabs, a_R;
    std::vector<float> R(2, 0); float z;
    std::vector<std::vector<double>> a(4, std::vector<double>(3, 0));
    for (size_t k=0; k<X.size(); k++)
    {
        r_temp[0] = X[k]; r_temp[1] = Y[k]; r_temp[2] = Z[k];
        v_temp[0] = vx[k]; v_temp[1] = vy[k]; v_temp[2]=vz[k];
        for (int i=0; i<n-1; i++)
        {
            kx1 = v_temp[0];
            ky1 = v_temp[1];
            kz1 = v_temp[2];
            R[0] = r_temp[0]; R[1] = r_temp[1]; z = r_temp[2];
            Rabs = func_abs(R);
            a_R = force_R({R[0], R[1], z}, z_ij); a[0][2] = force_z({static_cast<float>(R[0]),
                                                    static_cast<float>(R[1]), 
                                                    static_cast<float>(z)}, z_ij);
            a[0][0] = a_R*r_temp[0]/Rabs; a[0][1] = a_R*r_temp[1]/Rabs;
            
            R[0] = r_temp[0]+kx1*dt/2; R[1] = r_temp[1]+ky1*dt/2;
            Rabs = func_abs(R);
            z = r_temp[2]+kz1*dt/2;
            a_R = force_R({R[0], R[1], z}, z_ij); a[1][2] = force_z({static_cast<float>(R[0]),
                                                    static_cast<float>(R[1]), 
                                                    static_cast<float>(z)}, z_ij);
            a[1][0]=a_R*R[0]/Rabs;a[1][1]=a_R*R[1]/Rabs;
            kx2 = v_temp[0] + a[1][0] *dt/2;
            ky2 = v_temp[1] + a[1][1] *dt/2;
            kz2 = v_temp[2] + a[1][2] *dt/2;

            R[0] = r_temp[0]+kx2*dt/2; R[1] = r_temp[1]+ky2*dt/2;
            Rabs = func_abs(R);
            z = r_temp[2]+kz2*dt/2;
            a_R = force_R({R[0], R[1], z}, z_ij); a[2][2] = force_z({static_cast<float>(R[0]),
                                                    static_cast<float>(R[1]), 
                                                    static_cast<float>(z)}, z_ij);
            a[2][0] = a_R*R[0]/Rabs; a[2][1] = a_R*R[1]/Rabs;
            kx3 = v_temp[0] + a[2][0]*dt/2;
            ky3 = v_temp[1] + a[2][1] *dt/2;
            kz3 = v_temp[2] + a[2][2]*dt/2;
            
            R[0] = r_temp[0]+kx3*dt; R[1] = r_temp[1]+ky3*dt;
            Rabs = func_abs(R);
            z = r_temp[2]+kz3*dt;
            a_R = force_R({R[0], R[1], z}, z_ij); a[3][2] = force_z({static_cast<float>(R[0]),
                                                    static_cast<float>(R[1]), 
                                                    static_cast<float>(z)}, z_ij);
            a[3][0] = a_R*R[0]/Rabs; a[3][1] = a_R*R[1]/Rabs;
            kx4 = v_temp[0] + a[3][0] *dt;
            ky4 = v_temp[1] + a[3][1] *dt;
            kz4 = v_temp[2] + a[3][2]*dt;

            r_temp[0] += dt/6*(kx1+2*kx2+2*kx3+kx4);
            r_temp[1] += dt/6*(ky1+2*ky2+2*ky3+ky4);
            r_temp[2] += dt/6*(kz1+2*kz2+2*kz3+kz4);
            v_temp[0] += dt/6*(a[0][0]+2*a[1][0]+2*a[2][0]+a[3][0]);
            v_temp[1] += dt/6*(a[0][1]+2*a[1][1]+2*a[2][1]+a[3][1]);
            v_temp[2] += dt/6*(a[0][2]+2*a[1][2]+2*a[2][2]+a[3][2]);
        }
        r[k] = r_temp;
        v[k] = v_temp;
    }
    std::ofstream myfile;
    myfile.open("history.csv");
    myfile << Tmax << "\n";
    for (int j = 0; j<3; i++){
        for (size_t i = 0; i < X.size(); i++){
            myfile << r[i][j] << ", "
        }
        myfile << "\n";
    }
    for (int i = 0; i<z_ij.size(); i++){
        myfile << z_ij[i] << ",";
    }
    myfile << "\n";
    myfile.close();

    std::vector<std::vector<std::vector<double>>> result(2):
    result[0] = r;
    result[1] = v;
    return result;

}

/*
PYBIND11_MODULE(simple_projection, m) {
    m.def("RK4_cpp", &RK4_cpp, "Tmax, z, X, Y, vz, vt, vr, dt, frequency, z_ij, d, r");
    m.def("find_per2", &find_per2, "finding the nearest nbor");
    m.def("LKH", &LKH_main, "Return the logarifm likelihood function");
    m.def("find_dist", &dist, "Return the distancion between point to and point ");
    m.def("zeta", &Z_model);
    m.def("ind_for_matrix_cpp", &ind_for_matrix);
}
*/