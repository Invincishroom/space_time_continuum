#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<random>
#include<json/json.h>

int main(int argc, char** argv){ 
    //Usage: ./trial_generator <output_folder> <time_window> <time_step> <arclength_end> <arclength_step> <noise_stddev>
    std::ofstream fout(argv[1]+std::string("/vicon0.csv"));
    if(!fout.is_open()){
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }
    fout << "sensor type:, aurora" << std::endl;
    fout << "timestamp (s), arclength (m), data" << std::endl;

}