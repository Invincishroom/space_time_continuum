#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<random>
#include<json/json.h>

int main(int argc, char** argv){ 
    //Usage: ./trial_generator <output_folder> <time_window> <time_step> <arclength_end> <arclength_step> <noise_stddev> [measurement_per_st] [measurement_stddev]
    std::ofstream fout(argv[1]+std::string("/trial_data.csv"));
    if(!fout.is_open()){
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }
    fout << "sensor type:, aurora" << std::endl;
    fout << "timestamp (s), arclength (m), data" << std::endl;
    double time_window = std::stoi(argv[2]);
    std::cout<< "time_window:," << time_window << std::endl;
    double time_step = std::stod(argv[3]);
    double arclength_end = std::stod(argv[4]);
    double arclength_step = std::stod(argv[5]);
    double noise_stddev = std::stod(argv[6]);
    int measurement_per_st = 1;
    double measurement_stddev = 0.001;
    
    std::cout<< "time_step:," << time_step << std::endl;
    std::cout<< "arclength_end:," << arclength_end << std::endl;
    std::cout<< "arclength_step:," << arclength_step << std::endl;
    std::cout<< "noise_stddev:," << noise_stddev << std::endl;

    if(argc > 7){
        measurement_per_st = std::stoi(argv[7]);
    }
    if(argc > 8){
        measurement_stddev = std::stod(argv[8]);
    }
    std::cout << "measurement_per_st:," << measurement_per_st << std::endl;
    std::cout << "measurement_stddev:," << measurement_stddev << std::endl;
    std::normal_distribution<double> noise_dist(0.0, noise_stddev);
    std::normal_distribution<double> measurement_dist(0.0, measurement_stddev);
    std::default_random_engine generator;
    double current_vals[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for(double t=0.0; t<time_window; t+=time_step){
        for(double s=0.0; s<arclength_end; s+=arclength_step){
            for(int i=0; i<6; ++i){
                if(measurement_per_st == 1) current_vals[i] += noise_dist(generator);
            }
            for (int j=0; j<measurement_per_st; ++j){
                fout << t << "," << s << ",";
                for(int i=0; i<6; ++i){
                    double measurement = current_vals[i] + measurement_dist(generator);
                    fout << measurement;
                    if(i<5) fout << ",";
                }
                fout << std::endl;
            }
        }
    }
    fout.close();
    
    fout.open(argv[1]+std::string("/trial_config.json"));
    if(!fout.is_open()){
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }
    fout << "{\n";
    fout << " \"weights\": {\n";
    fout << "  \"P0\":[";
    for(int i=0; i<18; ++i){
        fout << noise_stddev / time_step;
        if(i<17) fout << ",";
    }
    fout << "],\n";
    fout << "  \"Q1\":[";
    for(int i=0; i<6; ++i){
        fout << noise_stddev / time_step;
        if(i<5) fout << ",";
    }
    fout << "],\n";
    fout << "  \"Q2\":[";
    for(int i=0; i<6; ++i){
        fout << noise_stddev / arclength_step;
        if(i<5) fout << ",";
    }
    fout << "],\n";
    fout << "  \"Q3\":[";
    for(int i=0; i<6; ++i){
        fout << noise_stddev / time_step;
        if(i<5) fout << ",";
    }
    fout << "],\n";
    fout << " \"prior_factor\": 1,\n";
    fout << " \"R_pose\": [1e-5, 1e-5, 1e-5, 2.5e-4, 2.5e-4, 2.5e-4],\n";
    fout << " \"R_gyro\": [1e-5, 1e-5, 1e-5],\n";
    fout << " \"measurement_factor\": 1\n";
    fout << " },\n";
    fout << " \"topology\": {\n";
    fout << " \"N\":" << std::max(1, int(arclength_end/arclength_step)) << ",\n";
    fout << " \"K_per_second\":" << std::max(1, int(1/time_step)) << ",\n";
    fout << " \"Ms\":1,\n";
    fout << " \"Mt\":1,\n";
    fout << " \"length\":" << arclength_end << ",\n";
    fout << " \"radius\":0.01,\n";
    fout << " \"lock_first_position\":false,\n";
    fout << " \"lock_first_pose\":false,\n";
    fout << " \"lock_last_strain\":false,\n";
    fout << " \"time_nodes_on_measurements\":false\n";
    fout << " }\n";
    fout << "}\n";
    fout.close();

    return 0;
}