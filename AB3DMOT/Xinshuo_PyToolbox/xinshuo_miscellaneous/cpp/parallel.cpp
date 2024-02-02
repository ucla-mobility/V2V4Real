#include<cstdio>
#include<cstdlib>
#include<cassert>
#include<string>
#include<vector>
#include<omp.h>
#include<chrono> 

using namespace std;
using namespace std::chrono; 

int main(int argc, char* argv[]){
	if( argc != 3 ){
		fprintf(stderr, "usage: %s commands threads\n", argv[0]);
		return -1;
	}
	
	fprintf(stderr, "reading command file.\n");
	FILE *in = fopen(argv[1], "r");
	assert(in != NULL);
	vector<string> cmds;
	char buf[10240];
	while(fscanf(in, "%[^\n]\n", buf) > 0 ){
		cmds.push_back(buf);
	}
	fclose(in);
	int threads;
	sscanf(argv[2], "%d", &threads);
	omp_set_num_threads(threads);
	int cmds_count = cmds.size();
	fprintf(stderr, "read %d commands, using %d threads\n", cmds_count, threads);
	// #pragma omp parallel for schedule(dynamic)
	// for(int i = 0; i < cmds_count; i++){
	// 	//string cmd = string("eval ") + cmds[i];
	// 	string cmd = cmds[i];
	// 	system(cmd.c_str());
	// 	if( i % 100 == 0 ){
	// 		fprintf(stderr, "completed %d/%d\n", i, cmds_count);
	// 	}
	// }

	// Get starting timepoint 
    auto start = high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < cmds_count; i++){
		string cmd = cmds[i];
		system(cmd.c_str());

		// Get ending timepoint 
		auto stop = high_resolution_clock::now(); 
		auto duration = duration_cast<seconds>(stop - start); 
		int seconds, minutes, hours;
		seconds = int(duration.count());

		minutes = seconds / 60;
		hours = minutes / 60;

		int sec_left, min_left, hour_left;
		sec_left = seconds * cmds_count / (i + 1) - seconds;
		min_left = sec_left / 60;
		hour_left = min_left / 60;

		fprintf(stderr, "completed %06d/%06d, EP [%d:%02d:%02d], ETA [%d:%02d:%02d] \r", i, cmds_count, hours, minutes%60, seconds%60, hour_left, min_left%60, sec_left%60);
	}

	return 0;
}