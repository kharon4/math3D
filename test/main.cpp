#include <iostream>
#include "vec3.cuh"
//#include "headers/linearMath.h"
//#include "headers/rotation.h"
#include "linearEqnSolver.h"

#include "cudaTest.cuh"

#include<fstream>

int main() {
		
	//cuda lib test
	//Main();

	//non cuda lib test
	//double arr[] = {10,19,23,45,20};
	
	vec3d f;
	std::cin >> f;

	vec3f vec2(1, 2, 3);
	std::cout << (f + vec2);
	std::cout << std::endl;
	//LES test
	/*std::ifstream file("res/LES/circle.txt",std::ios::in);
	LES::system sys(0,0);
	sys.load(file);
	sys.displayMatrix(std::cout);
	std::vector<double> rVal;
	sys.getSolution(rVal);
	std::cout << "\n solved version :\n";
	sys.displayMatrix(std::cout);
	std::cout << "\n solution :\n";

	for (unsigned long int i = 0; i < rVal.size(); ++i) {
		std::cout << rVal[i] << "  .  ";
	}
	//eq.displayEqn(std::cout);

	*/
	system("pause");
}