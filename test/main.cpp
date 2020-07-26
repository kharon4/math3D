#include <iostream>
//#include "vec3.cuh"
//#include "headers/linearMath.h"
//#include "headers/rotation.h"
//#include "linearEqnSolver.h"
//#include "rotation.cuh"
//#include "cudaTest.cuh"

#include<fstream>


#include "vec.cuh"
#include "vec3Specialized.cuh"
using namespace std;

int main() {
		
	//cuda lib test
	//Main();

	//non cuda lib test
	//double arr[] = {10,19,23,45,20};
	
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

	
	vec3S<float> v;
	cin >>v.V;

	system("pause");
}