#include <iostream>
#include "headers/vec3.h"
//#include "headers/linearMath.h"
//#include "headers/rotation.h"

#include "cudaTest.cuh"

int main() {
	
	vec3d temp(5, 6, 7);
	temp.x++;
	std::cout << temp.x << std::endl;
	//cuda lib test
	Main();

	//non cuda lib test
	
	
	system("pause");
}