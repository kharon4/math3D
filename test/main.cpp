#include <iostream>
#include "headers/vec3.h"
#include "headers/linearMath.h"
#include "headers/rotation.h"
#include "cudaTest.cuh"

int main() {
	Main();

	//non cuda lib test
	vec3f temp(10,2,0);
	temp = 5 * temp;
	temp = vec3f(1,0,0);

	temp = temp * vec3f(0, 1, 0);
	temp += vec3f(1, 1, 0);
	std::cout << temp.x<<" , "<< temp.y <<" , "<< temp.z << std::endl;
	std::cout << (temp / 2).z;
	
	system("pause");
}