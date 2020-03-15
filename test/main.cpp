#include <iostream>
#include "headers/vec3.h"
#include "headers/linearMath.h"


int main() {
	vec3d temp(0,0,0);
	std::cout<<temp.normalize();
	temp = vec3d::add(temp, vec3d(-10, -20, -50));
	std::cout << (int)defaultErrCode << std::endl;
	//vec3d::normalize(temp);
	std::cout << (int)defaultErrCode << std::endl;
	
	linearMathD::plane p(vec3d(0,0,0),vec3d(1,0,4));
	linearMathD::getPt(p, &temp, coordinateName::zCoordinate);

	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	system("pause");
}