#include <iostream>
#include "headers/vec3.h"
#include "headers/linearMath.h"


int main() {
	vec3ld temp(0,0,0);
	std::cout<<temp.normalize();
	temp = vec3ld::add(temp, vec3ld(-10, -20, -50));
	std::cout << (int)defaultErrCode << std::endl;
	//vec3d::normalize(temp);
	std::cout << (int)defaultErrCode << std::endl;
	
	linearMathLD::plane p(vec3ld(0,0,0),vec3ld(1,0,40));
	linearMathLD::getPt(p, &temp, coordinateName::zCoordinate);

	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	system("pause");
}