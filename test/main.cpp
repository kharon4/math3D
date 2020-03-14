#include <iostream>
#include "headers/vec3.h"

int main() {
	vec3d temp;
	std::cout<<temp.normalize();
	temp = vec3d::add(temp, vec3d(-10, -20, -50));
	std::cout << (int)defaultErrCode << std::endl;
	vec3d::normalize(temp);
	std::cout << (int)defaultErrCode << std::endl;
	
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	system("pause");
}