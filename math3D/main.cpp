#include <iostream>
#include "headers/vec3.h"

int main() {
	vec3 temp(10, 20, 50);
	temp = add(temp, vec3(-10, -20, -50));
	std::cout << vec3::errCode << std::endl;
	normalize(temp);
	std::cout << vec3::errCode << std::endl;
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	system("pause");
}