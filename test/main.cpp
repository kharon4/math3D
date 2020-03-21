#include <iostream>
#include "headers/vec3.h"
#include "headers/linearMath.h"
#include "headers/rotation.h"

int main() {
	vec3f temp(10,2,0);
	//temp = temp - vec3f(10,52,0);
	temp = 5 * temp;
	temp = vec3f(1,0,0);

	temp = temp * vec3f(0, 1, 0);
	temp += vec3f(1, 1, 0);
	std::cout << temp.x<<" , "<< temp.y <<" , "<< temp.z << std::endl;
	std::cout << (temp / 2).z;
	/*
	std::cout << temp.y<<std::endl;
	manipulation3dF::transform t;
	manipulation3dF::coordinateSystem c2(vec3f(1, 1, 1),vec3f(manipulation3dF::pi/2,0, manipulation3dF::pi / 2),vec3f(1,2,3));
	t.CS.set(c2);



	temp = t.CS.getAxis()[0];
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	temp = t.CS.getAxis()[1];
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	temp = t.CS.getAxis()[2];
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;

	temp = t.CS.getOrigin();
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	temp = t.CS.getAngle();
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	temp = t.CS.getScale();
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	t.CS.setAngle(vec3f(0, 0, 0));
	t.CS.setOrigin(vec3f(0, 0, 0));
	t.CS.setScale(vec3f(1, 1, 1));
	temp = vec3f(10, 0, 0);
	t.addVec(temp, &temp);
	t.addRelativeRot(vec3f(manipulation3dF::pi / 2, 0, 0));
	t.addRelativePos(vec3f(5, -10, 0));
	t.update();
	std::cout << temp.x << " , " << temp.y << " , " << temp.z << std::endl;
	*/
	system("pause");
}