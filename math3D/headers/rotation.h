#pragma once
#include"vec3.h"
#include<vector>


//Some rules / standards
//for eulers angles parent sequence is z , y , x hence z is yaw , y is pitch and x is roll
//every angle is measured from +ve dir of x axis or in the direction of z axis from the xy plane
//eulers angles are meaasured from respective axis + in the direction of the axis
//vec3 is used to store yaw , pitch and roll ans x , y , z components respectively
//assuming right handed coordinate system

namespace manipulation3dD {//double precision

	static char errCode = 0;

	static const float pi = 3.14159;

	double toDeg(double rad);
	double toRad(double deg);
	
	vec3d getDir(double yaw, double pitch);//get vector from 2 angles

	vec3d getRotation(vec3d a);//get 2 angles froma vector


	//coordinate system
	//a raw frame of reference that can be used to switch between global and frame of ref coordinate systems 
	class coordinateSystem {
	private:

	public:
		vec3d axis[3];
		vec3d origin;
		vec3d angle;
		vec3d scale;
		coordinateSystem(vec3d Origin = vec3d(0, 0, 0), vec3d Rot = vec3d(0, 0, 0), vec3d Scale = vec3d(1, 1, 1));
		void resetAxis();
		vec3d getInCoordinateSystem(vec3d realCoord);
		vec3d getRealWorldCoordinates(vec3d CSCoord);
	};


	class transform {
	private:
		std::vector <vec3d> data;
		std::vector <vec3d*> dataAdress;
	public:
		coordinateSystem CS;
		void addRelativeRot(vec3d rot);
		void addRelativePos(vec3d pos);

		void addVec(vec3d val, vec3d* adress);

		void update();

		std::vector<vec3d>* getData() { return &data; }
	};



}