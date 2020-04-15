#pragma once

//Some rules / standards
//for eulers angles parent sequence is z , y , x hence z is yaw , y is pitch and x is roll
//every angle is measured from +ve dir of x axis or in the direction of z axis from the xy plane
//eulers angles are meaasured from respective axis + in the direction of the axis
//vec3 is used to store yaw , pitch and roll as x , y , z components respectively
//assuming right handed coordinate system

#include "vec3.cuh"
#include<vector>

namespace manipulation3dD {//double precision

	static const float pi = 3.14159;

	__device__ __host__ double toDeg(double rad);
	__device__ __host__ double toRad(double deg);

	__device__ __host__ vec3d getDir(double yaw, double pitch);//get vector from 2 angles

	__device__ __host__ vec3d getRotation(vec3d a,bool * err);//get 2 angles froma vector
	__device__ __host__ vec3d getRotationRaw_s(vec3d a, vec3d defaultRVal = vec3d(0,0,0));//get 2 angles froma vector
	__device__ __host__ vec3d getRotationRaw(vec3d a);//get 2 angles froma vector



	//coordinate system
	//a raw frame of reference that can be used to switch between global and frame of ref coordinate systems 
	class coordinateSystem {
	private:
		vec3d origin;
		vec3d angle;
		vec3d scale;
		vec3d axis[3];
		bool reset = true;
	public:


		__device__ __host__ void setOrigin(vec3d);
		__device__ __host__ void setAngle(vec3d);
		__device__ __host__ void setScale(vec3d);
		__device__ __host__ void setAxis(vec3d*);


		__device__ __host__ vec3d getOrigin();
		__device__ __host__ vec3d getAngle();
		__device__ __host__ vec3d getScale();
		__device__ __host__ vec3d* getAxis();


		__device__ __host__ coordinateSystem(vec3d Origin = vec3d(0, 0, 0), vec3d Rot = vec3d(0, 0, 0), vec3d Scale = vec3d(1, 1, 1));
		__device__ __host__ void set(coordinateSystem& cs);
		__device__ __host__ vec3d getAngle(vec3d* axis);
		__device__ __host__ vec3d getScale(vec3d* axis);
		__device__ __host__ void resetAxis();
		__device__ __host__ vec3d getInCoordinateSystem(vec3d realCoord);
		__device__ __host__ vec3d getRealWorldCoordinates(vec3d CSCoord);
		__device__ __host__ void addRelativeRot(vec3d rot);
		__device__ __host__ void addRelativePos(vec3d pos);
		__device__ __host__ void addRotationAboutAxis(vec3d W);//rotation of |W| anticlocwise about w

	};


	class transform {
	private:
		std::vector <vec3d> data;
		std::vector <vec3d*> dataAddress;
	public:
		coordinateSystem CS;
		
		__host__ transform() {};
		__host__ void addVec(vec3d val, vec3d* address);
		__host__ void update();

		std::vector<vec3d>* getData() { return &data; }
	};

}

#ifndef math3D_DeclrationOnly
#ifndef INSIDE_ROTATION_CU_FILE
#include "rotation.cu"
#endif
#endif
