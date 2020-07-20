#pragma once

//Some rules / standards
//for eulers angles parent sequence is z , y , x hence z is yaw , y is pitch and x is roll
//every angle is measured from +ve dir of x axis or in the direction of z axis from the xy plane
//eulers angles are meaasured from respective axis + in the direction of the axis
//vec3 is used to store yaw , pitch and roll as x , y , z components respectively
//assuming right handed coordinate system

#include "vec3.cuh"
#include<vector>


#define math3D_pi 3.14159

namespace manipulation3d {

	__device__ __host__ double toDeg(double rad);
	__device__ __host__ double toRad(double deg);

	__device__ __host__ vec3d getDir(double yaw, double pitch);//get vector from 2 angles

	__device__ __host__ vec3d getRotation(const vec3d& a,bool * err);//get 2 angles from a vector (yaw , pitch, NA)
	__device__ __host__ vec3d getRotationRaw_s(const vec3d& a, const vec3d& defaultRVal = vec3d(0,0,0));//get 2 angles froma vector (yaw , pitch, NA)
	__device__ __host__ vec3d getRotationRaw(const vec3d& a);//get 2 angles froma vector (yaw , pitch, NA)



	//coordinate system
	//a raw frame of reference that can be used to switch between global and frame of ref coordinate systems 
	class coordinateSystem {
	private:
		vec3d origin;
		vec3d angle;
		vec3d scale;
		vec3d axis[3];
		bool reset = true;//calculate axis
	public:

		enum class transformationType:unsigned char { 
			translation = 1, rotation = 2 , scaling = 4, 
			TR = 3 , TS = 5 , RS = 6,
			none = 0, all = 7
		};

		__device__ __host__ void setOrigin(const vec3d&);
		__device__ __host__ void setAngle(const vec3d&);
		__device__ __host__ void setScale(const vec3d&);
		__device__ __host__ void setAxis(vec3d*);


		__device__ __host__ vec3d getOrigin() const;
		__device__ __host__ vec3d getAngle() const;
		__device__ __host__ vec3d getScale() const;
		__device__ __host__ const vec3d* getAxis();


		__device__ __host__ coordinateSystem(const vec3d& Origin = vec3d(0, 0, 0), const vec3d& Rot = vec3d(0, 0, 0), const vec3d& Scale = vec3d(1, 1, 1));
		__device__ __host__ void set(coordinateSystem& cs);
		__device__ __host__ vec3d getAngle(vec3d* axis) const;// gets angle of the axis wrt the coord system
		__device__ __host__ vec3d getScale(vec3d* axis) const;//gets global scale
		__device__ __host__ void resetAxis(); // recalculates axes based on scale and angle
		__device__ __host__ vec3d getInCoordinateSystem(const vec3d& realCoord , const transformationType type = transformationType::all);
		__device__ __host__ vec3d getRealWorldCoordinates(const vec3d& CSCoord , const transformationType type = transformationType::all);
		__device__ __host__ void addRelativeRot(const vec3d& rot);
		__device__ __host__ void addRelativePos(const vec3d& pos);
		__device__ __host__ void addRotationAboutAxis(const vec3d& W);//rotation of |W| anticlocwise about w

	};

}
