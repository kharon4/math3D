#pragma once
#include"vec3.h"
#include<vector>


//Some rules / standards
//for eulers angles parent sequence is z , y , x hence z is yaw , y is pitch and x is roll
//every angle is measured from +ve dir of x axis or in the direction of z axis from the xy plane
//eulers angles are meaasured from respective axis + in the direction of the axis
//vec3 is used to store yaw , pitch and roll as x , y , z components respectively
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
		vec3d origin;
		vec3d angle;
		vec3d scale;
		vec3d axis[3];
		bool reset = true;
	public:
		
		
		void setOrigin(vec3d);
		void setAngle(vec3d);
		void setScale(vec3d);
		void setAxis(vec3d*);


		vec3d getOrigin();
		vec3d getAngle();
		vec3d getScale();
		vec3d* getAxis();


		coordinateSystem(vec3d Origin = vec3d(0, 0, 0), vec3d Rot = vec3d(0, 0, 0), vec3d Scale = vec3d(1, 1, 1));
		void set(coordinateSystem& cs);
		vec3d getAngle(vec3d* axis);
		vec3d getScale(vec3d* axis);
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
		void addRotationAboutAxis(vec3d W);//rotation of |W| anticlocwise about w

		void addVec(vec3d val, vec3d* address);

		void update();

		std::vector<vec3d>* getData() { return &data; }
	};

}

namespace manipulation3dF {//single precision

	static char errCode = 0;

	static const float pi = 3.14159;

	float toDeg(float rad);
	float toRad(float deg);

	vec3f getDir(float yaw, float pitch);//get vector from 2 angles

	vec3f getRotation(vec3f a);//get 2 angles froma vector


	//coordinate system
	//a raw frame of reference that can be used to switch between global and frame of ref coordinate systems 
	class coordinateSystem {
	private:
		vec3f origin;
		vec3f angle;
		vec3f scale;
		vec3f axis[3];
		bool reset = true;
	public:


		void setOrigin(vec3f);
		void setAngle(vec3f);
		void setScale(vec3f);
		void setAxis(vec3f*);


		vec3f getOrigin();
		vec3f getAngle();
		vec3f getScale();
		vec3f* getAxis();


		coordinateSystem(vec3f Origin = vec3f(0, 0, 0), vec3f Rot = vec3f(0, 0, 0), vec3f Scale = vec3f(1, 1, 1));
		void set(coordinateSystem& cs);
		vec3f getAngle(vec3f* axis);
		vec3f getScale(vec3f* axis);
		void resetAxis();
		vec3f getInCoordinateSystem(vec3f realCoord);
		vec3f getRealWorldCoordinates(vec3f CSCoord);
	};


	class transform {
	private:
		std::vector <vec3f> data;
		std::vector <vec3f*> dataAdress;
	public:
		coordinateSystem CS;
		void addRelativeRot(vec3f rot);
		void addRelativePos(vec3f pos);
		void addRotationAboutAxis(vec3f W);//rotation of |W| anticlocwise about w

		void addVec(vec3f val, vec3f* address);

		void update();

		std::vector<vec3f>* getData() { return &data; }
	};

}

namespace manipulation3dLD {//long long double precision

	static char errCode = 0;

	static const float pi = 3.14159;

	long double toDeg(long double rad);
	long double toRad(long double deg);

	vec3ld getDir(long double yaw, long double pitch);//get vector from 2 angles

	vec3ld getRotation(vec3ld a);//get 2 angles froma vector


	//coordinate system
	//a raw frame of reference that can be used to switch between global and frame of ref coordinate systems 
	class coordinateSystem {
	private:
		vec3ld origin;
		vec3ld angle;
		vec3ld scale;
		vec3ld axis[3];
		bool reset = true;
	public:


		void setOrigin(vec3ld);
		void setAngle(vec3ld);
		void setScale(vec3ld);
		void setAxis(vec3ld*);


		vec3ld getOrigin();
		vec3ld getAngle();
		vec3ld getScale();
		vec3ld* getAxis();


		coordinateSystem(vec3ld Origin = vec3ld(0, 0, 0), vec3ld Rot = vec3ld(0, 0, 0), vec3ld Scale = vec3ld(1, 1, 1));
		void set(coordinateSystem& cs);
		vec3ld getAngle(vec3ld* axis);
		vec3ld getScale(vec3ld* axis);
		void resetAxis();
		vec3ld getInCoordinateSystem(vec3ld realCoord);
		vec3ld getRealWorldCoordinates(vec3ld CSCoord);
	};


	class transform {
	private:
		std::vector <vec3ld> data;
		std::vector <vec3ld*> dataAdress;
	public:
		coordinateSystem CS;
		void addRelativeRot(vec3ld rot);
		void addRelativePos(vec3ld pos);
		void addRotationAboutAxis(vec3ld W);//rotation of |W| anticlocwise about w

		void addVec(vec3ld val, vec3ld* address);

		void update();

		std::vector<vec3ld>* getData() { return &data; }
	};

}