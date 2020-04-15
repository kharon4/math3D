#pragma once

#define INSIDE_ROTATION_CU_FILE 1
#include "rotation.cuh"

#include<math.h>

namespace manipulation3dD {

	__device__ __host__ double toDeg(double rad) { return rad * 180 / pi; }
	__device__ __host__ double toRad(double deg) { return deg * pi / 180; }

	__device__ __host__ vec3d getDir(double yaw, double pitch) {
		vec3d newX = vec3d(cos(yaw), sin(yaw), 0);
		vec3d rval = vec3d::add(vec3d::multiply(newX, cos(pitch)), vec3d::multiply(vec3d::vec3(0, 0, 1), sin(pitch)));
		return rval;
	}

	__device__ __host__ vec3d getRotation(vec3d a,bool *err) {
		vec3d rval;
		if (a.mag2() == 0) {
			*err = true;//error
			return vec3d(0, 0, 0);
		}
		*err = false;//no error
		rval.y = (pi / 2) - vec3d::angleRaw(a, vec3d(0, 0, 1));
		rval.x = vec3d::angleRaw(vec3d(1, 0, 0), vec3d(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}

	__device__ __host__ vec3d getRotationRaw_s(vec3d a , vec3d defaultRVal) {
		vec3d rval;
		if (a.mag2() == 0) {
			return defaultRVal;
		}
		rval.y = (pi / 2) - vec3d::angleRaw(a, vec3d(0, 0, 1));
		rval.x = vec3d::angleRaw(vec3d(1, 0, 0), vec3d(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}
	
	__device__ __host__ vec3d getRotationRaw(vec3d a) {
		vec3d rval;
		rval.y = (pi / 2) - vec3d::angleRaw(a, vec3d(0, 0, 1));
		rval.x = vec3d::angleRaw(vec3d(1, 0, 0), vec3d(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}


	//coordinate system functions

	__device__ __host__ void coordinateSystem::setOrigin(vec3d vec) { origin = vec; }
	__device__ __host__ void coordinateSystem::setAngle(vec3d vec) { angle = vec; reset = true; }
	__device__ __host__ void coordinateSystem::setScale(vec3d vec) { scale = vec; reset = true; }
	__device__ __host__ void coordinateSystem::setAxis(vec3d* Axis) {
		axis[0] = Axis[0];
		axis[1] = Axis[1];
		axis[2] = Axis[2];
		//set scale
		scale = getScale(axis);

		//set angle
		angle = getAngle(axis);
	}

	__device__ __host__ vec3d coordinateSystem::getOrigin() { return origin; }
	__device__ __host__ vec3d coordinateSystem::getAngle() { return angle; }
	__device__ __host__ vec3d coordinateSystem::getScale() { return scale; }
	__device__ __host__ vec3d* coordinateSystem::getAxis() { if (reset)resetAxis(); reset = false; return axis; };

	__device__ __host__ coordinateSystem::coordinateSystem(vec3d Origin, vec3d Rot, vec3d Scale) {
		origin = Origin;
		angle = Rot;
		scale = Scale;
		resetAxis();
	}

	__device__ __host__ void coordinateSystem::resetAxis() {
		axis[0] = getDir(angle.x, angle.y);
		axis[1] = getDir(angle.x + pi / 2, 0);
		axis[2] = getDir(angle.x, angle.y + pi / 2);
		//axis[1] and axis[2] are temp axes
		vec3d cAxis[2];
		cAxis[0] = vec3d::add(vec3d::multiply(axis[1], cos(angle.z)), vec3d::multiply(axis[2], sin(angle.z)));
		cAxis[1] = vec3d::cross(axis[0], cAxis[0]);
		axis[0] = vec3d::multiply(axis[0], scale.x);
		axis[1] = vec3d::multiply(cAxis[0], scale.y);
		axis[2] = vec3d::multiply(cAxis[1], scale.z);
	}

	__device__ __host__ void coordinateSystem::set(coordinateSystem& cs) {
		//set origin
		origin = cs.getOrigin();
		scale = cs.getScale();
		angle = cs.getAngle();
		axis[0] = cs.getAxis()[0];
		axis[1] = cs.getAxis()[1];
		axis[2] = cs.getAxis()[2];
	}

	__device__ __host__ vec3d coordinateSystem::getScale(vec3d* axis) {
		return vec3d(axis[0].mag(), axis[1].mag(), axis[2].mag());
	}

	__device__ __host__ vec3d coordinateSystem::getAngle(vec3d* axis) {
		vec3d rVal;
		rVal = getRotationRaw_s(axis[0]);
		vec3d tempAxis = getDir(angle.x + pi / 2, 0);
		rVal.z = vec3d::angleRaw_s(tempAxis, axis[1]);
		tempAxis = getDir(angle.x, angle.y + pi / 2);
		if (vec3d::dot(tempAxis, axis[1]) < 0)rVal.z *= -1;
		return rVal;
	}

	__device__ __host__ vec3d coordinateSystem::getInCoordinateSystem(vec3d realCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		vec3d rVal;
		realCoord = vec3d::subtract(realCoord, origin);
		rVal.x = vec3d::componentRaw_s(realCoord, axis[0]);
		rVal.y = vec3d::componentRaw_s(realCoord, axis[1]);
		rVal.z = vec3d::componentRaw_s(realCoord, axis[2]);
		vec3d Scale = scale;
		if (Scale.x == 0)Scale.x == 1;
		if (Scale.y == 0)Scale.y == 1;
		if (Scale.z == 0)Scale.z == 1;
		rVal.x /= Scale.x;
		rVal.y /= Scale.y;
		rVal.z /= Scale.z;
		return rVal;
	}

	__device__ __host__ vec3d coordinateSystem::getRealWorldCoordinates(vec3d CSCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		return vec3d::add(origin, vec3d::add(vec3d::multiply(axis[0], CSCoord.x), vec3d::add(vec3d::multiply(axis[1], CSCoord.y), vec3d::multiply(axis[2], CSCoord.z))));
	}

	__device__ __host__ void coordinateSystem::addRelativeRot(vec3d rot) {

		vec3d oldAxis[3];
		vec3d Scale = scale;
		if (Scale.x == 0)Scale.x = 1;
		if (Scale.y == 0)Scale.y = 1;
		if (Scale.z == 0)Scale.z = 1;
		oldAxis[0] = vec3d::multiply(axis[0], 1 / Scale.x);
		oldAxis[1] = vec3d::multiply(axis[1], 1 / Scale.y);
		oldAxis[2] = vec3d::multiply(axis[2], 1 / Scale.z);


		vec3d dir[3];
		dir[0] = getDir(rot.x, rot.y);
		dir[1] = getDir(rot.x + pi / 2, 0/*rot.y*/);
		dir[2] = getDir(rot.x, rot.y + pi / 2);
		{
			vec3d temp[2];
			temp[0] = vec3d::add(vec3d::multiply(dir[1], cos(rot.z)), vec3d::multiply(dir[2], sin(rot.z)));
			temp[1] = vec3d::cross(dir[0], temp[1]);
			dir[1] = temp[0];
			dir[2] = temp[1];
		}

		vec3d newAxis[3];
		newAxis[0] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[0].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[0].y), vec3d::multiply(oldAxis[2], dir[0].z))), scale.x);
		newAxis[1] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[1].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[1].y), vec3d::multiply(oldAxis[2], dir[1].z))), scale.y);
		newAxis[2] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[2].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[2].y), vec3d::multiply(oldAxis[2], dir[2].z))), scale.z);
		setAxis(newAxis);
	}

	__device__ __host__ void coordinateSystem::addRelativePos(vec3d pos) {
		vec3d Axis[3];
		vec3d Scale = scale;
		if (Scale.x == 0)Scale.x = 1;
		if (Scale.y == 0)Scale.y = 1;
		if (Scale.z == 0)Scale.z = 1;
		Axis[0] = vec3d::multiply(axis[0], 1 / Scale.x);
		Axis[1] = vec3d::multiply(axis[1], 1 / Scale.y);
		Axis[2] = vec3d::multiply(axis[2], 1 / Scale.z);

		origin = vec3d::add(origin, vec3d::add(vec3d::multiply(Axis[0], pos.x), vec3d::add(vec3d::multiply(Axis[1], pos.y), vec3d::multiply(Axis[2], pos.z))));
	}

	__device__ __host__ void coordinateSystem::addRotationAboutAxis(vec3d W) {
		transform T;
		vec3d angle = getRotationRaw_s(W);
		T.CS.setAngle(angle);
		vec3d oldAxis[3] = { axis[0],axis[1],axis[2] };
		T.addVec(oldAxis[0], oldAxis);
		T.addVec(oldAxis[1], oldAxis + 1);
		T.addVec(oldAxis[2], oldAxis + 2);
		angle.z = W.mag();
		T.CS.setAngle(angle);
		T.update();
		setAxis(oldAxis);
	}
	
	//transform functions
	__host__ void transform::addVec(vec3d val, vec3d* adress) {
		data.push_back(CS.getInCoordinateSystem(val));
		dataAddress.push_back(adress);
	}

	__host__ void transform::update() {
		for (int i = 0; i < data.size(); ++i) {
			*(dataAddress[i]) = CS.getRealWorldCoordinates(data[i]);
		}
	}

	
}