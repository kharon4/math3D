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

namespace manipulation3dF {

	__device__ __host__ float toDeg(float rad) { return rad * 180 / pi; }
	__device__ __host__ float toRad(float deg) { return deg * pi / 180; }

	__device__ __host__ vec3f getDir(float yaw, float pitch) {
		vec3f newX = vec3f(cos(yaw), sin(yaw), 0);
		vec3f rval = vec3f::add(vec3f::multiply(newX, cos(pitch)), vec3f::multiply(vec3f::vec3(0, 0, 1), sin(pitch)));
		return rval;
	}

	__device__ __host__ vec3f getRotation(vec3f a, bool* err) {
		vec3f rval;
		if (a.mag2() == 0) {
			*err = true;//error
			return vec3f(0, 0, 0);
		}
		*err = false;//no error
		rval.y = (pi / 2) - vec3f::angleRaw(a, vec3f(0, 0, 1));
		rval.x = vec3f::angleRaw(vec3f(1, 0, 0), vec3f(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}

	__device__ __host__ vec3f getRotationRaw_s(vec3f a, vec3f defaultRVal) {
		vec3f rval;
		if (a.mag2() == 0) {
			return defaultRVal;
		}
		rval.y = (pi / 2) - vec3f::angleRaw(a, vec3f(0, 0, 1));
		rval.x = vec3f::angleRaw(vec3f(1, 0, 0), vec3f(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}

	__device__ __host__ vec3f getRotationRaw(vec3f a) {
		vec3f rval;
		rval.y = (pi / 2) - vec3f::angleRaw(a, vec3f(0, 0, 1));
		rval.x = vec3f::angleRaw(vec3f(1, 0, 0), vec3f(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}


	//coordinate system functions

	__device__ __host__ void coordinateSystem::setOrigin(vec3f vec) { origin = vec; }
	__device__ __host__ void coordinateSystem::setAngle(vec3f vec) { angle = vec; reset = true; }
	__device__ __host__ void coordinateSystem::setScale(vec3f vec) { scale = vec; reset = true; }
	__device__ __host__ void coordinateSystem::setAxis(vec3f* Axis) {
		axis[0] = Axis[0];
		axis[1] = Axis[1];
		axis[2] = Axis[2];
		//set scale
		scale = getScale(axis);

		//set angle
		angle = getAngle(axis);
	}

	__device__ __host__ vec3f coordinateSystem::getOrigin() { return origin; }
	__device__ __host__ vec3f coordinateSystem::getAngle() { return angle; }
	__device__ __host__ vec3f coordinateSystem::getScale() { return scale; }
	__device__ __host__ vec3f* coordinateSystem::getAxis() { if (reset)resetAxis(); reset = false; return axis; };

	__device__ __host__ coordinateSystem::coordinateSystem(vec3f Origin, vec3f Rot, vec3f Scale) {
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
		vec3f cAxis[2];
		cAxis[0] = vec3f::add(vec3f::multiply(axis[1], cos(angle.z)), vec3f::multiply(axis[2], sin(angle.z)));
		cAxis[1] = vec3f::cross(axis[0], cAxis[0]);
		axis[0] = vec3f::multiply(axis[0], scale.x);
		axis[1] = vec3f::multiply(cAxis[0], scale.y);
		axis[2] = vec3f::multiply(cAxis[1], scale.z);
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

	__device__ __host__ vec3f coordinateSystem::getScale(vec3f* axis) {
		return vec3f(axis[0].mag(), axis[1].mag(), axis[2].mag());
	}

	__device__ __host__ vec3f coordinateSystem::getAngle(vec3f* axis) {
		vec3f rVal;
		rVal = getRotationRaw_s(axis[0]);
		vec3f tempAxis = getDir(angle.x + pi / 2, 0);
		rVal.z = vec3f::angleRaw_s(tempAxis, axis[1]);
		tempAxis = getDir(angle.x, angle.y + pi / 2);
		if (vec3f::dot(tempAxis, axis[1]) < 0)rVal.z *= -1;
		return rVal;
	}

	__device__ __host__ vec3f coordinateSystem::getInCoordinateSystem(vec3f realCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		vec3f rVal;
		realCoord = vec3f::subtract(realCoord, origin);
		rVal.x = vec3f::componentRaw_s(realCoord, axis[0]);
		rVal.y = vec3f::componentRaw_s(realCoord, axis[1]);
		rVal.z = vec3f::componentRaw_s(realCoord, axis[2]);
		vec3f Scale = scale;
		if (Scale.x == 0)Scale.x == 1;
		if (Scale.y == 0)Scale.y == 1;
		if (Scale.z == 0)Scale.z == 1;
		rVal.x /= Scale.x;
		rVal.y /= Scale.y;
		rVal.z /= Scale.z;
		return rVal;
	}

	__device__ __host__ vec3f coordinateSystem::getRealWorldCoordinates(vec3f CSCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		return vec3f::add(origin, vec3f::add(vec3f::multiply(axis[0], CSCoord.x), vec3f::add(vec3f::multiply(axis[1], CSCoord.y), vec3f::multiply(axis[2], CSCoord.z))));
	}

	__device__ __host__ void coordinateSystem::addRelativeRot(vec3f rot) {

		vec3f oldAxis[3];
		vec3f Scale = scale;
		if (Scale.x == 0)Scale.x = 1;
		if (Scale.y == 0)Scale.y = 1;
		if (Scale.z == 0)Scale.z = 1;
		oldAxis[0] = vec3f::multiply(axis[0], 1 / Scale.x);
		oldAxis[1] = vec3f::multiply(axis[1], 1 / Scale.y);
		oldAxis[2] = vec3f::multiply(axis[2], 1 / Scale.z);


		vec3f dir[3];
		dir[0] = getDir(rot.x, rot.y);
		dir[1] = getDir(rot.x + pi / 2, 0/*rot.y*/);
		dir[2] = getDir(rot.x, rot.y + pi / 2);
		{
			vec3f temp[2];
			temp[0] = vec3f::add(vec3f::multiply(dir[1], cos(rot.z)), vec3f::multiply(dir[2], sin(rot.z)));
			temp[1] = vec3f::cross(dir[0], temp[1]);
			dir[1] = temp[0];
			dir[2] = temp[1];
		}

		vec3f newAxis[3];
		newAxis[0] = vec3f::multiply(vec3f::add(vec3f::multiply(oldAxis[0], dir[0].x), vec3f::add(vec3f::multiply(oldAxis[1], dir[0].y), vec3f::multiply(oldAxis[2], dir[0].z))), scale.x);
		newAxis[1] = vec3f::multiply(vec3f::add(vec3f::multiply(oldAxis[0], dir[1].x), vec3f::add(vec3f::multiply(oldAxis[1], dir[1].y), vec3f::multiply(oldAxis[2], dir[1].z))), scale.y);
		newAxis[2] = vec3f::multiply(vec3f::add(vec3f::multiply(oldAxis[0], dir[2].x), vec3f::add(vec3f::multiply(oldAxis[1], dir[2].y), vec3f::multiply(oldAxis[2], dir[2].z))), scale.z);
		setAxis(newAxis);
	}

	__device__ __host__ void coordinateSystem::addRelativePos(vec3f pos) {
		vec3f Axis[3];
		vec3f Scale = scale;
		if (Scale.x == 0)Scale.x = 1;
		if (Scale.y == 0)Scale.y = 1;
		if (Scale.z == 0)Scale.z = 1;
		Axis[0] = vec3f::multiply(axis[0], 1 / Scale.x);
		Axis[1] = vec3f::multiply(axis[1], 1 / Scale.y);
		Axis[2] = vec3f::multiply(axis[2], 1 / Scale.z);

		origin = vec3f::add(origin, vec3f::add(vec3f::multiply(Axis[0], pos.x), vec3f::add(vec3f::multiply(Axis[1], pos.y), vec3f::multiply(Axis[2], pos.z))));
	}

	__device__ __host__ void coordinateSystem::addRotationAboutAxis(vec3f W) {
		transform T;
		vec3f angle = getRotationRaw_s(W);
		T.CS.setAngle(angle);
		vec3f oldAxis[3] = { axis[0],axis[1],axis[2] };
		T.addVec(oldAxis[0], oldAxis);
		T.addVec(oldAxis[1], oldAxis + 1);
		T.addVec(oldAxis[2], oldAxis + 2);
		angle.z = W.mag();
		T.CS.setAngle(angle);
		T.update();
		setAxis(oldAxis);
	}

	//transform functions
	__host__ void transform::addVec(vec3f val, vec3f* adress) {
		data.push_back(CS.getInCoordinateSystem(val));
		dataAddress.push_back(adress);
	}

	__host__ void transform::update() {
		for (int i = 0; i < data.size(); ++i) {
			*(dataAddress[i]) = CS.getRealWorldCoordinates(data[i]);
		}
	}


}

namespace manipulation3dLD {

	__device__ __host__ long double toDeg(long double rad) { return rad * 180 / pi; }
	__device__ __host__ long double toRad(long double deg) { return deg * pi / 180; }

	__device__ __host__ vec3ld getDir(long double yaw, long double pitch) {
		vec3ld newX = vec3ld(cos(yaw), sin(yaw), 0);
		vec3ld rval = vec3ld::add(vec3ld::multiply(newX, cos(pitch)), vec3ld::multiply(vec3ld::vec3(0, 0, 1), sin(pitch)));
		return rval;
	}

	__device__ __host__ vec3ld getRotation(vec3ld a, bool* err) {
		vec3ld rval;
		if (a.mag2() == 0) {
			*err = true;//error
			return vec3ld(0, 0, 0);
		}
		*err = false;//no error
		rval.y = (pi / 2) - vec3ld::angleRaw(a, vec3ld(0, 0, 1));
		rval.x = vec3ld::angleRaw(vec3ld(1, 0, 0), vec3ld(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}

	__device__ __host__ vec3ld getRotationRaw_s(vec3ld a, vec3ld defaultRVal) {
		vec3ld rval;
		if (a.mag2() == 0) {
			return defaultRVal;
		}
		rval.y = (pi / 2) - vec3ld::angleRaw(a, vec3ld(0, 0, 1));
		rval.x = vec3ld::angleRaw(vec3ld(1, 0, 0), vec3ld(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}

	__device__ __host__ vec3ld getRotationRaw(vec3ld a) {
		vec3ld rval;
		rval.y = (pi / 2) - vec3ld::angleRaw(a, vec3ld(0, 0, 1));
		rval.x = vec3ld::angleRaw(vec3ld(1, 0, 0), vec3ld(a.x, a.y, 0));
		if (a.y < 0)rval.x = 2 * pi - rval.x;
		return rval;
	}


	//coordinate system functions

	__device__ __host__ void coordinateSystem::setOrigin(vec3ld vec) { origin = vec; }
	__device__ __host__ void coordinateSystem::setAngle(vec3ld vec) { angle = vec; reset = true; }
	__device__ __host__ void coordinateSystem::setScale(vec3ld vec) { scale = vec; reset = true; }
	__device__ __host__ void coordinateSystem::setAxis(vec3ld* Axis) {
		axis[0] = Axis[0];
		axis[1] = Axis[1];
		axis[2] = Axis[2];
		//set scale
		scale = getScale(axis);

		//set angle
		angle = getAngle(axis);
	}

	__device__ __host__ vec3ld coordinateSystem::getOrigin() { return origin; }
	__device__ __host__ vec3ld coordinateSystem::getAngle() { return angle; }
	__device__ __host__ vec3ld coordinateSystem::getScale() { return scale; }
	__device__ __host__ vec3ld* coordinateSystem::getAxis() { if (reset)resetAxis(); reset = false; return axis; };

	__device__ __host__ coordinateSystem::coordinateSystem(vec3ld Origin, vec3ld Rot, vec3ld Scale) {
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
		vec3ld cAxis[2];
		cAxis[0] = vec3ld::add(vec3ld::multiply(axis[1], cos(angle.z)), vec3ld::multiply(axis[2], sin(angle.z)));
		cAxis[1] = vec3ld::cross(axis[0], cAxis[0]);
		axis[0] = vec3ld::multiply(axis[0], scale.x);
		axis[1] = vec3ld::multiply(cAxis[0], scale.y);
		axis[2] = vec3ld::multiply(cAxis[1], scale.z);
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

	__device__ __host__ vec3ld coordinateSystem::getScale(vec3ld* axis) {
		return vec3ld(axis[0].mag(), axis[1].mag(), axis[2].mag());
	}

	__device__ __host__ vec3ld coordinateSystem::getAngle(vec3ld* axis) {
		vec3ld rVal;
		rVal = getRotationRaw_s(axis[0]);
		vec3ld tempAxis = getDir(angle.x + pi / 2, 0);
		rVal.z = vec3ld::angleRaw_s(tempAxis, axis[1]);
		tempAxis = getDir(angle.x, angle.y + pi / 2);
		if (vec3ld::dot(tempAxis, axis[1]) < 0)rVal.z *= -1;
		return rVal;
	}

	__device__ __host__ vec3ld coordinateSystem::getInCoordinateSystem(vec3ld realCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		vec3ld rVal;
		realCoord = vec3ld::subtract(realCoord, origin);
		rVal.x = vec3ld::componentRaw_s(realCoord, axis[0]);
		rVal.y = vec3ld::componentRaw_s(realCoord, axis[1]);
		rVal.z = vec3ld::componentRaw_s(realCoord, axis[2]);
		vec3ld Scale = scale;
		if (Scale.x == 0)Scale.x == 1;
		if (Scale.y == 0)Scale.y == 1;
		if (Scale.z == 0)Scale.z == 1;
		rVal.x /= Scale.x;
		rVal.y /= Scale.y;
		rVal.z /= Scale.z;
		return rVal;
	}

	__device__ __host__ vec3ld coordinateSystem::getRealWorldCoordinates(vec3ld CSCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		return vec3ld::add(origin, vec3ld::add(vec3ld::multiply(axis[0], CSCoord.x), vec3ld::add(vec3ld::multiply(axis[1], CSCoord.y), vec3ld::multiply(axis[2], CSCoord.z))));
	}

	__device__ __host__ void coordinateSystem::addRelativeRot(vec3ld rot) {

		vec3ld oldAxis[3];
		vec3ld Scale = scale;
		if (Scale.x == 0)Scale.x = 1;
		if (Scale.y == 0)Scale.y = 1;
		if (Scale.z == 0)Scale.z = 1;
		oldAxis[0] = vec3ld::multiply(axis[0], 1 / Scale.x);
		oldAxis[1] = vec3ld::multiply(axis[1], 1 / Scale.y);
		oldAxis[2] = vec3ld::multiply(axis[2], 1 / Scale.z);


		vec3ld dir[3];
		dir[0] = getDir(rot.x, rot.y);
		dir[1] = getDir(rot.x + pi / 2, 0/*rot.y*/);
		dir[2] = getDir(rot.x, rot.y + pi / 2);
		{
			vec3ld temp[2];
			temp[0] = vec3ld::add(vec3ld::multiply(dir[1], cos(rot.z)), vec3ld::multiply(dir[2], sin(rot.z)));
			temp[1] = vec3ld::cross(dir[0], temp[1]);
			dir[1] = temp[0];
			dir[2] = temp[1];
		}

		vec3ld newAxis[3];
		newAxis[0] = vec3ld::multiply(vec3ld::add(vec3ld::multiply(oldAxis[0], dir[0].x), vec3ld::add(vec3ld::multiply(oldAxis[1], dir[0].y), vec3ld::multiply(oldAxis[2], dir[0].z))), scale.x);
		newAxis[1] = vec3ld::multiply(vec3ld::add(vec3ld::multiply(oldAxis[0], dir[1].x), vec3ld::add(vec3ld::multiply(oldAxis[1], dir[1].y), vec3ld::multiply(oldAxis[2], dir[1].z))), scale.y);
		newAxis[2] = vec3ld::multiply(vec3ld::add(vec3ld::multiply(oldAxis[0], dir[2].x), vec3ld::add(vec3ld::multiply(oldAxis[1], dir[2].y), vec3ld::multiply(oldAxis[2], dir[2].z))), scale.z);
		setAxis(newAxis);
	}

	__device__ __host__ void coordinateSystem::addRelativePos(vec3ld pos) {
		vec3ld Axis[3];
		vec3ld Scale = scale;
		if (Scale.x == 0)Scale.x = 1;
		if (Scale.y == 0)Scale.y = 1;
		if (Scale.z == 0)Scale.z = 1;
		Axis[0] = vec3ld::multiply(axis[0], 1 / Scale.x);
		Axis[1] = vec3ld::multiply(axis[1], 1 / Scale.y);
		Axis[2] = vec3ld::multiply(axis[2], 1 / Scale.z);

		origin = vec3ld::add(origin, vec3ld::add(vec3ld::multiply(Axis[0], pos.x), vec3ld::add(vec3ld::multiply(Axis[1], pos.y), vec3ld::multiply(Axis[2], pos.z))));
	}

	__device__ __host__ void coordinateSystem::addRotationAboutAxis(vec3ld W) {
		transform T;
		vec3ld angle = getRotationRaw_s(W);
		T.CS.setAngle(angle);
		vec3ld oldAxis[3] = { axis[0],axis[1],axis[2] };
		T.addVec(oldAxis[0], oldAxis);
		T.addVec(oldAxis[1], oldAxis + 1);
		T.addVec(oldAxis[2], oldAxis + 2);
		angle.z = W.mag();
		T.CS.setAngle(angle);
		T.update();
		setAxis(oldAxis);
	}

	//transform functions
	__host__ void transform::addVec(vec3ld val, vec3ld* adress) {
		data.push_back(CS.getInCoordinateSystem(val));
		dataAddress.push_back(adress);
	}

	__host__ void transform::update() {
		for (int i = 0; i < data.size(); ++i) {
			*(dataAddress[i]) = CS.getRealWorldCoordinates(data[i]);
		}
	}


}