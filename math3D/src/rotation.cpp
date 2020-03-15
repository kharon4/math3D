#include "./../headers/rotation.h"

#include<math.h>

namespace manipulation3dD {

	double toDeg(double rad) { return rad * 180 / pi; }
	double toRad(double deg) { return deg * pi / 180; }
	
	vec3d getDir(double yaw, double pitch) {
		vec3d newX = vec3d(cos(yaw), sin(yaw), 0);
		vec3d rval = vec3d::add(vec3d::multiply(newX, cos(pitch)), vec3d::multiply(vec3d::vec3(0, 0, 1), sin(pitch)));
		return rval;
	}

	vec3d getRotation(vec3d a) {
		vec3d rval;
		errCode = 0;
		if (a.mag2() == 0) {
			errCode = 1;
			return vec3d(0, 0, 0);
		}
		rval.y = (pi / 2) - vec3d::angle(a, vec3d(0, 0, 1));
		rval.x = vec3d::angle(vec3d(1, 0, 0), vec3d(a.x, a.y, 0));
		if (defaultErrCode) rval.x = 0;
		else if (a.y < 0)rval.x = 2 * pi - rval.x;

		return rval;
	}


	//coordinate system functions

	void coordinateSystem::setOrigin(vec3d vec) { origin = vec; }
	void coordinateSystem::setAngle(vec3d vec) { angle = vec; reset = true; }
	void coordinateSystem::setScale(vec3d vec) { scale = vec; reset = true; }
	void coordinateSystem::setAxis(vec3d* Axis) {
		axis[0] = Axis[0];
		axis[1] = Axis[1];
		axis[2] = Axis[2];
		//set scale
		scale = getScale(axis);

		//set angle
		angle = getAngle(axis);
	}

	vec3d coordinateSystem::getOrigin() { return origin; }
	vec3d coordinateSystem::getAngle() { return angle; }
	vec3d coordinateSystem::getScale() { return scale; }
	vec3d* coordinateSystem::getAxis() { if (reset)resetAxis(); reset = false; return axis; };

	coordinateSystem::coordinateSystem(vec3d Origin, vec3d Rot, vec3d Scale) {
		origin = Origin;
		angle = Rot;
		scale = Scale;
		resetAxis();
	}

	void coordinateSystem::resetAxis() {
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

	void coordinateSystem::set(coordinateSystem& cs) {
		//set origin
		origin = cs.getOrigin();
		
		//set axis
		setAxis(cs.getAxis());
	}

	vec3d coordinateSystem::getScale(vec3d* axis) {
		return vec3d(axis[0].mag(), axis[1].mag(), axis[2].mag());
	}

	vec3d coordinateSystem::getAngle(vec3d* axis) {
		vec3d rVal;
		rVal = getRotation(axis[0]);
		vec3d tempAxis = getDir(angle.x + pi / 2, 0);
		rVal.z = vec3d::angle(tempAxis, axis[1]);
		tempAxis = getDir(angle.x, angle.y + pi / 2);
		if (vec3d::dot(tempAxis, axis[1]) < 0)rVal.z *= -1;
		return rVal;
	}

	vec3d coordinateSystem::getInCoordinateSystem(vec3d realCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		vec3d rVal;
		realCoord = vec3d::subtract(realCoord, origin);
		rVal.x = vec3d::component(realCoord, axis[0]);
		rVal.y = vec3d::component(realCoord, axis[1]);
		rVal.z = vec3d::component(realCoord, axis[2]);
		vec3d Scale = scale;
		if (Scale.x == 0)Scale.x == 1;
		if (Scale.y == 0)Scale.y == 1;
		if (Scale.z == 0)Scale.z == 1;
		rVal.x /= Scale.x;
		rVal.y /= Scale.y;
		rVal.z /= Scale.z;
		return rVal;
	}

	vec3d coordinateSystem::getRealWorldCoordinates(vec3d CSCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		return vec3d::add(origin, vec3d::add(vec3d::multiply(axis[0], CSCoord.x), vec3d::add(vec3d::multiply(axis[1], CSCoord.y), vec3d::multiply(axis[2], CSCoord.z))));
	}



	//transform functions

	void transform::addVec(vec3d val, vec3d* adress) {
		data.push_back(CS.getInCoordinateSystem(val));
		dataAdress.push_back(adress);
	}

	void transform::update() {
		for (int i = 0; i < data.size(); ++i) {
			*(dataAdress[i]) = CS.getRealWorldCoordinates(data[i]);
		}
	}

	void transform::addRelativeRot(vec3d rot) {
		
		vec3d oldAxis[3];
		vec3d scale = CS.getScale();
		if (scale.x == 0)scale.x = 1;
		if (scale.y == 0)scale.y = 1;
		if (scale.z == 0)scale.z = 1;
		oldAxis[0] = vec3d::multiply(CS.getAxis()[0], 1 / scale.x);
		oldAxis[1] = vec3d::multiply(CS.getAxis()[1], 1 / scale.y);
		oldAxis[2] = vec3d::multiply(CS.getAxis()[2], 1 / scale.z);
		

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

		scale = CS.getScale();
		vec3d newAxis[3];
		newAxis[0] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[0].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[0].y), vec3d::multiply(oldAxis[2], dir[0].z))), scale.x);
		newAxis[1] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[1].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[1].y), vec3d::multiply(oldAxis[2], dir[1].z))), scale.y);
		newAxis[2] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[2].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[2].y), vec3d::multiply(oldAxis[2], dir[2].z))), scale.z);
		CS.setAxis(newAxis);
	}


	void transform::addRelativePos(vec3d pos) {
		vec3d Axis[3];
		vec3d scale = CS.getScale();
		if (scale.x == 0)scale.x = 1;
		if (scale.y == 0)scale.y = 1;
		if (scale.z == 0)scale.z = 1;
		Axis[0] = vec3d::multiply(CS.getAxis()[0], 1 / scale.x);
		Axis[1] = vec3d::multiply(CS.getAxis()[1], 1 / scale.y);
		Axis[2] = vec3d::multiply(CS.getAxis()[2], 1 / scale.z);

		CS.setOrigin(vec3d::add(CS.getOrigin(), vec3d::add(vec3d::multiply(Axis[0], pos.x), vec3d::add(vec3d::multiply(Axis[1], pos.y), vec3d::multiply(Axis[2], pos.z)))));
	}
}


namespace manipulation3dF {

	float toDeg(float rad) { return rad * 180 / pi; }
	float toRad(float deg) { return deg * pi / 180; }

	vec3f getDir(float yaw, float pitch) {
		vec3f newX = vec3f(cos(yaw), sin(yaw), 0);
		vec3f rval = vec3f::add(vec3f::multiply(newX, cos(pitch)), vec3f::multiply(vec3f::vec3(0, 0, 1), sin(pitch)));
		return rval;
	}

	vec3f getRotation(vec3f a) {
		vec3f rval;
		errCode = 0;
		if (a.mag2() == 0) {
			errCode = 1;
			return vec3f(0, 0, 0);
		}
		rval.y = (pi / 2) - vec3f::angle(a, vec3f(0, 0, 1));
		rval.x = vec3f::angle(vec3f(1, 0, 0), vec3f(a.x, a.y, 0));
		if (defaultErrCode) rval.x = 0;
		else if (a.y < 0)rval.x = 2 * pi - rval.x;

		return rval;
	}


	//coordinate system functions

	void coordinateSystem::setOrigin(vec3f vec) { origin = vec; }
	void coordinateSystem::setAngle(vec3f vec) { angle = vec; reset = true; }
	void coordinateSystem::setScale(vec3f vec) { scale = vec; reset = true; }
	void coordinateSystem::setAxis(vec3f* Axis) {
		axis[0] = Axis[0];
		axis[1] = Axis[1];
		axis[2] = Axis[2];
		//set scale
		scale = getScale(axis);

		//set angle
		angle = getAngle(axis);
	}

	vec3f coordinateSystem::getOrigin() { return origin; }
	vec3f coordinateSystem::getAngle() { return angle; }
	vec3f coordinateSystem::getScale() { return scale; }
	vec3f* coordinateSystem::getAxis() { if (reset)resetAxis(); reset = false; return axis; };

	coordinateSystem::coordinateSystem(vec3f Origin, vec3f Rot, vec3f Scale) {
		origin = Origin;
		angle = Rot;
		scale = Scale;
		resetAxis();
	}

	void coordinateSystem::resetAxis() {
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

	void coordinateSystem::set(coordinateSystem& cs) {
		//set origin
		origin = cs.getOrigin();

		//set axis
		setAxis(cs.getAxis());
	}

	vec3f coordinateSystem::getScale(vec3f* axis) {
		return vec3f(axis[0].mag(), axis[1].mag(), axis[2].mag());
	}

	vec3f coordinateSystem::getAngle(vec3f* axis) {
		vec3f rVal;
		rVal = getRotation(axis[0]);
		vec3f tempAxis = getDir(angle.x + pi / 2, 0);
		rVal.z = vec3f::angle(tempAxis, axis[1]);
		tempAxis = getDir(angle.x, angle.y + pi / 2);
		if (vec3f::dot(tempAxis, axis[1]) < 0)rVal.z *= -1;
		return rVal;
	}

	vec3f coordinateSystem::getInCoordinateSystem(vec3f realCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		vec3f rVal;
		realCoord = vec3f::subtract(realCoord, origin);
		rVal.x = vec3f::component(realCoord, axis[0]);
		rVal.y = vec3f::component(realCoord, axis[1]);
		rVal.z = vec3f::component(realCoord, axis[2]);
		vec3f Scale = scale;
		if (Scale.x == 0)Scale.x == 1;
		if (Scale.y == 0)Scale.y == 1;
		if (Scale.z == 0)Scale.z == 1;
		rVal.x /= Scale.x;
		rVal.y /= Scale.y;
		rVal.z /= Scale.z;
		return rVal;
	}

	vec3f coordinateSystem::getRealWorldCoordinates(vec3f CSCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		return vec3f::add(origin, vec3f::add(vec3f::multiply(axis[0], CSCoord.x), vec3f::add(vec3f::multiply(axis[1], CSCoord.y), vec3f::multiply(axis[2], CSCoord.z))));
	}



	//transform functions

	void transform::addVec(vec3f val, vec3f* adress) {
		data.push_back(CS.getInCoordinateSystem(val));
		dataAdress.push_back(adress);
	}

	void transform::update() {
		for (int i = 0; i < data.size(); ++i) {
			*(dataAdress[i]) = CS.getRealWorldCoordinates(data[i]);
		}
	}

	void transform::addRelativeRot(vec3f rot) {

		vec3f oldAxis[3];
		vec3f scale = CS.getScale();
		if (scale.x == 0)scale.x = 1;
		if (scale.y == 0)scale.y = 1;
		if (scale.z == 0)scale.z = 1;
		oldAxis[0] = vec3f::multiply(CS.getAxis()[0], 1 / scale.x);
		oldAxis[1] = vec3f::multiply(CS.getAxis()[1], 1 / scale.y);
		oldAxis[2] = vec3f::multiply(CS.getAxis()[2], 1 / scale.z);


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

		scale = CS.getScale();
		vec3f newAxis[3];
		newAxis[0] = vec3f::multiply(vec3f::add(vec3f::multiply(oldAxis[0], dir[0].x), vec3f::add(vec3f::multiply(oldAxis[1], dir[0].y), vec3f::multiply(oldAxis[2], dir[0].z))), scale.x);
		newAxis[1] = vec3f::multiply(vec3f::add(vec3f::multiply(oldAxis[0], dir[1].x), vec3f::add(vec3f::multiply(oldAxis[1], dir[1].y), vec3f::multiply(oldAxis[2], dir[1].z))), scale.y);
		newAxis[2] = vec3f::multiply(vec3f::add(vec3f::multiply(oldAxis[0], dir[2].x), vec3f::add(vec3f::multiply(oldAxis[1], dir[2].y), vec3f::multiply(oldAxis[2], dir[2].z))), scale.z);
		CS.setAxis(newAxis);
	}


	void transform::addRelativePos(vec3f pos) {
		vec3f Axis[3];
		vec3f scale = CS.getScale();
		if (scale.x == 0)scale.x = 1;
		if (scale.y == 0)scale.y = 1;
		if (scale.z == 0)scale.z = 1;
		Axis[0] = vec3f::multiply(CS.getAxis()[0], 1 / scale.x);
		Axis[1] = vec3f::multiply(CS.getAxis()[1], 1 / scale.y);
		Axis[2] = vec3f::multiply(CS.getAxis()[2], 1 / scale.z);

		CS.setOrigin(vec3f::add(CS.getOrigin(), vec3f::add(vec3f::multiply(Axis[0], pos.x), vec3f::add(vec3f::multiply(Axis[1], pos.y), vec3f::multiply(Axis[2], pos.z)))));
	}
}


namespace manipulation3dLD {

	long double toDeg(long double rad) { return rad * 180 / pi; }
	long double toRad(long double deg) { return deg * pi / 180; }

	vec3ld getDir(long double yaw, long double pitch) {
		vec3ld newX = vec3ld(cos(yaw), sin(yaw), 0);
		vec3ld rval = vec3ld::add(vec3ld::multiply(newX, cos(pitch)), vec3ld::multiply(vec3ld::vec3(0, 0, 1), sin(pitch)));
		return rval;
	}

	vec3ld getRotation(vec3ld a) {
		vec3ld rval;
		errCode = 0;
		if (a.mag2() == 0) {
			errCode = 1;
			return vec3ld(0, 0, 0);
		}
		rval.y = (pi / 2) - vec3ld::angle(a, vec3ld(0, 0, 1));
		rval.x = vec3ld::angle(vec3ld(1, 0, 0), vec3ld(a.x, a.y, 0));
		if (defaultErrCode) rval.x = 0;
		else if (a.y < 0)rval.x = 2 * pi - rval.x;

		return rval;
	}


	//coordinate system functions

	void coordinateSystem::setOrigin(vec3ld vec) { origin = vec; }
	void coordinateSystem::setAngle(vec3ld vec) { angle = vec; reset = true; }
	void coordinateSystem::setScale(vec3ld vec) { scale = vec; reset = true; }
	void coordinateSystem::setAxis(vec3ld* Axis) {
		axis[0] = Axis[0];
		axis[1] = Axis[1];
		axis[2] = Axis[2];
		//set scale
		scale = getScale(axis);

		//set angle
		angle = getAngle(axis);
	}

	vec3ld coordinateSystem::getOrigin() { return origin; }
	vec3ld coordinateSystem::getAngle() { return angle; }
	vec3ld coordinateSystem::getScale() { return scale; }
	vec3ld* coordinateSystem::getAxis() { if (reset)resetAxis(); reset = false; return axis; };

	coordinateSystem::coordinateSystem(vec3ld Origin, vec3ld Rot, vec3ld Scale) {
		origin = Origin;
		angle = Rot;
		scale = Scale;
		resetAxis();
	}

	void coordinateSystem::resetAxis() {
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

	void coordinateSystem::set(coordinateSystem& cs) {
		//set origin
		origin = cs.getOrigin();

		//set axis
		setAxis(cs.getAxis());
	}

	vec3ld coordinateSystem::getScale(vec3ld* axis) {
		return vec3ld(axis[0].mag(), axis[1].mag(), axis[2].mag());
	}

	vec3ld coordinateSystem::getAngle(vec3ld* axis) {
		vec3ld rVal;
		rVal = getRotation(axis[0]);
		vec3ld tempAxis = getDir(angle.x + pi / 2, 0);
		rVal.z = vec3ld::angle(tempAxis, axis[1]);
		tempAxis = getDir(angle.x, angle.y + pi / 2);
		if (vec3ld::dot(tempAxis, axis[1]) < 0)rVal.z *= -1;
		return rVal;
	}

	vec3ld coordinateSystem::getInCoordinateSystem(vec3ld realCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		vec3ld rVal;
		realCoord = vec3ld::subtract(realCoord, origin);
		rVal.x = vec3ld::component(realCoord, axis[0]);
		rVal.y = vec3ld::component(realCoord, axis[1]);
		rVal.z = vec3ld::component(realCoord, axis[2]);
		vec3ld Scale = scale;
		if (Scale.x == 0)Scale.x == 1;
		if (Scale.y == 0)Scale.y == 1;
		if (Scale.z == 0)Scale.z == 1;
		rVal.x /= Scale.x;
		rVal.y /= Scale.y;
		rVal.z /= Scale.z;
		return rVal;
	}

	vec3ld coordinateSystem::getRealWorldCoordinates(vec3ld CSCoord) {
		if (reset) {
			resetAxis();
			reset = false;
		}
		return vec3ld::add(origin, vec3ld::add(vec3ld::multiply(axis[0], CSCoord.x), vec3ld::add(vec3ld::multiply(axis[1], CSCoord.y), vec3ld::multiply(axis[2], CSCoord.z))));
	}



	//transform functions

	void transform::addVec(vec3ld val, vec3ld* adress) {
		data.push_back(CS.getInCoordinateSystem(val));
		dataAdress.push_back(adress);
	}

	void transform::update() {
		for (int i = 0; i < data.size(); ++i) {
			*(dataAdress[i]) = CS.getRealWorldCoordinates(data[i]);
		}
	}

	void transform::addRelativeRot(vec3ld rot) {

		vec3ld oldAxis[3];
		vec3ld scale = CS.getScale();
		if (scale.x == 0)scale.x = 1;
		if (scale.y == 0)scale.y = 1;
		if (scale.z == 0)scale.z = 1;
		oldAxis[0] = vec3ld::multiply(CS.getAxis()[0], 1 / scale.x);
		oldAxis[1] = vec3ld::multiply(CS.getAxis()[1], 1 / scale.y);
		oldAxis[2] = vec3ld::multiply(CS.getAxis()[2], 1 / scale.z);


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

		scale = CS.getScale();
		vec3ld newAxis[3];
		newAxis[0] = vec3ld::multiply(vec3ld::add(vec3ld::multiply(oldAxis[0], dir[0].x), vec3ld::add(vec3ld::multiply(oldAxis[1], dir[0].y), vec3ld::multiply(oldAxis[2], dir[0].z))), scale.x);
		newAxis[1] = vec3ld::multiply(vec3ld::add(vec3ld::multiply(oldAxis[0], dir[1].x), vec3ld::add(vec3ld::multiply(oldAxis[1], dir[1].y), vec3ld::multiply(oldAxis[2], dir[1].z))), scale.y);
		newAxis[2] = vec3ld::multiply(vec3ld::add(vec3ld::multiply(oldAxis[0], dir[2].x), vec3ld::add(vec3ld::multiply(oldAxis[1], dir[2].y), vec3ld::multiply(oldAxis[2], dir[2].z))), scale.z);
		CS.setAxis(newAxis);
	}


	void transform::addRelativePos(vec3ld pos) {
		vec3ld Axis[3];
		vec3ld scale = CS.getScale();
		if (scale.x == 0)scale.x = 1;
		if (scale.y == 0)scale.y = 1;
		if (scale.z == 0)scale.z = 1;
		Axis[0] = vec3ld::multiply(CS.getAxis()[0], 1 / scale.x);
		Axis[1] = vec3ld::multiply(CS.getAxis()[1], 1 / scale.y);
		Axis[2] = vec3ld::multiply(CS.getAxis()[2], 1 / scale.z);

		CS.setOrigin(vec3ld::add(CS.getOrigin(), vec3ld::add(vec3ld::multiply(Axis[0], pos.x), vec3ld::add(vec3ld::multiply(Axis[1], pos.y), vec3ld::multiply(Axis[2], pos.z)))));
	}
}