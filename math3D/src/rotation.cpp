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


	vec3d coordinateSystem::getInCoordinateSystem(vec3d realCoord) {
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
		return vec3d::add(origin, vec3d::add(vec3d::multiply(axis[0], CSCoord.x), vec3d::add(vec3d::multiply(axis[1], CSCoord.y), vec3d::multiply(axis[2], CSCoord.z))));
	}


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
		oldAxis[0] = getDir(CS.angle.x, CS.angle.y);
		oldAxis[1] = getDir(CS.angle.x + pi / 2, 0/*CS.angle.y*/);
		oldAxis[2] = getDir(CS.angle.x, CS.angle.y + pi / 2);
		{
			vec3d temp[2];
			temp[0] = vec3d::add(vec3d::multiply(oldAxis[1], cos(CS.angle.z)), vec3d::multiply(oldAxis[2], sin(CS.angle.z)));
			temp[1] = vec3d::cross(oldAxis[0], temp[1]);
			oldAxis[1] = temp[0];
			oldAxis[2] = temp[1];
		}

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

		CS.axis[0] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[0].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[0].y), vec3d::multiply(oldAxis[2], dir[0].z))), CS.scale.x);
		CS.axis[1] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[1].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[1].y), vec3d::multiply(oldAxis[2], dir[1].z))), CS.scale.y);
		CS.axis[2] = vec3d::multiply(vec3d::add(vec3d::multiply(oldAxis[0], dir[2].x), vec3d::add(vec3d::multiply(oldAxis[1], dir[2].y), vec3d::multiply(oldAxis[2], dir[2].z))), CS.scale.z);

	}


	void transform::addRelativePos(vec3d pos) {
		vec3d Axis[3];
		Axis[0] = getDir(CS.angle.x, CS.angle.y);
		Axis[1] = getDir(CS.angle.x + pi / 2, 0);
		Axis[2] = getDir(CS.angle.x, CS.angle.y + pi / 2);
		{
			vec3d temp[2];
			temp[0] = vec3d::add(vec3d::multiply(Axis[1], cos(CS.angle.z)), vec3d::multiply(Axis[2], sin(CS.angle.z)));
			temp[1] = vec3d::cross(Axis[0], temp[0]);
			Axis[1] = temp[0];
			Axis[2] = temp[1];
		}
		CS.origin = vec3d::add(CS.origin, vec3d::add(vec3d::multiply(Axis[0], pos.x), vec3d::add(vec3d::multiply(Axis[1], pos.y), vec3d::multiply(Axis[2], pos.z))));
	}
}