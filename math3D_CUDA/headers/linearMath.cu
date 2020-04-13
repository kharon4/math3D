#pragma once
#define INSIDE_linearMath_CU_FILE 1
#include "linearMath.cuh"

#include <math.h>

namespace linearMathD {//double precesion


	//line class
	__host__ __device__ line::line() {
		setRaw(vec3d(0, 0, 0), vec3d(1, 0, 0));
	}

	__host__ __device__ line::line(vec3d PT, vec3d DR) {
		setRaw_s(PT, DR);
	}

	__host__ __device__ bool line::set(vec3d PT, vec3d DR) {
		pt = PT;
		if (vec3d::isNUL(DR)) {
			dr = vec3d(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void line::setRaw_s(vec3d PT, vec3d DR) {
		pt = PT;
		if (vec3d::isNUL(DR)) dr = vec3d(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void line::setRaw(vec3d PT, vec3d DR) {
		pt = PT;
		dr = DR;
	}

	

	__host__ __device__ void line::setPT(vec3d PT) {
		pt = PT;
	}

	__host__ __device__ bool line::setDR(vec3d DR) {
		if (vec3d::isNUL(DR)) {
			dr = vec3d(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void line::setDRRaw_s(vec3d DR) {
		if (vec3d::isNUL(DR))dr = vec3d(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void line::setDRRaw(vec3d DR) {
		dr = DR;
	}

	__host__ __device__ vec3d line::getPt() { return pt; }
	__host__ __device__ vec3d line::getDr() { return dr; }

	//plane class
	__host__ __device__ plane::plane() {
		setRaw(vec3d(0,0,0),vec3d(1,0,0));
	}

	__host__ __device__ plane::plane(vec3d PT, vec3d DR) {
		setRaw_s(PT, DR);
	}

	__host__ __device__ bool plane::set(vec3d PT, vec3d DR) {
		pt = PT;
		if (vec3d::isNUL(DR)) {
			dr = vec3d(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}
	__host__ __device__ void plane::setRaw_s(vec3d PT, vec3d DR) {
		pt = PT;
		if (vec3d::isNUL(DR))dr = vec3d(1, 0, 0);
		else dr = DR;
	}
	__host__ __device__ void plane::setRaw(vec3d PT, vec3d DR) {
		pt = PT;
		dr = DR;
	}

	__host__ __device__ void plane::setPT(vec3d PT) {
		pt = PT;
	}

	__host__ __device__ bool plane::setDR(vec3d DR) {
		if (vec3d::isNUL(DR)) {
			dr = vec3d(1, 0, 0);
			return true;
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void plane::setDRRaw_s(vec3d DR) {
		if (vec3d::isNUL(DR))dr = vec3d(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void plane::setDRRaw(vec3d DR) {
		dr = DR;
	}
	__host__ __device__ vec3d plane::getPt() { return pt; }
	__host__ __device__ vec3d plane::getDr() { return dr; }



	//line functions

	__host__ __device__ bool getPt(line l, vec3d& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3d(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3d(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3d(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
		return 0;//no error
	}

	__host__ __device__ void getPtRaw_s(line l, vec3d& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3d(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3d(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3d(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
	}

	__host__ __device__ void getPtRaw(line l, vec3d& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
				coord = vec3d(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
				coord = vec3d(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
		}
		else {//z coordinate
			coord = vec3d(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
		}
	}

	__host__ __device__ char getPtIn(vec3d start, vec3d end, vec3d& coord, coordinateName coordGiven) {
		char rval = 0;//no error
		if (getPt(line(start, vec3d::subtract(end, start)), coord, coordGiven)) {
			rval = 1;// zero / infinite ans
		}
		else
		{
			double ttlDist = vec3d::subtract(start, end).mag2();
			if (!(vec3d::subtract(coord, start).mag2() <= ttlDist) || !(vec3d::subtract(coord, end).mag2() <= ttlDist)) {
				rval = 2; // out of bounds
			}
		}
		return rval;
	}


	//plane functions

	__host__ __device__ bool getPt(plane p, vec3d& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z == 0) {
				return 1;
			}
			else {
				(coord).z = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
				return 0;
			}
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x == 0) {
				return 1;
			}
			else {
				(coord).x = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
				return 0;
			}
		}
		else {
			if (p.getDr().y == 0) {
				return 1;
			}
			else {
				(coord).y = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
				return 0;
			}
		}
	}

	__host__ __device__ void getPtRaw_s(plane p, vec3d& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z != 0)
				(coord).z = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x != 0)
				(coord).x = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		}
		else {
			if (p.getDr().y == 0)
				(coord).y = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
		}
	}

	__host__ __device__ void getPtRaw(plane p, vec3d& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate)
				(coord).z = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		else if (coordToFind == coordinateName::xCoordinate)
				(coord).x = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		else
				(coord).y = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
	}


	//point to point functions

	__host__ __device__ double distance(vec3d p1, vec3d p2) {
		return vec3d::subtract(p1, p2).mag();
	}


	//point and line functions

	__host__ __device__ double distance(vec3d p, line l) {
		return vec3d::cross(l.getDr(), vec3d::subtract(p, l.getPt())).mag() / l.getDr().mag();
	}


	//point and plane functions

	__host__ __device__ double aDistance(vec3d pt, plane p) { // algebraic distance
		return (vec3d::dot(pt, p.getDr()) - vec3d::dot(p.getDr(), p.getPt())) / p.getDr().mag();
	}

	__host__ __device__ vec3d getMirrorImage(vec3d pt, plane pl) {
		//get component perpendicular to pl
		double Component = vec3d::componentRaw_s(vec3d::subtract(pt, pl.getPt()), pl.getDr());
		vec3d normal = pl.getDr();
		normal.normalize();
		return (vec3d::subtract(pt, vec3d::multiply(normal, 2 * Component)));
	}


	//line and line functions

	__host__ __device__ bool coplanar(line l, line m) {
		if (vec3d::dot(vec3d::subtract(l.getPt(), m.getPt()), vec3d::cross(l.getDr(), m.getDr())) == 0) {
			return 1;
		}
		else {
			return 0;
		}
	}

	__host__ __device__ double distance(line l, line m) {
		if (vec3d::isEqual(l.getDr(), m.getDr())) {
			return (vec3d::cross(vec3d::subtract(l.getPt(), m.getPt()), l.getDr()).mag() / l.getDr().mag());
		}
		else {
			vec3d temp = vec3d::cross(l.getDr(), m.getDr());
			return fabs(vec3d::dot(vec3d::subtract(l.getPt(), m.getPt()), temp) / temp.mag());
		}
	}


	//plane plane functions

	__host__ __device__ double distance(plane p1, plane p2) {
		if (vec3d::isEqual(p1.getDr(), p2.getDr())) {
			return fabs((vec3d::dot(p1.getDr(), p1.getPt()) - vec3d::dot(p2.getDr(), p2.getPt())) / p1.getDr().mag());
		}
		else {
			return 0;
		}
	}


	//line and plane functions

	__host__ __device__ vec3d intersection(line l, plane p, bool* error) {
		if (vec3d::dot(l.getDr(), p.getDr()) == 0) {
			*error = true;//error , no solution of infinite solutions
			return l.getPt();
		}
		else {
			*error = false;//no error
			double lambda;
			lambda = (vec3d::dot(p.getPt(), p.getDr()) - vec3d::dot(p.getDr(), l.getPt())) / vec3d::dot(p.getDr(), l.getDr());
			vec3d rVal = vec3d::add(l.getPt(), vec3d::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	__host__ __device__ vec3d intersectionRaw_s(line l, plane p) {
		if (vec3d::dot(l.getDr(), p.getDr()) == 0) {
			return l.getPt();
		}
		else {
			double lambda;
			lambda = (vec3d::dot(p.getPt(), p.getDr()) - vec3d::dot(p.getDr(), l.getPt())) / vec3d::dot(p.getDr(), l.getDr());
			vec3d rVal = vec3d::add(l.getPt(), vec3d::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	__host__ __device__ vec3d intersectionRaw(line l, plane p) {
			double lambda;
			lambda = (vec3d::dot(p.getPt(), p.getDr()) - vec3d::dot(p.getDr(), l.getPt())) / vec3d::dot(p.getDr(), l.getDr());
			vec3d rVal = vec3d::add(l.getPt(), vec3d::multiply(l.getDr(), lambda));
			return rVal;
	}


	//ray cast

	__host__ __device__ bool rayCast(line l, plane p, vec3d& intersection) {
		if (vec3d::dot(l.getDr(), p.getDr()) == 0) {
			intersection = l.getPt();
			return 1;// multiple or no results
		}
		else {
			double lambda;
			lambda = (vec3d::dot(p.getPt(), p.getDr()) - vec3d::dot(p.getDr(), l.getPt())) / vec3d::dot(p.getDr(), l.getDr());
			if (lambda < 0) {
				intersection = l.getPt();
				return 1;//no results
			}
			intersection = vec3d::add(l.getPt(), vec3d::multiply(l.getDr(), lambda));
		}
	}
}



namespace linearMathF {//single precesion


	//line class
	__host__ __device__ line::line() {
		setRaw(vec3f(0, 0, 0), vec3f(1, 0, 0));
	}

	__host__ __device__ line::line(vec3f PT, vec3f DR) {
		setRaw_s(PT, DR);
	}

	__host__ __device__ bool line::set(vec3f PT, vec3f DR) {
		pt = PT;
		if (vec3f::isNUL(DR)) {
			dr = vec3f(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void line::setRaw_s(vec3f PT, vec3f DR) {
		pt = PT;
		if (vec3f::isNUL(DR)) dr = vec3f(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void line::setRaw(vec3f PT, vec3f DR) {
		pt = PT;
		dr = DR;
	}



	__host__ __device__ void line::setPT(vec3f PT) {
		pt = PT;
	}

	__host__ __device__ bool line::setDR(vec3f DR) {
		if (vec3f::isNUL(DR)) {
			dr = vec3f(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void line::setDRRaw_s(vec3f DR) {
		if (vec3f::isNUL(DR))dr = vec3f(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void line::setDRRaw(vec3f DR) {
		dr = DR;
	}

	__host__ __device__ vec3f line::getPt() { return pt; }
	__host__ __device__ vec3f line::getDr() { return dr; }

	//plane class
	__host__ __device__ plane::plane() {
		setRaw(vec3f(0, 0, 0), vec3f(1, 0, 0));
	}

	__host__ __device__ plane::plane(vec3f PT, vec3f DR) {
		setRaw_s(PT, DR);
	}

	__host__ __device__ bool plane::set(vec3f PT, vec3f DR) {
		pt = PT;
		if (vec3f::isNUL(DR)) {
			dr = vec3f(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}
	__host__ __device__ void plane::setRaw_s(vec3f PT, vec3f DR) {
		pt = PT;
		if (vec3f::isNUL(DR))dr = vec3f(1, 0, 0);
		else dr = DR;
	}
	__host__ __device__ void plane::setRaw(vec3f PT, vec3f DR) {
		pt = PT;
		dr = DR;
	}

	__host__ __device__ void plane::setPT(vec3f PT) {
		pt = PT;
	}

	__host__ __device__ bool plane::setDR(vec3f DR) {
		if (vec3f::isNUL(DR)) {
			dr = vec3f(1, 0, 0);
			return true;
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void plane::setDRRaw_s(vec3f DR) {
		if (vec3f::isNUL(DR))dr = vec3f(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void plane::setDRRaw(vec3f DR) {
		dr = DR;
	}
	__host__ __device__ vec3f plane::getPt() { return pt; }
	__host__ __device__ vec3f plane::getDr() { return dr; }



	//line functions

	__host__ __device__ bool getPt(line l, vec3f& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3f(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3f(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3f(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
		return 0;//no error
	}

	__host__ __device__ void getPtRaw_s(line l, vec3f& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3f(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3f(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3f(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
	}

	__host__ __device__ void getPtRaw(line l, vec3f& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			coord = vec3f(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			coord = vec3f(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
		}
		else {//z coordinate
			coord = vec3f(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
		}
	}

	__host__ __device__ char getPtIn(vec3f start, vec3f end, vec3f& coord, coordinateName coordGiven) {
		char rval = 0;//no error
		if (getPt(line(start, vec3f::subtract(end, start)), coord, coordGiven)) {
			rval = 1;// zero / infinite ans
		}
		else
		{
			float ttlDist = vec3f::subtract(start, end).mag2();
			if (!(vec3f::subtract(coord, start).mag2() <= ttlDist) || !(vec3f::subtract(coord, end).mag2() <= ttlDist)) {
				rval = 2; // out of bounds
			}
		}
		return rval;
	}


	//plane functions

	__host__ __device__ bool getPt(plane p, vec3f& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z == 0) {
				return 1;
			}
			else {
				(coord).z = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
				return 0;
			}
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x == 0) {
				return 1;
			}
			else {
				(coord).x = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
				return 0;
			}
		}
		else {
			if (p.getDr().y == 0) {
				return 1;
			}
			else {
				(coord).y = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
				return 0;
			}
		}
	}

	__host__ __device__ void getPtRaw_s(plane p, vec3f& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z != 0)
				(coord).z = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x != 0)
				(coord).x = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		}
		else {
			if (p.getDr().y == 0)
				(coord).y = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
		}
	}

	__host__ __device__ void getPtRaw(plane p, vec3f& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate)
			(coord).z = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		else if (coordToFind == coordinateName::xCoordinate)
			(coord).x = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		else
			(coord).y = (vec3f::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
	}


	//point to point functions

	__host__ __device__ float distance(vec3f p1, vec3f p2) {
		return vec3f::subtract(p1, p2).mag();
	}


	//point and line functions

	__host__ __device__ float distance(vec3f p, line l) {
		return vec3f::cross(l.getDr(), vec3f::subtract(p, l.getPt())).mag() / l.getDr().mag();
	}


	//point and plane functions

	__host__ __device__ float aDistance(vec3f pt, plane p) { // algebraic distance
		return (vec3f::dot(pt, p.getDr()) - vec3f::dot(p.getDr(), p.getPt())) / p.getDr().mag();
	}

	__host__ __device__ vec3f getMirrorImage(vec3f pt, plane pl) {
		//get component perpendicular to pl
		float Component = vec3f::componentRaw_s(vec3f::subtract(pt, pl.getPt()), pl.getDr());
		vec3f normal = pl.getDr();
		normal.normalize();
		return (vec3f::subtract(pt, vec3f::multiply(normal, 2 * Component)));
	}


	//line and line functions

	__host__ __device__ bool coplanar(line l, line m) {
		if (vec3f::dot(vec3f::subtract(l.getPt(), m.getPt()), vec3f::cross(l.getDr(), m.getDr())) == 0) {
			return 1;
		}
		else {
			return 0;
		}
	}

	__host__ __device__ float distance(line l, line m) {
		if (vec3f::isEqual(l.getDr(), m.getDr())) {
			return (vec3f::cross(vec3f::subtract(l.getPt(), m.getPt()), l.getDr()).mag() / l.getDr().mag());
		}
		else {
			vec3f temp = vec3f::cross(l.getDr(), m.getDr());
			return fabs(vec3f::dot(vec3f::subtract(l.getPt(), m.getPt()), temp) / temp.mag());
		}
	}


	//plane plane functions

	__host__ __device__ float distance(plane p1, plane p2) {
		if (vec3f::isEqual(p1.getDr(), p2.getDr())) {
			return fabs((vec3f::dot(p1.getDr(), p1.getPt()) - vec3f::dot(p2.getDr(), p2.getPt())) / p1.getDr().mag());
		}
		else {
			return 0;
		}
	}


	//line and plane functions

	__host__ __device__ vec3f intersection(line l, plane p, bool* error) {
		if (vec3f::dot(l.getDr(), p.getDr()) == 0) {
			*error = true;//error , no solution of infinite solutions
			return l.getPt();
		}
		else {
			*error = false;//no error
			float lambda;
			lambda = (vec3f::dot(p.getPt(), p.getDr()) - vec3f::dot(p.getDr(), l.getPt())) / vec3f::dot(p.getDr(), l.getDr());
			vec3f rVal = vec3f::add(l.getPt(), vec3f::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	__host__ __device__ vec3f intersectionRaw_s(line l, plane p) {
		if (vec3f::dot(l.getDr(), p.getDr()) == 0) {
			return l.getPt();
		}
		else {
			float lambda;
			lambda = (vec3f::dot(p.getPt(), p.getDr()) - vec3f::dot(p.getDr(), l.getPt())) / vec3f::dot(p.getDr(), l.getDr());
			vec3f rVal = vec3f::add(l.getPt(), vec3f::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	__host__ __device__ vec3f intersectionRaw(line l, plane p) {
		float lambda;
		lambda = (vec3f::dot(p.getPt(), p.getDr()) - vec3f::dot(p.getDr(), l.getPt())) / vec3f::dot(p.getDr(), l.getDr());
		vec3f rVal = vec3f::add(l.getPt(), vec3f::multiply(l.getDr(), lambda));
		return rVal;
	}


	//ray cast

	__host__ __device__ bool rayCast(line l, plane p, vec3f& intersection) {
		if (vec3f::dot(l.getDr(), p.getDr()) == 0) {
			intersection = l.getPt();
			return 1;// multiple or no results
		}
		else {
			float lambda;
			lambda = (vec3f::dot(p.getPt(), p.getDr()) - vec3f::dot(p.getDr(), l.getPt())) / vec3f::dot(p.getDr(), l.getDr());
			if (lambda < 0) {
				intersection = l.getPt();
				return 1;//no results
			}
			intersection = vec3f::add(l.getPt(), vec3f::multiply(l.getDr(), lambda));
		}
	}
}



namespace linearMathLD {//long double precesion


	//line class
	__host__ __device__ line::line() {
		setRaw(vec3ld(0, 0, 0), vec3ld(1, 0, 0));
	}

	__host__ __device__ line::line(vec3ld PT, vec3ld DR) {
		setRaw_s(PT, DR);
	}

	__host__ __device__ bool line::set(vec3ld PT, vec3ld DR) {
		pt = PT;
		if (vec3ld::isNUL(DR)) {
			dr = vec3ld(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void line::setRaw_s(vec3ld PT, vec3ld DR) {
		pt = PT;
		if (vec3ld::isNUL(DR)) dr = vec3ld(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void line::setRaw(vec3ld PT, vec3ld DR) {
		pt = PT;
		dr = DR;
	}



	__host__ __device__ void line::setPT(vec3ld PT) {
		pt = PT;
	}

	__host__ __device__ bool line::setDR(vec3ld DR) {
		if (vec3ld::isNUL(DR)) {
			dr = vec3ld(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void line::setDRRaw_s(vec3ld DR) {
		if (vec3ld::isNUL(DR))dr = vec3ld(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void line::setDRRaw(vec3ld DR) {
		dr = DR;
	}

	__host__ __device__ vec3ld line::getPt() { return pt; }
	__host__ __device__ vec3ld line::getDr() { return dr; }

	//plane class
	__host__ __device__ plane::plane() {
		setRaw(vec3ld(0, 0, 0), vec3ld(1, 0, 0));
	}

	__host__ __device__ plane::plane(vec3ld PT, vec3ld DR) {
		setRaw_s(PT, DR);
	}

	__host__ __device__ bool plane::set(vec3ld PT, vec3ld DR) {
		pt = PT;
		if (vec3ld::isNUL(DR)) {
			dr = vec3ld(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}
	__host__ __device__ void plane::setRaw_s(vec3ld PT, vec3ld DR) {
		pt = PT;
		if (vec3ld::isNUL(DR))dr = vec3ld(1, 0, 0);
		else dr = DR;
	}
	__host__ __device__ void plane::setRaw(vec3ld PT, vec3ld DR) {
		pt = PT;
		dr = DR;
	}

	__host__ __device__ void plane::setPT(vec3ld PT) {
		pt = PT;
	}

	__host__ __device__ bool plane::setDR(vec3ld DR) {
		if (vec3ld::isNUL(DR)) {
			dr = vec3ld(1, 0, 0);
			return true;
		}
		dr = DR;
		return false;
	}

	__host__ __device__ void plane::setDRRaw_s(vec3ld DR) {
		if (vec3ld::isNUL(DR))dr = vec3ld(1, 0, 0);
		else dr = DR;
	}

	__host__ __device__ void plane::setDRRaw(vec3ld DR) {
		dr = DR;
	}
	__host__ __device__ vec3ld plane::getPt() { return pt; }
	__host__ __device__ vec3ld plane::getDr() { return dr; }



	//line functions

	__host__ __device__ bool getPt(line l, vec3ld& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3ld(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3ld(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3ld(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
		return 0;//no error
	}

	__host__ __device__ void getPtRaw_s(line l, vec3ld& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3ld(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3ld(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3ld(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
	}

	__host__ __device__ void getPtRaw(line l, vec3ld& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			coord = vec3ld(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			coord = vec3ld(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
		}
		else {//z coordinate
			coord = vec3ld(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
		}
	}

	__host__ __device__ char getPtIn(vec3ld start, vec3ld end, vec3ld& coord, coordinateName coordGiven) {
		char rval = 0;//no error
		if (getPt(line(start, vec3ld::subtract(end, start)), coord, coordGiven)) {
			rval = 1;// zero / infinite ans
		}
		else
		{
			long double ttlDist = vec3ld::subtract(start, end).mag2();
			if (!(vec3ld::subtract(coord, start).mag2() <= ttlDist) || !(vec3ld::subtract(coord, end).mag2() <= ttlDist)) {
				rval = 2; // out of bounds
			}
		}
		return rval;
	}


	//plane functions

	__host__ __device__ bool getPt(plane p, vec3ld& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z == 0) {
				return 1;
			}
			else {
				(coord).z = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
				return 0;
			}
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x == 0) {
				return 1;
			}
			else {
				(coord).x = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
				return 0;
			}
		}
		else {
			if (p.getDr().y == 0) {
				return 1;
			}
			else {
				(coord).y = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
				return 0;
			}
		}
	}

	__host__ __device__ void getPtRaw_s(plane p, vec3ld& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z != 0)
				(coord).z = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x != 0)
				(coord).x = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		}
		else {
			if (p.getDr().y == 0)
				(coord).y = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
		}
	}

	__host__ __device__ void getPtRaw(plane p, vec3ld& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate)
			(coord).z = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		else if (coordToFind == coordinateName::xCoordinate)
			(coord).x = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		else
			(coord).y = (vec3ld::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
	}


	//point to point functions

	__host__ __device__ long double distance(vec3ld p1, vec3ld p2) {
		return vec3ld::subtract(p1, p2).mag();
	}


	//point and line functions

	__host__ __device__ long double distance(vec3ld p, line l) {
		return vec3ld::cross(l.getDr(), vec3ld::subtract(p, l.getPt())).mag() / l.getDr().mag();
	}


	//point and plane functions

	__host__ __device__ long double aDistance(vec3ld pt, plane p) { // algebraic distance
		return (vec3ld::dot(pt, p.getDr()) - vec3ld::dot(p.getDr(), p.getPt())) / p.getDr().mag();
	}

	__host__ __device__ vec3ld getMirrorImage(vec3ld pt, plane pl) {
		//get component perpendicular to pl
		long double Component = vec3ld::componentRaw_s(vec3ld::subtract(pt, pl.getPt()), pl.getDr());
		vec3ld normal = pl.getDr();
		normal.normalize();
		return (vec3ld::subtract(pt, vec3ld::multiply(normal, 2 * Component)));
	}


	//line and line functions

	__host__ __device__ bool coplanar(line l, line m) {
		if (vec3ld::dot(vec3ld::subtract(l.getPt(), m.getPt()), vec3ld::cross(l.getDr(), m.getDr())) == 0) {
			return 1;
		}
		else {
			return 0;
		}
	}

	__host__ __device__ long double distance(line l, line m) {
		if (vec3ld::isEqual(l.getDr(), m.getDr())) {
			return (vec3ld::cross(vec3ld::subtract(l.getPt(), m.getPt()), l.getDr()).mag() / l.getDr().mag());
		}
		else {
			vec3ld temp = vec3ld::cross(l.getDr(), m.getDr());
			return fabs(vec3ld::dot(vec3ld::subtract(l.getPt(), m.getPt()), temp) / temp.mag());
		}
	}


	//plane plane functions

	__host__ __device__ long double distance(plane p1, plane p2) {
		if (vec3ld::isEqual(p1.getDr(), p2.getDr())) {
			return fabs((vec3ld::dot(p1.getDr(), p1.getPt()) - vec3ld::dot(p2.getDr(), p2.getPt())) / p1.getDr().mag());
		}
		else {
			return 0;
		}
	}


	//line and plane functions

	__host__ __device__ vec3ld intersection(line l, plane p, bool* error) {
		if (vec3ld::dot(l.getDr(), p.getDr()) == 0) {
			*error = true;//error , no solution of infinite solutions
			return l.getPt();
		}
		else {
			*error = false;//no error
			long double lambda;
			lambda = (vec3ld::dot(p.getPt(), p.getDr()) - vec3ld::dot(p.getDr(), l.getPt())) / vec3ld::dot(p.getDr(), l.getDr());
			vec3ld rVal = vec3ld::add(l.getPt(), vec3ld::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	__host__ __device__ vec3ld intersectionRaw_s(line l, plane p) {
		if (vec3ld::dot(l.getDr(), p.getDr()) == 0) {
			return l.getPt();
		}
		else {
			long double lambda;
			lambda = (vec3ld::dot(p.getPt(), p.getDr()) - vec3ld::dot(p.getDr(), l.getPt())) / vec3ld::dot(p.getDr(), l.getDr());
			vec3ld rVal = vec3ld::add(l.getPt(), vec3ld::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	__host__ __device__ vec3ld intersectionRaw(line l, plane p) {
		long double lambda;
		lambda = (vec3ld::dot(p.getPt(), p.getDr()) - vec3ld::dot(p.getDr(), l.getPt())) / vec3ld::dot(p.getDr(), l.getDr());
		vec3ld rVal = vec3ld::add(l.getPt(), vec3ld::multiply(l.getDr(), lambda));
		return rVal;
	}


	//ray cast

	__host__ __device__ bool rayCast(line l, plane p, vec3ld& intersection) {
		if (vec3ld::dot(l.getDr(), p.getDr()) == 0) {
			intersection = l.getPt();
			return 1;// multiple or no results
		}
		else {
			long double lambda;
			lambda = (vec3ld::dot(p.getPt(), p.getDr()) - vec3ld::dot(p.getDr(), l.getPt())) / vec3ld::dot(p.getDr(), l.getDr());
			if (lambda < 0) {
				intersection = l.getPt();
				return 1;//no results
			}
			intersection = vec3ld::add(l.getPt(), vec3ld::multiply(l.getDr(), lambda));
		}
	}
}