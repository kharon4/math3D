#include "./../headers/linearMath.h"
#include <math.h>

namespace linearMathD {//double precesion
	

	//line class
	line::line(vec3d PT, vec3d DR) {
		set(PT, DR);
	}

	void line::set(vec3d PT, vec3d DR) {
		pt = PT;
		errCode = 0;
		if (vec3d::isNUL(DR)) {
			errCode = 1;
			dr = vec3d(1, 0, 0);
		}
		else {
			dr = DR;
		}
	}

	void line::setPT(vec3d PT) {
		pt = PT;
	}

	void line::setDR(vec3d DR) {
		errCode = 0;
		if (vec3d::isNUL(DR)) {
			errCode = 1;
			dr = vec3d(1, 0, 0);
		}
		else {
			dr = DR;
		}
	}

	vec3d line::getPt() { return pt; }
	vec3d line::getDr() { return dr; }

	//plane class
	plane::plane(vec3d PT , vec3d DR) {
		set(PT, DR);
	}

	void plane::set(vec3d PT, vec3d DR) {
		pt = PT;
		errCode = 0;
		if (vec3d::isNUL(DR)) {
			errCode = 1;
			dr = vec3d(1, 0, 0);
		}
		else {
			dr = DR;
		}
	}

	void plane::setPT(vec3d PT) {
		pt = PT;
	}
	void plane::setDR(vec3d DR) {
		errCode = 0;
		if (vec3d::isNUL(DR)) {
			errCode = 1;
			dr = vec3d(1, 0, 0);
		}
		else {
			dr = DR;
		}
	}
	vec3d plane::getPt() { return pt; }
	vec3d plane::getDr() { return dr; }



	//line functions

	vec3d getPt(line l, double coord, coordinateName coordGiven) {
		errCode = 0;
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				errCode = 1;
				return (l.getPt());
			}
			else {
				return vec3d(coord, l.getPt().y + l.getDr().y * (coord - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				errCode = 1;
				return (l.getPt());
			}
			else {
				return vec3d(l.getPt().x + l.getDr().x * (coord - l.getPt().y) / l.getDr().y, coord, l.getPt().z + l.getDr().z * (coord - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				errCode = 1;
				return (l.getPt());
			}
			else {
				return vec3d(l.getPt().x + l.getDr().x * (coord - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord - l.getPt().z) / l.getDr().z, coord);
			}
		}
	}

	char getPtIn(vec3d start, vec3d end, double coord, coordinateName coordGiven, vec3d* ans){
		char rval = 0;
		*ans = getPt(line(start, vec3d::subtract(end, start)), coord, coordGiven);
		if (errCode == 1) {
			rval = 1;
		}
		else
		{
			double ttlDist = vec3d::subtract(start, end).mag();
			if (!(vec3d::subtract(*ans, start).mag() <= ttlDist) || !(vec3d::subtract(*ans, end).mag() <= ttlDist)) {
				rval = 2;
			}

		}
		return rval;
	}
	
	
	//plane functions

	bool getPt(plane p, vec3d* coord, coordinateName coordToFind){
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z == 0) {
				return 1;
			}
			else {
				(*coord).z = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (*coord).x + p.getDr().y * (*coord).y)) / p.getDr().z;
				return 0;
			}
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x == 0) {
				return 1;
			}
			else {
				(*coord).x = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().z * (*coord).z + p.getDr().y * (*coord).y)) / p.getDr().x;
				return 0;
			}
		}
		else {
			if (p.getDr().y == 0) {
				return 1;
			}
			else {
				(*coord).y = (vec3d::dot(p.getPt(), p.getDr()) - (p.getDr().x * (*coord).x + p.getDr().z * (*coord).z)) / p.getDr().y;
				return 0;
			}
		}
	}


	//point to point functions
	
	double distance(vec3d p1, vec3d p2) {
		return vec3d::subtract(p1, p2).mag();
	}

	
	//point and line functions

	double distance(vec3d p, line l) {
		return vec3d::cross(l.getDr(), vec3d::subtract(p, l.getPt())).mag() / l.getDr().mag();
	}


	//point and plane functions
	
	double aDistance(vec3d pt, plane p) { // algebraic distance
		return (vec3d::dot(pt, p.getDr()) - vec3d::dot(p.getDr(), p.getPt())) / p.getDr().mag();
	}

	vec3d getMirrorImage(vec3d pt, plane pl) {
		//get component perpendicular to pl
		float Component = vec3d::component(vec3d::subtract(pt, pl.getPt()), pl.getDr());
		vec3d normal = pl.getDr();
		normal.normalize();
		return (vec3d::subtract(pt, vec3d::multiply(normal, 2 * Component)));
	}

	
	//line and line functions

	bool coplanar(line l, line m) {
		if (vec3d::dot(vec3d::subtract(l.getPt(), m.getPt()), vec3d::cross(l.getDr(), m.getDr())) == 0) {
			return 1;
		}
		else {
			return 0;
		}
	}

	double distance(line l, line m) {
		if (vec3d::isEqual(l.getDr(), m.getDr())) {
			return (vec3d::cross(vec3d::subtract(l.getPt(), m.getPt()), l.getDr()).mag() / l.getDr().mag());
		}
		else {
			vec3d temp = vec3d::cross(l.getDr(), m.getDr());
			return fabs(vec3d::dot(vec3d::subtract(l.getPt(), m.getPt()), temp) / temp.mag());
		}
	}


	//plane plane functions
	
	double distance(plane p1, plane p2) {
		if (vec3d::isEqual(p1.getDr(), p2.getDr())) {
			return fabs((vec3d::dot(p1.getDr(), p1.getPt()) - vec3d::dot(p2.getDr(), p2.getPt())) / p1.getDr().mag());
		}
		else {
			return 0;
		}
	}


	//line and plane functions

	vec3d intersection(line l, plane p) {
		errCode = 0;
		if (vec3d::dot(l.getDr(), p.getDr()) == 0) {
			errCode = 1;
			return l.getPt();
		}
		else {
			double lambda;
			lambda = (vec3d::dot(p.getPt(), p.getDr()) - vec3d::dot(p.getDr(), l.getPt())) / vec3d::dot(p.getDr(), l.getDr());
			vec3d rVal = vec3d::add(l.getPt(), vec3d::multiply(l.getDr(), lambda));
			return rVal;
		}
	}


	//ray cast
	
	vec3d rayCast(line l, plane p) {
		errCode = 0;
		if (vec3d::dot(l.getDr(), p.getDr()) == 0) {
			errCode = 1;
			return l.getPt();
		}
		else {
			double lambda;
			lambda = (vec3d::dot(p.getPt(), p.getDr()) - vec3d::dot(p.getDr(), l.getPt())) / vec3d::dot(p.getDr(), l.getDr());
			if (lambda < 0) {
				errCode = 1;
			}
			vec3d rVal = vec3d::add(l.getPt(), vec3d::multiply(l.getDr(), lambda));
			return rVal;
		}
	}
}