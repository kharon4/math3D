#pragma once
//deals with planes , lines and points


#include "vec3.h"

enum class coordinateName
{
	xCoordinate = 0,
	yCoordinate = 1,
	zCoordinate = 2
};

namespace linearMathD {//double precesion
	static int errCode = 0;

	class line {
	public:
		line(vec3d PT = vec3d(0, 0, 0), vec3d DR = vec3d(1, 0, 0));

		void set(vec3d PT, vec3d DR);
		void setPT(vec3d PT);
		void setDR(vec3d DR);
		vec3d getPt();
		vec3d getDr();

	private:
		vec3d dr;
		vec3d pt;
	};

	class plane
	{
	public:
		plane(vec3d PT = vec3d(0, 0, 0), vec3d DR = vec3d(0, 0, 1));

		void set(vec3d PT, vec3d DR);
		void setPT(vec3d PT);
		void setDR(vec3d DR);

		vec3d getPt();
		vec3d getDr();

	private:
		vec3d pt;
		vec3d dr;

	};


	//line functions
	vec3d getPt(line l, double coord, coordinateName coordGiven = coordinateName::xCoordinate);
	char getPtIn(vec3d start, vec3d end, double coord, coordinateName coordGiven, vec3d* ans);


	//plane functions
	bool getPt(plane p, vec3d* coord, coordinateName coordToFind = coordinateName::xCoordinate);

	//point to point functions
	double distance(vec3d p1, vec3d p2);
	vec3d getMirrorImage(vec3d pt, plane pl);

	//point and line functions
	double distance(vec3d p, line l);

	//point and plane functions
	double aDistance(vec3d pt, plane p);

	//line and line functions
	bool coplanar(line l, line m);
	double distance(line l, line m);


	//plane plane functions
	double distance(plane p1, plane p2);

	//line and plane functions
	vec3d intersection(line l, plane p);

	//ray cast
	vec3d rayCast(line l, plane p);
}