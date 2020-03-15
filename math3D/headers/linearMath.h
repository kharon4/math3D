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

namespace linearMathF {//single precesion
	static int errCode = 0;

	class line {
	public:
		line(vec3f PT = vec3f(0, 0, 0), vec3f DR = vec3f(1, 0, 0));

		void set(vec3f PT, vec3f DR);
		void setPT(vec3f PT);
		void setDR(vec3f DR);
		vec3f getPt();
		vec3f getDr();

	private:
		vec3f dr;
		vec3f pt;
	};

	class plane
	{
	public:
		plane(vec3f PT = vec3f(0, 0, 0), vec3f DR = vec3f(0, 0, 1));

		void set(vec3f PT, vec3f DR);
		void setPT(vec3f PT);
		void setDR(vec3f DR);

		vec3f getPt();
		vec3f getDr();

	private:
		vec3f pt;
		vec3f dr;

	};


	//line functions
	vec3f getPt(line l, double coord, coordinateName coordGiven = coordinateName::xCoordinate);
	char getPtIn(vec3f start, vec3f end, double coord, coordinateName coordGiven, vec3f* ans);


	//plane functions
	bool getPt(plane p, vec3f* coord, coordinateName coordToFind = coordinateName::xCoordinate);

	//point to point functions
	double distance(vec3f p1, vec3f p2);
	vec3f getMirrorImage(vec3f pt, plane pl);

	//point and line functions
	double distance(vec3f p, line l);

	//point and plane functions
	double aDistance(vec3f pt, plane p);

	//line and line functions
	bool coplanar(line l, line m);
	double distance(line l, line m);


	//plane plane functions
	double distance(plane p1, plane p2);

	//line and plane functions
	vec3f intersection(line l, plane p);

	//ray cast
	vec3f rayCast(line l, plane p);
}

namespace linearMathLD {//long double precesion
	static int errCode = 0;

	class line {
	public:
		line(vec3ld PT = vec3ld(0, 0, 0), vec3ld DR = vec3ld(1, 0, 0));

		void set(vec3ld PT, vec3ld DR);
		void setPT(vec3ld PT);
		void setDR(vec3ld DR);
		vec3ld getPt();
		vec3ld getDr();

	private:
		vec3ld dr;
		vec3ld pt;
	};

	class plane
	{
	public:
		plane(vec3ld PT = vec3ld(0, 0, 0), vec3ld DR = vec3ld(0, 0, 1));

		void set(vec3ld PT, vec3ld DR);
		void setPT(vec3ld PT);
		void setDR(vec3ld DR);

		vec3ld getPt();
		vec3ld getDr();

	private:
		vec3ld pt;
		vec3ld dr;

	};


	//line functions
	vec3ld getPt(line l, double coord, coordinateName coordGiven = coordinateName::xCoordinate);
	char getPtIn(vec3ld start, vec3ld end, double coord, coordinateName coordGiven, vec3ld* ans);


	//plane functions
	bool getPt(plane p, vec3ld* coord, coordinateName coordToFind = coordinateName::xCoordinate);

	//point to point functions
	double distance(vec3ld p1, vec3ld p2);
	vec3ld getMirrorImage(vec3ld pt, plane pl);

	//point and line functions
	double distance(vec3ld p, line l);

	//point and plane functions
	double aDistance(vec3ld pt, plane p);

	//line and line functions
	bool coplanar(line l, line m);
	double distance(line l, line m);


	//plane plane functions
	double distance(plane p1, plane p2);

	//line and plane functions
	vec3ld intersection(line l, plane p);

	//ray cast
	vec3ld rayCast(line l, plane p);
}