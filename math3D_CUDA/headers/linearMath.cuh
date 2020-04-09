#pragma once

#include "vec3.cuh"

enum class coordinateName
{
	xCoordinate = 0,
	yCoordinate = 1,
	zCoordinate = 2
};

namespace linearMathD {//double precesion

	class line {
	public:


		__host__ __device__ line();
		__host__ __device__ line(vec3d PT , vec3d DR);

		__host__ __device__ bool set(vec3d PT, vec3d DR);
		__host__ __device__ void setRaw_s(vec3d PT, vec3d DR);
		__host__ __device__ void setRaw(vec3d PT, vec3d DR);
		__host__ __device__ void setPT(vec3d PT);
		__host__ __device__ bool setDR(vec3d DR);
		__host__ __device__ void setDRRaw_s(vec3d DR);
		__host__ __device__ void setDRRaw(vec3d DR);
		__host__ __device__ vec3d getPt();
		__host__ __device__ vec3d getDr();

	private:
		vec3d dr;
		vec3d pt;
	};

	class plane
	{
	public:
		__host__ __device__ plane();
		__host__ __device__ plane(vec3d PT, vec3d DR);

		__host__ __device__ bool set(vec3d PT, vec3d DR);
		__host__ __device__ void setRaw_s(vec3d PT, vec3d DR);
		__host__ __device__ void setRaw(vec3d PT, vec3d DR);
		__host__ __device__ void setPT(vec3d PT);
		__host__ __device__ bool setDR(vec3d DR);
		__host__ __device__ void setDRRaw_s(vec3d DR);
		__host__ __device__ void setDRRaw(vec3d DR);

		__host__ __device__ vec3d getPt();
		__host__ __device__ vec3d getDr();

	private:
		vec3d pt;
		vec3d dr;

	};


	//line functions
	__host__ __device__ bool getPt(line l, vec3d& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw_s(line l, vec3d& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw(line l, vec3d& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	
	__host__ __device__ char getPtIn(vec3d start, vec3d end, vec3d& coord, coordinateName coordGiven);


	//plane functions
	__host__ __device__ bool getPt(plane p, vec3d& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw_s(plane p, vec3d& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw(plane p, vec3d& coord, coordinateName coordToFind = coordinateName::xCoordinate);

	//point to point functions
	__host__ __device__ double distance(vec3d p1, vec3d p2);
	__host__ __device__ vec3d getMirrorImage(vec3d pt, plane pl);

	//point and line functions
	__host__ __device__ double distance(vec3d p, line l);

	//point and plane functions
	__host__ __device__ double aDistance(vec3d pt, plane p);

	//line and line functions
	__host__ __device__ bool coplanar(line l, line m);
	__host__ __device__ double distance(line l, line m);


	//plane plane functions
	__host__ __device__ double distance(plane p1, plane p2);

	//line and plane functions
	__host__ __device__ vec3d intersection(line l, plane p, bool* error);
	__host__ __device__ vec3d intersectionRaw_s(line l, plane p);
	__host__ __device__ vec3d intersectionRaw(line l, plane p);

	//ray cast
	__host__ __device__ bool rayCast(line l, plane p, vec3d& intersection);
}

namespace linearMathF {//single precesion

	class line {
	public:


		__host__ __device__ line();
		__host__ __device__ line(vec3f PT, vec3f DR);

		__host__ __device__ bool set(vec3f PT, vec3f DR);
		__host__ __device__ void setRaw_s(vec3f PT, vec3f DR);
		__host__ __device__ void setRaw(vec3f PT, vec3f DR);
		__host__ __device__ void setPT(vec3f PT);
		__host__ __device__ bool setDR(vec3f DR);
		__host__ __device__ void setDRRaw_s(vec3f DR);
		__host__ __device__ void setDRRaw(vec3f DR);
		__host__ __device__ vec3f getPt();
		__host__ __device__ vec3f getDr();

	private:
		vec3f dr;
		vec3f pt;
	};

	class plane
	{
	public:
		__host__ __device__ plane();
		__host__ __device__ plane(vec3f PT, vec3f DR);

		__host__ __device__ bool set(vec3f PT, vec3f DR);
		__host__ __device__ void setRaw_s(vec3f PT, vec3f DR);
		__host__ __device__ void setRaw(vec3f PT, vec3f DR);
		__host__ __device__ void setPT(vec3f PT);
		__host__ __device__ bool setDR(vec3f DR);
		__host__ __device__ void setDRRaw_s(vec3f DR);
		__host__ __device__ void setDRRaw(vec3f DR);

		__host__ __device__ vec3f getPt();
		__host__ __device__ vec3f getDr();

	private:
		vec3f pt;
		vec3f dr;

	};


	//line functions
	__host__ __device__ bool getPt(line l, vec3f& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw_s(line l, vec3f& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw(line l, vec3f& coord, coordinateName coordGiven = coordinateName::xCoordinate);

	__host__ __device__ char getPtIn(vec3f start, vec3f end, vec3f& coord, coordinateName coordGiven);


	//plane functions
	__host__ __device__ bool getPt(plane p, vec3f& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw_s(plane p, vec3f& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw(plane p, vec3f& coord, coordinateName coordToFind = coordinateName::xCoordinate);

	//point to point functions
	__host__ __device__ float distance(vec3f p1, vec3f p2);
	__host__ __device__ vec3f getMirrorImage(vec3f pt, plane pl);

	//point and line functions
	__host__ __device__ float distance(vec3f p, line l);

	//point and plane functions
	__host__ __device__ float aDistance(vec3f pt, plane p);

	//line and line functions
	__host__ __device__ bool coplanar(line l, line m);
	__host__ __device__ float distance(line l, line m);


	//plane plane functions
	__host__ __device__ float distance(plane p1, plane p2);

	//line and plane functions
	__host__ __device__ vec3f intersection(line l, plane p, bool* error);
	__host__ __device__ vec3f intersectionRaw_s(line l, plane p);
	__host__ __device__ vec3f intersectionRaw(line l, plane p);

	//ray cast
	__host__ __device__ bool rayCast(line l, plane p, vec3f& intersection);
}


namespace linearMathLD {//long double precesion

	class line {
	public:


		__host__ __device__ line();
		__host__ __device__ line(vec3ld PT, vec3ld DR);

		__host__ __device__ bool set(vec3ld PT, vec3ld DR);
		__host__ __device__ void setRaw_s(vec3ld PT, vec3ld DR);
		__host__ __device__ void setRaw(vec3ld PT, vec3ld DR);
		__host__ __device__ void setPT(vec3ld PT);
		__host__ __device__ bool setDR(vec3ld DR);
		__host__ __device__ void setDRRaw_s(vec3ld DR);
		__host__ __device__ void setDRRaw(vec3ld DR);
		__host__ __device__ vec3ld getPt();
		__host__ __device__ vec3ld getDr();

	private:
		vec3ld dr;
		vec3ld pt;
	};

	class plane
	{
	public:
		__host__ __device__ plane();
		__host__ __device__ plane(vec3ld PT, vec3ld DR);

		__host__ __device__ bool set(vec3ld PT, vec3ld DR);
		__host__ __device__ void setRaw_s(vec3ld PT, vec3ld DR);
		__host__ __device__ void setRaw(vec3ld PT, vec3ld DR);
		__host__ __device__ void setPT(vec3ld PT);
		__host__ __device__ bool setDR(vec3ld DR);
		__host__ __device__ void setDRRaw_s(vec3ld DR);
		__host__ __device__ void setDRRaw(vec3ld DR);

		__host__ __device__ vec3ld getPt();
		__host__ __device__ vec3ld getDr();

	private:
		vec3ld pt;
		vec3ld dr;

	};


	//line functions
	__host__ __device__ bool getPt(line l, vec3ld& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw_s(line l, vec3ld& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw(line l, vec3ld& coord, coordinateName coordGiven = coordinateName::xCoordinate);

	__host__ __device__ char getPtIn(vec3ld start, vec3ld end, vec3ld& coord, coordinateName coordGiven);


	//plane functions
	__host__ __device__ bool getPt(plane p, vec3ld& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw_s(plane p, vec3ld& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	__host__ __device__ void getPtRaw(plane p, vec3ld& coord, coordinateName coordToFind = coordinateName::xCoordinate);

	//point to point functions
	__host__ __device__ long double distance(vec3ld p1, vec3ld p2);
	__host__ __device__ vec3ld getMirrorImage(vec3ld pt, plane pl);

	//point and line functions
	__host__ __device__ long double distance(vec3ld p, line l);

	//point and plane functions
	__host__ __device__ long double aDistance(vec3ld pt, plane p);

	//line and line functions
	__host__ __device__ bool coplanar(line l, line m);
	__host__ __device__ long double distance(line l, line m);


	//plane plane functions
	__host__ __device__ long double distance(plane p1, plane p2);

	//line and plane functions
	__host__ __device__ vec3ld intersection(line l, plane p, bool* error);
	__host__ __device__ vec3ld intersectionRaw_s(line l, plane p);
	__host__ __device__ vec3ld intersectionRaw(line l, plane p);

	//ray cast
	__host__ __device__ bool rayCast(line l, plane p, vec3ld& intersection);
}