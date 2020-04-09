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