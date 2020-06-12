#pragma once

#include "vec3.cuh"

enum class coordinateName
{
	xCoordinate = 0,
	yCoordinate = 1,
	zCoordinate = 2
};

namespace linearMath {//double precesion

	template<typename T>
	class line {
	public:


		__host__ __device__ line();
		__host__ __device__ line(const vec3<T>& PT, const vec3<T>& DR);

		__host__ __device__ bool set(const vec3<T>& PT, const vec3<T>& DR);
		inline __host__ __device__ void setRaw_s(const vec3<T>& PT, const vec3<T>& DR);
		inline __host__ __device__ void setRaw(const vec3<T>& PT, const vec3<T>& DR);
		inline __host__ __device__ void setPT(const vec3<T>& PT);
		__host__ __device__ bool setDR(const vec3<T>& DR);
		inline __host__ __device__ void setDRRaw_s(const vec3<T>& DR);
		inline __host__ __device__ void setDRRaw(const vec3<T>& DR);
		inline __host__ __device__ vec3<T> getPt() const;
		inline __host__ __device__ vec3<T> getDr() const;

	private:
		vec3<T> dr;
		vec3<T> pt;
	};


	template<typename T>
	class plane
	{
	public:
		__host__ __device__ plane();
		__host__ __device__ plane(const vec3<T>& PT, const vec3<T>& DR);

		__host__ __device__ bool set(const vec3<T>& PT, const vec3<T>& DR);
		inline __host__ __device__ void setRaw_s(const vec3<T>& PT, const vec3<T>& DR);
		inline __host__ __device__ void setRaw(const vec3<T>& PT, const vec3<T>& DR);
		inline __host__ __device__ void setPT(const vec3<T>& PT);
		__host__ __device__ bool setDR(const vec3<T>& DR);
		inline __host__ __device__ void setDRRaw_s(const vec3<T>& DR);
		inline __host__ __device__ void setDRRaw(const vec3<T>& DR);

		inline __host__ __device__ vec3<T> getPt() const;
		inline __host__ __device__ vec3<T> getDr() const;

	private:
		vec3<T> pt;
		vec3<T> dr;

	};


	//line functions
	template<typename T>
	__host__ __device__ bool getPt(const line<T>& l, vec3<T>& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	template<typename T>
	__host__ __device__ void getPtRaw_s(const line<T>& l, vec3<T>& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	template<typename T>
	__host__ __device__ void getPtRaw(const line<T>& l, vec3<T>& coord, coordinateName coordGiven = coordinateName::xCoordinate);
	template<typename T>
	__host__ __device__ char getPtIn(const vec3<T>& start,const vec3<T>& end, vec3<T>& coord, coordinateName coordGiven);


	//plane functions
	template<typename T>
	__host__ __device__ bool getPt(const plane<T>& p, vec3<T>& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	template<typename T>
	__host__ __device__ void getPtRaw_s(const plane<T>& p, vec3<T>& coord, coordinateName coordToFind = coordinateName::xCoordinate);
	template<typename T>
	__host__ __device__ void getPtRaw(const plane<T>& p, vec3<T>& coord, coordinateName coordToFind = coordinateName::xCoordinate);

	//point to point functions
	template<typename T>
	__host__ __device__ double distance(const vec3<T>& p1, const vec3<T>& p2);

	//point and line functions
	template<typename T>
	__host__ __device__ double distance(const vec3<T>& p, const line<T>& l);

	//point and plane functions
	template<typename T>
	__host__ __device__ double aDistance(const vec3<T>& pt, const plane<T>& p);
	template<typename T>
	__host__ __device__ vec3<T> getMirrorImage(const vec3<T>& pt, const plane<T>& pl);

	//line and line functions
	template<typename T>
	__host__ __device__ bool coplanar(const line<T>& l, const line<T>& m);
	template<typename T>
	__host__ __device__ double distance(const line<T>& l, const line<T>& m);


	//plane plane functions
	template<typename T>
	__host__ __device__ double distance(const plane<T>& p1, const plane<T>& p2);

	//line and plane functions
	template<typename T>
	__host__ __device__ vec3<T> intersection(const line<T>& l, const plane<T>& p, bool* error);
	template<typename T>
	__host__ __device__ vec3<T> intersectionRaw_s(const line<T>& l, const plane<T>& p);
	template<typename T>
	__host__ __device__ vec3<T> intersectionRaw(const line<T>& l, const plane<T>& p);

	template<typename T>
	__host__ __device__ bool intersectionLambda(const line<T>& l, const plane<T>& p, double& OUTlambda);
	template<typename T>
	__host__ __device__ void intersectionLambdaRaw_s(const line<T>& l, const plane<T>& p, double& OUTlambda, double defaultVal = -1);
	template<typename T>
	__host__ __device__ double intersectionLambdaRaw(const line<T>& l, const plane<T>& p);
	template<typename T>
	__host__ __device__ vec3<T> getPt(const line<T>& l, double lambda);

	//ray cast
	template<typename T>
	__host__ __device__ bool rayCast(const line<T>& l, const plane<T>& p, vec3<T>& intersection);



	typedef line<double>		lineD;
	typedef line<long double>	lineLD;
	typedef line<float>			linef;
	typedef line<short>			lineS;
	typedef line<int>			lineI;
	typedef line<char>			lineC;
	typedef line<unsigned char> lineUC;


	typedef plane<double>		planeD;
	typedef plane<long double>	planeLD;
	typedef plane<float>		planef;
	typedef plane<short>		planeS;
	typedef plane<int>			planeI;
	typedef plane<char>			planeC;
	typedef plane<unsigned char>planeUC;
}



#ifndef math3D_DeclrationOnly
#ifndef INSIDE_linearMath_CU_FILE
#include "linearMath.cu"
#endif
#endif