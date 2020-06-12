#pragma once
#define INSIDE_linearMath_CU_FILE 1
#include "linearMath.cuh"

#include <math.h>

namespace linearMath {//double precesion


	//line class
	template<typename T>
	__host__ __device__ line<T>::line() {
		setRaw(vec3<T>(0, 0, 0), vec3<T>(1, 0, 0));
	}

	template<typename T>
	__host__ __device__ line<T>::line(const vec3<T>& PT, const vec3<T>& DR) {
		setRaw_s(PT, DR);
	}

	template<typename T>
	__host__ __device__ bool line<T>::set(const vec3<T>& PT, const vec3<T>& DR) {
		pt = PT;
		if (vec3<T>::isNUL(DR)) {
			dr = vec3<T>(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	template<typename T>
	inline __host__ __device__ void line<T>::setRaw_s(const vec3<T>& PT, const vec3<T>& DR) {
		pt = PT;
		if (vec3<T>::isNUL(DR)) dr = vec3<T>(1, 0, 0);
		else dr = DR;
	}

	template<typename T>
	inline __host__ __device__ void line<T>::setRaw(const vec3<T>& PT, const vec3<T>& DR) {
		pt = PT;
		dr = DR;
	}

	
	template<typename T>
	inline __host__ __device__ void line<T>::setPT(const vec3<T>& PT) {
		pt = PT;
	}

	template<typename T>
	__host__ __device__ bool line<T>::setDR(const vec3<T>& DR) {
		if (vec3<T>::isNUL(DR)) {
			dr = vec3<T>(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	template<typename T>
	inline __host__ __device__ void line<T>::setDRRaw_s(const vec3<T>& DR) {
		if (vec3<T>::isNUL(DR))dr = vec3<T>(1, 0, 0);
		else dr = DR;
	}

	template<typename T>
	inline __host__ __device__ void line<T>::setDRRaw(const vec3<T>& DR) {
		dr = DR;
	}

	template<typename T>
	inline __host__ __device__ vec3<T> line<T>::getPt() const{ return pt; }

	template<typename T>
	inline __host__ __device__ vec3<T> line<T>::getDr() const{ return dr; }

	//plane class
	template<typename T>
	__host__ __device__ plane<T>::plane() {
		setRaw(vec3<T>(0,0,0),vec3<T>(1,0,0));
	}

	template<typename T>
	__host__ __device__ plane<T>::plane(const vec3<T>& PT, const vec3<T>& DR) {
		setRaw_s(PT, DR);
	}

	template<typename T>
	__host__ __device__ bool plane<T>::set(const vec3<T>& PT, const vec3<T>& DR) {
		pt = PT;
		if (vec3<T>::isNUL(DR)) {
			dr = vec3<T>(1, 0, 0);
			return true;//error
		}
		dr = DR;
		return false;
	}

	template<typename T>
	inline __host__ __device__ void plane<T>::setRaw_s(const vec3<T>& PT, const vec3<T>& DR) {
		pt = PT;
		if (vec3<T>::isNUL(DR))dr = vec3<T>(1, 0, 0);
		else dr = DR;
	}

	template<typename T>
	inline __host__ __device__ void plane<T>::setRaw(const vec3<T>& PT, const vec3<T>& DR) {
		pt = PT;
		dr = DR;
	}

	template<typename T>
	inline __host__ __device__ void plane<T>::setPT(const vec3<T>& PT) {
		pt = PT;
	}

	template<typename T>
	__host__ __device__ bool plane<T>::setDR(const vec3<T>& DR) {
		if (vec3<T>::isNUL(DR)) {
			dr = vec3<T>(1, 0, 0);
			return true;
		}
		dr = DR;
		return false;
	}

	template<typename T>
	inline __host__ __device__ void plane<T>::setDRRaw_s(const vec3<T>& DR) {
		if (vec3<T>::isNUL(DR))dr = vec3<T>(1, 0, 0);
		else dr = DR;
	}

	template<typename T>
	inline __host__ __device__ void plane<T>::setDRRaw(const vec3<T>& DR) {
		dr = DR;
	}

	template<typename T>
	inline __host__ __device__ vec3<T> plane<T>::getPt() const{ return pt; }

	template<typename T>
	inline __host__ __device__ vec3<T> plane<T>::getDr() const{ return dr; }



	//helper math functions
	template<typename T> 
	inline __host__ __device__ T absVal(T val) {
		if (val < 0)return (val * (-1));
		return val;
	}

	//line functions
	template<typename T>
	__host__ __device__ bool getPt(const line<T>& l, vec3<T>& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3<T>(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3<T>(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
				return 1;//error
			}
			else {
				coord = vec3<T>(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
		return 0;//no error
	}

	template<typename T>
	__host__ __device__ void getPtRaw_s(const line<T>& l, vec3<T>& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
			if (l.getDr().x == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3<T>(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
			}
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
			if (l.getDr().y == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3<T>(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
			}
		}
		else {//z coordinate
			if (l.getDr().z == 0) {
				coord = (l.getPt());
			}
			else {
				coord = vec3<T>(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
			}
		}
	}

	template<typename T>
	__host__ __device__ void getPtRaw(const line<T>& l, vec3<T>& coord, coordinateName coordGiven) {
		if (coordGiven == coordinateName::xCoordinate) {//coord is x coord
				coord = vec3<T>(coord.x, l.getPt().y + l.getDr().y * (coord.x - l.getPt().x) / l.getDr().x, l.getPt().z + l.getDr().z * (coord.x - l.getPt().x) / l.getDr().x);
		}
		else if (coordGiven == coordinateName::yCoordinate) {//coord is y coord
				coord = vec3<T>(l.getPt().x + l.getDr().x * (coord.y - l.getPt().y) / l.getDr().y, coord.y, l.getPt().z + l.getDr().z * (coord.y - l.getPt().y) / l.getDr().y);
		}
		else {//z coordinate
			coord = vec3<T>(l.getPt().x + l.getDr().x * (coord.z - l.getPt().z) / l.getDr().z, l.getPt().y + l.getDr().y * (coord.z - l.getPt().z) / l.getDr().z, coord.z);
		}
	}

	template<typename T>
	__host__ __device__ char getPtIn(const vec3<T>& start, const vec3<T>& end, vec3<T>& coord, coordinateName coordGiven) {
		char rval = 0;//no error
		if (getPt(line<T>(start, vec3<T>::subtract(end, start)), coord, coordGiven)) {
			rval = 1;// zero / infinite ans
		}
		else
		{
			double ttlDist = vec3<T>::subtract(start, end).mag2();
			if (!(vec3<T>::subtract(coord, start).mag2() <= ttlDist) || !(vec3<T>::subtract(coord, end).mag2() <= ttlDist)) {
				rval = 2; // out of bounds
			}
		}
		return rval;
	}


	//plane functions
	template<typename T>
	__host__ __device__ bool getPt(const plane<T>& p, vec3<T>& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z == 0) {
				return 1;
			}
			else {
				(coord).z = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
				return 0;
			}
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x == 0) {
				return 1;
			}
			else {
				(coord).x = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
				return 0;
			}
		}
		else {
			if (p.getDr().y == 0) {
				return 1;
			}
			else {
				(coord).y = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
				return 0;
			}
		}
	}

	template<typename T>
	__host__ __device__ void getPtRaw_s(const plane<T>& p, vec3<T>& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate) {
			if (p.getDr().z != 0)
				(coord).z = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		}
		else if (coordToFind == coordinateName::xCoordinate) {
			if (p.getDr().x != 0)
				(coord).x = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		}
		else {
			if (p.getDr().y == 0)
				(coord).y = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
		}
	}

	template<typename T>
	__host__ __device__ void getPtRaw(const plane<T>& p, vec3<T>& coord, coordinateName coordToFind) {
		if (coordToFind == coordinateName::zCoordinate)
				(coord).z = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().y * (coord).y)) / p.getDr().z;
		else if (coordToFind == coordinateName::xCoordinate)
				(coord).x = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().z * (coord).z + p.getDr().y * (coord).y)) / p.getDr().x;
		else
				(coord).y = (vec3<T>::dot(p.getPt(), p.getDr()) - (p.getDr().x * (coord).x + p.getDr().z * (coord).z)) / p.getDr().y;
	}


	//point to point functions
	template<typename T>
	__host__ __device__ double distance(const vec3<T>& p1, const vec3<T>& p2) {
		return vec3<T>::subtract(p1, p2).mag();
	}


	//point and line functions
	template<typename T>
	__host__ __device__ double distance(const vec3<T>& p, const line<T>& l) {
		return vec3<T>::cross(l.getDr(), vec3<T>::subtract(p, l.getPt())).mag() / l.getDr().mag();
	}


	//point and plane functions
	template<typename T>
	__host__ __device__ double aDistance(const vec3<T>& pt, const plane<T>& p) { // algebraic distance
		return (vec3<T>::dot(pt, p.getDr()) - vec3<T>::dot(p.getDr(), p.getPt())) / p.getDr().mag();
	}

	template<typename T>
	__host__ __device__ vec3<T> getMirrorImage(const vec3<T>& pt, const plane<T>& pl) {
		//get component perpendicular to pl
		double Component = vec3<T>::componentRaw_s(vec3<T>::subtract(pt, pl.getPt()), pl.getDr());
		vec3<T> normal = pl.getDr();
		normal.normalize();
		return (vec3<T>::subtract(pt, vec3<T>::multiply(normal, 2 * Component)));
	}


	//line and line functions
	template<typename T>
	__host__ __device__ bool coplanar(const line<T>& l, const line<T>& m) {
		if (vec3<T>::dot(vec3<T>::subtract(l.getPt(), m.getPt()), vec3<T>::cross(l.getDr(), m.getDr())) == 0) {
			return 1;
		}
		else {
			return 0;
		}
	}

	template<typename T>
	__host__ __device__ double distance(const line<T>& l, const line<T>& m) {
		if (vec3<T>::isEqual(l.getDr(), m.getDr())) {
			return (vec3<T>::cross(vec3<T>::subtract(l.getPt(), m.getPt()), l.getDr()).mag() / l.getDr().mag());
		}
		else {
			vec3<T> temp = vec3<T>::cross(l.getDr(), m.getDr());
			//replacing fabs with absVal
			return absVal(vec3<T>::dot(vec3<T>::subtract(l.getPt(), m.getPt()), temp) / temp.mag());
		}
	}


	//plane plane functions
	template<typename T>
	__host__ __device__ double distance(const plane<T>& p1, const plane<T>& p2) {
		if (vec3<T>::isEqual(p1.getDr(), p2.getDr())) {
			return absVal((vec3<T>::dot(p1.getDr(), p1.getPt()) - vec3<T>::dot(p2.getDr(), p2.getPt())) / p1.getDr().mag());
		}
		else {
			return 0;
		}
	}


	//line and plane functions
	template<typename T>
	__host__ __device__ vec3<T> intersection(const line<T>& l, const plane<T>& p, bool* error) {
		if (vec3<T>::dot(l.getDr(), p.getDr()) == 0) {
			*error = true;//error , no solution of infinite solutions
			return l.getPt();
		}
		else {
			*error = false;//no error
			double lambda;
			lambda = vec3<T>::dot(p.getPt()-l.getPt(), p.getDr()) / vec3<T>::dot(p.getDr(), l.getDr());
			vec3<T> rVal = vec3<T>::add(l.getPt(), vec3<T>::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	template<typename T>
	__host__ __device__ vec3<T> intersectionRaw_s(const line<T>& l, const plane<T>& p) {
		if (vec3<T>::dot(l.getDr(), p.getDr()) == 0) {
			return l.getPt();
		}
		else {
			double lambda;
			lambda = vec3<T>::dot(p.getPt()-l.getPt(), p.getDr()) / vec3<T>::dot(p.getDr(), l.getDr());
			vec3<T> rVal = vec3<T>::add(l.getPt(), vec3<T>::multiply(l.getDr(), lambda));
			return rVal;
		}
	}

	template<typename T>
	__host__ __device__ vec3<T> intersectionRaw(const line<T>& l, const plane<T>& p) {
			double lambda;
			lambda = vec3<T>::dot(p.getPt()-l.getPt(), p.getDr()) / vec3<T>::dot(p.getDr(), l.getDr());
			vec3<T> rVal = vec3<T>::add(l.getPt(), vec3<T>::multiply(l.getDr(), lambda));
			return rVal;
	}

	template<typename T>
	__host__ __device__ bool intersectionLambda(const line<T>& l, const plane<T>& p, double& OUTlambda) {
		if (vec3<T>::dot(l.getDr(), p.getDr()) == 0) {
			return true;
		}
		else {
			OUTlambda = vec3<T>::dot(p.getPt() - l.getPt(), p.getDr()) / vec3<T>::dot(p.getDr(), l.getDr());
			return false;
		}
	}

	template<typename T>
	__host__ __device__ void intersectionLambdaRaw_s(const line<T>& l, const plane<T>& p, double& OUTlambda, double defaultVal) {
		if (vec3<T>::dot(l.getDr(), p.getDr()) == 0) {
			OUTlambda = defaultVal;
		}
		else {
			OUTlambda = vec3<T>::dot(p.getPt() - l.getPt(), p.getDr()) / vec3<T>::dot(p.getDr(), l.getDr());
		}
	}
	
	template<typename T>
	__host__ __device__ double intersectionLambdaRaw(const line<T>& l, const plane<T>& p) {
			return (vec3<T>::dot(p.getPt() - l.getPt(), p.getDr()) / vec3<T>::dot(p.getDr(), l.getDr()));
	}

	template<typename T>
	__host__ __device__ vec3<T> getPt(const line<T>& l, double lambda) {
		return (l.getPt() + lambda * l.getDr());
	}

	//ray cast
	template<typename T>
	__host__ __device__ bool rayCast(const line<T>& l, const plane<T>& p, vec3<T>& intersection) {
		if (vec3<T>::dot(l.getDr(), p.getDr()) == 0) {
			intersection = l.getPt();
			return 1;// multiple or no results
		}
		else {
			double lambda;
			lambda = vec3<T>::dot(p.getPt()-l.getPt(), p.getDr()) / vec3<T>::dot(p.getDr(), l.getDr());
			if (lambda < 0) {
				intersection = l.getPt();
				return 1;//no results
			}
			intersection = vec3<T>::add(l.getPt(), vec3<T>::multiply(l.getDr(), lambda));
		}
	}

}