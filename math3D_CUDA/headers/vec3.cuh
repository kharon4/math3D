#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum errCodes { noErr = 0, devideByZero = 1 };

template <typename T>
class vec3
{
public:
	T x, y, z;
	__host__ __device__ vec3();
	__host__ __device__ vec3(T, T, T);
	__host__ __device__ double mag();
	__host__ __device__ double mag2();
	__host__ __device__ bool normalize();
	__host__ __device__ void multiply(double);
	
	static __host__ __device__ double dot(vec3<T> a, vec3<T> b);//dot product

	static __host__ __device__ vec3<T> cross(vec3<T> a, vec3<T> b);//cross product

	static __host__ __device__ vec3<T> multiply(vec3<T> a, double factor);

	static __host__ __device__ vec3<T> add(vec3<T> a, vec3<T> b);

	static __host__ __device__ vec3<T> subtract(vec3<T> a, vec3<T> b);//a-b

	/*
	static __host__ __device__ double angle(vec3<T> a, vec3<T> b, errCodes* err = &defaultErrCode);//in radian

	static __host__ __device__ double component(vec3<T> of, vec3<T> along, errCodes* err = &defaultErrCode);

	static __host__ __device__ vec3<T> normalize(vec3<T> a, errCodes* err = &defaultErrCode);

	static __host__ __device__ bool isNUL(vec3<T> a);

	static __host__ __device__ bool isEqual(vec3<T> a, vec3<T> b);
	
	//dir based vectors
	vec3<T> vecX(T val) { return vec3<T>(val, 0, 0); }
	vec3<T> vecY(T val) { return vec3<T>(0, val, 0); }
	vec3<T> vecZ(T val) { return vec3<T>(0, 0, val); }


	template<typename N>
	void convert(vec3<N> in) {
		x = (T)in.x;
		y = (T)in.y;
		z = (T)in.z;
	}

	vec3<T> operator + (vec3<T> const other)const { return add(*this, other); }

	void operator += (vec3<T> const other) { *this = add(*this, other); }

	vec3<T> operator - (vec3<T> const other)const { return subtract(*this, other); }

	vec3<T> operator - ()const { return multiply(*this, -1); }

	void operator -= (vec3<T> const other) { *this = subtract(*this, other); }

	vec3<T> operator * (double const other)const { return multiply(*this, other); }

	void operator *= (double const other) { *this = multiply(*this, other); }

	vec3<T> operator * (vec3<T> const other)const { return cross(*this, other); }

	void operator *= (vec3<T> const other) { *this = cross(*this, other); }

	vec3<T> operator / (double const other)const { return multiply(*this, 1 / other); }

	void operator /= (double const other) { *this = multiply(*this, 1 / other); }

	double operator / (vec3<T> const other)const { return dot(*this, other); }

	bool operator == (vec3<T> const other)const { return isEqual(*this, other); }

	bool operator != (vec3<T> const other)const { return !isEqual(*this, other); }

	*/
};

//template <typename T>
//vec3<T> operator * (double const other, vec3<T> vec) { return vec3<T>::multiply(vec, other); }


//declaration macro
#define declareVec3(type , name) typedef vec3<type> name;


//declare vec3 types here
declareVec3(double, vec3d)
declareVec3(long double, vec3ld)
declareVec3(float, vec3f)
declareVec3(int, vec3i)
declareVec3(unsigned int, vec3ui)
declareVec3(long, vec3l)
declareVec3(unsigned long, vec3ul)
declareVec3(long long, vec3ll)
declareVec3(unsigned long long, vec3ull)
declareVec3(short, vec3s)
declareVec3(unsigned short, vec3us)
declareVec3(char, vec3c)
declareVec3(unsigned char, vec3uc)

#ifndef INSIDE_CU_FILE
#include "vec3.cu"
#endif