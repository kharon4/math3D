#pragma once

#include <iostream>

#ifdef __NVCC__
#pragma message("NVCC found . Compilling CUDA code : " __FILE__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#else
#pragma message("nvcc NOT found !!!. Compilling only Host code : " __FILE__)

#define __host__
#define __device__

#endif

enum errCodes { noErr = 0, devideByZero = 1 };

template <typename T>
class vec3
{
public:
	T x, y, z;
	__host__ __device__ vec3();
	__host__ __device__ vec3(T, T, T);
	template <typename N>
	__host__ __device__ vec3<T>(const vec3<N>& other) { x = other.x; y = other.y; z = other.z; }
	__host__ __device__ double mag() const;
	__host__ __device__ double mag2() const;
	__host__ __device__ bool normalize();
	__host__ __device__ void normalizeRaw();
	__host__ __device__ void multiply(double);
	
	static __host__ __device__ double dot(const vec3<T>& a, const vec3<T>& b);//dot product
	static __host__ __device__ vec3<T> cross(const vec3<T>& a, const vec3<T>& b);//cross product
	static __host__ __device__ vec3<T> multiply(const vec3<T>& a, const double factor);
	static __host__ __device__ vec3<T> add(const vec3<T>& a, const vec3<T>& b);
	static __host__ __device__ vec3<T> subtract(const vec3<T>& a, const vec3<T>& b);//a-b

	
	static __host__ __device__ double angle(const vec3<T>& a, const vec3<T>& b, errCodes* err);//in radian
	inline static __host__ __device__ double angleRaw(const vec3<T>& a, const vec3<T>& b);//no err checking
	static __host__ __device__ double angleRaw_s(const vec3<T>& a, const vec3<T>& b);//safe
	static __host__ __device__ double component(const vec3<T>& of, const vec3<T>& along, errCodes* err);
	inline static __host__ __device__ double componentRaw(const vec3<T>& of, const vec3<T>& along);//no err checking
	static __host__ __device__ double componentRaw_s(const vec3<T>& of, const vec3<T>& along);//safe
	static __host__ __device__ vec3<T> normalize(vec3<T> a, errCodes* err);
	inline static __host__ __device__ vec3<T> normalizeRaw(vec3<T> a);//no err checking
	static __host__ __device__ vec3<T> normalizeRaw_s(vec3<T> a);//safe

	static __host__ __device__ bool isNUL(const vec3<T>& a);

	static __host__ __device__ bool isEqual(const vec3<T>& a, const vec3<T>& b);
	
	//dir based vectors
	inline static __host__ __device__ vec3<T> vecX(T val) { return vec3<T>(val, 0, 0); }
	inline static __host__ __device__ vec3<T> vecY(T val) { return vec3<T>(0, val, 0); }
	inline static __host__ __device__ vec3<T> vecZ(T val) { return vec3<T>(0, 0, val); }

	inline __host__ void print(std::ostream& f) { f << "( " << x << " , " << y << " , " << z << " )"; }

	__host__ __device__ vec3<T> operator + (const vec3<T>& other)const { return add(*this, other); }

	__host__ __device__ void operator += (const vec3<T>& other) { *this = add(*this, other); }

	__host__ __device__ vec3<T> operator - (const vec3<T>& other)const { return subtract(*this, other); }

	__host__ __device__ vec3<T> operator - ()const { return multiply(*this, -1); }

	__host__ __device__ void operator -= (const vec3<T>& other) { *this = subtract(*this, other); }

	__host__ __device__ vec3<T> operator * (const double& other)const { return multiply(*this, other); }

	__host__ __device__ void operator *= (const double& other) { *this = multiply(*this, other); }

	__host__ __device__ vec3<T> operator * (const vec3<T>& other)const { return cross(*this, other); }

	__host__ __device__ void operator *= (const vec3<T>& other) { *this = cross(*this, other); }

	__host__ __device__ vec3<T> operator / (const double& other)const { return multiply(*this, 1 / other); }

	__host__ __device__ void operator /= (const double& other) { *this = multiply(*this, 1 / other); }

	__host__ __device__ double operator / (const vec3<T>& other)const { return dot(*this, other); }

	__host__ __device__ bool operator == (const vec3<T>& other)const { return isEqual(*this, other); }

	__host__ __device__ bool operator != (const vec3<T>& other)const { return !isEqual(*this, other); }
	
};

template <typename T>
__host__ __device__ vec3<T> operator * (const double& other,const vec3<T>& vec) { return vec3<T>::multiply(vec, other); }


//declaration macro
#define declareVec3(type , name) typedef vec3<type> name;


//declare vec3 types here
declareVec3(double, vec3d)
//declareVec3(long double, vec3ld) // does not work with cuda 10.2 long double is converted to double
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

#ifndef math3D_DeclrationOnly
#ifndef INSIDE_VEC3_CU_FILE
#include "vec3.cu"
#endif
#endif
