#pragma once
#define INSIDE_CU_FILE 1
#include "vec3.cuh"

template <typename T>
__host__ __device__ inline vec3<T>::vec3(){}

template <typename T>
__host__ __device__ vec3<T>::vec3(T X, T Y, T Z)
{
	x = X;
	y = Y;
	z = Z;
}


template <typename T>
__host__ __device__ double vec3<T>::mag()
{
	return sqrt(x * x + y * y + z * z);
}

template <typename T>
__host__ __device__ double vec3<T>::mag2()
{
	return (x * x + y * y + z * z);
}


template <typename T>
__host__ __device__ bool vec3<T>::normalize() {
	double m = mag();
	if (m == 0) {
		return 1;
	}
	else {
		x /= m;
		y /= m;
		z /= m;
		return 0;
	}
}

template <typename T>
__host__ __device__ void vec3<T>::normalizeRaw() {
	double m = mag();
	x /= m;
	y /= m;
	z /= m;
}

template <typename T>
__host__ __device__ void vec3<T>::multiply(double factor) {
	x *= factor;
	y *= factor;
	z *= factor;
}

//vec functions

template <typename T>
__host__ __device__ double vec3<T>::dot(vec3<T> a, vec3<T> b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

template <typename T>
__host__ __device__ vec3<T> vec3<T>::cross(vec3<T> a, vec3<T> b) {
	return vec3<T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

template <typename T>
__host__ __device__ vec3<T> vec3<T>::multiply(vec3<T> a, double factor) {
	return vec3<T>(a.x * factor, a.y * factor, a.z * factor);
}

template <typename T>
__host__ __device__ vec3<T> vec3<T>::add(vec3<T> a, vec3<T> b) {
	return vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template <typename T>
__host__ __device__ vec3<T> vec3<T>::subtract(vec3<T> a, vec3<T> b) {
	return vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}



//functions with err codes
template <typename T>
__host__ __device__ double vec3<T>::angle(vec3<T> a, vec3<T> b, errCodes* err) {//in radian
	(*err) = errCodes::noErr;
	double m = a.mag() * b.mag();
	if (m == 0)
	{
		(*err) = errCodes::devideByZero;
		return 0;
	}
	else
	{
		return acos(dot(a, b) / m);
	}

}

template <typename T>
__host__ __device__ double vec3<T>::component(vec3<T> of, vec3<T> along, errCodes* err) {
	(*err) = errCodes::noErr;
	double m = along.mag();
	if (m == 0)
	{
		(*err) = errCodes::devideByZero;
		return of.mag();
	}
	else
	{
		return (dot(of, along) / m);
	}
}

template <typename T>
__host__ __device__ vec3<T> vec3<T>::normalize(vec3<T> a, errCodes* err) {
	(*err) = errCodes::noErr;
	if (a.normalize()) {
		(*err) = errCodes::devideByZero;
	}
	return a;
}

//safe raw versions
template <typename T>
__host__ __device__ double vec3<T>::angleRaw_s(vec3<T> a, vec3<T> b) {//in radian
	double m = a.mag() * b.mag();
	if (m == 0)
	{
		return 0;
	}
	else
	{
		return acos(dot(a, b) / m);
	}

}

template <typename T>
__host__ __device__ double vec3<T>::componentRaw_s(vec3<T> of, vec3<T> along) {
	double m = along.mag();
	if (m == 0)
	{
		return of.mag();
	}
	else
	{
		return (dot(of, along) / m);
	}
}

template <typename T>
__host__ __device__ vec3<T> vec3<T>::normalizeRaw_s(vec3<T> a) {
	a.normalize();
	return a;
}

//raw versions
template <typename T>
inline __host__ __device__ double vec3<T>::angleRaw(vec3<T> a, vec3<T> b) {//in radian
		return acos(dot(a, b) / a.mag() * b.mag());
}

template <typename T>
inline __host__ __device__ double vec3<T>::componentRaw(vec3<T> of, vec3<T> along) {
	return (dot(of, along) / along.mag());
	
}

template <typename T>
inline __host__ __device__ vec3<T> vec3<T>::normalizeRaw(vec3<T> a) {
	a.normalizeRaw();
	return a;
}

template <typename T>
__host__ __device__ bool vec3<T>::isNUL(vec3<T> a) {
	if ((a.x == 0) && (a.y == 0) && (a.z == 0)) {
		return 1;
	}
	else {
		return 0;
	}
}

template <typename T>
__host__ __device__ bool vec3<T>::isEqual(vec3<T> a, vec3<T> b) {
	if ((a.x == b.x) && (a.y == b.y) && (a.z == b.z)) {
		return 1;
	}
	else {
		return 0;
	}
}

