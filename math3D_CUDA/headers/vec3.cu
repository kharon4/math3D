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
