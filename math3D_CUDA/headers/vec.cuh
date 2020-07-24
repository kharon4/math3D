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

//enum errCodes { noErr = 0, devideByZero = 1 };



template<unsigned char dim,class T>
class vec {
private:
	T data;
	vec<dim - 1, T> next;

	
public:
	
	__forceinline void copy(const vec<dim, T>& other) { data = other.data; next.copy(other.next);}

	vec():data(T()){}
	
	template<typename ... ArgTypes>
	vec(T val, ArgTypes ... args) :data(val),next(args...) {};
	
	vec(const vec<dim, T>& other) { copy(other);}

	constexpr __forceinline unsigned char getDim() const{ return dim; }

	//getter + setter
	template<unsigned char index>
	constexpr __forceinline T& at(){
		static_assert(index < dim, "OUT OF BOUNDS");
		return next.at<index - 1>();
	}

	template<>
	constexpr __forceinline T& at<0>(){
		return data;
	}


	//const getter
	template<unsigned char index>
	constexpr __forceinline const T& at() const{
		static_assert(index < dim, "OUT OF BOUNDS");
		return next.at<index - 1>();
	}

	template<>
	constexpr __forceinline const T& at<0>() const{
		return data;
	}

	//magnitude
	__forceinline double mag2() const{
		return ((data * data) + next.mag2());
	}

	__forceinline double mag() const{
		return sqrt(mag2());
	}

	//multiply
	__forceinline void multiply(double f) {
		data *= f; next.multiply(f);
	}

	static vec<dim, T> multiply(const vec<dim, T>& A, double f) {
		vec<dim, T> rVal = A;
		rVal.multiply(f);
		return rVal;
	}

	//normalize
	bool normalize() {
		double m = mag();
		if (m == 0)return true;
		multiply(1 / m);
		return false;
	}

	void normalizeRaw() {
		multiply(1 / mag());
	}


	//add + subtract
	__forceinline void add(const vec<dim, T>& other) { data += other.data; next.add(other.next); }
	__forceinline void subtract(const vec<dim, T>& other) { data -= other.data; next.subtract(other.next); }

	static vec<dim, T> add(const vec<dim, T>& A, const vec<dim, T>& B) { vec<dim, T> rVal = A; rVal.add(B); return rVal; }
	static vec<dim, T> subtract(const vec<dim, T>& A, const vec<dim, T>& B) { vec<dim, T> rVal = A; rVal.subtract(B); return rVal; }

	//dot
	static __forceinline T dot(const vec<dim, T>& A, const vec<dim, T>& B) {
		return ((A.data * B.data) + vec<dim-1,T>::dot(A.next, B.next));
	}

	//commulative multiplication
	__forceinline void multiply(const vec<dim, T>& other) { data *= other.data; next.multiply(other.next); }

	static vec<dim,T> commulativeMul(const vec<dim, T>& A, const vec<dim, T>& B) {
		vec<dim, T> rVal = A;
		rVal.multiply(B);
		return rVal;
	}



};



template<class T>
class vec<1,T>{
private:
	T data;
public:
	vec() :data(T()) {}

	vec(T val) : data(val){};

	__forceinline void copy(const vec<1, T>& other) { data = other.data;}
	vec(const vec<1, T>& other) { copy(other); }

	constexpr __forceinline unsigned char getDim() const { return 1; }


	//getter + setter
	template<unsigned char index>
	constexpr __forceinline T& at() {
		static_assert(index == 0, "OUT OF BOUNDS");
		return data;
	}

	//const getter
	template<unsigned char index>
	constexpr __forceinline const T& at() const{
		static_assert(index == 0, "OUT OF BOUNDS");
		return data;
	}

	//magnitude
	__forceinline double mag2() const{
		return (data * data);
	}

	__forceinline double mag() const{
		return data;
	}

	//multiply
	__forceinline void multiply(double f) {
		data *= f;
	}

	static vec<1, T> multiply(const vec<1, T>& A, double f) {
		vec<1, T> rVal = A;
		rVal.multiply(f);
		return rVal;
	}

	//normalize

	bool normalize() {
		if (data == 0)return true;
		else if (data > 0)data = 1;
		else data = -1;
		return false;
	}

	void normalizeRaw() {
		if (data > 0)data = 1;
		else data = -1;
	}

	//add + subtract
	__forceinline void add(const vec<1, T>& other) { data += other.data;}
	__forceinline void subtract(const vec<1, T>& other) { data -= other.data;}

	static vec<1, T> add(const vec<1, T>& A, const vec<1, T>& B) { vec<1, T> rVal = A; rVal.add(B); return rVal; }
	static vec<1, T> subtract(const vec<1, T>& A, const vec<1, T>& B) { vec<1, T> rVal = A; rVal.subtract(B); return rVal; }
	
	//dot
	static __forceinline T dot(const vec<1, T>& A, const vec<1, T>& B) {
		return ((A.data * B.data));
	}

	//commulative multiplication
	__forceinline void multiply(const vec<1, T>& other) { data *= other.data; }

	static vec<1, T> commulativeMul(const vec<1, T>& A, const vec<1, T>& B) {
		vec<1, T> rVal = A;
		rVal.multiply(B);
		return rVal;
	}

};