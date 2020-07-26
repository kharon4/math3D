#pragma once

#include <iostream>
#include "math3D_Constants.h"

#ifdef __NVCC__
#pragma message("NVCC found . Compilling CUDA code : " __FILE__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _hd_ __host__ __device__

#else
#pragma message("nvcc NOT found !!!. Compilling only Host code : " __FILE__)
#define __host__
#define __device__
#define _hd_

#endif

template<unsigned char dim,class T>
class vec {
private:
	T data;
	vec<dim - 1, T> next;
public:
	
	__forceinline void copy(const vec<dim, T>& other) { data = other.data; next.copy(other.next);}

	template<class N>
	__forceinline void castCopy(const vec<dim, N>& other) { data = other.data; next.castCopy(other.next); }
	

	vec():data(T()){}
	
	template<typename ... ArgTypes>
	vec(T val, ArgTypes ... args) :data(val),next(args...) {};
	
	vec(const vec<dim, T>& other) { copy(other);}

	template<class N>
	vec(const vec<dim, N>& other) { castCopy(other); }

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

	//add + subtract
	__forceinline void add(const vec<dim, T>& other) { data += other.data; next.add(other.next); }
	__forceinline void subtract(const vec<dim, T>& other) { data -= other.data; next.subtract(other.next); }

	//multiply
	__forceinline void multiply(double f) {
		data *= f; next.multiply(f);
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

	void normalizeRaw_s(const vec<dim, T>& Default) {
		double m = mag();
		if (m == 0) copy(Default);
		else multiply(1 / m);
	}


	//commulative multiplication
	__forceinline void multiply(const vec<dim, T>& other) { data *= other.data; next.multiply(other.next); }


	//decisions
	__forceinline bool isNUL() const{
		if (data != 0)return false;//not null
		return next.isNUL();
	}

	__forceinline bool isEqual(const vec<dim,T>& other) const{
		if (data != other.data)return false;//not equal
		return next.isEqual(other.next);
	}

	///Static functions

	//add + subtract
	static vec<dim, T> add(const vec<dim, T>& A, const vec<dim, T>& B) { vec<dim, T> rVal = A; rVal.add(B); return rVal; }
	static vec<dim, T> subtract(const vec<dim, T>& A, const vec<dim, T>& B) { vec<dim, T> rVal = A; rVal.subtract(B); return rVal; }

	//multiply
	static vec<dim, T> multiply(const vec<dim, T>& A, double f) {
		vec<dim, T> rVal = A;
		rVal.multiply(f);
		return rVal;
	}

	//dot
	static __forceinline T dot(const vec<dim, T>& A, const vec<dim, T>& B) {
		return ((A.data * B.data) + vec<dim - 1, T>::dot(A.next, B.next));
	}

	//commulative multiplication
	static vec<dim, T> commulativeMul(const vec<dim, T>& A, const vec<dim, T>& B) {
		vec<dim, T> rVal = A;
		rVal.multiply(B);
		return rVal;
	}

	//normalize
	static vec<dim,T> normalize(vec<dim,T> a, bool* err) {
		*err = a.normalize();
		return a;
	}

	static vec<dim, T> normalizeRaw(vec<dim, T> a) {
		a.normalizeRaw();
		return a;
	}
	
	static vec<dim, T> normalizeRaw_s(vec<dim, T> a) {
		a.normalize();
		return a;
	}

	static vec<dim, T> normalizeRaw_s(vec<dim, T> a, const vec<dim, T>& Default) {
		a.normalizeRaw_s(Default);
		return a;
	}


	//component
	static double component(const vec<dim, T>& of, const vec<dim, T>& along, bool* err) {
		double m = along.mag();
		if (m == 0) {
			*err = true;
			return of.mag();
		}
		else {
			*err = false;
			return (dot(of, along) / m);
		}
	}

	static double componentRaw(const vec<dim, T>& of, const vec<dim, T>& along) {//no err checking
		return (dot(of, along) / along.mag());
	}

	static double componentRaw_s(const vec<dim, T>& of, const vec<dim, T>& along) {//safe
		double m = along.mag();
		if (m == 0) return of.mag();
		else return (dot(of, along) / m);
	}

	static double componentRaw_s(const vec<dim, T>& of, const vec<dim, T>& along, double Default) {//safe
		double m = along.mag();
		if (m == 0) return Default;
		else return (dot(of, along) / m);
	}

	//angle
	static double angle(const vec<dim,T>& a, const vec<dim,T>& b, bool* err) {//in radian
		double m = a.mag2() * b.mag2();
		if (m == 0) {
			*err = true;
			return 0;
		}
		else {
			*err = false;
			double dt = dot(a, b);
			return acos(sqrt((dt * dt) / m));
		}
	}

	static double angleRaw(const vec<dim, T>& a, const vec<dim, T>& b) {//no err checking
		double dt = dot(a, b);
		return acos(sqrt((dt * dt) / (a.mag2() * b.mag2())));
	}

	static double angleRaw_s(const vec<dim, T>& a, const vec<dim, T>& b) {//safe
		double m = a.mag2() * b.mag2();
		if (m == 0) return 0;
		else {
			double dt = dot(a, b);
			return acos(sqrt((dt * dt) / m));
		}
	}

	static double angleRaw_s(const vec<dim, T>& a, const vec<dim, T>& b, double Default) {//safe
		double m = a.mag2() * b.mag2();
		if (m == 0) return Default;
		else {
			double dt = dot(a, b);
			return acos(sqrt((dt * dt) / m));
		}
	}

	//decisions
	static bool isNUL(const vec<dim, T>& a) {
		return a.isNUL();
	}
	static bool isEqual(const vec<dim,T>& a, const vec<dim,T>& b) {
		return a.isEqual(b);
	}

	//operators
	vec<dim,T> operator + (const vec<dim,T>& other)const { return add(*this, other); }

	void operator += (const vec<dim, T>& other) {(*this).add(other);}

	vec<dim,T> operator - (const vec<dim,T>& other)const { return subtract(*this, other); }

	vec<dim,T> operator - ()const { return multiply(*this, -1); }

	void operator -= (const vec<dim,T>& other) { (*this).subtract(other); }

	vec<dim,T> operator * (const double& other)const { return multiply(*this, other); }

	void operator *= (const double& other) { (*this).multiply(other); }

	vec<dim,T> operator * (const vec<dim,T>& other)const { return commulativeMul(*this, other); }

	void operator *= (const vec<dim,T>& other) { (*this).multiply(other); }

	vec<dim,T> operator / (const double& other)const { return multiply(*this, 1 / other); }

	void operator /= (const double& other) { (*this).multiply(1 / other); }

	bool operator == (const vec<dim,T>& other)const { return isEqual(*this, other); }

	bool operator != (const vec<dim,T>& other)const { return !isEqual(*this, other); }


	//misc
	template<bool first = true>
	void print(std::ostream& f) const{
		if (first)f << '[';
		f << data<<',';
		next.print<false>(f);
	}

	void insert(std::istream& f) {
		f >> data;
		next.insert(f);
		
	}
};



template<class T>
class vec<1,T>{
private:
	T data;
public:


	__forceinline void copy(const vec<1, T>& other) { data = other.data; }

	template<class N>
	__forceinline void castCopy(const vec<1, N>& other) { data = other.data; }

	vec() :data(T()) {}

	vec(T val) : data(val){};

	vec(const vec<1, T>& other) { copy(other); }
	
	template<class N>
	vec(const vec<1, N>& other) { castCopy(other); }



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

	//add + subtract
	__forceinline void add(const vec<1, T>& other) { data += other.data; }
	__forceinline void subtract(const vec<1, T>& other) { data -= other.data; }

	//multiply
	__forceinline void multiply(double f) {
		data *= f;
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

	void normalizeRaw_s(const vec<1, T>& Default) {
		if (data == 0) copy(Default);
		else normalizeRaw();
	}

	

	//commulative multiplication
	__forceinline void multiply(const vec<1, T>& other) { data *= other.data; }

	//decisions
	__forceinline bool isNUL() const{
		if (data != 0)return false;//not null
		return true;
	}
	
	__forceinline bool isEqual(const vec<1, T>& other) const{
		if (data != other.data)return false;//not equal
		return true;
	}

	///Static functions

	//add + subtract
	static vec<1, T> add(const vec<1, T>& A, const vec<1, T>& B) { vec<1, T> rVal = A; rVal.add(B); return rVal; }
	static vec<1, T> subtract(const vec<1, T>& A, const vec<1, T>& B) { vec<1, T> rVal = A; rVal.subtract(B); return rVal; }

	//multiply
	static vec<1, T> multiply(const vec<1, T>& A, double f) {
		vec<1, T> rVal = A;
		rVal.multiply(f);
		return rVal;
	}

	//dot
	static __forceinline T dot(const vec<1, T>& A, const vec<1, T>& B) {
		return ((A.data * B.data));
	}

	//commulative multiplication
	static vec<1, T> commulativeMul(const vec<1, T>& A, const vec<1, T>& B) {
		vec<1, T> rVal = A;
		rVal.multiply(B);
		return rVal;
	}

	//normalize
	static vec<1, T> normalize(vec<1, T> a, bool* err) {
		*err = a.normalize();
		return a;
	}

	static vec<1, T> normalizeRaw(vec<1, T> a) {
		a.normalizeRaw();
		return a;
	}

	static vec<1, T> normalizeRaw_s(vec<1, T> a) {
		a.normalize();
		return a;
	}

	static vec<1, T> normalizeRaw_s(vec<1, T> a, const vec<1, T>& Default) {
		a.normalizeRaw_s(Default);
		return a;
	}

	//component
	static double component(const vec<1, T>& of, const vec<1, T>& along, bool* err) {
		double m = along.mag();
		if (m == 0) {
			*err = true;
			return of.mag();
		}
		else {
			*err = false;
			return (dot(of, along) / m);
		}
	}

	static double componentRaw(const vec<1, T>& of, const vec<1, T>& along) {//no err checking
		return (dot(of, along) / along.mag());
	}

	static double componentRaw_s(const vec<1, T>& of, const vec<1, T>& along) {//safe
		double m = along.mag();
		if (m == 0) return of.mag();
		else return (dot(of, along) / m);
	}

	static double componentRaw_s(const vec<1, T>& of, const vec<1, T>& along, double Default) {//safe
		double m = along.mag();
		if (m == 0) return Default;
		else return (dot(of, along) / m);
	}

	//angle
	static double angle(const vec<1, T>& a, const vec<1, T>& b, bool* err) {//in radian
		double m = a.mag() * b.mag();
		if (m == 0) {
			*err = true;
			return 0;
		}
		else {
			*err = false;
			if (m > 0)return 0;
			else return math3D_pi;
		}
	}

	static double angleRaw(const vec<1, T>& a, const vec<1, T>& b) {//no err checking
		double m = a.mag() * b.mag();
		if (m > 0) return 0;
		else return math3D_pi;
		
	}

	static double angleRaw_s(const vec<1, T>& a, const vec<1, T>& b) {//safe
		double m = a.mag() * b.mag();
		if (m >= 0) return 0;
		else return math3D_pi;
	}

	static double angleRaw_s(const vec<1, T>& a, const vec<1, T>& b, double Default) {//safe
		double m = a.mag() * b.mag();
		if (m == 0) {
			return Default;
		}
		else {
			if (m > 0)return 0;
			else return math3D_pi;
		}
	}

	//decisions
	static bool isNUL(const vec<1, T>& a) {
		return a.isNUL();
	}
	static bool isEqual(const vec<1, T>& a, const vec<1, T>& b) {
		return a.isEqual(b);
	}


	//operators
	vec<1, T> operator + (const vec<1, T>& other)const { return add(*this, other); }

	void operator += (const vec<1, T>& other) { (*this).add(other); }

	vec<1, T> operator - (const vec<1, T>& other)const { return subtract(*this, other); }

	vec<1, T> operator - ()const { return multiply(*this, -1); }

	void operator -= (const vec<1, T>& other) { (*this).subtract(other); }

	vec<1, T> operator * (const double& other)const { return multiply(*this, other); }

	void operator *= (const double& other) { (*this).multiply(other); }

	vec<1, T> operator * (const vec<1, T>& other)const { return commulativeMul(*this, other); }

	void operator *= (const vec<1, T>& other) { (*this).multiply(other); }

	vec<1, T> operator / (const double& other)const { return multiply(*this, 1 / other); }

	void operator /= (const double& other) { (*this).multiply(1 / other); }

	bool operator == (const vec<1, T>& other)const { return isEqual(*this, other); }

	bool operator != (const vec<1, T>& other)const { return !isEqual(*this, other); }


	//misc
	template<bool first = true>
	void print(std::ostream& f) const{
		if (first)f << '[';
		f << data;
		f << ']';
	}

	void insert(std::istream& f) {
		f >> data;
	}
};

//global operators

template <unsigned char dim,typename T>
vec<dim,T> operator * (const double& other, const vec<dim,T>& VEC) { return vec<dim,T>::multiply(VEC, other); }


//misc operators
template<unsigned char dim, typename T>
std::ostream& operator << (std::ostream& f, const vec<dim, T>& vec) {
	vec.print(f);
	return f;
}

//misc operators
template<unsigned char dim, typename T>
std::istream& operator >> (std::istream& f, vec<dim, T>& vec) {
	vec.insert(f);
	return f;
}