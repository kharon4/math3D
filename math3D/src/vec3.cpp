#include "../headers/vec3.h"

#include<math.h>


template<typename T>
T vec3<T>::dVal;

//class procedures

template <typename T>
vec3<T>::vec3()
{
	x = dVal;
	y = dVal;
	z = dVal;
}

template <typename T>
vec3<T>::vec3(T X, T Y, T Z)
{
	x = X;
	y = Y;
	z = Z;
}


template <typename T>
double vec3<T>::mag()
{
	return sqrt(x * x + y * y + z * z);
}

template <typename T>
double vec3<T>::mag2()
{
	return (x * x + y * y + z * z);
}


template <typename T>
bool vec3<T>::normalize() {
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
void vec3<T>::multiply(double factor) {
	x *= factor;
	y *= factor;
	z *= factor;
}

//vec functions

template <typename T>
double vec3<T>::dot(vec3<T> a, vec3<T> b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

template <typename T>
vec3<T> vec3<T>::cross(vec3<T> a, vec3<T> b) {
	return vec3<T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

template <typename T>
vec3<T> vec3<T>::multiply(vec3<T> a, double factor) {
	return vec3<T>(a.x * factor, a.y * factor, a.z * factor);
}

template <typename T>
vec3<T> vec3<T>::add(vec3<T> a, vec3<T> b) {
	return vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template <typename T>
vec3<T> vec3<T>::subtract(vec3<T> a, vec3<T> b) {
	return vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}

template <typename T>
double vec3<T>::angle(vec3<T> a, vec3<T> b , errCodes* err) {//in radian
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
double vec3<T>::component(vec3<T> of, vec3<T> along , errCodes* err) {
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
vec3<T> vec3<T>::normalize(vec3<T> a, errCodes* err) {
	(*err) = errCodes::noErr;
	if (a.normalize()) {
		(*err) = errCodes::devideByZero;
	}
	return a;
}

template <typename T>
bool vec3<T>::isNUL(vec3<T> a) {
	if ((a.x == 0) && (a.y == 0) && (a.z == 0)) {
		return 1;
	}
	else {
		return 0;
	}
}

template <typename T>
bool vec3<T>::isEqual(vec3<T> a, vec3<T> b) {
	if ((a.x == b.x) && (a.y == b.y) && (a.z == b.z)) {
		return 1;
	}
	else {
		return 0;
	}
}


//specify all the types here
long double vec3ld::dVal = 0;
double vec3d::dVal = 0;
float vec3f::dVal = 0;
int vec3i::dVal = 0;
unsigned int vec3ui::dVal = 0;
long vec3l::dVal = 0;
unsigned long vec3ul::dVal = 0;
long long vec3ll::dVal = 0;
unsigned long long vec3ull::dVal = 0;
short vec3s::dVal = 0;
unsigned short vec3us::dVal = 0;
char vec3c::dVal = 0;
unsigned char vec3uc::dVal = 0;

void init() {
	{
		vec3d d;
		vec3c c;
		vec3f f;
		vec3i i;
		vec3l l;
		vec3ld ld;
		vec3ll ll;
		vec3s s;
		vec3uc uc;
		vec3ui ui;
		vec3ul ul;
		vec3ull ull;
		vec3us us;

		d.mag();
		c.mag();
		f.mag();
		i.mag();
		l.mag();
		ld.mag();
		ll.mag();
		s.mag();
		uc.mag();
		ui.mag();
		ul.mag();
		ull.mag();
		us.mag();

		d.mag2();
		c.mag2();
		f.mag2();
		i.mag2();
		l.mag2();
		ld.mag2();
		ll.mag2();
		s.mag2();
		uc.mag2();
		ui.mag2();
		ul.mag2();
		ull.mag2();
		us.mag2();
		
		d.normalize();
		c.normalize();
		f.normalize();
		i.normalize();
		l.normalize();
		ld.normalize();
		ll.normalize();
		s.normalize();
		uc.normalize();
		ui.normalize();
		ul.normalize();
		ull.normalize();
		us.normalize();

		d.multiply(1);
		c.multiply(1);
		f.multiply(1);
		i.multiply(1);
		l.multiply(1);
		ld.multiply(1);
		ll.multiply(1);
		s.multiply(1);
		uc.multiply(1);
		ui.multiply(1);
		ul.multiply(1);
		ull.multiply(1);
		us.multiply(1);
	} {
		vec3d d(0,0,0);
		vec3c c(0, 0, 0);
		vec3f f(0, 0, 0);
		vec3i i(0, 0, 0);
		vec3l l(0, 0, 0);
		vec3ld ld(0, 0, 0);
		vec3ll ll(0, 0, 0);
		vec3s s(0, 0, 0);
		vec3uc uc(0, 0, 0);
		vec3ui ui(0, 0, 0);
		vec3ul ul(0, 0, 0);
		vec3ull ull(0, 0, 0);
		vec3us us(0, 0, 0);
	} {
		vec3d::dot(vec3d(), vec3d());
		vec3d::cross(vec3d(), vec3d());
		vec3d::multiply(vec3d(), 1);
		vec3d::add(vec3d(), vec3d());
		vec3d::subtract(vec3d(), vec3d());
		vec3d::angle(vec3d(), vec3d());
		vec3d::component(vec3d(), vec3d());
		vec3d::normalize(vec3d());
		vec3d::isNUL(vec3d());
		vec3d::isEqual(vec3d(), vec3d());
	} {
		vec3c::dot(vec3c(), vec3c());
		vec3c::cross(vec3c(), vec3c());
		vec3c::multiply(vec3c(), 1);
		vec3c::add(vec3c(), vec3c());
		vec3c::subtract(vec3c(), vec3c());
		vec3c::angle(vec3c(), vec3c());
		vec3c::component(vec3c(), vec3c());
		vec3c::normalize(vec3c());
		vec3c::isNUL(vec3c());
		vec3c::isEqual(vec3c(), vec3c());
	} {
		vec3f::dot(vec3f(), vec3f());
		vec3f::cross(vec3f(), vec3f());
		vec3f::multiply(vec3f(), 1);
		vec3f::add(vec3f(), vec3f());
		vec3f::subtract(vec3f(), vec3f());
		vec3f::angle(vec3f(), vec3f());
		vec3f::component(vec3f(), vec3f());
		vec3f::normalize(vec3f());
		vec3f::isNUL(vec3f());
		vec3f::isEqual(vec3f(), vec3f());
	} {
		vec3i::dot(vec3i(), vec3i());
		vec3i::cross(vec3i(), vec3i());
		vec3i::multiply(vec3i(), 1);
		vec3i::add(vec3i(), vec3i());
		vec3i::subtract(vec3i(), vec3i());
		vec3i::angle(vec3i(), vec3i());
		vec3i::component(vec3i(), vec3i());
		vec3i::normalize(vec3i());
		vec3i::isNUL(vec3i());
		vec3i::isEqual(vec3i(), vec3i());
	} {
		vec3l::dot(vec3l(), vec3l());
		vec3l::cross(vec3l(), vec3l());
		vec3l::multiply(vec3l(), 1);
		vec3l::add(vec3l(), vec3l());
		vec3l::subtract(vec3l(), vec3l());
		vec3l::angle(vec3l(), vec3l());
		vec3l::component(vec3l(), vec3l());
		vec3l::normalize(vec3l());
		vec3l::isNUL(vec3l());
		vec3l::isEqual(vec3l(), vec3l());
	} {
		vec3ld::dot(vec3ld(), vec3ld());
		vec3ld::cross(vec3ld(), vec3ld());
		vec3ld::multiply(vec3ld(), 1);
		vec3ld::add(vec3ld(), vec3ld());
		vec3ld::subtract(vec3ld(), vec3ld());
		vec3ld::angle(vec3ld(), vec3ld());
		vec3ld::component(vec3ld(), vec3ld());
		vec3ld::normalize(vec3ld());
		vec3ld::isNUL(vec3ld());
		vec3ld::isEqual(vec3ld(), vec3ld());
	} {
		vec3ll::dot(vec3ll(), vec3ll());
		vec3ll::cross(vec3ll(), vec3ll());
		vec3ll::multiply(vec3ll(), 1);
		vec3ll::add(vec3ll(), vec3ll());
		vec3ll::subtract(vec3ll(), vec3ll());
		vec3ll::angle(vec3ll(), vec3ll());
		vec3ll::component(vec3ll(), vec3ll());
		vec3ll::normalize(vec3ll());
		vec3ll::isNUL(vec3ll());
		vec3ll::isEqual(vec3ll(), vec3ll());
	} {
		vec3s::dot(vec3s(), vec3s());
		vec3s::cross(vec3s(), vec3s());
		vec3s::multiply(vec3s(), 1);
		vec3s::add(vec3s(), vec3s());
		vec3s::subtract(vec3s(), vec3s());
		vec3s::angle(vec3s(), vec3s());
		vec3s::component(vec3s(), vec3s());
		vec3s::normalize(vec3s());
		vec3s::isNUL(vec3s());
		vec3s::isEqual(vec3s(), vec3s());
	} {
		vec3uc::dot(vec3uc(), vec3uc());
		vec3uc::cross(vec3uc(), vec3uc());
		vec3uc::multiply(vec3uc(), 1);
		vec3uc::add(vec3uc(), vec3uc());
		vec3uc::subtract(vec3uc(), vec3uc());
		vec3uc::angle(vec3uc(), vec3uc());
		vec3uc::component(vec3uc(), vec3uc());
		vec3uc::normalize(vec3uc());
		vec3uc::isNUL(vec3uc());
		vec3uc::isEqual(vec3uc(), vec3uc());
	} {
		vec3ui::dot(vec3ui(), vec3ui());
		vec3ui::cross(vec3ui(), vec3ui());
		vec3ui::multiply(vec3ui(), 1);
		vec3ui::add(vec3ui(), vec3ui());
		vec3ui::subtract(vec3ui(), vec3ui());
		vec3ui::angle(vec3ui(), vec3ui());
		vec3ui::component(vec3ui(), vec3ui());
		vec3ui::normalize(vec3ui());
		vec3ui::isNUL(vec3ui());
		vec3ui::isEqual(vec3ui(), vec3ui());
	} {
		vec3ul::dot(vec3ul(), vec3ul());
		vec3ul::cross(vec3ul(), vec3ul());
		vec3ul::multiply(vec3ul(), 1);
		vec3ul::add(vec3ul(), vec3ul());
		vec3ul::subtract(vec3ul(), vec3ul());
		vec3ul::angle(vec3ul(), vec3ul());
		vec3ul::component(vec3ul(), vec3ul());
		vec3ul::normalize(vec3ul());
		vec3ul::isNUL(vec3ul());
		vec3ul::isEqual(vec3ul(), vec3ul());
	} {
		vec3ull::dot(vec3ull(), vec3ull());
		vec3ull::cross(vec3ull(), vec3ull());
		vec3ull::multiply(vec3ull(), 1);
		vec3ull::add(vec3ull(), vec3ull());
		vec3ull::subtract(vec3ull(), vec3ull());
		vec3ull::angle(vec3ull(), vec3ull());
		vec3ull::component(vec3ull(), vec3ull());
		vec3ull::normalize(vec3ull());
		vec3ull::isNUL(vec3ull());
		vec3ull::isEqual(vec3ull(), vec3ull());
	} {
		vec3us::dot(vec3us(), vec3us());
		vec3us::cross(vec3us(), vec3us());
		vec3us::multiply(vec3us(), 1);
		vec3us::add(vec3us(), vec3us());
		vec3us::subtract(vec3us(), vec3us());
		vec3us::angle(vec3us(), vec3us());
		vec3us::component(vec3us(), vec3us());
		vec3us::normalize(vec3us());
		vec3us::isNUL(vec3us());
		vec3us::isEqual(vec3us(), vec3us());
	}
}
