#include "../headers/vec3.h"

#include<math.h>


//clss static variables
vec3::errCodes vec3::errCode = vec3::errCodes::noErr;

//class procedures

vec3::vec3(double X = 0, double Y = 0, double Z = 0)
{
	x = X;
	y = Y;
	z = Z;
}

double vec3::mag()
{
	return sqrt(x * x + y * y + z * z);
}

double vec3::mag2()
{
	return (x * x + y * y + z * z);
}

bool vec3::normalize() {
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

void vec3::multiply(double factor) {
	x *= factor;
	y *= factor;
	z *= factor;
}

//vec functions

double dot(vec3 a, vec3 b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

vec3 cross(vec3 a, vec3 b) {
	return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

vec3 multiply(vec3 a, double factor) {
	return vec3(a.x * factor, a.y * factor, a.z * factor);
}

vec3 add(vec3 a, vec3 b) {
	return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

vec3 subtract(vec3 a, vec3 b) {
	return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

double angle(vec3 a, vec3 b , vec3::errCodes* err) {//in radian
	(*err) = vec3::noErr;
	double m = a.mag() * b.mag();
	if (m == 0)
	{
		(*err) = vec3::devideByZero;
		return 0;
	}
	else
	{
		return acos(dot(a, b) / m);
	}

}

double component(vec3 of, vec3 along , vec3::errCodes* err) {
	(*err) = vec3::noErr;
	double m = along.mag();
	if (m == 0)
	{
		(*err) = vec3::devideByZero;
		return of.mag();
	}
	else
	{
		return (dot(of, along) / m);
	}
}

vec3 normalize(vec3 a, vec3::errCodes* err) {
	(*err) = vec3::errCodes::noErr;
	if (a.normalize()) {
		(*err) = vec3::errCodes::devideByZero;
	}
	return a;
}

bool isNUL(vec3 a) {
	if ((a.x == 0) && (a.y == 0) && (a.z == 0)) {
		return 1;
	}
	else {
		return 0;
	}
}

bool isEqual(vec3 a, vec3 b) {
	if ((a.x == b.x) && (a.y == b.y) && (a.z == b.z)) {
		return 1;
	}
	else {
		return 0;
	}
}