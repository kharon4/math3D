#pragma once



class vec3
{
public:
	enum errCodes{noErr = 0 , devideByZero = 1};

	static errCodes errCode;
	double x, y, z;
	vec3(double, double, double);
	double mag();
	double mag2();
	bool normalize();
	void multiply(double);
};


//non class functions

double dot(vec3 a, vec3 b);//dot product

vec3 cross(vec3 a, vec3 b);//cross product

vec3 multiply(vec3 a, double factor);

vec3 add(vec3 a, vec3 b);

vec3 subtract(vec3 a, vec3 b);//a-b

double angle(vec3 a, vec3 b, vec3::errCodes* err = &vec3::errCode);//in radian

double component(vec3 of, vec3 along, vec3::errCodes* err = &vec3::errCode);

vec3 normalize(vec3 a, vec3::errCodes* err = &vec3::errCode);

bool isNUL(vec3 a);

bool isEqual(vec3 a, vec3 b);