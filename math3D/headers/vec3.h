#pragma once
enum errCodes { noErr = 0, devideByZero = 1 };


static errCodes defaultErrCode = errCodes::noErr;


template <typename T>
class vec3
{
public:
	static T dVal;
	T x, y, z;
	vec3();
	vec3(T, T, T);
	double mag();
	double mag2();
	bool normalize();
	void multiply(double);

	static double dot(vec3<T> a, vec3<T> b);//dot product

	static vec3<T> cross(vec3<T> a, vec3<T> b);//cross product

	static vec3<T> multiply(vec3<T> a, double factor);

	static vec3<T> add(vec3<T> a, vec3<T> b);

	static vec3<T> subtract(vec3<T> a, vec3<T> b);//a-b

	static double angle(vec3<T> a, vec3<T> b, errCodes* err = &defaultErrCode);//in radian

	static double component(vec3<T> of, vec3<T> along, errCodes* err = &defaultErrCode);

	static vec3<T> normalize(vec3<T> a, errCodes* err = &defaultErrCode);

	static bool isNUL(vec3<T> a);

	static bool isEqual(vec3<T> a, vec3<T> b);
};


typedef vec3<long double> vec3ld;

typedef vec3<double> vec3d;

typedef vec3<float> vec3f;

typedef vec3<int> vec3i;

typedef vec3<unsigned int> vec3ui;

typedef vec3<long> vec3l;

typedef vec3<unsigned long> vec3ul;

typedef vec3<long long> vec3ll;

typedef vec3<unsigned long long> vec3ull;

typedef vec3<short> vec3s;

typedef vec3<unsigned short> vec3us;

typedef vec3<char> vec3c;

typedef vec3<unsigned char> vec3uc;
