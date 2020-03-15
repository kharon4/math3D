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

	template<typename N>
	void convert(vec3<N> in) {
		x = (T)in.x;
		y = (T)in.y;
		z = (T)in.z;
	}
};

//declaration macro
#define declareVec3(type , name) typedef vec3<type> name;

//defination macro
#define defineVec3(type , name , val)type name::dVal = val;\
void init_##name(){\
name obj;\
name obj1(name::dVal, name::dVal, name::dVal);\
obj.mag();\
obj.mag2();\
obj.normalize();\
obj.multiply(1);\
name::dot(name() ,name());\
name::cross(name(),name());\
name::multiply(name(),1);\
name::add(name(),name());\
name::subtract(name(),name());\
name::angle(name(),name());\
name::component(name(),name());\
name::normalize(name());\
name::isNUL(name());\
name::isEqual(name(),name());\
}


//declare vec3 types here
declareVec3(double , vec3d)
declareVec3(long double , vec3ld)
declareVec3(float , vec3f)
declareVec3(int , vec3i)
declareVec3(unsigned int , vec3ui)
declareVec3(long , vec3l)
declareVec3(unsigned long , vec3ul)
declareVec3(long long , vec3ll)
declareVec3(unsigned long long , vec3ull)
declareVec3(short, vec3s)
declareVec3(unsigned short, vec3us)
declareVec3(char, vec3c)
declareVec3(unsigned char, vec3uc)

