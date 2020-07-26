
#include "vec.cuh"


template<typename T>
union vec3S
{
	vec<3, T> V;
	struct {
		T x,y,z;
	};

	operator vec<3, T>&() {
		return V;
	}

	vec3S<T>():x(0),y(0),z(0) {};

	static vec3S cross(const vec3S& a, const vec3S& b) {
		vec3S rVal;
		rVal.x = a.y * b.z - a.z * b.y;
		rVal.y = a.z * b.x - a.x * b.z;
		rVal.z = a.x * b.y - a.y * b.x;
		return
	}
};