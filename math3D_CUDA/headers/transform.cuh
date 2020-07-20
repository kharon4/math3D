
#include "rotation.cuh"

namespace manipulation3d{

template<typename T>
class transform {
private:
	std::vector <vec3<T>> data;
	std::vector <vec3<T>*> dataAddress;
	std::vector <coordinateSystem::transformationType> types;
public:
	manipulation3d::coordinateSystem CS;

	__host__ transform() {};
	__host__ void addVec(vec3<T> val, vec3<T>* address , coordinateSystem::transformationType type = coordinateSystem::transformationType::all) {
		data.push_back(CS.getInCoordinateSystem(val,type));
		dataAddress.push_back(address);
		types.push_back(type);
	}
	__host__ void update() {
		for (int i = 0; i < data.size(); ++i) *(dataAddress[i]) = CS.getRealWorldCoordinates(data[i],types[i]);
	}

	std::vector<vec3<T>>* getData() { return &data; }
};

typedef transform<double> transformd;
typedef transform<float> transformf;


template<typename T>
class transformExternal {
private:
	std::vector <vec3<T>> data;
	std::vector <vec3<T>*> dataAddress;
	std::vector <coordinateSystem::transformationType> types;
public:
	manipulation3d::coordinateSystem &CS;

	__host__ transformExternal(coordinateSystem& system):CS(system) {};
	__host__ void addVec(vec3<T> val, vec3<T>* address, coordinateSystem::transformationType type = coordinateSystem::transformationType::all) {
		data.push_back(CS.getInCoordinateSystem(val, type));
		dataAddress.push_back(address);
		types.push_back(type);
	}
	__host__ void update() {
		for (int i = 0; i < data.size(); ++i) *(dataAddress[i]) = CS.getRealWorldCoordinates(data[i], types[i]);
	}

	std::vector<vec3<T>>* getData() { return &data; }
};

typedef transformExternal<double> transformEd;
typedef transformExternal<float> transformEf;

}