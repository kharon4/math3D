
#include "rotation.cuh"

namespace manipulation3d{

template<typename T>
class transform {
private:
	std::vector <vec3<T>> data;
	std::vector <vec3<T>*> dataAddress;
public:
	manipulation3d::coordinateSystem CS;

	__host__ transform() {};
	__host__ void addVec(vec3<T> val, vec3<T>* address) {
		data.push_back(CS.getInCoordinateSystem(val));
		dataAddress.push_back(address);
	}
	__host__ void update() {
		for (int i = 0; i < data.size(); ++i) *(dataAddress[i]) = CS.getRealWorldCoordinates(data[i]);
	}

	std::vector<vec3<T>>* getData() { return &data; }
};

typedef transform<double> transformd;
typedef transform<float> transformf;
}