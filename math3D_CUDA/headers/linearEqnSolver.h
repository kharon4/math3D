#pragma once

#include<vector>

namespace LES {//Linear equation solver

	class eqn {
	public:
		std::vector<double> coefficients;
		double constant = 0;
	};


	class system {
	public:
		enum solType : unsigned char { uniqueSol = 0, inconsistant = 1, infiniteSols = 2 };
		enum sysType : unsigned char { criticallyDefined = 0, underDefined = 1, overDefined = 2 };
		solType solutionType;
		sysType systemType;

		std::vector<double> sols;
		eqn* eqns = nullptr;

		unsigned long int noEqns = 0;
		unsigned long int noCoef = 0;

		system();
		~system();
	};

}
